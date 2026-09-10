#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""The pre-comparison power gate for H4, written before any quantum arm is scored.

The pre-registration (docs/protocol.md, H4) commits to declaring the tensor-network
comparison underpowered *in advance* if the minimum detectable effect exceeds 0.023 average
precision.  The point of a gate is that it binds: running the comparison first and then
computing the MDE would let the observed result influence whether the test is deemed
informative, which is the thing the gate exists to prevent.

The standard deviation the MDE needs must therefore be estimated without looking at the
quantum arm.  Two proxies are computed, and the binding one is the larger.

**Within-family** -- two gradient-boosted models differing only in random seed.  This is the
floor: the irreducible noise of scoring the same rows twice.

**Cross-family** -- a regularised logistic model against the same gradient-boosted baseline.
This is the one that binds, and the first version of this script did not have it.  Using the
within-family figure alone understated the noise of the comparison H4 actually makes by
about eightfold (standard error 0.0025 against a measured 0.020), because two seeds of one
model family disagree far less than two different families do.  A gate calibrated on
seed noise passes almost anything.  Neither proxy involves a quantum model or touches
``D_test``.  See ``docs/decisions.md`` D-030.

    .venv/bin/python scripts/run_power.py

Writes ``results/tables/power.csv``.  ``scripts/run_mps.py`` refuses to run the H4
comparison until this file exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hsbcfraud.config import load_config
from hsbcfraud.data.ieee_cis import IEEE_CIS_ZIP, load_ieee_cis

# The gate must be computed on the same rows, features and scaling as the comparison it
# gates, or it prices a different experiment.  This script used to carry its own copy of the
# band pipeline; the copies agreed, which is exactly what would have made a later divergence
# invisible -- run_mps.py reads power.csv and refuses H4 without it.
from hsbcfraud.features.band import band_edges, prepare, rows_in_band, select_band_features
from hsbcfraud.paths import display_path, require_run_artefact
from hsbcfraud.progress import run_log
from hsbcfraud.stats import (
    clustered_bootstrap_difference,
    minimum_detectable_effect_from_standard_error,
)

REPO = Path(__file__).resolve().parents[1]

# The pre-registered ceiling.  An MDE above this means the comparison cannot resolve an
# effect of the size that would matter, and H4 is reported as underpowered rather than null.
MDE_CEILING = 0.023


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--zip", type=Path, default=REPO / "datasets" / IEEE_CIS_ZIP)
    parser.add_argument("--runs", type=Path, default=REPO / "results" / "runs")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "tables")
    parser.add_argument("--arm", default="temporal")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--band-features", type=int, default=8)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)

    import xgboost as xgb

    cfg = load_config(args.config)
    # `or` treats --seed 0 as absent and silently substitutes the configured seed,
    # which is then written into the output's `seed` column as though it were asked for.
    seed = cfg.split.seeds[0] if args.seed is None else args.seed
    scores = pd.read_parquet(
        require_run_artefact(
            args.runs / f"scores_{args.arm}_{seed}.parquet", produced_by="baseline"
        )
    )
    # Wrapped because the expensive parts reported nothing.  The IEEE-CIS load is 590,540
    # rows with identity columns, and the bootstrap that follows resamples the evaluation
    # block thousands of times; until both finished the script was indistinguishable from a
    # hang.  `run_log` writes to results/runs/ and to stdout, so a redirected run is still
    # followable, and its `info` calls carry the timestamps a bare `print` does not.
    with run_log("run_power", directory=args.runs) as run:
        run.info(f"loading {args.zip.name}")
        loaded = load_ieee_cis(args.zip, None, with_identity=True)
        frame = loaded.frame
        for column in frame.columns:
            if frame[column].dtype == "object" or str(frame[column].dtype) == "str":
                frame[column] = frame[column].astype("category").cat.codes.astype("int32")

        all_numeric = [
            c
            for c in frame.select_dtypes(include=[np.number]).columns
            if c not in {"isFraud", "day", "TransactionDT", "TransactionID"}
        ]
        y = frame["isFraud"].to_numpy()
        train_rows = scores[scores["block"] == "train"]["row"].to_numpy()

        band_block = scores[scores["block"] == "band"]
        lo, hi = band_edges(
            band_block["score"].to_numpy(), cfg.decline_rate_budget, cfg.band.traffic_budget
        )

        band_columns = select_band_features(
            frame, train_rows, y, all_numeric, k=args.band_features, seed=seed
        )

        fit_rows = rows_in_band(scores, "band", lo, hi)
        eval_rows = rows_in_band(scores, "cal", lo, hi)

        x_fit, scaler = prepare(frame, fit_rows, band_columns, None)
        x_eval, _ = prepare(frame, eval_rows, band_columns, scaler)
        y_fit, y_eval = y[fit_rows], y[eval_rows]
        clusters = frame.iloc[eval_rows]["card1"].to_numpy()

        run.info(
            f"power gate for H4: {len(x_fit):,} fit rows, {len(x_eval):,} evaluation rows "
            f"({int(y_eval.sum())} fraud, {len(np.unique(clusters)):,} card clusters)"
        )

        def gbdt(model_seed: int) -> np.ndarray:
            model = xgb.XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                tree_method="hist",
                device="cuda",
                eval_metric="aucpr",
                random_state=model_seed,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=0,
            )
            model.fit(x_fit, y_fit)
            return model.predict_proba(x_eval)[:, 1]

        def logistic() -> np.ndarray:
            """A different inductive bias, fitted on the identical rows.

            Standardised because a penalised linear model is not scale-free; the scaler is
            fitted on the fit block only, exactly as the min-max scaler in ``features.band`` is.
            """
            pipeline = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            )
            pipeline.fit(x_fit, y_fit)
            return pipeline.predict_proba(x_eval)[:, 1]

        baseline = gbdt(seed)
        within = clustered_bootstrap_difference(
            average_precision_score, y_eval, baseline, gbdt(seed + 1), clusters,
            n_resamples=args.resamples, seed=seed,
        )
        across = clustered_bootstrap_difference(
            average_precision_score, y_eval, logistic(), baseline, clusters,
            n_resamples=args.resamples, seed=seed,
        )
        print(
            f"  within-family (two GBDT seeds):   dAP {within.point:+.4f} "
            f"[{within.low:+.4f}, {within.high:+.4f}]  SE {within.spread:.4f}"
        )
        print(
            f"  cross-family (logistic vs GBDT):  dAP {across.point:+.4f} "
            f"[{across.low:+.4f}, {across.high:+.4f}]  SE {across.spread:.4f}"
        )
        # The gate must be calibrated on the noisier of the two.  H4 compares model families,
        # not seeds, so the within-family figure is a floor and not a description.
        reference = within if within.spread >= across.spread else across
        binding = "within-family" if reference is within else "cross-family"
        print(f"  binding proxy: {binding}")
        # The bootstrap spread is already a standard error -- the sample size and the card
        # clustering are inside it -- so it must not be divided by sqrt(n) a second time.
        standard_error = reference.spread
        mde = minimum_detectable_effect_from_standard_error(standard_error)
        powered = mde <= MDE_CEILING

        run.info(
            f"  standard error {standard_error:.4f}  ->  "
            f"MDE {mde:.4f} AP at 80% power, alpha 0.05"
        )
        print(f"  pre-registered ceiling {MDE_CEILING:.4f}: {'PASS' if powered else 'FAIL'}")
        if not powered:
            print(
                "  H4 is declared underpowered in advance. A null result from the comparison\n"
                "  will be reported as uninformative, not as evidence of no effect."
            )

        args.out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "hypothesis": "H4",
                    "comparison": (
                        "band-conditional average precision, tensor network vs GBDT"
                    ),
                    "n_eval": len(y_eval),
                    "n_fraud": int(y_eval.sum()),
                    "n_clusters": len(np.unique(clusters)),
                    "noise_source": binding,
                    "se_within_family": within.spread,
                    "se_cross_family": across.spread,
                    "observed_difference": reference.point,
                    "ci_low": reference.low,
                    "ci_high": reference.high,
                    "n_resamples_usable": reference.n_resamples,
                    "standard_error_paired_difference": standard_error,
                    "alpha": 0.05,
                    "power": 0.80,
                    "minimum_detectable_effect": mde,
                    "preregistered_ceiling": MDE_CEILING,
                    "adequately_powered": powered,
                    "seed": seed,
                }
            ]
        ).to_csv(args.out / "power.csv", index=False)
        print(f"\nWrote {display_path(args.out / 'power.csv')}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
