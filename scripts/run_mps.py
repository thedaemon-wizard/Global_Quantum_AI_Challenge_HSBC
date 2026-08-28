#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""E10 -- the matrix-product-state arm, in the band and at full scale.

Writes ``results/tables/mps_band.csv`` (the in-band re-ranking comparison, which is the body
claim) and ``results/tables/mps_full.csv`` (a standalone classifier on all features, which is
the appendix claim).

Two experiments, because they answer different questions.

**In the band.**  The same band, the same features and the same evaluation as the quantum
kernel arm, so the three candidates -- MPS, quantum kernel, gradient boosting -- are directly
comparable.  The quantum kernel was rejected by its a-priori screens, so the honest question
is what a quantum-inspired model does in the place the kernel could not be used.

**At full scale.**  All features, no band, against the tuned gradient-boosted baseline.  A
literature check on 2026-08-28 found published matrix-product-state work on tabular data to be
largely *generative* -- synthetic-data modelling scored on fidelity and privacy -- rather than
discriminative against a tuned gradient-boosted baseline on imbalanced payment data.  That is
a statement about what was found, not about what exists.  The arm is affordable because an MPS
has no qubit ceiling: 431 sites is a long chain, not an intractable state space.

Both sweep the bond dimension rather than tuning it, so the capacity dependence is measured.

    .venv/bin/python scripts/run_mps.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from hsbcfraud.config import load_config
from hsbcfraud.data.ieee_cis import IEEE_CIS_ZIP, load_ieee_cis
from hsbcfraud.progress import ProgressReporter, SweepTimer
from hsbcfraud.quantum.mps import MPSConfig, fit_mps
from hsbcfraud.stats import clustered_bootstrap_difference, holm_bonferroni

REPO = Path(__file__).resolve().parents[1]


def require_power_gate(tables: Path) -> pd.DataFrame:
    """Refuse to run the H4 comparison until the power gate has been recorded.

    The pre-registration commits to deciding whether this comparison can resolve an effect
    *before* seeing its outcome.  Enforcing the order in code is the only version of that
    commitment that survives contact with a deadline: a gate that can be computed afterwards
    is not a gate, because by then the result is already known.
    """
    path = tables / "power.csv"
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. The H4 power gate is pre-registered to run before the "
            "comparison it governs.\n  Run: .venv/bin/python scripts/run_power.py"
        )
    return pd.read_csv(path)


def prepare(frame: pd.DataFrame, rows: np.ndarray, columns: list[str], scaler: MinMaxScaler | None):
    """Numeric matrix scaled to [0, 1], which is the domain the local feature map needs.

    The scaler is fitted once on the training block and reused everywhere else.  Fitting it
    per block would leak the deployment distribution into the encoding.
    """
    numeric = frame.iloc[rows][columns]
    filled = (
        numeric.fillna(numeric.median(numeric_only=True))
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    if scaler is None:
        scaler = MinMaxScaler(clip=True).fit(filled)
    return scaler.transform(filled).astype(np.float32), scaler


def progress_probe(x_eval: np.ndarray, y_eval: np.ndarray, n_rows: int, seed: int):
    """A per-epoch ranking metric, on a stratified subsample.

    Loss is not evidence of learning.  Two silent bugs in this classifier produced steadily
    falling loss with an AUC of exactly 0.5000, and only a ranking metric distinguishes
    "training" from "the contraction no longer depends on the input".

    The subsample is stratified because average precision on a 3.4 percent positive rate is
    unstable under simple random sampling, and it is drawn once and reused so the number
    moves between epochs only when the model does.  Returns None when a probe would be
    degenerate rather than reporting a metric computed on one class.
    """
    if len(np.unique(y_eval)) < 2:
        return None
    if n_rows <= 0 or n_rows >= len(y_eval):
        rows = np.arange(len(y_eval))
    else:
        rng = np.random.default_rng(seed)
        positive = np.flatnonzero(y_eval == 1)
        negative = np.flatnonzero(y_eval == 0)
        take_pos = max(1, round(n_rows * len(positive) / len(y_eval)))
        take_neg = max(1, n_rows - take_pos)
        rows = np.concatenate(
            [
                rng.choice(positive, size=min(take_pos, len(positive)), replace=False),
                rng.choice(negative, size=min(take_neg, len(negative)), replace=False),
            ]
        )
    x_probe, y_probe = x_eval[rows], y_eval[rows]

    def probe(model) -> dict[str, float]:
        scores = model.predict_proba(x_probe)
        return {
            "auc": float(roc_auc_score(y_probe, scores)),
            "ap": float(average_precision_score(y_probe, scores)),
        }

    return probe


def evaluate(y_true, scores) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--zip", type=Path, default=REPO / "datasets" / IEEE_CIS_ZIP)
    parser.add_argument("--runs", type=Path, default=REPO / "results" / "runs")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "tables")
    parser.add_argument("--arm", default="temporal")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bonds", type=int, nargs="*", default=[4, 8, 16, 32])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--band-features", type=int, default=8)
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument(
        "--eval-rows",
        type=int,
        default=20_000,
        help=(
            "rows used for the per-epoch progress metric, stratified on the label. "
            "0 uses the whole evaluation block, which costs about 11 percent of an epoch "
            "at full scale against about 2 percent for the default."
        ),
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    # results/tables/ holds the pre-registered configuration and nothing else.  An
    # exploratory run -- a shorter sweep, fewer epochs, a smoke test -- produces numbers that
    # are not the study's, and writing them to the same path silently replaces the committed
    # result with something that merely looks like it.  That happened once: a six-epoch
    # single-bond smoke test overwrote a thirty-epoch four-bond table, and only the git
    # history distinguished them.  Non-default runs are diverted to results/runs/.
    defaults = parser.parse_args([])
    exploratory = (args.bonds, args.epochs, args.band_features) != (
        defaults.bonds,
        defaults.epochs,
        defaults.band_features,
    )
    if exploratory and args.out == defaults.out:
        args.out = args.runs / "exploratory"
        args.out.mkdir(parents=True, exist_ok=True)
        print(
            f"Not the pre-registered configuration (bonds={args.bonds}, "
            f"epochs={args.epochs}, band_features={args.band_features}); writing tables to "
            f"{args.out.relative_to(REPO)} instead of results/tables/."
        )

    power = require_power_gate(defaults.out)
    mde = float(power["minimum_detectable_effect"].iloc[0])
    powered = bool(power["adequately_powered"].iloc[0])
    print(
        f"H4 power gate (recorded before this run): MDE {mde:.4f} AP against a ceiling of "
        f"{float(power['preregistered_ceiling'].iloc[0]):.4f} -- "
        f"{'adequately powered' if powered else 'UNDERPOWERED, a null result is uninformative'}"
    )

    seed = args.seed or cfg.split.seeds[0]
    scores = pd.read_parquet(args.runs / f"scores_{args.arm}_{seed}.parquet")
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
    train_rows = scores[scores["block"] == "train"]["row"].to_numpy()
    y = frame["isFraud"].to_numpy()

    # ---------------------------------------------------------------- in-band comparison
    # The band is the region between the decline threshold and the approve threshold, fixed
    # on D_band exactly as scripts/run_conformal.py fixes it.
    band_block = scores[scores["block"] == "band"]
    s_band = band_block["score"].to_numpy()
    hi = float(np.quantile(s_band, 1.0 - cfg.decline_rate_budget))
    lo = float(np.quantile(s_band, 1.0 - cfg.decline_rate_budget - cfg.band.traffic_budget))

    from sklearn.feature_selection import mutual_info_classif

    rng = np.random.default_rng(seed)
    probe = rng.choice(len(train_rows), size=min(20_000, len(train_rows)), replace=False)
    probe_x, _ = prepare(frame, train_rows[probe], all_numeric, None)
    mi = mutual_info_classif(probe_x, y[train_rows[probe]], random_state=seed)
    band_columns = [all_numeric[i] for i in np.argsort(mi)[::-1][: args.band_features]]

    rows: list[dict] = []
    band_train = scores[(scores["block"] == "band")]
    band_train_rows = band_train[
        (band_train["score"] >= lo) & (band_train["score"] < hi)
    ]["row"].to_numpy()
    band_eval = scores[scores["block"] == "cal"]
    band_eval_rows = band_eval[
        (band_eval["score"] >= lo) & (band_eval["score"] < hi)
    ]["row"].to_numpy()

    x_bt, scaler = prepare(frame, band_train_rows, band_columns, None)
    x_be, _ = prepare(frame, band_eval_rows, band_columns, scaler)
    y_bt, y_be = y[band_train_rows], y[band_eval_rows]
    print(
        f"in-band: fit on {len(x_bt):,} band rows from D_band ({int(y_bt.sum())} fraud), "
        f"evaluate on {len(x_be):,} from D_cal ({int(y_be.sum())} fraud), "
        f"{args.band_features} features"
    )

    band_predictions: dict[int, np.ndarray] = {}
    band_probe = progress_probe(x_be, y_be, args.eval_rows, seed)
    band_sweep = SweepTimer(len(args.bonds))
    args.runs.mkdir(parents=True, exist_ok=True)
    for chi in args.bonds:
        start = time.perf_counter()
        reporter = ProgressReporter(
            f"band chi={chi}",
            args.epochs,
            stream=sys.stdout,
            log_path=args.runs / f"mps_band_chi{chi}_{seed}.jsonl",
            context={"experiment": "band", "bond_dimension": chi, "seed": seed,
                     "n_train": len(x_bt), "n_features": args.band_features},
            label_width=14,
        )
        with reporter:
            model, history = fit_mps(
                x_bt,
                y_bt,
                MPSConfig(bond_dimension=chi, epochs=args.epochs, seed=seed, device="cuda"),
                reporter=reporter,
                evaluate=band_probe,
            )
        elapsed = time.perf_counter() - start
        band_sweep.record(elapsed)
        band_predictions[chi] = model.predict_proba(x_be)
        metrics = evaluate(y_be, band_predictions[chi])
        rows.append(
            {
                "experiment": "band",
                "model": "mps",
                "bond_dimension": chi,
                "n_features": args.band_features,
                "n_train": len(x_bt),
                "n_eval": len(x_be),
                "seed": seed,
                "fit_seconds": elapsed,
                "final_loss": history[-1],
                **metrics,
            }
        )
        print(
            f"  chi={chi:3d}  {elapsed:6.1f}s  loss {history[0]:.4f} -> {history[-1]:.4f}  "
            f"AUC {metrics['roc_auc']:.4f}  AP {metrics['average_precision']:.4f}"
        )

    # The in-band GBDT control, on the identical features and rows.
    import xgboost as xgb

    control = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05, tree_method="hist",
        device="cuda", eval_metric="aucpr", random_state=seed, verbosity=0,
    )
    start = time.perf_counter()
    control.fit(x_bt, y_bt)
    elapsed = time.perf_counter() - start
    control_prediction = control.predict_proba(x_be)[:, 1]
    metrics = evaluate(y_be, control_prediction)
    rows.append(
        {
            "experiment": "band", "model": "xgboost", "bond_dimension": np.nan,
            "n_features": args.band_features, "n_train": len(x_bt), "n_eval": len(x_be),
            "seed": seed, "fit_seconds": elapsed, "final_loss": np.nan, **metrics,
        }
    )
    print(
        f"  xgboost   {elapsed:6.1f}s  AUC {metrics['roc_auc']:.4f}  "
        f"AP {metrics['average_precision']:.4f}   (same rows, same features)"
    )

    pd.DataFrame([r for r in rows if r["experiment"] == "band"]).to_csv(
        args.out / "mps_band.csv", index=False
    )

    # ------------------------------------------------------------------------- H4
    # The null is that the tensor network does not improve band-conditional average precision
    # over the tuned baseline.  Testing it at every bond dimension rather than at the best one
    # avoids the winner's curse; Holm controls the family-wise error across the sweep.  Cards
    # are the resampling unit because rows within a card are not independent.
    band_clusters = frame.iloc[band_eval_rows]["card1"].to_numpy()
    h4_rows, p_values = [], {}
    for chi, prediction in band_predictions.items():
        result = clustered_bootstrap_difference(
            average_precision_score,
            y_be,
            prediction,
            control_prediction,
            band_clusters,
            n_resamples=args.resamples,
            seed=seed,
        )
        # One-sided: H4 asks whether the MPS *improves* on the baseline, so the evidence
        # against it is the resampled mass at or below zero.
        p_values[f"chi={chi}"] = float(
            stats.norm.sf(result.point / result.spread) if result.spread > 0 else 1.0
        )
        h4_rows.append(
            {
                "hypothesis": "H4",
                "bond_dimension": chi,
                "ap_mps": float(average_precision_score(y_be, prediction)),
                "ap_gbdt": float(average_precision_score(y_be, control_prediction)),
                "ap_difference": result.point,
                "ci_low": result.low,
                "ci_high": result.high,
                "standard_error": result.spread,
                "n_clusters": result.n_clusters,
                "n_resamples_usable": result.n_resamples,
                "minimum_detectable_effect": mde,
                "resolvable": bool(abs(result.point) >= mde),
                "seed": seed,
            }
        )
    rejected = holm_bonferroni(p_values, alpha=0.05)
    for row in h4_rows:
        row["p_value"] = p_values[f"chi={row['bond_dimension']}"]
        row["rejects_null_holm"] = rejected[f"chi={row['bond_dimension']}"]
    pd.DataFrame(h4_rows).to_csv(args.out / "mps_h4.csv", index=False)

    print("\n  H4: does the tensor network improve band-conditional AP over the baseline?")
    for row in h4_rows:
        verdict = "improves" if row["rejects_null_holm"] else "no improvement"
        print(
            f"    chi={row['bond_dimension']:3d}  dAP {row['ap_difference']:+.4f} "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]  p={row['p_value']:.3f}  "
            f"{verdict}"
        )

    # ------------------------------------------------------------------ full-scale arm
    if not args.skip_full:
        test_rows = scores[scores["block"] == "test"]["row"].to_numpy()
        x_tr, scaler_full = prepare(frame, train_rows, all_numeric, None)
        x_te, _ = prepare(frame, test_rows, all_numeric, scaler_full)
        print(
            f"\nfull scale: {len(x_tr):,} train rows x {len(all_numeric)} features, "
            f"evaluate on {len(x_te):,} test rows"
        )
        full_rows = []
        full_probe = progress_probe(x_te, y[test_rows], args.eval_rows, seed)
        # The sweep estimate extrapolates from finished jobs, which is defensible here
        # because per-step cost is flat in the bond dimension at this chain length: 139.5,
        # 148.8, 161.4 and 158.8 ms at chi = 4, 8, 16, 32 against the 1x/4x/16x/64x a chi^2
        # model predicts. The chain is bound by 431 sequential kernel launches. See D-032.
        full_sweep = SweepTimer(len(args.bonds))
        for chi in args.bonds:
            start = time.perf_counter()
            reporter = ProgressReporter(
                f"full chi={chi}",
                args.epochs,
                stream=sys.stdout,
                log_path=args.runs / f"mps_full_chi{chi}_{seed}.jsonl",
                context={"experiment": "full", "bond_dimension": chi, "seed": seed,
                         "n_train": len(x_tr), "n_features": len(all_numeric)},
                label_width=14,
            )
            with reporter:
                model, history = fit_mps(
                    x_tr,
                    y[train_rows],
                    MPSConfig(bond_dimension=chi, epochs=args.epochs, seed=seed,
                              device="cuda"),
                    reporter=reporter,
                    evaluate=full_probe,
                )
            elapsed = time.perf_counter() - start
            full_sweep.record(elapsed)
            print(f"  sweep: {full_sweep.summary()}")
            metrics = evaluate(y[test_rows], model.predict_proba(x_te))
            full_rows.append(
                {
                    "experiment": "full", "model": "mps", "bond_dimension": chi,
                    "n_features": len(all_numeric), "n_train": len(x_tr), "n_eval": len(x_te),
                    "seed": seed, "fit_seconds": elapsed, "final_loss": history[-1], **metrics,
                }
            )
            print(
                f"  chi={chi:3d}  {elapsed:6.1f}s  loss {history[0]:.4f} -> {history[-1]:.4f}  "
                f"AUC {metrics['roc_auc']:.4f}  AP {metrics['average_precision']:.4f}"
            )
        pd.DataFrame(full_rows).to_csv(args.out / "mps_full.csv", index=False)
        print(
            "\n  The tuned gradient-boosted baseline on the same split reaches AUC 0.8837-0.8884 "
            "and AP 0.5055-0.5114 (results/tables/baselines.csv)."
        )

    written = ["mps_band.csv", "mps_h4.csv"] + ([] if args.skip_full else ["mps_full.csv"])
    print("\nWrote " + ", ".join(str(args.out / name) for name in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
