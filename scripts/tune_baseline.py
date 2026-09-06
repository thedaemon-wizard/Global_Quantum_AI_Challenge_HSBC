#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Ask whether the shipped classical hyperparameters are near-optimal, without spending a fold.

The challenge statement's secondary objective 4.2 asks for improvement over "**tuned** classical
baselines".  This study's XGBoost settings are seven literals chosen from the values the
IEEE-CIS public solutions converged on -- a defensible choice, and not a search.  The whole
comparison rests on the classical arm being at least as well served as the quantum one, whose
kernel bandwidth *is* swept over six values, so the asymmetry is worth measuring rather than
asserting.

**Where this searches, and why it matters.**  Entirely inside ``D_train``: the last
``VALIDATION_DAYS`` days of the training block are held out, models are fitted on the rest, and
ranked on that slice.  ``D_band``, ``D_cal`` and ``D_test`` are never read.

That is not fussiness.  Selecting on ``D_band`` would set the band edges on data already used
to choose the model; selecting on ``D_cal`` would certify lambda on data used the same way; and
selecting on ``D_test`` would be a second evaluation of the held-out fold, which the
pre-registration permits once and ``TestFoldGuard`` enforces.  A tuning run that quietly spends
one of those is worse than no tuning run, because the guarantee is stated in terms of them.

The result is a ranking, not a new model.  Adopting a winner would change the scorer, and with
it the band edges, lambda, the certificate, every downstream table and the single-evaluation
record -- so that decision is the reader's, made against the margin this prints.

    .venv/bin/python scripts/tune_baseline.py

Writes ``results/tables/baseline_tuning.csv``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from hsbcfraud.config import load_config
from hsbcfraud.data.ieee_cis import IEEE_CIS_ZIP, load_ieee_cis
from hsbcfraud.data.splits import temporal_blocks
from hsbcfraud.paths import display_path
from hsbcfraud.progress import run_log

REPO = Path(__file__).resolve().parents[1]

# The shipped configuration, copied from `run_baselines.fit_xgboost`.  Named here so the search
# always contains the incumbent and the comparison is against what actually ships, not against
# a re-typed approximation of it.
SHIPPED = {
    "n_estimators": 1000,
    "max_depth": 10,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.5,
    "min_child_weight": 4,
    "reg_lambda": 1.0,
}

# One axis at a time around the incumbent, rather than a full factorial.  A full grid over these
# five axes is 324 fits for a question that a coordinate sweep answers: is the incumbent on a
# plateau, or is there a direction that clearly improves it.
SWEEP = {
    "max_depth": (6, 8, 10, 12),
    "learning_rate": (0.03, 0.05, 0.08),
    "subsample": (0.7, 0.9, 1.0),
    "colsample_bytree": (0.3, 0.5, 0.7),
    "min_child_weight": (1, 4, 8),
}

# The tail of D_train held out to rank on.  Days rather than rows because the split is snapped
# to day boundaries and a row-count cut would put one day on both sides.
VALIDATION_DAYS = 20


def candidates() -> list[dict]:
    """The incumbent first, then one variant per axis value that differs from it."""
    seen = {tuple(sorted(SHIPPED.items()))}
    out = [dict(SHIPPED)]
    for axis, values in SWEEP.items():
        for value in values:
            trial = dict(SHIPPED, **{axis: value})
            key = tuple(sorted(trial.items()))
            if key not in seen:
                seen.add(key)
                out.append(trial)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--zip", type=Path, default=REPO / "datasets" / IEEE_CIS_ZIP)
    parser.add_argument("--runs", type=Path, default=REPO / "results" / "runs")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "tables")
    args = parser.parse_args(argv)

    import xgboost as xgb

    cfg = load_config(args.config)
    seed = cfg.split.seeds[0]
    grid = candidates()

    with run_log("tune_baseline", directory=args.runs) as run:
        run.info(f"loading {args.zip.name}")
        loaded = load_ieee_cis(args.zip, None, with_identity=True)
        frame = loaded.frame
        for column in frame.columns:
            if frame[column].dtype == "object" or str(frame[column].dtype) == "str":
                frame[column] = frame[column].astype("category").cat.codes.astype("int32")

        blocks = temporal_blocks(frame["day"], cfg.split)
        train_idx = blocks["train"]
        days = frame["day"].to_numpy()
        cut = days[train_idx].max() - VALIDATION_DAYS
        fit_idx = train_idx[days[train_idx] <= cut]
        rank_idx = train_idx[days[train_idx] > cut]
        run.info(
            f"searching inside D_train only: fit on {len(fit_idx):,} rows to day {int(cut)}, "
            f"rank on {len(rank_idx):,} rows after it. D_band, D_cal and D_test are not read."
        )

        features = [
            c for c in frame.select_dtypes(include=[np.number]).columns
            if c not in {"isFraud", "day", "TransactionDT", "TransactionID"}
        ]
        x = frame[features]
        y = frame["isFraud"].to_numpy()
        x_fit, y_fit = x.iloc[fit_idx], y[fit_idx]
        x_rank, y_rank = x.iloc[rank_idx], y[rank_idx]

        rows = []
        started = time.monotonic()
        for index, trial in enumerate(grid, start=1):
            model = xgb.XGBClassifier(
                **trial, tree_method="hist", device="cuda", eval_metric="aucpr",
                random_state=seed, n_jobs=0,
            )
            model.fit(x_fit, y_fit, verbose=False)
            scores = model.predict_proba(x_rank)[:, 1]
            auc = float(roc_auc_score(y_rank, scores))
            ap = float(average_precision_score(y_rank, scores))
            rows.append({**trial, "roc_auc": auc, "average_precision": ap,
                         "is_shipped": trial == SHIPPED})
            elapsed = time.monotonic() - started
            run.info(
                f"  {index:>2}/{len(grid)}  AUC {auc:.4f}  AP {ap:.4f}  "
                f"{'(shipped)' if trial == SHIPPED else ''}  "
                f"{elapsed:.0f}s elapsed, {elapsed / index * (len(grid) - index):.0f}s left"
            )

        frame_out = pd.DataFrame(rows).sort_values("average_precision", ascending=False,
                                                   ignore_index=True)
        args.out.mkdir(parents=True, exist_ok=True)
        target = args.out / "baseline_tuning.csv"
        frame_out.to_csv(target, index=False)

        shipped = frame_out[frame_out["is_shipped"]].iloc[0]
        best = frame_out.iloc[0]
        rank = int(frame_out.index[frame_out["is_shipped"]][0]) + 1
        run.info(
            f"shipped configuration ranks {rank} of {len(frame_out)}; "
            f"AP {shipped['average_precision']:.4f} against best {best['average_precision']:.4f} "
            f"(delta {best['average_precision'] - shipped['average_precision']:+.4f})"
        )
        print(f"\nWrote {display_path(target)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
