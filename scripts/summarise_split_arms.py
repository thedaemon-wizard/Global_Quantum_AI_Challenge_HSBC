#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Aggregate the per-seed baseline runs into the split-arm comparison the proposal prints.

``split_arm_baselines.csv`` backs Table 1, which reports the largest measured effect in the
study -- a stratified random split inflating average precision over a forward holdout.  The
table was committed and no script in the repository wrote it, so ``make reproduce`` could not
regenerate it and ``freeze.py --check`` passed it trivially: a file nothing rewrites cannot
differ from its own hash.

Nothing is measured here.  Every value is an aggregate of ``baselines.csv``, which
``run_baselines.py`` produces, so the arithmetic is the whole content of this file.

    .venv/bin/python scripts/summarise_split_arms.py

Writes ``results/tables/split_arm_baselines.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hsbcfraud.paths import display_path

REPO = Path(__file__).resolve().parents[1]

# The block the comparison is about, and the scorer the study reports.  Both are stated rather
# than inferred: aggregating over every block would average a fitted arm with a held-out one.
REPORTED_BLOCK = "test"
REPORTED_MODEL = "xgboost"

# The arm every other arm is compared against, because it is the only one that orders time.
REFERENCE_ARM = "temporal"

# Presentation order: the inflated arm first, the reference second, the control last, which is
# the order the argument is made in.
ARM_ORDER = ("stratified", REFERENCE_ARM, "card_disjoint")

METRICS = ("roc_auc", "average_precision")


def summarise(baselines: pd.DataFrame) -> pd.DataFrame:
    """Mean, minimum and maximum of each metric across seeds, by arm."""
    rows = baselines[
        (baselines["block"] == REPORTED_BLOCK) & (baselines["model"] == REPORTED_MODEL)
    ]
    if rows.empty:
        raise SystemExit(f"no {REPORTED_MODEL} rows on the {REPORTED_BLOCK} block in baselines.csv")

    aggregation = {f"{metric}_{stat}": (metric, stat)
                   for metric in METRICS for stat in ("mean", "min", "max")}
    summary = rows.groupby("arm").agg(**aggregation, n_seeds=("seed", "nunique")).reset_index()

    reference = summary[summary["arm"] == REFERENCE_ARM]
    if len(reference) != 1:
        raise SystemExit(f"expected exactly one {REFERENCE_ARM} row, found {len(reference)}")
    for metric, column in (("average_precision", "ap"), ("roc_auc", "auc")):
        baseline = float(reference[f"{metric}_mean"].iloc[0])
        summary[f"{column}_inflation_vs_{REFERENCE_ARM}"] = summary[f"{metric}_mean"] - baseline

    ordered = summary.set_index("arm").reindex(ARM_ORDER)
    missing = ordered[ordered["n_seeds"].isna()].index.tolist()
    if missing:
        raise SystemExit(f"baselines.csv has no rows for {missing}")
    ordered["n_seeds"] = ordered["n_seeds"].astype(int)
    return ordered.reset_index()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, default=REPO / "results" / "tables")
    args = parser.parse_args(argv)

    summary = summarise(pd.read_csv(args.tables / "baselines.csv"))
    target = args.tables / "split_arm_baselines.csv"
    summary.to_csv(target, index=False)

    inflation = f"ap_inflation_vs_{REFERENCE_ARM}"
    for row in summary.itertuples():
        print(
            f"  {row.arm:<14s} ROC AUC {row.roc_auc_mean:.4f}  "
            f"AP {row.average_precision_mean:.4f}  "
            f"over {REFERENCE_ARM} {getattr(row, inflation):+.4f}"
        )
    print(f"\nWrote {display_path(target)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
