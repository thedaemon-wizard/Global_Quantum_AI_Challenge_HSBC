#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Aggregate the full-scale bond-dimension sweep into the per-chi summary the documents quote.

``mps_seed_sweep_summary.csv`` carries six bound claims and backs the results section's
statement that seed noise is several times the capacity signal.  It was committed and no script
in the repository wrote it, so ``make reproduce`` could not regenerate it and
``freeze.py --check`` passed it trivially: a file nothing rewrites cannot differ from its hash.

Nothing is measured here.  Every value aggregates ``mps_seed_sweep.csv``, which
``run_seed_sweep.py`` produces over thirteen GPU-hours, so this file is the arithmetic between
that run and the prose.

    .venv/bin/python scripts/summarise_seed_sweep.py

Writes ``results/tables/mps_seed_sweep_summary.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hsbcfraud.paths import display_path

REPO = Path(__file__).resolve().parents[1]

# A fit that never left chance.  The sweep separates with a wide gap and no borderline case:
# two fits end at ROC AUC 0.488 and 0.533 with final losses of 0.67 and 0.65, and every other
# fit reaches 0.764 or better with a loss at or below 0.37.  The constant sits in that gap
# rather than at a tuned value -- any threshold between 0.54 and 0.76 returns the same count,
# which is what makes it a reportable property of the ansatz rather than a choice.
STALLED_BELOW_AUC = 0.65

METRICS = ("roc_auc", "average_precision")


def summarise(sweep: pd.DataFrame) -> pd.DataFrame:
    """Per bond dimension: the spread across seeds, and how many fits never trained."""
    if sweep.empty:
        raise SystemExit("mps_seed_sweep.csv is empty; run `make seedsweep` first")

    aggregation = {f"{metric}_{stat}": (metric, stat)
                   for metric in METRICS for stat in ("mean", "min", "max")}
    summary = (
        sweep.assign(stalled=sweep["roc_auc"] < STALLED_BELOW_AUC)
        .groupby("bond_dimension")
        .agg(n_seeds=("seed", "nunique"), n_stalled=("stalled", "sum"), **aggregation)
        .reset_index()
    )

    # Two spreads, and the comparison between them is the point: the first is noise at fixed
    # capacity, the second is the whole capacity signal the sweep was run to resolve.
    summary["ap_seed_spread"] = summary["average_precision_max"] - summary["average_precision_min"]
    summary["ap_across_chi_spread"] = (
        summary["average_precision_mean"].max() - summary["average_precision_mean"].min()
    )

    ordered = [
        "bond_dimension", "n_seeds", "n_stalled",
        *(f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "min", "max")),
        "ap_seed_spread", "ap_across_chi_spread",
    ]
    return summary[ordered]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, default=REPO / "results" / "tables")
    args = parser.parse_args(argv)

    summary = summarise(pd.read_csv(args.tables / "mps_seed_sweep.csv"))
    target = args.tables / "mps_seed_sweep_summary.csv"
    summary.to_csv(target, index=False)

    for row in summary.itertuples():
        print(
            f"  chi {row.bond_dimension:>3d}  AP {row.average_precision_mean:.4f}  "
            f"seed spread {row.ap_seed_spread:.4f}  stalled {row.n_stalled} of {row.n_seeds}"
        )
    across = float(summary["ap_across_chi_spread"].iloc[0])
    widest = float(summary["ap_seed_spread"].max())
    print(
        f"\nseed noise is {widest / across:.1f} times the capacity signal "
        f"({widest:.4f} against {across:.4f})"
    )
    print(f"Wrote {display_path(target)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
