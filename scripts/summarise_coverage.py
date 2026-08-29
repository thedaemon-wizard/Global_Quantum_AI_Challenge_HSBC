#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Summarise the per-seed coverage runs, because the single-split table oversold two arms.

``coverage_by_arm.csv`` reports one seed per arm and level.  Quoting it led to two statements
the five-seed data does not support:

* the temporal ratio at ``alpha = 0.01`` was reported as **1.494**, which is the *maximum*
  across seeds, not the centre.  D-020 retracted exactly this and the retraction never reached
  the table.  The five seeds give a mean of 1.436 with a standard deviation of 0.055.
* "the stratified and card-disjoint arms sit inside the interval at every level" is true of
  the stratified arm at every level and every seed, and **false** of the card-disjoint arm,
  which falls outside on two of five seeds at ``alpha = 0.01`` and on one of five at
  ``alpha = 0.001``.

The per-seed counts are also stronger evidence than the single split they replace: the
temporal arm breaches on five of five seeds at the two loosest levels while the stratified arm
breaches on none at any level.  That contrast is what the three-arm design was built to
produce and it was sitting unreported.

    .venv/bin/python scripts/summarise_coverage.py

Writes ``results/tables/coverage_seed_summary.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

LABELS = {"temporal": "temporal", "stratified": "stratified", "card_disjoint": "card-disjoint"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, default=REPO / "results" / "tables")
    args = parser.parse_args(argv)

    seeds = pd.read_csv(args.tables / "coverage_by_arm_seeds.csv")
    rows = []
    for (arm, alpha), group in seeds.groupby(["arm", "alpha"]):
        inside = group["inside"].astype(bool)
        rows.append(
            {
                "arm": arm,
                "alpha": alpha,
                "n_seeds": len(group),
                "n_outside": int((~inside).sum()),
                "ratio_mean": float(group["ratio"].mean()),
                "ratio_sd": float(group["ratio"].std(ddof=1)),
                "ratio_min": float(group["ratio"].min()),
                "ratio_max": float(group["ratio"].max()),
            }
        )

    frame = pd.DataFrame(rows).sort_values(
        ["arm", "alpha"], ascending=[True, False], ignore_index=True
    )
    target = args.tables / "coverage_seed_summary.csv"
    frame.to_csv(target, index=False)

    print(f"Wrote {target.relative_to(REPO)}\n")
    header = (
        f"{'arm':>14s} {'alpha':>7s} {'outside':>9s} {'ratio mean':>11s} {'sd':>7s} {'range':>18s}"
    )
    print(header)
    for row in frame.itertuples():
        print(
            f"{LABELS.get(row.arm, row.arm):>14s} {row.alpha:7.3f} "
            f"{row.n_outside:4d}/{row.n_seeds:<4d} {row.ratio_mean:11.4f} {row.ratio_sd:7.4f} "
            f"  [{row.ratio_min:.4f}, {row.ratio_max:.4f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
