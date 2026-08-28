#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Draw the submission figures from the result tables.

Figures are generated rather than drawn so they cannot drift from the numbers.  A figure that
disagrees with its table is worse than no figure: it is the one artefact a reader trusts
without checking.

    .venv/bin/python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on this host, and none needed
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# Print-safe and legible in greyscale: the submission may well be read on paper.
INSIDE, BREACH, REFERENCE = "#3b6ea5", "#b03a2e", "#555555"


def coverage_figure(tables: Path, out: Path) -> Path:
    """Empirical over nominal rate by arm, against the Beta-Binomial interval.

    The interval is what makes the plot honest.  A bar chart of ratios alone invites the eye
    to read any departure from 1.0 as a breach, when what counts is whether the error count
    falls outside the exact predictive law for that sample size.
    """
    frame = pd.read_csv(tables / "coverage_by_arm.csv")
    arms = ["temporal", "stratified", "card_disjoint"]
    labels = {"temporal": "temporal", "stratified": "stratified", "card_disjoint": "card-disjoint"}

    figure, axes = plt.subplots(1, 3, figsize=(9.5, 3.0), sharey=True)
    for axis, arm in zip(axes, arms, strict=True):
        rows = frame[frame["arm"] == arm].sort_values("alpha")
        alphas = rows["alpha"].to_numpy()
        positions = range(len(alphas))
        # Normalise the interval by the nominal count so every alpha is comparable on one axis.
        nominal = alphas * rows["n_test_legit"].to_numpy()
        low = rows["band_low"].to_numpy() / nominal
        high = rows["band_high"].to_numpy() / nominal
        for position, lo, hi in zip(positions, low, high, strict=True):
            axis.plot([position, position], [lo, hi], color=REFERENCE, linewidth=6, alpha=0.25,
                      solid_capstyle="butt")
        colours = [INSIDE if ok else BREACH for ok in rows["finite_sample_ok"]]
        axis.scatter(positions, rows["ratio"], c=colours, s=34, zorder=3)
        axis.axhline(1.0, color=REFERENCE, linewidth=0.8, linestyle="--")
        axis.set_xticks(list(positions))
        axis.set_xticklabels([f"{a:g}" for a in alphas], fontsize=8)
        axis.set_xlabel(r"nominal $\alpha$", fontsize=9)
        axis.set_title(labels[arm], fontsize=10)
        axis.tick_params(labelsize=8)
    axes[0].set_ylabel("empirical / nominal", fontsize=9)
    figure.suptitle(
        "Grey bands are the exact Beta-Binomial interval; red points fall outside it",
        fontsize=8.5, y=1.02, color=REFERENCE,
    )
    figure.tight_layout()
    target = out / "coverage_by_arm.pdf"
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    return target


def mps_figure(tables: Path, out: Path) -> Path:
    """The H4 difference with its interval, against the effect the study could resolve."""
    frame = pd.read_csv(tables / "mps_h4.csv").sort_values("bond_dimension")
    figure, axis = plt.subplots(figsize=(5.2, 2.8))
    positions = range(len(frame))
    for position, (_, row) in zip(positions, frame.iterrows(), strict=True):
        axis.plot([row["ci_low"], row["ci_high"]], [position, position],
                  color=REFERENCE, linewidth=2, alpha=0.55)
        axis.scatter([row["ap_difference"]], [position], color=BREACH, s=34, zorder=3)
    axis.axvline(0.0, color=REFERENCE, linewidth=0.9, linestyle="--")
    axis.set_yticks(list(positions))
    axis.set_yticklabels([f"$\\chi={int(c)}$" for c in frame["bond_dimension"]], fontsize=9)
    axis.set_xlabel(r"$\Delta$ average precision, MPS $-$ GBDT", fontsize=9)
    axis.tick_params(labelsize=8)
    axis.set_title("Every interval contains zero", fontsize=9.5, color=REFERENCE)
    figure.tight_layout()
    target = out / "mps_h4.pdf"
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, default=REPO / "results" / "tables")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "figures")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    for builder in (coverage_figure, mps_figure):
        target = builder(args.tables, args.out)
        print(f"Wrote {target.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
