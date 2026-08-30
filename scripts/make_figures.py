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
from matplotlib.figure import Figure

from hsbcfraud.paths import display_path

matplotlib.use("Agg")  # no display on this host, and none needed
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# Print-safe and legible in greyscale: the submission may well be read on paper.
INSIDE, BREACH, REFERENCE = "#3b6ea5", "#b03a2e", "#555555"


def coverage_figure(tables: Path) -> Figure:
    """Empirical over nominal rate by arm, against the Beta-Binomial interval.

    The interval is what makes the plot honest.  A bar chart of ratios alone invites the eye
    to read any departure from 1.0 as a breach, when what counts is whether the error count
    falls outside the exact predictive law for that sample size.
    """
    frame = pd.read_csv(tables / "coverage_by_arm.csv")
    arms = ["temporal", "stratified", "card_disjoint"]
    labels = {"temporal": "temporal", "stratified": "stratified", "card_disjoint": "card-disjoint"}

    # Sized to be *placed* near full text width rather than scaled down to fit it.  At 9.5 in
    # the figure had to be shrunk to about two thirds of the text block to leave room on the
    # page, which took its 8 pt tick labels below 5 pt in the rendered PDF -- legible on screen
    # at 400 % and not on paper.  A narrower figure with larger type renders larger.
    figure, axes = plt.subplots(1, 3, figsize=(8.6, 2.0), sharey=True)
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
        axis.set_xticklabels([f"{a:g}" for a in alphas], fontsize=11)
        axis.set_xlabel(r"nominal $\alpha$", fontsize=10.5)
        axis.set_title(labels[arm], fontsize=11)
        axis.tick_params(labelsize=9.5)
    axes[0].set_ylabel("empirical / nominal", fontsize=10.5)
    figure.suptitle(
        "Grey bands are the exact Beta-Binomial interval; red points fall outside it",
        fontsize=9, y=1.03, color=REFERENCE,
    )
    figure.tight_layout()
    return figure, "coverage_by_arm"


def mps_figure(tables: Path) -> Figure:
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
    return figure, "mps_h4"


def tradeoff_figure(tables: Path) -> Figure:
    """Which (band budget, alpha) pairs certify, and at which missed-fraud budget.

    The certificate's reach is the study's headline deliverable and the easiest thing to
    overstate: a reader told "the certificate holds" will assume it holds everywhere.  It does
    not.  Drawing the whole pre-registered grid, with the certified cells filled and the rest
    left open, makes the sparsity the first thing visible rather than a caveat in prose.

    The PNG of this one is a portal artefact in its own right: the portal accepts images but
    not a standalone PDF figure alongside the two document PDFs.
    """
    frame = pd.read_csv(tables / "riskcontrol.csv")
    budgets = sorted(frame["budget"].unique())
    alphas = sorted(frame["alpha"].unique())
    fn_budgets = sorted(frame["alpha_fn"].unique())

    figure, axes = plt.subplots(
        1, len(fn_budgets), figsize=(2.6 * len(fn_budgets), 2.9), sharey=True
    )
    for axis, alpha_fn in zip(axes, fn_budgets, strict=True):
        panel = frame[frame["alpha_fn"] == alpha_fn]
        for _, row in panel.iterrows():
            certified = bool(row["certified"])
            axis.scatter(
                [alphas.index(row["alpha"])], [budgets.index(row["budget"])],
                marker="s", s=340,
                facecolor=INSIDE if certified else "none",
                edgecolor=INSIDE if certified else REFERENCE,
                linewidth=1.0 if certified else 0.7,
            )
        axis.set_xticks(range(len(alphas)))
        axis.set_xticklabels([f"{a:g}" for a in alphas], fontsize=8)
        axis.set_xlabel(r"false-decline budget $\alpha$", fontsize=8.5)
        axis.set_title(rf"$\alpha_{{\mathrm{{FN}}}} = {alpha_fn:g}$", fontsize=9.5)
        axis.set_xlim(-0.6, len(alphas) - 0.4)
        axis.set_ylim(-0.6, len(budgets) - 0.4)
        axis.tick_params(labelsize=8)
    axes[0].set_yticks(range(len(budgets)))
    axes[0].set_yticklabels([f"{b:g}" for b in budgets], fontsize=8)
    axes[0].set_ylabel("band traffic budget", fontsize=8.5)

    certified = int(frame["certified"].astype(bool).sum())
    figure.suptitle(
        f"Filled: certified ({certified} of {len(frame)} pre-registered configurations). "
        "Open: no admissible threshold.",
        fontsize=8.5, y=1.04, color=REFERENCE,
    )
    figure.tight_layout()
    return figure, "certified_region"


# Every figure is written in both formats, from one draw.  PDF is what LaTeX includes; PNG is
# what GitHub renders in the README and what the portal accepts as an upload.  Producing them
# from the same figure object is what stops the two from ever showing different numbers -- the
# failure this whole script exists to prevent, one level up.
FORMATS = ((".pdf", {}), (".png", {"dpi": 200}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, default=REPO / "results" / "tables")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "figures")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    for builder in (coverage_figure, mps_figure, tradeoff_figure):
        figure, stem = builder(args.tables)
        for suffix, options in FORMATS:
            target = args.out / f"{stem}{suffix}"
            figure.savefig(target, bbox_inches="tight", **options)
            print(f"Wrote {display_path(target)}")
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
