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

# The proposal's text block, in inches: A4 at an 18 mm margin.  Figures are authored at exactly
# this width so that `\includegraphics[width=\textwidth]` scales them by 1.0 and the point sizes
# below are the point sizes that reach the page.  Authoring wider and letting LaTeX shrink is
# what put 5.1 % of the document's characters under the 10 pt floor -- the gate tolerated it
# because its small-text allowance exists for mathematical sub- and superscripts, not for
# figure labels that were simply authored too small.
TEXT_WIDTH_IN = (210.0 - 2 * 18.0) / 25.4


# Neutral fill for the blocks that carry no verdict, so colour is reserved for the two that do.
NEUTRAL = "#dfe6ee"


# The method as a flow, for the reader who has not yet read the method.  Deliberately separate
# from `draw_split_row`: that helper draws the four blocks and is shared with `overview_figure`,
# and this figure is about what happens *to* the data rather than how it is cut.  Reusing it
# here would couple two pictures that answer different questions, and changing it would silently
# redraw the README's overview.
#
# Laid out in one flat band rather than a tall graph because the proposal's page 2 has no spare
# lines: a 3-inch figure would push text off the page, and the argument in that text is what the
# certificate rests on.
METHOD_STAGES = (
    ("IEEE-CIS", "<rows> rows\ndays <first>-<last>", NEUTRAL),
    ("temporal\nsplit", "4 blocks\nnever shuffled", NEUTRAL),
    ("scorer $f$", "gradient-boosted\non all traffic", NEUTRAL),
    ("band $B$", "edges frozen\non $D_{\\mathrm{band}}$", INSIDE),
    ("Learn-then-\nTest", "2 risks, 11 $\\lambda$\nHolm-corrected", INSIDE),
    ("certificate", "$R(\\lambda) \\leq \\alpha$ w.p.\n$1-\\delta$ on $D_{\\mathrm{cal}}$", INSIDE),
    ("decisions", "approve / step-up\n/ decline", NEUTRAL),
)

# Both quantum arms enter at the band and neither reached the decision.  Drawn beneath it with
# dashed arrows, because a solid arrow would claim a contribution that the results section
# spends a page retracting.
METHOD_DEAD_ENDS = (
    "quantum kernel: 0 of 120 passed",
    "tensor network: no improvement",
)


def substitute(text: str, numbers: dict[str, str]) -> str:
    """Replace `<name>` tokens, and refuse silently leaving one behind."""
    for name, value in numbers.items():
        text = text.replace(f"<{name}>", value)
    if "<" in text and ">" in text:
        raise ValueError(f"unsubstituted token in figure label: {text!r}")
    return text


def method_figure(tables: Path) -> tuple[Figure, str]:
    """The method end to end: what enters, what is frozen where, what comes out.

    Section 2 tells the reader that the geometry of the split is what licenses the guarantee.
    That is a claim about *order* -- edges before certification, certification before the single
    test read -- and order is the one thing prose conveys worst and a flow conveys immediately.

    Shaded stages are the ones that carry the guarantee, matching the convention
    `draw_split_row` already uses, so a reader who sees both figures reads the same colour the
    same way.

    Row and feature counts come from ``splits.csv`` rather than being typed, so this cannot
    drift from the table it summarises.
    """
    # The input box carries only properties of the file: how many rows and how many days. An
    # earlier version put "431 features" here, which is the tensor network's *site* count -- the
    # baseline uses 439 -- so the box describing the dataset was labelled with one arm's
    # downstream number. The clean room surfaced it by logging 439 on the line above.
    #
    # Substituted rather than typed, and through angle-bracket tokens rather than `str.format`:
    # the detail strings carry LaTeX like `$D_{\mathrm{band}}$`, whose braces `format` reads as
    # field names and rejects.  A literal-replace fallback was the first attempt and is worse
    # still -- it is a silent no-op the day the number moves, which is how a figure comes to
    # disagree with the table beneath it.
    frame = pd.read_csv(tables / "splits.csv")
    temporal = frame[frame["arm"] == "temporal"]
    numbers = {
        "rows": f"{int(temporal['n_rows'].sum()):,}",
        "first": str(int(temporal["day_first"].min())),
        "last": str(int(temporal["day_last"].max())),
    }

    # 1.25 in, not the 1.52 the first draft used.  The old split figure was 1.24 and proposal
    # page 2 has no spare line, so a taller figure pushes text off the page -- and the text in
    # section 2 is the argument the certificate rests on.  The y-range shrinks with the figure
    # so every box keeps its height in inches and the point sizes still fit inside it.
    figure = plt.figure(figsize=(TEXT_WIDTH_IN, 1.10))
    axis = figure.add_axes([0.004, 0.02, 0.992, 0.96])
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 29)
    axis.axis("off")

    # Seven stages across the text width leaves 0.90 in per box.  The two-line titles and the
    # 8.8 pt face are what keep "Learn-then-Test" and "certificate" inside their rectangles at
    # that width; the first attempt at 9.6 pt on one line overflowed both.
    width, gap = 13.2, 1.2
    top, height = 12.0, 16.0
    for index, (name, detail, fill) in enumerate(METHOD_STAGES):
        x = index * (width + gap)
        carries = fill == INSIDE
        axis.add_patch(
            plt.Rectangle((x, top), width, height, facecolor=fill,
                          edgecolor=REFERENCE, linewidth=0.9)
        )
        ink = "white" if carries else "black"
        axis.text(x + width / 2, top + height * 0.72, name,
                  ha="center", va="center", fontsize=8.8, color=ink, fontweight="bold",
                  linespacing=1.15)
        axis.text(x + width / 2, top + height * 0.28,
                  substitute(detail, numbers), ha="center", va="center",
                  fontsize=7.2, color=ink, linespacing=1.3)
        if index < len(METHOD_STAGES) - 1:
            axis.annotate("", xy=(x + width + gap, top + height / 2),
                          xytext=(x + width, top + height / 2),
                          arrowprops={"arrowstyle": "-|>", "color": REFERENCE,
                                      "linewidth": 1.1, "shrinkA": 0, "shrinkB": 0})

    band_centre = 3 * (width + gap) + width / 2
    # One line each, not two.  Both arms are a single fact -- what was tried and that it did not
    # arrive -- and the second line cost 0.27 in of figure height that page 2 does not have.
    dead_width, dead_gap = 29.5, 3.0
    span = 2 * dead_width + dead_gap
    for index, label in enumerate(METHOD_DEAD_ENDS):
        x = band_centre - span / 2 + index * (dead_width + dead_gap)
        axis.add_patch(
            plt.Rectangle((x, 1.0), dead_width, 6.2, facecolor="#f6ecec",
                          edgecolor=BREACH, linewidth=0.9, linestyle="--")
        )
        axis.text(x + dead_width / 2, 4.1, label, ha="center", va="center",
                  fontsize=7.0, color=BREACH, fontweight="bold")
        axis.annotate("", xy=(band_centre, top - 0.3), xytext=(x + dead_width / 2, 7.4),
                      arrowprops={"arrowstyle": "-|>", "color": BREACH, "linewidth": 0.9,
                                  "linestyle": "--", "shrinkA": 1, "shrinkB": 1})

    axis.text(100, 9.4, "where a quantum model would enter; neither did",
              ha="right", va="center", fontsize=7.4, color=BREACH, style="italic")
    return figure, "method"


def architecture_figure(tables: Path) -> tuple[Figure, str]:
    """The four-block temporal split, and which block may touch which parameter.

    This is the argument for why the guarantee holds, not decoration.  Two exchangeability
    breaks have to be kept apart: the temporal one, that calibration precedes test, and the
    selective one, that conditioning on band membership is conditioning on a data-dependent
    event *if* the edges were estimated on the data used to certify.  Freezing the edges on
    their own block removes the second by construction, and that is a fact about the geometry
    of the split -- which a reader absorbs from this picture and reconstructs only slowly from
    prose.

    Row counts and day ranges are read from ``splits.csv``, so the figure cannot disagree with
    the table it illustrates.

    Two deliberate constraints.  It is **not to scale**: an earlier version sized each block by
    its day span, and the two narrow blocks are precisely the two that carry the guarantee, so
    their labels collided.  And it is authored at ``TEXT_WIDTH_IN`` so LaTeX includes it at
    scale 1.0; authoring wider and letting LaTeX shrink is what drives figure labels under the
    document's 10 pt floor.
    """
    figure = plt.figure(figsize=(TEXT_WIDTH_IN, 1.24))
    axis = figure.add_axes([0.005, 0.02, 0.99, 0.84])
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 10)
    axis.axis("off")
    draw_split_row(axis, tables, bottom=1.6, height=6.9)
    axis.text(50, 9.6, "time, forward only --- blocks are contiguous and never shuffled",
              ha="center", va="center", fontsize=9.2, color=REFERENCE)
    return figure, "architecture"


# What each block is allowed to touch.  Shared by both figures that draw the split, because two
# pictures of the same four blocks disagreeing with each other would be worse than one picture.
SPLIT_BLOCKS = (
    ("train", "fits the scorer $f$", False),
    ("band", "sets the band edges\nand every threshold", True),
    ("cal", "certifies $\\lambda$", True),
    ("test", "read once; nothing\nis selected here", False),
)


def draw_split_row(axis, tables: Path, *, bottom: float, height: float) -> None:
    """Draw the four-block split across ``axis`` in a 0-100 x-range, arrows between blocks.

    Counts and day ranges come from ``splits.csv`` on every call, so neither figure can carry a
    typed number and the two cannot drift apart from each other.
    """
    frame = pd.read_csv(tables / "splits.csv")
    temporal = frame[frame["arm"] == "temporal"].set_index("block")

    width, gap = 22.3, 2.9
    middle = bottom + height / 2
    for index, (name, role, carries) in enumerate(SPLIT_BLOCKS):
        row = temporal.loc[name]
        x = index * (width + gap)
        axis.add_patch(
            plt.Rectangle((x, bottom), width, height,
                          facecolor=INSIDE if carries else NEUTRAL,
                          edgecolor=REFERENCE, linewidth=0.9)
        )
        ink = "white" if carries else "black"
        axis.text(x + width / 2, bottom + height * 0.80,
                  f"$D_{{\\mathrm{{{name}}}}}$   {int(row['n_rows']):,}",
                  ha="center", va="center", fontsize=10.5, color=ink, fontweight="bold")
        axis.text(x + width / 2, bottom + height * 0.40, role, ha="center", va="center",
                  fontsize=9.5, color=ink, linespacing=1.35)
        axis.text(x + width / 2, bottom + height * 0.09,
                  f"days {int(row['day_first'])}-{int(row['day_last'])}",
                  ha="center", va="center", fontsize=9, color=ink)
        if index < len(SPLIT_BLOCKS) - 1:
            axis.annotate("", xy=(x + width + gap, middle), xytext=(x + width, middle),
                          arrowprops={"arrowstyle": "-|>", "color": REFERENCE, "linewidth": 1.1,
                                      "shrinkA": 0, "shrinkB": 0})


def overview_figure(tables: Path) -> tuple[Figure, str]:
    """The whole picture for the README: the split above, the decision it produces below.

    The proposal has six pages and its version of this had to drop the decision flow.  The
    README has no page limit, so the two halves sit together -- which is the only place a
    reader can see that the blocks exist *in order to* license the middle branch, rather than
    as a data-handling convention that happens to precede it.
    """
    figure = plt.figure(figsize=(TEXT_WIDTH_IN, 3.5))
    axis = figure.add_axes([0.005, 0.01, 0.99, 0.98])
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 100)
    axis.axis("off")

    axis.text(50, 98, "time, forward only --- blocks are contiguous and never shuffled",
              ha="center", va="center", fontsize=9.2, color=REFERENCE)
    draw_split_row(axis, tables, bottom=74, height=21)

    # A rail under the whole row, so the connector reads as "the split, all of it" rather than
    # as an arrow leaving D_cal -- which is what a single line dropped at x=50 looked like.
    axis.plot([11, 89], [71.5, 71.5], color=REFERENCE, linewidth=1.0)
    axis.annotate("", xy=(50, 66), xytext=(50, 71.5),
                  arrowprops={"arrowstyle": "-|>", "color": REFERENCE, "linewidth": 1.1,
                              "shrinkA": 0, "shrinkB": 0})
    axis.text(51.5, 68.6, "the split licenses the rule below", ha="left", va="center",
              fontsize=9, color=REFERENCE)

    # One authorisation, left to right: score it, then act on where the score falls.
    axis.add_patch(plt.Rectangle((4, 52), 24, 13, facecolor=NEUTRAL,
                                 edgecolor=REFERENCE, linewidth=0.9))
    axis.text(16, 58.5, "one authorisation $x$", ha="center", va="center", fontsize=10)
    axis.annotate("", xy=(34, 58.5), xytext=(28, 58.5),
                  arrowprops={"arrowstyle": "-|>", "color": REFERENCE, "linewidth": 1.1,
                              "shrinkA": 0, "shrinkB": 0})
    axis.add_patch(plt.Rectangle((34, 52), 32, 13, facecolor=NEUTRAL,
                                 edgecolor=REFERENCE, linewidth=0.9))
    axis.text(50, 58.5, "scorer $f$ on $D_{\\mathrm{train}}$: $s = f(x)$", ha="center",
              va="center", fontsize=10)

    # The three branches fan from the SCORE, not from the transaction: an arrow leaving the
    # input box would say "one authorisation, therefore approve", which means nothing.
    centres = (16, 50, 84)
    axis.annotate("", xy=(50, 46), xytext=(50, 51.6),
                  arrowprops={"arrowstyle": "-", "color": REFERENCE, "linewidth": 1.1,
                              "shrinkA": 0, "shrinkB": 0})
    axis.plot([centres[0], centres[-1]], [46, 46], color=REFERENCE, linewidth=1.0)

    branches = (
        (4, 24, NEUTRAL, "black", "approve", "$s < \\tau_{\\mathrm{lo}}$"),
        (30, 40, INSIDE, "white", "abstention band $B$", "$\\tau_{\\mathrm{lo}} \\leq s "
         "< \\tau_{\\mathrm{hi}}$"),
        (72, 24, NEUTRAL, "black", "decline", "$s \\geq \\tau_{\\mathrm{hi}}$"),
    )
    for centre, (x, width, fill, ink, label, rule) in zip(centres, branches, strict=True):
        axis.annotate("", xy=(centre, 39), xytext=(centre, 46),
                      arrowprops={"arrowstyle": "-|>", "color": REFERENCE, "linewidth": 1.1,
                                  "shrinkA": 0, "shrinkB": 0})
        axis.text(centre + 1.5, 42.5, rule, ha="left", va="center", fontsize=9.5,
                  color=REFERENCE)
        axis.add_patch(plt.Rectangle((x, 26), width, 13, facecolor=fill,
                                     edgecolor=REFERENCE, linewidth=0.9))
        axis.text(x + width / 2, 32.5, label, ha="center", va="center", fontsize=10,
                  color=ink, fontweight="bold")

    axis.annotate("", xy=(50, 13), xytext=(50, 25.6),
                  arrowprops={"arrowstyle": "-|>", "color": REFERENCE, "linewidth": 1.1,
                              "shrinkA": 0, "shrinkB": 0})
    # Wide enough for its own caption: at 40 units the certified-threshold line overhung the
    # box on both sides, which is the one defect in a diagram that no test catches.
    axis.add_patch(plt.Rectangle((21, 0), 58, 13, facecolor="white",
                                 edgecolor=INSIDE, linewidth=1.3))
    axis.text(50, 8.6, "in-band rule $g$, threshold $\\lambda$ certified on $D_{\\mathrm{cal}}$",
              ha="center", va="center", fontsize=10)
    axis.text(50, 3.4, "challenge through 3-D Secure, or approve", ha="center", va="center",
              fontsize=10, color=REFERENCE)
    return figure, "overview"


def coverage_figure(tables: Path) -> tuple[Figure, str]:
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
    figure, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 1.95), sharey=True)
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


def mps_figure(tables: Path) -> tuple[Figure, str]:
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


def tradeoff_figure(tables: Path) -> tuple[Figure, str]:
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
#
# ``CreationDate: None`` omits the timestamp matplotlib otherwise stamps into every PDF.  Without
# it two consecutive runs of this script produce different bytes for identical pictures, and
# ``scripts/freeze.py`` classifies figures as scientific artefacts that must be bit-identical --
# so every rebuild reported eleven artefacts as changed, which teaches a reader to re-freeze
# without reading the list.  A reproducibility check that always fires checks nothing.
FORMATS = ((".pdf", {"metadata": {"CreationDate": None}}), (".png", {"dpi": 200}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, default=REPO / "results" / "tables")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "figures")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    for builder in (method_figure, architecture_figure, overview_figure,
                    coverage_figure, mps_figure, tradeoff_figure):
        figure, stem = builder(args.tables)
        for suffix, options in FORMATS:
            target = args.out / f"{stem}{suffix}"
            figure.savefig(target, bbox_inches="tight", **options)
            print(f"Wrote {display_path(target)}")
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
