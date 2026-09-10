#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Draw the screened encodings, because a table of depths is not a circuit design.

The challenge statement's §5.2 asks for a "description of quantum approach, encoding strategy,
and circuit design choices", and lists "qubit count and circuit depth" among the metrics that
make near-term feasibility assessable.  Both are answered in prose and in
``results/tables/circuits.csv``.  Neither lets a reader *see* what was screened, and the
repository had no picture of a circuit anywhere.

**This is not a portal deliverable and is not staged as one.**  The statement never asks for a
figure -- the words "diagram" and "figure" do not occur in it -- the five upload slots are full,
and the arm drawn here was rejected by its own screens before it ran.  Spending a slot on a
method that produced no result, in place of the figure that shows the certificate, would be the
wrong trade.  This belongs in the repository, which the proposal links, and costs nothing there.

Two rows, answering different questions:

* the three feature maps drawn gate by gate at their smallest screened width, which is what
  "encoding strategy" means and is the part prose describes worst;
* transpiled depth and two-qubit count against qubit count over the whole grid, which is what
  near-term feasibility actually turns on and which no single drawing shows.

The widest configuration screened is **not** drawn.  Its text rendering is 526 columns, so a
reader would learn less from it than from the numbers already in the proposal, and the scaling
row carries the same information legibly.

Circuits come from ``build_feature_map`` and the transpilation constants are imported from
``report_circuits.py`` rather than restated, so the drawing cannot depict a different circuit
from the one the table measured.  Qiskit's text drawer is used deliberately: the matplotlib
drawer needs ``pylatexenc``, and a new dependency for a documentation figure is a poor trade
against the clean-room reproduction this project maintains.

    .venv/bin/python scripts/plot_circuits.py

Writes ``results/figures/circuits.png`` and ``.pdf``.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from hsbcfraud.paths import display_path
from hsbcfraud.quantum.featuremaps import build_feature_map

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from report_circuits import PORTABLE_BASIS  # noqa: E402

# Entanglement pattern to prefer when an encoding was screened both ways: the entangling
# variant is the one whose structure is worth seeing, and the control differs from it only by
# the omitted layer.
PREFERRED_ENTANGLEMENT = "linear"

MARKERS = {"linear": "o", "none": "s"}

# A one-qubit circuit has no entangling layer to display, which defeats the purpose
# of the drawing row.
MINIMUM_QUBITS_TO_SHOW_STRUCTURE = 2

FIGURE_WIDTH_INCHES = 13.0
POINTS_PER_INCH = 72.0

# Width of a monospace glyph as a fraction of its point size.  Close enough for DejaVu Sans
# Mono, which is matplotlib's default and what this renders with.
MONOSPACE_ASPECT = 0.60

# Above this the drawings crowd their titles, and nothing in the grid is narrow enough to need
# it.
MAXIMUM_DRAWING_POINTS = 9.0

TITLE_WRAP_COLUMNS = 150

# Panel titles read better with the encoding names the screens use, but "zz" and "z" alone are
# opaque in a figure a reviewer skims.
ENCODING_LABELS = {
    "z": "z: product encoding, no entanglement",
    "zz": "zz: pairwise phase, linear entanglement",
    "dense_angle": "dense-angle: two features per qubit",
}


def smallest_per_encoding(frame: pd.DataFrame) -> list[pd.Series]:
    """One representative configuration per encoding, at the smallest width screened.

    Smallest rather than widest because the point of the row is to show *structure*, and every
    configuration in the grid repeats the same block; the widest only adds columns.

    At least two qubits, because a one-qubit circuit cannot show an entangling layer -- and
    dense-angle packs two features per qubit, so its smallest screened configuration is exactly
    that degenerate case.
    """
    chosen = []
    for _encoding, rows in frame.groupby("encoding", sort=False):
        wide_enough = rows[rows["n_qubits"] >= MINIMUM_QUBITS_TO_SHOW_STRUCTURE]
        rows = wide_enough if not wide_enough.empty else rows
        preferred = rows[rows["entanglement"] == PREFERRED_ENTANGLEMENT]
        candidates = preferred if not preferred.empty else rows
        chosen.append(candidates.sort_values(["n_qubits", "n_features"]).iloc[0])
    return chosen


def coincident_series(frame: pd.DataFrame, metric: str):
    """Group the grid into series, merging any that would plot exactly on top of each other.

    Two configurations here are not merely close, they are identical: with the entangling layer
    removed, ``zz`` *is* ``z`` -- depth 7 and no two-qubit gates at every width -- which is
    precisely what makes it the control the screens compare against.

    Plotting both draws one invisibly beneath the other and leaves the legend naming a line the
    reader cannot find. No amount of dashing or hollow markers fixes that, because the data are
    equal; the only honest presentation is one line labelled with both names.

    Yields ``(label, entanglement, rows)``, ordered so entangling variants come first.
    """
    grouped = []
    for (encoding, entanglement), rows in frame.groupby(["encoding", "entanglement"]):
        rows = rows.sort_values("n_qubits")
        signature = tuple(zip(rows["n_qubits"], rows[metric], strict=True))
        grouped.append((signature, f"{encoding}, {entanglement}", entanglement, rows))

    merged: dict[tuple, tuple[list[str], str, pd.DataFrame]] = {}
    for signature, label, entanglement, rows in grouped:
        if signature in merged:
            merged[signature][0].append(label)
        else:
            merged[signature] = ([label], entanglement, rows)

    ordered = sorted(
        merged.values(), key=lambda item: item[1] != PREFERRED_ENTANGLEMENT
    )
    for labels, entanglement, rows in ordered:
        if len(labels) == 1:
            yield labels[0], entanglement, rows
        else:
            encodings = ", ".join(sorted(label.split(",")[0] for label in labels))
            yield f"{encodings} ({entanglement}) -- identical", entanglement, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, default=REPO / "results" / "tables")
    parser.add_argument("--figures", type=Path, default=REPO / "results" / "figures")
    args = parser.parse_args(argv)

    source = args.tables / "circuits.csv"
    if not source.exists():
        raise SystemExit(
            f"{display_path(source)} is missing, so there is no grid to draw.\n"
            "  Run: .venv/bin/python scripts/report_circuits.py"
        )
    frame = pd.read_csv(source)
    representatives = smallest_per_encoding(frame)

    figure = plt.figure(figsize=(FIGURE_WIDTH_INCHES, 9.5), constrained_layout=True)
    outer = figure.add_gridspec(2, 1, height_ratios=[1.25, 1])
    drawings = outer[0].subgridspec(len(representatives), 1)
    scaling = outer[1].subgridspec(1, 2)

    rendered = [
        build_feature_map(
            row["encoding"], int(row["n_features"]),
            reps=int(row["reps"]), entanglement=row["entanglement"],
        )
        .draw("text", fold=-1)
        .single_string()
        for row in representatives
    ]
    # One font size for all three panels, chosen so the widest drawing spans the axes without
    # overflowing.  Fixing it by hand instead would silently clip the moment the grid gains a
    # wider configuration, which is exactly the kind of drift this project keeps finding.
    widest_columns = max(len(line) for drawing in rendered for line in drawing.splitlines())
    monospace_points = min(
        MAXIMUM_DRAWING_POINTS,
        FIGURE_WIDTH_INCHES * POINTS_PER_INCH / (widest_columns * MONOSPACE_ASPECT),
    )

    for position, (row, drawing) in enumerate(zip(representatives, rendered, strict=True)):
        panel = figure.add_subplot(drawings[position])
        panel.axis("off")
        panel.text(
            0.0,
            0.5,
            drawing,
            family="monospace",
            fontsize=monospace_points,
            va="center",
            ha="left",
        )
        panel.set_title(
            f"{ENCODING_LABELS.get(row['encoding'], row['encoding'])}  "
            f"[{int(row['n_features'])} features, {int(row['n_qubits'])} qubits, "
            f"logical depth {int(row['logical_depth'])}, "
            f"transpiled depth {int(row['transpiled_depth'])}]",
            fontsize=8.5,
            loc="left",
        )

    for column, (metric, label) in enumerate(
        [("transpiled_depth", "transpiled depth"), ("two_qubit_gates", "two-qubit gates")]
    ):
        axis = figure.add_subplot(scaling[column])
        for series, entanglement, rows in coincident_series(frame, metric):
            axis.plot(
                rows["n_qubits"],
                rows[metric],
                marker=MARKERS.get(entanglement, "^"),
                label=series,
                linestyle="-" if entanglement == PREFERRED_ENTANGLEMENT else "--",
                markersize=6,
            )
        axis.set_xlabel("qubits")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        if column == 0:
            axis.legend(fontsize=7, ncol=2)
    caption = (
        f"Screened encodings and their cost after transpilation to a "
        f"{', '.join(PORTABLE_BASIS)} basis. {len(frame)} configurations, "
        f"{frame['n_qubits'].min()}-{frame['n_qubits'].max()} qubits, at most depth "
        f"{frame['transpiled_depth'].max()} and {frame['two_qubit_gates'].max()} two-qubit "
        f"gates: the arm was rejected by its own screens, not by circuit size. "
        f"The unentangled series coincide -- with its entangling layer removed zz is z, which "
        f"is what makes it the control."
    )
    # Wrapped rather than left to run: an unwrapped suptitle is centred on a width matplotlib
    # does not clip to, so the left end had been running off the canvas.
    figure.suptitle("\n".join(textwrap.wrap(caption, width=TITLE_WRAP_COLUMNS)), fontsize=9)

    args.figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(args.figures / f"circuits.{suffix}", dpi=200)
    plt.close(figure)

    for row in representatives:
        print(
            f"  drawn: {row['encoding']:<12s} {int(row['n_features'])} features, "
            f"{row['entanglement']:<7s} -> {int(row['n_qubits'])} qubits, "
            f"transpiled depth {int(row['transpiled_depth'])}"
        )
    print(f"\nWrote {display_path(args.figures / 'circuits.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
