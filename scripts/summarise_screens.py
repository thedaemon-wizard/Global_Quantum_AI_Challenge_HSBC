#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""E13 -- collapse the 120-configuration screen grid onto the axis that orders it.

``screens.csv`` records every configuration but answers no question on its own: 120 rows of
(encoding, qubits, bandwidth, entanglement) with two pass flags cannot be read for a trend.  The
challenge statement asks participants to *characterize under what conditions* quantum approaches
perform differently, and the honest answer here is not "none passed" but "none passed, and here
is the direction in which the boundary moves".

Bandwidth is that axis.  Both screens improve with it monotonically in pass-rate, and the best
distinctness sits at the **largest bandwidth the grid contains** -- so the search stopped exactly
where the trend was most favourable.  That is a named, falsifiable condition rather than a
shrug, and a reviewer can check it against the same file.

The entanglement column carries a second result.  Linear entanglement lowers the RBF correlation
at every narrower bandwidth, which is the intuition; at the endpoint the ordering **reverses**,
so the configurations that come closest to passing are the unentangled ones.

    .venv/bin/python scripts/summarise_screens.py

Writes ``results/tables/screen_bandwidth.csv``.  Derived from ``screens.csv`` alone -- it reads
no model, fits nothing, and touches no held-out data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hsbcfraud.paths import display_path

REPO = Path(__file__).resolve().parents[1]

# The distinctness bar the protocol fixes.  Imported rather than restated would be better, but
# the screens module exposes it as a threshold on the comparison, not as a module constant, and
# duplicating a number the protocol hashes is worse than naming it here with its source.
DISTINCTNESS_BAR = 0.60


def summarise(screens: pd.DataFrame) -> pd.DataFrame:
    """One row per bandwidth: how close the grid came, and how often it cleared conditioning."""
    rows = []
    for bandwidth, block in screens.groupby("bandwidth", sort=True):
        closest = block.loc[block["rbf_correlation"].idxmin()]
        rows.append(
            {
                "bandwidth": bandwidth,
                "n_configurations": len(block),
                "conditioning_pass_rate": block["passes_conditioning"].mean(),
                "distinctness_pass_rate": block["passes_distinctness"].mean(),
                "min_rbf_correlation": block["rbf_correlation"].min(),
                "distinctness_bar": DISTINCTNESS_BAR,
                "closest_encoding": closest["name"],
                "closest_entanglement": closest["entanglement"],
                "closest_n_features": int(closest["n_features"]),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, default=REPO / "results" / "tables")
    args = parser.parse_args(argv)

    screens = pd.read_csv(args.tables / "screens.csv")
    frame = summarise(screens)
    target = args.tables / "screen_bandwidth.csv"
    frame.to_csv(target, index=False)

    print(f"{'bandwidth':>10s} {'cond':>6s} {'dist':>6s} {'min rho':>9s}  closest")
    for row in frame.itertuples():
        print(
            f"{row.bandwidth:10.5f} {row.conditioning_pass_rate:6.2f} "
            f"{row.distinctness_pass_rate:6.2f} {row.min_rbf_correlation:9.4f}  "
            f"{row.closest_encoding}/{row.closest_entanglement}/{row.closest_n_features}f"
        )

    best = frame.loc[frame["min_rbf_correlation"].idxmin()]
    widest = frame["bandwidth"].max()
    print(
        f"\n  Closest approach at bandwidth {best.bandwidth:g}, which is "
        f"{'the widest the grid contains' if best.bandwidth == widest else 'interior to the grid'}."
    )
    print(f"  Wrote {display_path(target)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
