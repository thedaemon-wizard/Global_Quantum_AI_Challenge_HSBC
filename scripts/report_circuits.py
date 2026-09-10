#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Qubit count and circuit depth for every screened encoding, which the statement asks for.

The challenge statement lists "qubit count and circuit depth" among the metrics that make a
proposal assessable for near-term hardware feasibility.  The screens recorded the qubit count
and not the depth, so the question "could this have run on hardware?" had no answer in the
repository even though every circuit was constructible from the configuration grid.

Two depths are reported and they answer different questions.

**Logical depth** is the depth of the circuit as written.  It is the honest description of the
encoding and it is what changes when the encoding changes.

**Transpiled depth** is the depth after decomposition to a common ``rz, sx, x, cx`` basis with
all-to-all connectivity.  It is the portable lower bound on what a device would run: a real
backend with limited connectivity would be deeper, never shallower.  Reporting it without a
named device is deliberate -- a depth quoted against one machine's coupling map is a statement
about that machine, and no hardware run was made here.

The configuration grid is read from ``screens.csv`` rather than restated, so this cannot drift
from the arm it describes.

    .venv/bin/python scripts/report_circuits.py

Writes ``results/tables/circuits.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from qiskit import transpile

from hsbcfraud.config import load_config
from hsbcfraud.paths import display_path
from hsbcfraud.quantum.featuremaps import build_feature_map

REPO = Path(__file__).resolve().parents[1]

# A device-independent basis.  Every superconducting backend on Braket decomposes to something
# close to this, and quoting a coupling-map-specific depth would imply a hardware run.
PORTABLE_BASIS = ("rz", "sx", "x", "cx")
# Transpilation is a search; fixing the seed keeps the reported depth reproducible.
TRANSPILE_SEED = 20260828
TRANSPILE_LEVEL = 3


def measure(name: str, n_features: int, entanglement: str, reps: int) -> dict[str, object]:
    """Structural properties of one encoding, before and after decomposition.

    ``reps`` is passed rather than defaulted, and recorded in the row.  Depth and gate counts
    are close to linear in it, and neither this table nor ``screens.csv`` used to say which
    value produced them -- so a config setting ``quantum.reps: 3`` would have made
    ``screen_kernels.py`` (which does read it) describe three-repetition circuits while this
    table reported two-repetition depths, with nothing in either artefact to show the
    disagreement.
    """
    circuit = build_feature_map(name, n_features, reps=reps, entanglement=entanglement)
    decomposed = transpile(
        circuit,
        basis_gates=list(PORTABLE_BASIS),
        optimization_level=TRANSPILE_LEVEL,
        seed_transpiler=TRANSPILE_SEED,
    )
    two_qubit = sum(count for gate, count in decomposed.count_ops().items() if gate == "cx")
    return {
        "encoding": name,
        "n_features": n_features,
        "entanglement": entanglement,
        "reps": reps,
        "n_qubits": circuit.num_qubits,
        "logical_depth": circuit.depth(),
        "logical_gates": circuit.size(),
        "transpiled_depth": decomposed.depth(),
        "transpiled_gates": decomposed.size(),
        "two_qubit_gates": two_qubit,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--tables", type=Path, default=REPO / "results" / "tables")
    args = parser.parse_args(argv)

    screens = args.tables / "screens.csv"
    if not screens.exists():
        raise SystemExit(
            f"{display_path(screens)} is missing. The configuration grid is read from the "
            "screen results so that this table cannot describe a different arm.\n"
            "  Run: .venv/bin/python scripts/screen_kernels.py"
        )
    grid = (
        pd.read_csv(screens)[["name", "n_features", "entanglement"]]
        .drop_duplicates()
        .sort_values(["name", "n_features", "entanglement"], ignore_index=True)
    )

    cfg = load_config(args.config)
    frame = pd.DataFrame(
        [
            measure(row.name, int(row.n_features), row.entanglement, cfg.quantum.reps)
            for row in grid.itertuples()
        ]
    )
    target = args.tables / "circuits.csv"
    frame.to_csv(target, index=False)

    print(f"{len(frame)} distinct encodings from {display_path(screens)}\n")
    header = (
        f"{'encoding':>12s} {'features':>9s} {'entangle':>9s} {'qubits':>7s} "
        f"{'depth':>6s} {'gates':>6s} {'t.depth':>8s} {'t.gates':>8s} {'2q':>5s}"
    )
    print(header)
    for row in frame.itertuples():
        print(
            f"{row.encoding:>12s} {row.n_features:9d} {row.entanglement:>9s} "
            f"{row.n_qubits:7d} {row.logical_depth:6d} {row.logical_gates:6d} "
            f"{row.transpiled_depth:8d} {row.transpiled_gates:8d} {row.two_qubit_gates:5d}"
        )
    print(
        f"\n  qubits {frame.n_qubits.min()} to {frame.n_qubits.max()}, "
        f"transpiled depth {frame.transpiled_depth.min()} to {frame.transpiled_depth.max()}"
    )
    print(f"  Wrote {display_path(target)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
