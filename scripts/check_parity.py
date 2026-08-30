#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""E11 -- assert that four independent routes to the same overlap agree.

The fidelity kernel is computed by four implementations: an exact statevector, Amazon Braket's
``LocalSimulator``, and Qiskit Aer on CPU and on GPU.  Four independent implementations of one
mathematical object either agree to numerical precision or one of them is wrong, so running
them against each other is a stronger statement than trusting any one alone.  It is also the
substitute this study offers for hardware execution, which the challenge statement explicitly
does not penalise omitting.

D-022 recorded the measured agreement and named this script as the thing that asserts it.
The script did not exist, so the proposal shipped a parity figure backed by prose alone.  This
is that script, and it writes the table the figure now resolves to.

**Backend availability is reported, not worked around.**  ``qiskit-aer`` and its GPU wheel are
an optional extra (``make venv-gpu``); the default environment has the exact statevector and
Braket only.  A backend that cannot be imported is recorded as unavailable with the reason,
and the run still fails if either *required* backend is missing -- a parity check that quietly
compares one implementation against itself would report success and mean nothing.

    .venv/bin/python scripts/check_parity.py

Writes ``results/tables/parity.csv``.  Exit 0 when every available backend agrees with the
reference within tolerance, 1 otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hsbcfraud.paths import display_path
from hsbcfraud.quantum.featuremaps import build_feature_map
from hsbcfraud.quantum.kernel import BACKENDS, KernelBackendError, fidelity_gram

REPO = Path(__file__).resolve().parents[1]

# The exact statevector is the reference: it computes the overlap in linear algebra with no
# simulator in the path, so a disagreement is the other backend's.
REFERENCE = "statevector"
# Braket is the platform the challenge statement names, so its absence is a failure rather
# than a missing optional cross-check.
REQUIRED = (REFERENCE, "braket")

# Double-precision statevector arithmetic over a few dozen gates accumulates error at the
# 1e-13 scale; anything larger is a difference in the mathematics, not in the rounding.
TOLERANCE = 1e-10

# Configurations to compare over.  Small on purpose: parity is a property of the
# implementations, not of the data, and every pair costs a full state preparation on three
# simulators.  The grid still spans both entangling and product-state encodings, because a
# translation bug in the entangling layer would not show up on a product state.
CONFIGURATIONS = (
    ("zz", 2, "linear"),
    ("zz", 4, "linear"),
    ("z", 4, "none"),
    ("dense_angle", 4, "linear"),
)
N_POINTS = 6
SEED = 20260828


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "results" / "tables")
    parser.add_argument("--points", type=int, default=N_POINTS)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    unavailable: dict[str, str] = {}

    for name, n_features, entanglement in CONFIGURATIONS:
        circuit = build_feature_map(name, n_features, entanglement=entanglement)
        x = rng.uniform(0.0, 1.0, size=(args.points, n_features))
        reference = fidelity_gram(circuit, x, backend=REFERENCE)

        for backend in BACKENDS:
            if backend == REFERENCE or backend in unavailable:
                continue
            try:
                gram = fidelity_gram(circuit, x, backend=backend)
            except KernelBackendError as error:
                unavailable[backend] = str(error).splitlines()[0]
                continue
            difference = float(np.abs(gram - reference).max())
            rows.append(
                {
                    "encoding": name,
                    "n_features": n_features,
                    "entanglement": entanglement,
                    "n_qubits": circuit.num_qubits,
                    "backend": backend,
                    "reference": REFERENCE,
                    "n_points": args.points,
                    "max_abs_difference": difference,
                    "agrees": difference <= TOLERANCE,
                    "tolerance": TOLERANCE,
                }
            )

    missing = [b for b in REQUIRED if b != REFERENCE and b in unavailable]
    if missing:
        print(f"Required backend(s) unavailable: {missing}", flush=True)
        for backend in missing:
            print(f"  {backend}: {unavailable[backend]}")
        return 1
    if not rows:
        print("No backend could be compared against the reference.")
        return 1

    frame = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "parity.csv"
    frame.to_csv(target, index=False)

    print(f"Reference: {REFERENCE}. Tolerance {TOLERANCE:.0e}.\n")
    print(f"{'encoding':>12s} {'qubits':>7s} {'backend':>10s} {'max |diff|':>12s}  verdict")
    for row in frame.itertuples():
        print(
            f"{row.encoding:>12s} {row.n_qubits:7d} {row.backend:>10s} "
            f"{row.max_abs_difference:12.3e}  {'agrees' if row.agrees else 'DISAGREES'}"
        )
    for backend, reason in unavailable.items():
        print(f"{'':>12s} {'':>7s} {backend:>10s} {'unavailable':>12s}  {reason}")

    worst = frame.groupby("backend")["max_abs_difference"].max()
    print("\n  worst disagreement per backend:")
    for backend, value in worst.items():
        print(f"    {backend:>10s} {value:.3e}")
    print(f"  Wrote {display_path(target)}")

    if not frame["agrees"].all():
        failed = frame[~frame["agrees"]]
        print(f"\n{len(failed)} comparison(s) exceed the tolerance.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
