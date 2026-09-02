# SPDX-License-Identifier: Apache-2.0
"""Fidelity quantum kernels, computed independently by four simulators.

The kernel entry is the state overlap ``|<phi(x_i)|phi(x_j)>|^2`` for an encoding circuit
``phi``.  Four independent routes to the same quantity are implemented, and
``scripts/check_parity.py`` asserts they agree:

``statevector``  Exact, via ``qiskit.quantum_info.Statevector``.  No sampling, no simulator
                 backend.  This is the reference.
``braket``       Amazon Braket's ``LocalSimulator`` on the ``braket_sv`` backend, driven from
                 an OpenQASM 3 translation of the same circuit.
``aer_cpu``      Qiskit Aer statevector on CPU.
``aer_gpu``      Qiskit Aer statevector on GPU with cuStateVec.

Why four, and why Braket specifically
-------------------------------------
The challenge statement asks participants to use Amazon Braket, and puts simulator-based
execution in scope.  An earlier design decision here (``docs/decisions.md`` D-006) chose the
pure-numpy path on the grounds that it needs no proprietary binary and that a GPU does not
accelerate a pairwise Python loop.  Both grounds are still true, but the conclusion was too
narrow: it optimised the compute and dropped a stated requirement.

Braket is therefore a first-class execution path, and the agreement check turns what would
otherwise be a compliance box into evidence.  Four independent implementations of the same
mathematical object either agree to numerical precision or one of them is wrong; running them
against each other is a stronger statement than running any one of them alone, and it is the
substitute this study offers for hardware execution, which the challenge explicitly does not
penalise omitting.

Cost
----
A fidelity Gram matrix has ``n(n-1)/2`` distinct off-diagonal entries, but this
implementation does not run a circuit per pair.  The statevector route computes each state
**once** and then forms a single ``(n, 2**q) x (2**q, n)`` product, so the number of circuit
evaluations is linear in ``n`` and the per-pair cost falls as the band grows.  Measured at
8 qubits on the 300-point screening block, ``results/tables/screens.csv`` records a mean
``gram_seconds`` of 0.186 s -- about 0.62 ms per state, or 4.2 microseconds per pair at that
size -- so a 2,400-point band is a few seconds, not minutes.  The 19.5 microseconds per pair
recorded in ``docs/decisions.md`` D-006 is a measurement of the pairwise
``FidelityStatevectorKernel``, which is no longer in this tree; quoting it against the route
below would attribute a quadratic loop's cost to the linear one.  The naive route of building
a compute-uncompute circuit per pair is quadratic in circuit executions and is what makes
hardware implementations of this so expensive.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from qiskit import QuantumCircuit

__all__ = ["BACKENDS", "KernelBackendError", "fidelity_gram", "statevectors"]

Backend = Literal["statevector", "braket", "aer_cpu", "aer_gpu"]
BACKENDS: tuple[Backend, ...] = ("statevector", "braket", "aer_cpu", "aer_gpu")


class KernelBackendError(RuntimeError):
    """A requested backend is unavailable or did not behave as requested.

    Raised rather than falling back.  A silent downgrade from GPU to CPU, or from Braket to
    Qiskit, would make the agreement check vacuous: it would be comparing a backend against
    itself under two names.
    """


def _bind(circuit: QuantumCircuit, row: np.ndarray) -> QuantumCircuit:
    return circuit.assign_parameters(np.asarray(row, dtype=float))


def statevectors(circuit: QuantumCircuit, x: np.ndarray, backend: Backend) -> np.ndarray:
    """Amplitudes for every row of ``x``, one circuit evaluation each.

    Returns an ``(n, 2**q)`` complex array.  Computing states once and forming overlaps
    afterwards is what keeps the Gram matrix linear in circuit evaluations.
    """
    rows = np.asarray(x, dtype=float)
    if rows.shape[1] != circuit.num_parameters:
        raise ValueError(
            f"circuit takes {circuit.num_parameters} parameters, got {rows.shape[1]} features"
        )

    if backend == "statevector":
        from qiskit.quantum_info import Statevector

        return np.stack([Statevector(_bind(circuit, r)).data for r in rows])

    if backend in ("aer_cpu", "aer_gpu"):
        try:
            from qiskit import transpile
            from qiskit_aer import AerSimulator
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise KernelBackendError(
                f"{backend} needs the gpu-crosscheck extra: pip install -e '.[gpu-crosscheck]'"
            ) from exc

        device = "GPU" if backend == "aer_gpu" else "CPU"
        options = {"cuStateVec_enable": True} if device == "GPU" else {}
        simulator = AerSimulator(method="statevector", device=device, **options)
        if device == "GPU" and "GPU" not in AerSimulator().available_devices():
            raise KernelBackendError(
                "Aer reports no GPU device. Install order matters: the GPU wheel must be "
                "reinstalled last (see docs/decisions.md D-007)."
            )

        out = []
        for r in rows:
            probe = _bind(circuit, r)
            probe.save_statevector()
            result = simulator.run(transpile(probe, simulator)).result()
            if result.results[0].metadata.get("device") != device:
                raise KernelBackendError(
                    f"asked Aer for {device}, it dispatched to "
                    f"{result.results[0].metadata.get('device')!r}"
                )
            out.append(np.asarray(result.get_statevector()))
        return np.stack(out)

    if backend == "braket":
        try:
            from braket.circuits import Circuit
            from braket.devices import LocalSimulator
        except ImportError as exc:
            raise KernelBackendError("amazon-braket-sdk is not installed") from exc

        device = LocalSimulator("braket_sv")
        out = []
        for r in rows:
            bound = _bind(circuit, r)
            braket_circuit = Circuit.from_ir(_to_openqasm(bound))
            braket_circuit.state_vector()
            result = device.run(braket_circuit, shots=0).result()
            out.append(np.asarray(result.values[0]))
        return np.stack(out)

    raise KernelBackendError(f"unknown backend {backend!r}; expected one of {BACKENDS}")


# Braket's OpenQASM dialect names two of the gates this study uses differently from Qiskit's,
# and does not resolve the `stdgates.inc` include.  The mapping is a literal rename of gate
# tokens, not a reconstruction of the circuit: gate order, qubit indices and parameter values
# pass through untouched, so the two backends really do execute the same circuit.  The parity
# check in scripts/check_parity.py is what confirms that claim rather than assuming it.
_BRAKET_GATE_NAMES = {
    "p": "phaseshift",
    "cx": "cnot",
    "sdg": "si",
    "tdg": "ti",
    "u1": "phaseshift",
}


def _to_openqasm(circuit: QuantumCircuit) -> str:
    """Qiskit circuit to the OpenQASM 3 dialect Braket's local simulator accepts.

    Going through OpenQASM rather than rebuilding the circuit with Braket's Python API is
    deliberate.  A circuit assembled twice by hand can differ in gate order or in a rotation
    convention, and the agreement check would then report a real discrepancy as a numerical
    one -- or, worse, hide a real one behind a compensating difference.
    """
    import re

    from qiskit.qasm3 import dumps

    lines = []
    for line in dumps(circuit).splitlines():
        if "stdgates.inc" in line:
            continue  # Braket provides the standard gates without an include
        # Rename only a leading gate token, so a qubit register called `p` is untouched.
        lines.append(
            re.sub(
                r"^(\s*)([a-z][a-z0-9]*)(\s*[\(\s])",
                lambda m: m.group(1) + _BRAKET_GATE_NAMES.get(m.group(2), m.group(2)) + m.group(3),
                line,
            )
        )
    return "\n".join(lines)


def fidelity_gram(
    circuit: QuantumCircuit,
    x: np.ndarray,
    *,
    backend: Backend = "statevector",
    y: np.ndarray | None = None,
) -> np.ndarray:
    """Fidelity kernel matrix ``|<phi(x_i)|phi(y_j)>|^2``.

    With ``y`` omitted this is the symmetric Gram matrix of ``x``; with ``y`` supplied it is
    the cross-kernel needed to score new points against support vectors.

    The diagonal of the symmetric case is set to exactly 1 rather than left to accumulate
    floating-point error.  ``|<phi|phi>|^2`` is 1 analytically, and a diagonal that reads
    0.9999999997 propagates into the eigenspectrum and shifts the effective-rank screen.
    """
    states_x = statevectors(circuit, x, backend)
    if y is None:
        gram = np.abs(states_x.conj() @ states_x.T) ** 2
        np.fill_diagonal(gram, 1.0)
        return gram
    states_y = statevectors(circuit, y, backend)
    return np.abs(states_x.conj() @ states_y.T) ** 2
