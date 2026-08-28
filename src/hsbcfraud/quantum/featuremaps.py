# SPDX-License-Identifier: Apache-2.0
"""Feature maps for the quantum kernel, including the entanglement-removed controls.

Encoding choices, and why each is here
--------------------------------------
``zz``  The standard second-order Pauli-Z map.  Entangling, and the most commonly used
        encoding in the fraud-detection quantum literature, which is why it is the arm a
        reviewer will expect to see.

``z``   First-order Pauli-Z: a product state, no entanglement at all.  This is the control
        Bowles, Ahmed and Schuld (arXiv:2403.07059) demand -- they found that removing
        entanglement from a quantum model "often results in as good or better performance",
        so a quantum arm reported without a product-state control has not been tested.

``dense_angle``  One rotation per feature per qubit, packing two features into each qubit
        through Ry and Rz.  Included because it uses half the qubits for the same feature
        count, which moves the concentration behaviour, and because it is the encoding the
        quantum-autoencoder line of work favours.

Bandwidth
---------
Every map takes a bandwidth ``c`` that scales the input angles.  This is not a convenience
knob.  Fidelity kernels concentrate exponentially in qubit count (Thanasilp, Wang, Cerezo and
Holmes, *Nature Communications* 15:5200, 2024), and bandwidth tuning is the standard escape
-- but Slattery et al. (*Physical Review A* 107:062417, 2023) and Florez-Ablan, Roth and
Schnabel (arXiv:2503.05602) show that the bandwidth which restores conditioning also drives
the kernel toward a radial basis function.

So bandwidth is swept rather than tuned, and the resulting kernels are scored on **two** axes
at once by :mod:`hsbcfraud.quantum.screens`: whether they are well conditioned, and whether
they remain distinguishable from an RBF kernel.  A map that needs a bandwidth at which it has
become an approximate RBF has not been rescued; it has been dequantised.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

__all__ = ["FEATURE_MAPS", "build_feature_map", "scale_features"]

FeatureMapName = Literal["zz", "z", "dense_angle"]
Entanglement = Literal["linear", "full", "none"]

FEATURE_MAPS: tuple[FeatureMapName, ...] = ("zz", "z", "dense_angle")


def scale_features(x: np.ndarray, bandwidth: float) -> np.ndarray:
    """Map features into rotation angles at the given bandwidth.

    Inputs are assumed already min-max scaled to [0, 1] on the block that fixed the band;
    this multiplies into [0, pi * bandwidth].  Fitting the scaler anywhere other than that
    block would leak, so the scaler is fitted upstream and passed in.
    """
    return np.asarray(x, dtype=float) * (np.pi * bandwidth)


def build_feature_map(
    name: FeatureMapName,
    n_features: int,
    *,
    reps: int = 2,
    entanglement: Entanglement = "linear",
) -> QuantumCircuit:
    """Construct a parameterised encoding circuit.

    Written explicitly rather than assembled from library builders so that the
    entanglement-removed control is the *same circuit* with the entangling layer omitted,
    rather than a different library function that might also differ in rotation structure.
    An ablation that changes two things at once measures neither.
    """
    if n_features < 1:
        raise ValueError(f"need at least one feature, got {n_features}")
    # Dense angle encoding packs two features per qubit; every other map is one-to-one.
    n_qubits = (n_features + 1) // 2 if name == "dense_angle" else n_features

    params = ParameterVector("x", n_features)
    circuit = QuantumCircuit(n_qubits, name=f"{name}_r{reps}_{entanglement}")

    for _ in range(reps):
        if name == "dense_angle":
            for q in range(n_qubits):
                first = 2 * q
                second = 2 * q + 1
                circuit.ry(params[first], q)
                if second < n_features:
                    circuit.rz(params[second], q)
        else:
            circuit.h(range(n_qubits))
            for q in range(n_qubits):
                circuit.p(2.0 * params[q], q)

        if name == "zz" and entanglement != "none":
            for a, b in _entangling_pairs(n_qubits, entanglement):
                circuit.cx(a, b)
                # The (pi - x_a)(pi - x_b) product is the second-order term that makes this
                # map non-factorising; without it the circuit is a product state regardless
                # of the CX gates.
                circuit.p(2.0 * (np.pi - params[a]) * (np.pi - params[b]), b)
                circuit.cx(a, b)
        elif name == "dense_angle" and entanglement != "none":
            for a, b in _entangling_pairs(n_qubits, entanglement):
                circuit.cx(a, b)

    return circuit


def _entangling_pairs(n_qubits: int, entanglement: Entanglement) -> list[tuple[int, int]]:
    if entanglement == "none" or n_qubits < 2:
        return []
    if entanglement == "linear":
        return [(q, q + 1) for q in range(n_qubits - 1)]
    if entanglement == "full":
        return [(a, b) for a in range(n_qubits) for b in range(a + 1, n_qubits)]
    raise ValueError(f"unknown entanglement {entanglement!r}")
