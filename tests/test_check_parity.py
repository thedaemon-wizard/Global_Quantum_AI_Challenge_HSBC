# SPDX-License-Identifier: Apache-2.0
"""The cross-backend parity gate must fail on disagreement even when it declines to write.

`check_parity.py` gained a branch that refuses to overwrite the committed twelve-row table with
the four rows a default environment can produce.  That branch returned 0 **before** the
agreement check twenty lines below it, and printed "Every comparison that did run agreed with
the reference" as an unguarded string literal.

So in exactly the environment the documented procedure produces -- `make venv` installs
`.[dev]`, and `qiskit-aer` lives in the optional `gpu-crosscheck` extra -- a real numerical
disagreement in the Braket rows exited 0 with a success message.  The four-backend agreement is
this study's stated substitute for hardware execution, so the single gate behind that claim
could not fail.  Nothing tested this script at all before these two tests.

`fidelity_gram` is stubbed rather than run: what is under test is `main`'s control flow, not the
linear algebra, and stubbing keeps the test free of any quantum backend.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# `main` skips the write only when a *wider* table is already committed, so the fixture has to be
# wider than the four rows the stub produces.
COMMITTED_ROWS = 12


def _load():
    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "check_parity_under_test", REPO / "scripts" / "check_parity.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_backends(monkeypatch, module, *, braket_offset: float) -> None:
    """Reference and Braket return Grams differing by `braket_offset`; Aer is unavailable.

    This reproduces the default environment: `REQUIRED` is (statevector, braket), so the two Aer
    backends raising is not fatal and execution reaches the skip branch.
    """
    from hsbcfraud.quantum.kernel import KernelBackendError

    def fake_gram(circuit, x, *, backend):
        n = len(x)
        if backend == module.REFERENCE:
            return np.zeros((n, n))
        if backend == "braket":
            return np.full((n, n), braket_offset)
        raise KernelBackendError(f"{backend} is not installed in this environment")

    monkeypatch.setattr(module, "fidelity_gram", fake_gram)


def _committed_wider(directory: Path) -> Path:
    target = directory / "parity.csv"
    pd.DataFrame(
        [
            {
                "encoding": "z", "n_qubits": 4, "backend": "aer_cpu",
                "max_abs_difference": 0.0, "agrees": True,
            }
            for _ in range(COMMITTED_ROWS)
        ]
    ).to_csv(target, index=False)
    return target


def test_a_disagreement_fails_even_though_the_write_is_skipped(monkeypatch, tmp_path) -> None:
    """The regression: a narrower run, a wider committed table, and rows that disagree."""
    module = _load()
    _stub_backends(monkeypatch, module, braket_offset=0.5)
    target = _committed_wider(tmp_path)
    before = target.read_bytes()

    assert module.main(["--out", str(tmp_path)]) == 1, (
        "a comparison exceeding the tolerance must fail the gate whether or not the narrower "
        "result is written; this returned 0 and printed a success message"
    )
    assert target.read_bytes() == before, "the committed table must not be overwritten"


def test_agreement_still_skips_the_write_and_passes(monkeypatch, tmp_path) -> None:
    """The branch must keep doing its job: D-135 exists because overwriting broke `make check`."""
    module = _load()
    _stub_backends(monkeypatch, module, braket_offset=0.0)
    target = _committed_wider(tmp_path)
    before = target.read_bytes()

    assert module.main(["--out", str(tmp_path)]) == 0
    assert target.read_bytes() == before, (
        "a narrower but agreeing run must still leave the wider committed table alone"
    )
