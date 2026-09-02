# SPDX-License-Identifier: Apache-2.0
"""The guards around the matrix-product-state fit.

These pin behaviour that was wrong once.  Two silent bugs in this classifier produced
plausible near-chance numbers rather than errors, and a third let a non-finite gradient reach
the optimiser before anything noticed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import hsbcfraud.quantum.mps as mps

# Only the two fits below need a GPU.  A module-level skip put the CPU-refusal test behind the
# same condition it exists to exercise, and its own inner guard skipped on the complement, so
# `pytest.raises` in that test was unreachable on every machine: with CUDA the module mark
# fired, without CUDA the inner guard did.
needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="this fit runs on the GPU by design"
)


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(2048, 64)).astype(np.float32)
    y = (rng.uniform(size=2048) < 0.2).astype(int)
    return x, y


@needs_cuda
def test_non_finite_gradient_stops_before_the_optimiser_writes(data, monkeypatch) -> None:
    """The loss check alone fires one step too late, with the parameters already corrupted.

    With a non-finite total norm the clip coefficient is ``1.0 / inf = 0``, so every gradient
    is multiplied by zero except the offending element, where ``inf * 0`` is nan -- and Adam
    writes that nan into a parameter.  Measured on a 64-site chain: at the moment the norm
    goes non-finite the loss is still around 0.46 and every parameter is finite; one step
    later a parameter is nan and the next forward pass is non-finite.  So the norm is the
    signal that permits a clean stop, and the loss is not.
    """
    x, y = data
    real_clip = torch.nn.utils.clip_grad_norm_
    captured: dict[str, object] = {"calls": 0, "model": None}

    def poisoned(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        captured["calls"] += 1
        if captured["calls"] == 40:
            target = next(p for p in parameters if p.grad is not None and p.dim() >= 3)
            target.grad.view(-1)[0] = float("inf")
        return real_clip(parameters, max_norm, *args, **kwargs)

    real_init = mps.MPSClassifier.__init__

    def capturing_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        captured["model"] = self

    monkeypatch.setattr(mps.torch.nn.utils, "clip_grad_norm_", poisoned)
    monkeypatch.setattr(mps.MPSClassifier, "__init__", capturing_init)

    with pytest.raises(RuntimeError, match="gradient norm became non-finite"):
        mps.fit_mps(x, y, mps.MPSConfig(bond_dimension=8, epochs=20, seed=0, device="cuda"))

    model = captured["model"]
    assert model is not None
    non_finite = sum(int((~torch.isfinite(p)).sum()) for p in model.parameters())
    assert non_finite == 0, "the run must stop with parameters that are still usable"
    with torch.no_grad():
        probe = model(torch.as_tensor(x[:8]).cuda())
    assert bool(torch.isfinite(probe).all())


@needs_cuda
def test_the_failure_names_the_step_and_the_loss_at_that_step(data, monkeypatch) -> None:
    """A post-mortem needs the step, not just the fact.

    The loss is reported alongside because its plausibility is the point: it is what makes
    the loss an unusable trigger.
    """
    x, y = data
    real_clip = torch.nn.utils.clip_grad_norm_
    calls = {"n": 0}

    def poisoned(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        calls["n"] += 1
        if calls["n"] == 40:
            next(p for p in parameters if p.grad is not None and p.dim() >= 3).grad.view(-1)[
                0
            ] = float("inf")
        return real_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(mps.torch.nn.utils, "clip_grad_norm_", poisoned)
    with pytest.raises(RuntimeError, match=r"step 39, while the loss was still 0\.\d+"):
        mps.fit_mps(x, y, mps.MPSConfig(bond_dimension=8, epochs=20, seed=0, device="cuda"))


def test_fit_refuses_the_cpu_rather_than_falling_back(monkeypatch) -> None:
    """A run silently demoted to CPU reports a timing wrong by an order of magnitude, and
    the bond-dimension sweep is partly a compute-cost measurement.

    The absence of CUDA is simulated rather than waited for.  Gating this on a CPU-only host
    made the assertion unreachable, because that condition was the exact complement of the
    module-level skip: the guard shipped with nothing exercising it on either kind of machine.
    """
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(64, 8)).astype(np.float32)
    y = (rng.uniform(size=64) < 0.2).astype(int)
    config = mps.MPSConfig(bond_dimension=4, epochs=1, seed=0, device="cuda")
    monkeypatch.setattr(mps.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="refusing to silently use the CPU"):
        mps.fit_mps(x, y, config)
