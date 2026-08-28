# SPDX-License-Identifier: Apache-2.0
"""A matrix-product-state classifier, trained by gradient descent on the RTX 6000.

What this is, and what it is not
--------------------------------
Each feature is mapped to a two-dimensional local spin, and the classifier is the contraction
of those spins with a matrix product state carrying one open output index.  With bond
dimension ``chi`` the model has ``O(d * chi^2 * 2)`` parameters and represents a specific
low-rank slice of the full ``2^d``-dimensional product basis.  The construction is
Stoudenmire and Schwab, "Supervised Learning with Tensor Networks" (NeurIPS 2016,
arXiv:1605.05775); the local feature map is theirs.

**It runs entirely on classical hardware and claims no quantum advantage.**  Calling it
"quantum-inspired" is defensible only in the narrow sense that the ansatz and the truncation
argument come from many-body physics -- an MPS is the classically tractable corner of the same
Hilbert space a quantum circuit explores, and its bond dimension is exactly the entanglement
budget.  That is a statement about where the mathematics came from, not about what the
hardware is doing.  The challenge statement's In Scope list names ITensor alongside Qiskit and
Braket, so the arm is in scope; the honest framing is that it is the classically simulable end
of the same family.

Why it is worth running here
----------------------------
The quantum kernel arm was rejected by its a-priori screens: all 120 candidates failed, and
the failure mode was that no feature map is simultaneously well conditioned and
distinguishable from a radial basis function.  That failure is specific to *fidelity kernels*,
whose Gram entries concentrate exponentially in qubit count.  An MPS classifier has neither
property: there is no qubit ceiling, no shot noise, and the bond dimension is a continuous
capacity knob rather than a fixed exponential.  So it tests a different hypothesis on the same
band, which is the point of running it rather than a way of rescuing a negative result.

Optimisation
------------
Trained by Adam on the log-loss rather than by DMRG-style sweeping.  Sweeping is the canonical
method and adapts the bond dimension automatically, but it needs a careful gauge and a
truncation schedule; on 439 features and an 18-day budget a fixed-``chi`` gradient fit is the
defensible choice, and the bond dimension is swept explicitly instead so the capacity
dependence is measured rather than adapted away.

Numerical conditioning is the one thing this ansatz does badly by default.  A product of ``d``
matrices underflows or overflows long before ``d = 439``, so the contraction carries a running
log-norm and renormalises at every site.  Without it the loss is ``nan`` within a few steps,
silently, which is exactly the failure this project's discipline exists to catch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from hsbcfraud.progress import DivergenceWatch, ProgressReporter

__all__ = ["LEARNING_RATE_SCALE", "MPSClassifier", "MPSConfig", "local_feature_map"]

# Numerator of the depth-scaled learning rate; see MPSConfig.learning_rate.
LEARNING_RATE_SCALE = 0.05


@dataclass(frozen=True)
class MPSConfig:
    """Hyperparameters, all swept rather than tuned to a single value."""

    bond_dimension: int = 16
    epochs: int = 40
    batch_size: int = 512
    # None means "scale with chain length", which is what long chains require.  A single
    # fixed rate does not work across the range this study uses: measured on 2,000 rows over
    # five epochs, 3e-3 trains a 24-site chain but leaves a 431-site chain at its initial loss
    # (0.7113 -> 0.6899), while 3e-4 takes the same 431-site chain to 0.3573.  The gradient
    # passes through one einsum per site, so the effective step compounds with depth.
    # LEARNING_RATE_SCALE / n_features sets the initial rate, which then decays; the numerator
    # comes from a 30,000-row probe at 431 sites where 1e-4 reached loss 0.199 against 0.308
    # at 3.5e-4, so lower is both more stable and better at this depth.  Pass a float to
    # override.
    learning_rate: float | None = None
    weight_decay: float = 1e-5
    # Positive-class weight in the loss.  At a 3.5 % base rate an unweighted fit collapses to
    # the majority class; this is the study's documented class-imbalance handling for this arm.
    positive_weight: float = 8.0
    init_scale: float = 3e-3
    seed: int = 20260828
    device: str = "cuda"

    def resolved_learning_rate(self, n_features: int) -> float:
        """The learning rate actually used, given the chain length."""
        if self.learning_rate is not None:
            return self.learning_rate
        return LEARNING_RATE_SCALE / n_features


def local_feature_map(x: torch.Tensor) -> torch.Tensor:
    """Map each scaled feature to a two-dimensional local spin.

    ``phi(x) = [cos(pi x / 2), sin(pi x / 2)]`` for ``x`` in [0, 1], from Stoudenmire and
    Schwab.  The map is unit-norm at every site, which is what keeps the contraction bounded
    once the running renormalisation is in place, and it is injective on [0, 1] so no two
    feature values collide.

    Inputs are expected already min-max scaled on the block that fixed the band.  Values
    outside [0, 1] are clamped rather than extrapolated: beyond that range the map folds back
    on itself and two different feature values would map to the same spin.
    """
    x = torch.clamp(x, 0.0, 1.0)
    angle = (np.pi / 2.0) * x
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


class MPSClassifier(torch.nn.Module):
    """Binary classifier whose decision function is an MPS contraction.

    The network is ``d`` rank-4 cores of shape ``(chi, 2, chi)`` with one carrying an extra
    output leg, contracted left to right against the local feature maps.  Boundary cores are
    ``(1, 2, chi)`` and ``(chi, 2, 1)`` so the contraction closes to a scalar per class.
    """

    def __init__(self, n_features: int, config: MPSConfig) -> None:
        super().__init__()
        if n_features < 2:
            raise ValueError(f"an MPS needs at least two sites, got {n_features}")
        self.n_features = n_features
        self.config = config
        chi = config.bond_dimension
        generator = torch.Generator().manual_seed(config.seed)

        # Initialisation is the whole difficulty of this ansatz, and getting it wrong fails
        # silently.  Two constraints pull against each other.
        #
        # A fully random initialisation at this depth produces a contraction whose magnitude
        # varies over hundreds of orders of magnitude between samples, and the fit never
        # recovers.  So the network starts near a product state, with an identity backbone in
        # the bond indices.
        #
        # But the identity must be placed on ONE physical channel only.  An earlier version
        # set both `core[:, 0, :]` and `core[:, 1, :]` to the identity plus small noise; the
        # contraction then reduces to a scalar multiple of a fixed vector, the running
        # renormalisation divides that scalar out, and the model becomes exactly
        # input-independent.  Measured: logits bit-identical for all-zeros and all-ones input,
        # loss flat at 0.6931, holdout AUC 0.5000.  Nothing raised.
        #
        # The `cos` channel therefore carries the identity and the `sin` channel starts as a
        # small learnable perturbation, so the physical index carries signal from step one.
        cores = []
        for site in range(n_features):
            left = 1 if site == 0 else chi
            # Every right bond stays at chi, including the last.  Closing the final bond to 1
            # destroys the model: the partial contraction becomes a scalar, the running
            # renormalisation maps it to +/-1, and every sample collapses onto one of two
            # values before the head ever sees it.  Measured that way: AUC 0.4990, 0.4970 and
            # 0.5000 at bond dimensions 4, 12 and 32 on a linearly separable task.  The head
            # closes the network instead.
            right = chi
            core = torch.zeros(left, 2, right)
            core[:, 0, :] = torch.eye(left, right)
            core[:, 1, :] = config.init_scale * torch.randn(left, right, generator=generator)
            core[:, 0, :] = core[:, 0, :] + config.init_scale * torch.randn(
                left, right, generator=generator
            )
            cores.append(torch.nn.Parameter(core))
        self.cores = torch.nn.ParameterList(cores)

        # The output leg closes the network: a (chi, 2) head on the final open bond, kept
        # separate from the cores so that sweeping the bond dimension changes one thing.
        self.head = torch.nn.Parameter(
            torch.randn(chi, 2, generator=generator) / np.sqrt(chi)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Contract the network against a batch, returning two class logits.

        The running log-norm is what makes this trainable at 439 sites.  After each site the
        partial contraction is divided by its per-sample norm and the log of that norm is
        accumulated; the final logit is the log of the closed scalar plus the accumulated log
        norms.  Omitting this underflows to zero within about forty sites in float32.
        """
        phi = local_feature_map(x)  # (batch, d, 2)
        batch = phi.shape[0]

        # (batch, 1) left boundary
        partial = torch.ones(batch, 1, device=x.device, dtype=x.dtype)
        log_norm = torch.zeros(batch, device=x.device, dtype=x.dtype)

        for site in range(self.n_features):
            core = self.cores[site]  # (left, 2, right)
            # Contract the physical index with this site's spin, then the bond with the
            # running partial contraction.
            site_matrix = torch.einsum("lpr,bp->blr", core, phi[:, site, :])
            partial = torch.einsum("bl,blr->br", partial, site_matrix)

            norm = torch.linalg.vector_norm(partial, dim=-1, keepdim=True)
            norm = torch.clamp(norm, min=1e-12)
            partial = partial / norm
            log_norm = log_norm + torch.log(norm.squeeze(-1))

        # The accumulated log-norm is deliberately discarded rather than added back.  It is
        # identical across both logits, so it cancels in the softmax and cannot affect a
        # decision; and it reaches around -330 at 431 sites, so any attempt to restore the
        # scale multiplicatively overflows.  An earlier version kept it as
        # `log_norm.unsqueeze(-1) * 0.0`, which is not the same as dropping it: if log_norm
        # ever became infinite, `inf * 0.0` is nan and would silently poison the loss.
        del log_norm
        return torch.einsum("bl,lc->bc", partial, self.head)

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Positive-class probability for each row."""
        self.eval()
        device = next(self.parameters()).device
        out = []
        for start in range(0, len(x), batch_size):
            chunk = torch.as_tensor(
                np.asarray(x[start : start + batch_size], dtype=np.float32), device=device
            )
            out.append(torch.softmax(self(chunk), dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(out)


def fit_mps(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: MPSConfig,
    *,
    reporter: ProgressReporter | None = None,
    evaluate: Callable[[MPSClassifier], dict[str, float]] | None = None,
) -> tuple[MPSClassifier, list[float]]:
    """Fit by Adam on a class-weighted cross-entropy.

    Raises rather than falling back if the requested device is unavailable: a run silently
    demoted from GPU to CPU would report a timing that is wrong by an order of magnitude, and
    the bond-dimension sweep is partly a compute-cost measurement.

    ``reporter`` receives one tick per epoch and any divergence warning.  It is optional and
    supplied by the caller rather than constructed here, because nothing under
    ``src/hsbcfraud/`` writes or prints -- the library returns data and the scripts report it.
    Passing ``None`` leaves the loop as it was.

    ``evaluate`` is called once per epoch with the model in eval mode and its result attached
    to the tick.  It exists because a falling loss is not evidence of learning: two silent
    bugs in this classifier produced steadily decreasing loss with an AUC of exactly 0.5000,
    and only a ranking metric would have caught them.  It is optional because it costs a
    forward pass over the evaluation set -- measured at roughly 11 % of an epoch on the full
    115,534-row block, or about 2 % on a stratified subsample -- and the caller is the one
    who knows which trade it wants.

    Per-step telemetry is deliberately not conditional on being cheap: measured on this host
    with interleaved trials to cancel GPU contention, capturing the loss, the pre-clip
    gradient norm and the learning rate on every step costs 1.3 % against a 146 ms step. At
    431 sites the step is dominated by kernel launches, so the record is effectively free.
    """
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing to silently use the CPU")

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model = MPSClassifier(x_train.shape[1], config).to(device)

    xt = torch.as_tensor(np.asarray(x_train, dtype=np.float32), device=device)
    yt = torch.as_tensor(np.asarray(y_train, dtype=np.int64), device=device)
    weight = torch.tensor([1.0, config.positive_weight], device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    learning_rate = config.resolved_learning_rate(x_train.shape[1])
    optimiser = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=config.weight_decay
    )
    n = len(xt)
    steps = max(1, config.epochs * ((n + config.batch_size - 1) // config.batch_size))
    # Cosine decay to zero.  A fixed rate that is stable for a few hundred steps is not
    # necessarily stable for tens of thousands: the in-band arm (2,916 rows, 30 epochs, ~180
    # steps) trains at any rate tried, while the full-scale arm (356,216 rows, 30 epochs,
    # ~21,000 steps) trained for twenty minutes and then went non-finite at the same rate.
    # The failure is step-count dependent, not configuration dependent -- a 30,000-row probe
    # at 431 sites survived twelve epochs at every learning rate and clipping threshold
    # tested, including the one the full run died at.  Decaying the step is the standard
    # remedy for "trains well, then diverges", and unlike per-core renormalisation it does
    # not hurt the loss: that alternative was measured and left the probe at 0.588 against
    # 0.307 for the unmodified run.
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps)

    history: list[float] = []
    watch = DivergenceWatch()
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    step = 0
    for _ in range(config.epochs):
        model.train()
        order = torch.randperm(n, generator=generator).to(device)
        epoch_loss = 0.0
        epoch_norms: list[float] = []
        for start in range(0, n, config.batch_size):
            idx = order[start : start + config.batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[idx]), yt[idx])
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"MPS loss became non-finite at {x_train.shape[1]} sites, step {step}, "
                    f"with initial learning rate {learning_rate:.2e} and initialisation "
                    f"scale {config.init_scale:.2e}. The usable rate falls with both chain "
                    "length and step count; see MPSConfig."
                )
            loss.backward()
            # clip_grad_norm_ returns the total norm BEFORE clipping. The loop used to discard
            # it, which threw away the earliest available signal that a step was about to be
            # taken into a bad region -- the one that leads the loss by roughly a hundred
            # steps. See DivergenceWatch.
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimiser.step()
            schedule.step()
            step += 1

            batch_loss = float(loss.detach())
            epoch_loss += batch_loss * len(idx)
            epoch_norms.append(grad_norm)
            warning = watch.observe(batch_loss, grad_norm)
            if warning is not None and reporter is not None:
                reporter.note(warning, step=step, learning_rate=schedule.get_last_lr()[0])

        mean_loss = epoch_loss / n
        history.append(mean_loss)
        if reporter is not None:
            metrics: dict[str, float] = {
                "loss": mean_loss,
                "lr": schedule.get_last_lr()[0],
                "grad_norm_median": float(np.median(epoch_norms)),
                "grad_norm_max": float(np.max(epoch_norms)),
            }
            if evaluate is not None:
                metrics.update(evaluate(model))
            reporter.tick(**metrics)
    return model, history
