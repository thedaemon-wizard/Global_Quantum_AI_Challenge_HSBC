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

import math
import time
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
    # Sites folded into one reduction tree before the running vector is renormalised.  None
    # means measure it: the best width depends on the bond dimension and on the device, and a
    # table baked from one GPU is a number that silently stops being true on another.  1 is
    # the original sequential fold.  See MPSClassifier.forward.
    contraction_chunk: int | None = None
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

        # Held on the model rather than read from the config at call time, so predict_proba
        # contracts the same way the fit did.  A model that was never tuned falls back to the
        # sequential fold, which is correct everywhere and fastest at large bond dimensions.
        self.contraction_chunk = config.contraction_chunk or 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Contract the network against a batch, returning two class logits.

        The contraction is reassociated into a binary reduction tree, which is where almost
        all of the runtime went.

        **Why reassociating is legitimate.**  The accumulated log-norm is discarded rather
        than returned (D-031), and the loop's last act is to divide by the running norm, so
        the output is exactly ``normalise(v0 @ M_1 @ ... @ M_d) @ head`` where ``M_i`` is the
        transfer matrix at site ``i``.  The per-site renormalisation exists only to keep the
        product inside float32 range; it cannot change the direction of the result.  Matrix
        multiplication is associative, so any bracketing of that chain computes the same
        direction, and a pairwise tree finishes in ``log2(d)`` rounds instead of ``d``.

        **Why it is worth doing.**  The site loop was issuing about seven kernels per site,
        each on a ``(batch, chi, chi)`` tensor far too small to occupy the device.  Measured
        at 439 sites, batch 512: per-step time was flat across bond dimensions 4 to 128
        (137, 126, 131, 137, 131, 127 ms) despite a 1024-fold change in arithmetic, and
        linear in the number of sites at about 300 microseconds each.  Achieved throughput
        was 0.0067 TFLOPS against roughly 125 TFLOPS of fp32 peak -- about 0.005 per cent.
        The work was launch-bound, not compute-bound.

        **Why the tree is not always faster, and what ``contraction_chunk`` is for.**  The
        fold does ``d`` matrix-vector products, ``6.b.d.chi^2`` flops in total; the tree does
        about ``d`` matrix-matrix products, ``4.b.d.chi^2 + 2.b.d.chi^3``.  The tree
        therefore performs ``(2 + chi)/3`` times the arithmetic -- twice as much at ``chi=4``
        and forty-three times as much at ``chi=128`` -- in exchange for roughly three thousand
        fewer launches.  A crossover exists by arithmetic alone.  Measured per-step time at
        439 sites, batch 512, with the conditions interleaved to cancel drift:

        =====  ==========  ==========  ==========  ===========  ===========
        chi    chunk 1     chunk 8     chunk 32    chunk 128    chunk 439
        =====  ==========  ==========  ==========  ===========  ===========
        4      134.8 ms    60.0 ms     27.6 ms     17.0 ms      11.5 ms
        8      145.5 ms    66.0 ms     29.7 ms     17.5 ms      12.2 ms
        16     148.1 ms    64.1 ms     30.6 ms     17.9 ms      17.0 ms
        32     144.6 ms    67.9 ms     47.9 ms     66.0 ms      82.3 ms
        64     143.0 ms    79.0 ms     119.9 ms    138.6 ms     155.7 ms
        128    142.3 ms    459.0 ms    546.7 ms    574.1 ms     627.2 ms
        =====  ==========  ==========  ==========  ===========  ===========

        The best width falls as the bond dimension rises, and by ``chi = 128`` the original
        sequential fold is already optimal.  ``contraction_chunk = 1`` reproduces it exactly,
        which is why it is the safe value rather than a special case.

        Memory moves the other way: the tree materialises every transfer matrix at once, so
        peak allocation grows with the chunk width.  At ``chi = 32`` the fold peaks at
        0.99 GiB and the full tree at 3.93 GiB.  Memory, not time, is what bounds the bond
        dimension here.
        """
        chunk = max(1, min(self.contraction_chunk, self.n_features))
        phi = local_feature_map(x)  # (batch, d, 2)

        # Site 0 has a left bond of 1 while every other site has chi, so it cannot join the
        # stacked batch.  It is contracted separately rather than padded: a padded core would
        # put zero rows into the reduction, and a zero row that later picks up a nan is
        # exactly the kind of defect this file already carries two entries about.
        partial = torch.einsum("pr,bp->br", self.cores[0][0], phi[:, 0, :])
        partial = self._renormalise(partial)
        if self.n_features == 1:
            return torch.einsum("bl,lc->bc", partial, self.head)

        # One batched einsum builds every remaining transfer matrix, replacing d separate
        # launches with one.
        stack = torch.stack(list(self.cores[1:]), dim=0)  # (d-1, chi, 2, chi)
        n_sites = stack.shape[0]

        for start in range(0, n_sites, chunk):
            stop = min(start + chunk, n_sites)
            mats = torch.einsum(
                "dlpr,bdp->bdlr", stack[start:stop], phi[:, 1 + start : 1 + stop, :]
            )
            partial = torch.einsum("bl,blr->br", partial, self._reduce(mats))
            partial = self._renormalise(partial)

        return torch.einsum("bl,lc->bc", partial, self.head)

    @staticmethod
    def _reduce(mats: torch.Tensor) -> torch.Tensor:
        """Collapse ``(batch, k, chi, chi)`` to ``(batch, chi, chi)`` by pairwise products.

        Site order is preserved because the even-indexed factor is always the LEFT operand:
        ``mats[:, 0::2] @ mats[:, 1::2]`` pairs sites (0,1), (2,3), ... in place, and an odd
        tail is carried forward unpaired so it stays rightmost.  Matrix multiplication is
        associative but not commutative, so a tail appended on the wrong side would silently
        permute the chain -- the model would still train and still report plausible metrics.
        ``tests/test_mps_contraction.py`` pins this against a deliberate site swap, which
        moves the output by order 1 while the reassociation moves it by order 1e-5.

        Each round divides by the largest absolute entry rather than a Frobenius norm.  The
        products square in magnitude every round, so the quantity that must stay in range is
        the extreme entry, and max-abs bounds it directly.
        """
        while mats.shape[1] > 1:
            count = mats.shape[1]
            paired = torch.matmul(mats[:, 0 : count - count % 2 : 2], mats[:, 1:count:2])
            if count % 2:
                paired = torch.cat([paired, mats[:, -1:]], dim=1)
            scale = torch.clamp(paired.abs().amax(dim=(-2, -1), keepdim=True), min=1e-12)
            mats = paired / scale
        return mats[:, 0]

    @staticmethod
    def _renormalise(partial: torch.Tensor) -> torch.Tensor:
        """Divide by the per-sample norm, clamped away from zero.

        Without it the product underflows to zero within about forty sites in float32.  The
        discarded scale is identical across both logits and cancels in the softmax.
        """
        norm = torch.clamp(torch.linalg.vector_norm(partial, dim=-1, keepdim=True), min=1e-12)
        return partial / norm

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


# Candidate reduction widths.  Powers of two spanning the sequential fold to the full tree;
# the measured optimum moves across this range with the bond dimension, so the set has to be
# wide rather than centred on any one machine's answer.
CHUNK_CANDIDATES = (1, 8, 32, 128, 512)


def tune_contraction_chunk(
    n_features: int, config: MPSConfig, *, batch_size: int | None = None, repeats: int = 3
) -> tuple[int, dict[int, float]]:
    """Time each candidate reduction width and return the fastest, with the measurements.

    Measured rather than tabulated.  The best width depends on the bond dimension, the number
    of sites, the batch size and the device, and the crossover is sharp: at 439 sites the full
    tree is 11.7 times faster than the fold at ``chi = 4`` and 4.4 times *slower* at
    ``chi = 128``.  A table baked from one GPU is a number that silently stops being true on
    another, and being wrong here costs hours.

    Tuning runs on a throwaway model with random inputs, so it cannot perturb the fit: no
    parameter of the real model is touched and no data is read.  The cost is
    ``len(CHUNK_CANDIDATES) * (repeats + 1)`` steps, about twenty against the twenty-one
    thousand a full-scale job runs.

    Candidates that exhaust memory are skipped and reported as such rather than silently
    dropped, because "the tree did not help" and "the tree did not fit" are different facts.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("chunk tuning requires CUDA; pass contraction_chunk explicitly")

    batch = batch_size or config.batch_size
    device = torch.device(config.device)
    probe = MPSClassifier(n_features, config).to(device)
    x = torch.rand(batch, n_features, device=device)
    y = torch.randint(0, 2, (batch,), device=device)
    loss_fn = torch.nn.CrossEntropyLoss()

    # The full tree is always a candidate.  Without it the search tops out at the largest
    # power of two below the site count, and at small bond dimensions the full tree is the
    # fastest option by a clear margin.
    candidates = sorted({c for c in CHUNK_CANDIDATES if c <= n_features} | {1, n_features})
    timings: dict[int, float] = {}
    for candidate in candidates:
        probe.contraction_chunk = candidate
        try:
            for _ in range(2):  # warmup: the first pass pays allocator and kernel setup
                probe.zero_grad(set_to_none=True)
                loss_fn(probe(x), y).backward()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(repeats):
                probe.zero_grad(set_to_none=True)
                loss_fn(probe(x), y).backward()
            torch.cuda.synchronize()
            timings[candidate] = (time.perf_counter() - start) / repeats
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

    if not timings:
        raise RuntimeError(
            f"no reduction width fitted in memory at {n_features} sites, bond dimension "
            f"{config.bond_dimension}, batch {batch}. Reduce the batch size."
        )
    del probe, x, y
    torch.cuda.empty_cache()
    return min(timings, key=timings.__getitem__), timings


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

    if config.contraction_chunk is None:
        chunk, timings = tune_contraction_chunk(x_train.shape[1], config)
        model.contraction_chunk = chunk
        if reporter is not None:
            reporter.note(
                "reduction width "
                + ", ".join(f"{c}: {t * 1000:.1f} ms" for c, t in sorted(timings.items()))
                + f" -- using {chunk}",
                contraction_chunk=chunk,
            )

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
                # Dump what the watch is already holding.  Without it the post-mortem knows
                # only which epoch failed, and an epoch is hundreds of steps.
                if reporter is not None:
                    reporter.note(
                        f"loss became non-finite at step {step}; dumping the preceding steps",
                        step=step,
                        learning_rate=schedule.get_last_lr()[0],
                    )
                    for record in watch.recent_trace():
                        reporter.trace(record)
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
            if not math.isfinite(grad_norm):
                # Stop here, before the optimiser writes.  With a non-finite total norm the
                # clip coefficient is 1.0/inf = 0, so every gradient is multiplied by zero
                # except the offending element, where inf * 0 is nan -- and Adam then writes
                # that nan into a parameter.  Measured on a 64-site chain: at this point the
                # loss is still 0.5927 and all 8,096 parameters are finite; one step later a
                # parameter is nan and the next forward pass is non-finite.  Checking the
                # loss therefore stops the run one step too late, with the state already
                # corrupted, and reports the wrong step as the failure.
                if reporter is not None:
                    reporter.note(
                        f"gradient norm became non-finite at step {step}, before the "
                        f"optimiser step; parameters are still finite",
                        step=step,
                        learning_rate=schedule.get_last_lr()[0],
                        loss=float(loss.detach()),
                    )
                    for record in watch.recent_trace():
                        reporter.trace(record)
                raise RuntimeError(
                    f"MPS gradient norm became non-finite at {x_train.shape[1]} sites, step "
                    f"{step}, while the loss was still {float(loss.detach()):.4f}. The "
                    "optimiser step was not taken, so the model parameters are unchanged "
                    "and finite."
                )
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
