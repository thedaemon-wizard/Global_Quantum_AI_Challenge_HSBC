# SPDX-License-Identifier: Apache-2.0
"""The reassociated contraction, and the properties that make it safe to use.

The site loop was rewritten as a binary reduction tree because the original was bound by
kernel launches rather than arithmetic. Reassociating a matrix chain is only legitimate
because the accumulated log-norm is discarded, so the output depends on the DIRECTION of the
product and nothing else; and it is only safe if the site ordering survives, which is the
class of defect this classifier has already shipped twice.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hsbcfraud.quantum.mps import (
    CHUNK_CANDIDATES,
    MPSClassifier,
    MPSConfig,
    tune_contraction_chunk,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fit requires CUDA")

WIDTHS = (1, 8, 32, 128, 439)


def build(sites: int, chi: int, chunk: int) -> MPSClassifier:
    torch.manual_seed(0)
    model = MPSClassifier(sites, MPSConfig(bond_dimension=chi, seed=0, contraction_chunk=chunk))
    return model.cuda()


@pytest.mark.parametrize(("sites", "chi"), [(439, 4), (439, 32), (431, 16), (64, 8), (9, 4)])
def test_every_width_agrees_with_the_sequential_fold(sites: int, chi: int) -> None:
    """Reassociation changes the rounding, not the answer.

    Tolerance is on the softmax probability rather than the logit: logits pass through a
    softmax before any decision, and a relative tolerance on a logit near zero is not a
    meaningful quantity.
    """
    reference = build(sites, chi, 1)
    x = torch.rand(256, sites, device="cuda")
    with torch.no_grad():
        expected = torch.softmax(reference(x), dim=-1)[:, 1]
    for width in WIDTHS:
        if width > sites:
            continue
        reference.contraction_chunk = width
        with torch.no_grad():
            actual = torch.softmax(reference(x), dim=-1)[:, 1]
        assert float((expected - actual).abs().max()) < 1e-4, f"width {width}"


def test_site_ordering_survives_the_reduction(sites: int = 439) -> None:
    """Matrix multiplication is associative but not commutative.

    A tail handled on the wrong side, or a stride that pairs the wrong neighbours, permutes
    the chain silently: the model still trains and still reports plausible metrics. This
    classifier has shipped two bugs of that shape, one of which gave an AUC of exactly 0.5000
    from an input-independent contraction.

    The test establishes its own teeth. Random cores replace the identity backbone so that
    order genuinely matters, and swapping two interior sites must move the output by orders
    of magnitude more than the reassociation does. Without that check the agreement below
    would be consistent with a chain whose factors happen to commute.
    """
    model = build(sites, 8, 1)
    with torch.no_grad():
        for core in model.cores:
            core.copy_(torch.randn_like(core) * 0.4)
    x = torch.rand(64, sites, device="cuda")

    with torch.no_grad():
        reference = model(x)
    model.contraction_chunk = sites
    with torch.no_grad():
        reassociated = model(x)

    swapped = build(sites, 8, 1)
    swapped.load_state_dict(model.state_dict())
    left, right = sites // 3, 2 * sites // 3
    with torch.no_grad():
        held = swapped.cores[left].clone()
        swapped.cores[left].copy_(swapped.cores[right])
        swapped.cores[right].copy_(held)
        permuted = swapped(x)

    reassociation = float((reference - reassociated).abs().max())
    permutation = float((reference - permuted).abs().max())
    assert permutation > 1e-2, "the fixture does not distinguish orderings; the test is vacuous"
    assert reassociation < permutation / 1000, (
        f"reassociation moved the output by {reassociation:.2e} against {permutation:.2e} for a "
        "single interior swap; that is too close to rule out a reordering"
    )


def test_gradients_agree_so_the_training_trajectory_is_unchanged() -> None:
    """A matching forward is not sufficient evidence that the model trains the same.

    The normalisation points differ between the fold and the tree, which changes the scale of
    intermediate gradients even when the output agrees. What matters downstream is the
    gradient direction and its norm, because the training loop now guards on the norm.
    """
    x = torch.rand(256, 439, device="cuda")
    y = torch.randint(0, 2, (256,), device="cuda")
    loss_fn = torch.nn.CrossEntropyLoss()

    def gradients(width: int) -> list[torch.Tensor]:
        model = build(439, 32, width)
        model.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        return [p.grad.detach().clone() for p in model.parameters()]

    reference, tree = gradients(1), gradients(439)
    dot = sum(float((a * b).sum()) for a, b in zip(reference, tree, strict=True))
    norm_a = sum(float((a * a).sum()) for a in reference) ** 0.5
    norm_b = sum(float((b * b).sum()) for b in tree) ** 0.5
    assert dot / (norm_a * norm_b) > 1 - 1e-6
    assert abs(norm_b / norm_a - 1) < 1e-4


def test_width_one_is_the_sequential_fold() -> None:
    """The default path must be the shipped behaviour, not an approximation of it.

    A model that has never been tuned contracts with width 1, so this is what runs unless a
    caller opts in.
    """
    assert build(64, 8, 1).contraction_chunk == 1
    assert MPSConfig().contraction_chunk is None
    assert MPSClassifier(8, MPSConfig(bond_dimension=4)).contraction_chunk == 1


def test_tuning_prefers_the_tree_at_small_bond_and_the_fold_at_large() -> None:
    """The crossover is arithmetic, not an artefact of one machine.

    The fold does 6.b.d.chi^2 flops; the tree does 4.b.d.chi^2 + 2.b.d.chi^3, so the tree
    performs (2 + chi)/3 times the work in exchange for about three thousand fewer launches.
    A crossover therefore exists by construction, and the tuner has to find it rather than
    assume a direction.
    """
    small, small_timings = tune_contraction_chunk(439, MPSConfig(bond_dimension=4, seed=0))
    large, large_timings = tune_contraction_chunk(439, MPSConfig(bond_dimension=128, seed=0))
    assert small > 1, f"the tree should win at chi=4; timings {small_timings}"
    assert large < small, (
        f"the fold should be relatively better at chi=128: picked {large} against {small} "
        f"at chi=4; timings {large_timings}"
    )


def test_tuning_reports_every_width_it_could_measure() -> None:
    """A width skipped for memory and a width that lost are different facts."""
    _, timings = tune_contraction_chunk(64, MPSConfig(bond_dimension=4, seed=0))
    assert 1 in timings
    assert set(timings) <= {c for c in CHUNK_CANDIDATES if c <= 64} | {64}
    assert all(np.isfinite(v) and v > 0 for v in timings.values())
