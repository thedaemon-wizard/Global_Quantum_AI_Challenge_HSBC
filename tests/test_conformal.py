# SPDX-License-Identifier: Apache-2.0
"""Split conformal, the class-conditional construction, and the exact coverage law.

Covers CP-1 (the order-statistic threshold), CP-2 (the Beta-Binomial coverage law), CP-3 (the
Mondrian construction), CP-4 (the degeneracy floor) and CP-5 (the weighted quantile with its
point mass at infinity).

``docs/decisions.md`` D-013 stated that "the tests assert that it reduces exactly to standard
split conformal at ``w_i = 1`` (verified to machine precision at three (n, alpha)
combinations)". That test did not exist. It is here, at exactly three combinations, because a
sentence that describes a test should describe the test that runs.
"""

from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

REPO = Path(__file__).resolve().parents[1]

from hsbcfraud.conformal.coverage import (
    beta_binomial_pmf,
    coverage_band,
    tail_probability,
    verify,
)
from hsbcfraud.conformal.split import (
    conformal_threshold,
    degeneracy_floor,
    mondrian_thresholds,
)
from hsbcfraud.conformal.weighted import geometric_weights, weighted_conformal_threshold

SEED = 20260828


# ------------------------------------------------------------------ CP-1, the threshold

@pytest.mark.parametrize(("n", "alpha"), [(100, 0.1), (847, 0.05), (2259, 0.01)])
def test_the_threshold_is_the_ceiling_order_statistic(n, alpha) -> None:
    """``qhat`` is the ``ceil((1-alpha)(n+1))``-th smallest calibration score."""
    rng = np.random.default_rng(SEED)
    scores = rng.random(n)
    qhat, k, reported_n = conformal_threshold(scores, alpha)
    assert reported_n == n
    assert k == int(np.ceil((n + 1) * (1 - alpha)))
    assert qhat == pytest.approx(np.sort(scores)[k - 1])


def test_the_threshold_is_infinite_when_the_sample_is_too_small() -> None:
    """"Never flag" is a real answer; clipping to the maximum would fabricate a guarantee."""
    qhat, k, n = conformal_threshold(np.array([0.1, 0.5, 0.9]), alpha=0.01)
    assert np.isinf(qhat)
    assert k > n


def test_a_non_finite_score_is_refused() -> None:
    """A NaN sorts to the end and would silently become the threshold."""
    with pytest.raises(ValueError):
        conformal_threshold(np.array([0.1, np.nan, 0.9]), alpha=0.1)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
def test_an_alpha_outside_the_unit_interval_is_refused(alpha) -> None:
    with pytest.raises(ValueError, match="alpha"):
        conformal_threshold(np.array([0.1, 0.5]), alpha)


# ----------------------------------------------------------------- CP-4, the exact floor

@pytest.mark.parametrize(
    ("alpha", "expected"), [(0.05, 19.0), (0.10, 9.0), (0.25, 3.0), (0.001, 999.0)]
)
def test_the_floor_is_the_exact_form_not_the_ceiling(alpha, expected) -> None:
    """``(1/alpha) - 1``, not ``ceil(1/alpha) - 1``.

    The README printed the ceiling form for most of this project's life, and attributed the
    band-conditional alpha limit to a floor that on that grid is 19 rows against 847 available
    and therefore cannot bind (D-044).
    """
    assert degeneracy_floor(alpha) == pytest.approx(expected)


def test_the_floor_is_not_rounded() -> None:
    """Rounding changes which borderline configurations are reported as degenerate."""
    assert degeneracy_floor(0.15) == pytest.approx(1 / 0.15 - 1)
    assert degeneracy_floor(0.15) != round(degeneracy_floor(0.15))


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.25, 0.01, 0.001])
def test_the_floor_is_a_strict_bound(alpha) -> None:
    """``n < floor`` is infinite; ``n == floor`` is already finite.

    Writing this test is what found the off-by-one: the docstring said "at or below this
    value has qhat = +inf", and at exact equality the quantile is finite. The algebra is
    ``ceil((n+1)(1-alpha)) > n  <=>  n < (1/alpha) - 1``. No reported configuration sits on
    the boundary, so nothing measured moved -- but the bound's whole content is where it sits.
    """
    rng = np.random.default_rng(SEED)
    floor = degeneracy_floor(alpha)
    at_floor = int(floor)
    assert np.isinf(conformal_threshold(rng.random(at_floor - 1), alpha)[0])
    if floor == at_floor:  # an integral floor is attainable exactly
        assert np.isfinite(conformal_threshold(rng.random(at_floor), alpha)[0])


# ------------------------------------------------------------- CP-3, the Mondrian split

def test_thresholds_are_computed_per_class() -> None:
    """A marginal threshold at this class balance can be met by the majority class alone."""
    rng = np.random.default_rng(SEED)
    labels = np.concatenate([np.zeros(2000, dtype=int), np.ones(80, dtype=int)])
    scores = np.concatenate([rng.normal(0.2, 0.1, 2000), rng.normal(0.8, 0.1, 80)])
    per_class = mondrian_thresholds(scores, labels, alpha=0.1)
    assert set(per_class) == {0, 1}
    # Each class's threshold is that class's own order statistic, not the pooled one.
    for label, entry in per_class.items():
        subset = np.sort(scores[labels == label])
        expected, k, _ = conformal_threshold(subset, 0.1)
        assert entry.threshold == pytest.approx(expected)
        assert entry.order_index == k


def test_a_class_too_small_to_certify_gets_an_infinite_threshold() -> None:
    labels = np.concatenate([np.zeros(500, dtype=int), np.ones(3, dtype=int)])
    scores = np.linspace(0.0, 1.0, 503)
    per_class = mondrian_thresholds(scores, labels, alpha=0.05)
    assert np.isfinite(per_class[0].threshold)
    assert np.isinf(per_class[1].threshold)


# ---------------------------------------------------------- CP-2, the exact coverage law

@pytest.mark.parametrize(("m", "n", "k"), [(100, 500, 476), (2537, 847, 806), (50, 100, 91)])
def test_the_beta_binomial_pmf_is_a_distribution(m, n, k) -> None:
    pmf = beta_binomial_pmf(m, a=n + 1 - k, b=k)
    assert pmf.shape == (m + 1,)
    assert pmf.sum() == pytest.approx(1.0)
    assert (pmf >= 0).all()


def test_it_matches_scipy_betabinom() -> None:
    """An independent implementation of the same law, which is the point of checking it."""
    m, n, k = 200, 500, 476
    a, b = n + 1 - k, k
    ours = beta_binomial_pmf(m, a=a, b=b)
    theirs = stats.betabinom.pmf(np.arange(m + 1), m, a, b)
    assert np.abs(ours - theirs).max() < 1e-12


def test_the_band_is_the_equal_tailed_one_and_not_merely_wide_enough() -> None:
    """Mass alone does not pin a band, and the shape is what the verdicts are defined against.

    The two assertions this test used to make -- at least ``level`` of the mass, and bounds
    inside ``[0, m]`` -- are both satisfied by returning ``(0, m)``, whose mass is 1.0. Four
    further mutants also passed them: ignoring the ``level`` argument for 0.999 or 0.9999, and
    putting all the excluded mass in one tail. The one-sided mutant survived every test in this
    file, and it is the damaging one, because ``band_low = 0`` makes ``observed < band_low``
    unreachable and so deletes the ``conservative`` verdict -- the over-covering case this
    module's docstring keeps separate precisely so that it cannot be read as a pass.

    So both tails are bounded above by ``(1 - level) / 2``, which is what "equal-tailed" means,
    and both are bounded below by moving each endpoint one step inward, which is what makes the
    interval the smallest such one. Together they admit exactly one band: the real
    implementation passes, and full-support, both one-sided variants, both wrong levels and an
    off-by-one at either endpoint are all rejected.
    """
    m, n, k = 500, 847, 806
    low, high = coverage_band(n, k, m, level=0.99)
    pmf = beta_binomial_pmf(m, a=n + 1 - k, b=k)
    tail = (1.0 - 0.99) / 2.0
    assert pmf[low : high + 1].sum() >= 0.99
    assert 0 <= low <= high <= m
    assert pmf[:low].sum() <= tail, "more than half the excluded mass is below the band"
    assert pmf[high + 1 :].sum() <= tail, "more than half the excluded mass is above the band"
    assert pmf[: low + 1].sum() >= tail, "the band starts later than the equal-tailed one"
    assert pmf[high:].sum() >= tail, "the band ends earlier than the equal-tailed one"


def test_the_tail_probability_is_one_sided_and_upper() -> None:
    """Over-covering wastes budget; under-covering breaks the certificate."""
    m, n, k = 300, 847, 806
    assert tail_probability(n, k, m, observed=0) == pytest.approx(1.0)
    assert tail_probability(n, k, m, observed=m + 5) == pytest.approx(0.0, abs=1e-12)
    values = [tail_probability(n, k, m, observed=o) for o in range(0, m, 20)]
    assert all(a >= b for a, b in pairwise(values))


def test_a_naive_rate_check_is_not_the_same_as_the_exact_law() -> None:
    """The reason the project judges against the law: on a calibrated system the naive check
    passes only about half the time, so passing it is not evidence of calibration."""
    m, n, alpha = 400, 847, 0.05
    k = int(np.ceil((n + 1) * (1 - alpha)))
    inside = verify(
        n_calibration=n, order_index=k, n_test=m, observed_errors=int(alpha * m), alpha=alpha
    )
    assert inside.finite_sample_ok
    # An error count well above nominal fails the exact law even though it is "close" to alpha.
    breached = verify(
        n_calibration=n, order_index=k, n_test=m, observed_errors=int(2.0 * alpha * m),
        alpha=alpha,
    )
    assert not breached.finite_sample_ok


# ------------------------------------------------- CP-5, the weighted quantile (D-013)

@pytest.mark.parametrize(("n", "alpha"), [(100, 0.1), (500, 0.05), (1000, 0.01)])
def test_unit_weights_reduce_exactly_to_split_conformal(n, alpha) -> None:
    """D-013's sentence, at exactly the three combinations it claims, to machine precision."""
    rng = np.random.default_rng(SEED)
    scores = rng.random(n)
    plain, _, _ = conformal_threshold(scores, alpha)
    weighted, mass = weighted_conformal_threshold(scores, np.ones(n), alpha)
    assert weighted == pytest.approx(plain, abs=0.0, rel=0.0)
    assert mass == pytest.approx(1.0 / (n + 1))


def test_the_point_mass_at_infinity_is_included() -> None:
    """Omitting the test point's own weight is the standard way to get this wrong."""
    n, alpha = 20, 0.01
    scores = np.linspace(0.0, 1.0, n)
    qhat, mass = weighted_conformal_threshold(scores, np.ones(n), alpha)
    # 1/(n+1) = 0.0476 exceeds alpha = 0.01, so the atom alone forces an infinite threshold.
    assert mass > alpha
    assert np.isinf(qhat)


def test_geometric_weights_favour_recent_observations() -> None:
    weights = geometric_weights(50, rho=0.99)
    assert weights.shape == (50,)
    assert (np.diff(weights) > 0).all(), "weights must increase toward the present"


def test_unit_rho_gives_unit_weights() -> None:
    """rho = 1 is plain split conformal, which is the sanity anchor for the whole family."""
    assert np.allclose(geometric_weights(30, rho=1.0), np.ones(30))


@pytest.mark.parametrize("alpha", [0.5, 0.25, 0.2, 0.1, 0.05])
def test_degeneracy_is_strict_at_the_floor(alpha: float) -> None:
    """A class holding exactly ``floor`` calibration points is not degenerate.

    ``degeneracy_floor`` documents the bound as strict and derives it -- an earlier version of
    that docstring said "at or below", which is off by one at exact equality. The flag in
    ``mondrian_thresholds`` still read ``n <= floor``, so the module contradicted itself twice
    in one file and disagreed with its own ``headroom`` property, which calls a class
    degenerate only when the headroom is negative.

    Nothing measured moves: the smallest headroom in ``degeneracy.csv`` is 1122 rows. That is
    exactly why a test is the only thing that would ever have caught it.
    """
    at_floor = math.ceil(degeneracy_floor(alpha))

    # Ground truth: a finite quantile exists, so the class is not degenerate.
    qhat, k, n = conformal_threshold(np.arange(at_floor, dtype=float), alpha)
    assert math.isfinite(qhat), f"n = floor = {at_floor} should admit a finite quantile"
    assert k <= n

    labels = np.zeros(at_floor, dtype=int)
    result = mondrian_thresholds(np.arange(at_floor, dtype=float), labels, alpha)[0]
    assert not result.degenerate, "a class exactly at the floor is not degenerate"
    assert result.headroom >= 0

    # One row below it is, and the two statements have to agree.
    below = at_floor - 1
    if below > 0:
        qhat_below, _, _ = conformal_threshold(np.arange(below, dtype=float), alpha)
        assert not math.isfinite(qhat_below), f"n = {below} should have no finite quantile"
        thin = mondrian_thresholds(np.arange(below, dtype=float),
                                   np.zeros(below, dtype=int), alpha)[0]
        assert thin.degenerate and thin.headroom < 0


# The worked example in README section 4.3 and in D-012, with the parameters those documents
# state. Neither file is scanned by any gate: `check_claims.py` binds figures that appear in
# the PDFs and `check_pdf.py` reads only `submission/`, so a recomputable number in the README
# is checked by nothing -- and the proposal's first page sends reviewers to exactly that file.
# Both percentages were wrong by more than a point and survived every round until 2026-09-06.
WORKED_EXAMPLE = {"n": 5000, "alpha": 0.01, "m": 20000, "naive_pass": 52.55, "band_mass": 99.05}


def test_the_readme_worked_example_recomputes() -> None:
    """The illustration must come out of the module it illustrates.

    It exists to show that `rate <= alpha` passes about half the time on a sound system, which
    is the argument for using the exact law instead. A wrong percentage there does not change
    the conclusion, and that is exactly why nothing caught it.
    """
    n, alpha, m = WORKED_EXAMPLE["n"], WORKED_EXAMPLE["alpha"], WORKED_EXAMPLE["m"]
    order_index = int(np.ceil((1 - alpha) * (n + 1)))
    pmf = beta_binomial_pmf(m, n + 1 - order_index, order_index)
    low, high = coverage_band(n, order_index, m, 0.99)

    naive = 100 * pmf[: int(alpha * m) + 1].sum()
    mass = 100 * pmf[low : high + 1].sum()
    assert round(naive, 2) == WORKED_EXAMPLE["naive_pass"], (
        f"the naive pass rate is {naive:.2f} %, and README section 4.3 and D-012 say "
        f"{WORKED_EXAMPLE['naive_pass']} %"
    )
    assert round(mass, 2) == WORKED_EXAMPLE["band_mass"], (
        f"the 99 % band holds {mass:.2f} % of draws, and D-012 says "
        f"{WORKED_EXAMPLE['band_mass']} %"
    )

    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert f"{WORKED_EXAMPLE['naive_pass']} % of the time" in readme, (
        "README section 4.3 no longer quotes the figure this test pins"
    )
