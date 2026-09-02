# SPDX-License-Identifier: Apache-2.0
"""Learn-then-Test and the Hoeffding-Bentkus p-value.

``docs/REFERENCES.md`` stated that ``learn_then_test`` was "validated by simulation (0
violations in 3,000 replications at delta = 0.05)" and that ``hoeffding_bentkus_p_value`` was
"validated at the null boundary". Neither validation was committed. These are those two
validations, written so that the sentences describe something a reader can run.

The Monte-Carlo test is the load-bearing one. It is the only check here that would catch a
procedure using an invalid p-value; the multiplicity correction is conservative enough to pass
it while missing entirely, so that is pinned separately by
``test_correction_is_over_every_grid_point_and_risk_pair``.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from scipy import stats

from hsbcfraud.conformal.riskcontrol import (
    RiskDefinition,
    abstention_rate,
    band_conditional_false_decline,
    hoeffding_bentkus_p_value,
    in_band_grid,
    learn_then_test,
    missed_fraud_rate,
)

DELTA = 0.05
# Replications for the coverage simulation.  The reference sentence says three thousand; the
# check below is exact rather than approximate, so this is a budget rather than a requirement.
REPLICATIONS = 3000
SEED = 20260828


# --------------------------------------------------------------- Hoeffding-Bentkus, CP-8

def test_no_evidence_when_the_empirical_risk_already_exceeds_alpha() -> None:
    """The null is ``R > alpha``. An empirical risk above alpha is evidence *for* it."""
    assert hoeffding_bentkus_p_value(0.30, 1000, 0.25) == 1.0
    assert hoeffding_bentkus_p_value(0.25, 1000, 0.25) == 1.0


def test_the_p_value_is_monotone_in_the_observed_risk() -> None:
    """More observed risk is less evidence against the null, always."""
    values = [hoeffding_bentkus_p_value(r, 500, 0.25) for r in np.linspace(0.0, 0.24, 25)]
    assert all(a <= b for a, b in pairwise(values)), values


def test_the_p_value_falls_as_the_sample_grows() -> None:
    """The same observed risk is stronger evidence from more data."""
    values = [hoeffding_bentkus_p_value(0.10, n, 0.25) for n in (50, 100, 400, 1600)]
    assert all(a > b for a, b in pairwise(values)), values


def test_it_is_bounded_below_by_the_binomial_tail() -> None:
    """The Bentkus branch is a constant multiple of an exact binomial tail.

    If the returned value ever fell *below* that tail the p-value would be anti-conservative,
    which is the failure mode that matters: it would certify configurations it should not.
    """
    for n, alpha, risk in ((200, 0.25, 0.10), (847, 0.10, 0.05), (2259, 0.15, 0.08)):
        observed = int(np.ceil(risk * n))
        binomial = float(stats.binom.cdf(observed, n, alpha))
        assert hoeffding_bentkus_p_value(risk, n, alpha) >= binomial * 0.999


def test_an_empty_sample_carries_no_evidence() -> None:
    assert hoeffding_bentkus_p_value(0.0, 0, 0.25) == 1.0


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_an_alpha_outside_the_unit_interval_is_refused(alpha) -> None:
    with pytest.raises(ValueError, match="alpha"):
        hoeffding_bentkus_p_value(0.1, 100, alpha)


# ------------------------------------------------------- Learn-then-Test coverage, CP-7

def _bernoulli_risk(rng: np.random.Generator, true_rate: float, n: int):
    """A risk whose loss really is Bernoulli at ``true_rate``, independent of lambda.

    Independence from lambda is deliberate: every grid point then tests the same true risk, so
    the family-wise statement is exercised at its hardest, and any grid point declared
    admissible when ``true_rate > alpha`` is a genuine violation.
    """
    draws = rng.random(n)

    def evaluate(_lam: float) -> tuple[float, int]:
        return float((draws < true_rate).mean()), n

    return evaluate


def test_the_family_wise_guarantee_holds_at_the_null_boundary() -> None:
    """The validation the reference list described: violations at or below delta.

    The true risk sits exactly at alpha, which is the hardest point of the null -- any
    admissible set returned here is a violation, and the guarantee allows them at rate delta.
    """
    rng = np.random.default_rng(SEED)
    alpha, n = 0.25, 400
    grid = np.linspace(0.0, 1.0, 11)
    violations = 0
    for _ in range(REPLICATIONS):
        risk = RiskDefinition("r", alpha, _bernoulli_risk(rng, alpha, n))
        if learn_then_test(grid, [risk], delta=DELTA).any_admissible:
            violations += 1
    # Binomial upper bound at the 1e-6 level: a correct procedure effectively never fails it.
    # It is not a test of the multiplicity correction, which is pinned separately below.  The
    # ceiling of 210 is a violation rate of 7 %, and Hoeffding-Bentkus is conservative enough
    # that an entirely uncorrected procedure violates only 32 times here and would pass; a
    # plain binomial tail with neither the Bentkus factor nor the correction violates 113 times
    # and would also pass.  What this catches is a procedure that has stopped using a valid
    # p-value at all: `empirical < alpha` violates 1,371 times of 3,000.  All four counts were
    # measured against this fixture.
    ceiling = stats.binom.ppf(1 - 1e-6, REPLICATIONS, DELTA)
    assert violations <= ceiling, f"{violations} violations in {REPLICATIONS} at delta={DELTA}"


def test_a_risk_far_below_its_target_is_certified() -> None:
    """The negative control: a procedure that never certifies would pass the test above."""
    rng = np.random.default_rng(SEED)
    risk = RiskDefinition("r", 0.25, _bernoulli_risk(rng, 0.02, 2000))
    result = learn_then_test(np.linspace(0.0, 1.0, 11), [risk], delta=DELTA)
    assert result.any_admissible
    assert result.admissible.size == 11


def test_two_risks_must_both_be_rejected() -> None:
    """Intersection-union: one satisfied risk does not carry an unsatisfied one."""
    rng = np.random.default_rng(SEED)
    grid = np.linspace(0.0, 1.0, 11)
    easy = RiskDefinition("easy", 0.25, _bernoulli_risk(rng, 0.01, 2000))
    hard = RiskDefinition("hard", 0.05, _bernoulli_risk(rng, 0.30, 2000))
    assert not learn_then_test(grid, [easy, hard], delta=DELTA).any_admissible
    assert learn_then_test(grid, [easy], delta=DELTA).any_admissible


def test_correction_is_over_every_grid_point_and_risk_pair() -> None:
    """Correcting only across grid points would ignore that each is tested more than once.

    The fixture is chosen so the correction bites.  The shared p-value is 0.0033, which clears
    ``delta / 11 = 0.00455`` but not ``delta / 22 = 0.00227``, so the second risk is what
    empties the admissible set.  Because the synthetic risk is independent of lambda every grid
    point carries the same p-value and the outcome is all-or-nothing, so a fixture outside that
    window cannot separate the two families: the earlier one certified nothing in either call,
    and the comparison it made was ``0 <= 0``, which the uncorrected procedure also satisfies.
    """
    rng = np.random.default_rng(SEED)
    grid = np.linspace(0.0, 1.0, 11)
    risk = RiskDefinition("r", 0.25, _bernoulli_risk(rng, 0.18, 250))
    one = learn_then_test(grid, [risk], delta=DELTA)
    two = learn_then_test(grid, [risk, RiskDefinition("s", 0.25, risk.evaluate)], delta=DELTA)
    assert one.any_admissible, "the fixture is vacuous; the comparison below proves nothing"
    assert two.admissible.size < one.admissible.size


@pytest.mark.parametrize("delta", [0.0, 1.0, -0.5])
def test_a_delta_outside_the_unit_interval_is_refused(delta) -> None:
    risk = RiskDefinition("r", 0.25, lambda _l: (0.01, 100))
    with pytest.raises(ValueError, match="delta"):
        learn_then_test(np.array([0.5]), [risk], delta=delta)


def test_no_risks_is_refused() -> None:
    """Certifying against an empty family would return everything as admissible."""
    with pytest.raises(ValueError, match="at least one risk"):
        learn_then_test(np.array([0.5]), [], delta=DELTA)


# ------------------------------------------------------------------- the risk definitions

def test_the_certified_risk_is_a_mean_of_per_observation_losses() -> None:
    """Hoeffding-Bentkus applies to a mean of [0, 1] losses and to nothing else.

    ``SUBMISSION_CHECKLIST.md`` carried this as verified by reading the source. It is checked
    here instead: the risk at a threshold must equal the mean of the indicator, exactly.
    """
    rng = np.random.default_rng(SEED)
    n = 500
    y = (rng.random(n) < 0.2).astype(int)
    score = rng.random(n)
    in_band = rng.random(n) < 0.6

    evaluate = band_conditional_false_decline(y, score, in_band)
    legit_band = (y == 0) & in_band
    for lam in (0.1, 0.5, 0.9):
        risk, n_eff = evaluate(lam)
        expected = (score[legit_band] >= lam).mean()
        assert risk == pytest.approx(expected)
        assert n_eff == int(legit_band.sum())


def test_the_effective_sample_size_is_the_conditioned_subset() -> None:
    """Using the full row count would claim more evidence than the data contains."""
    rng = np.random.default_rng(SEED)
    n = 400
    y = (rng.random(n) < 0.5).astype(int)
    in_band = rng.random(n) < 0.25
    _, n_eff = band_conditional_false_decline(y, rng.random(n), in_band)(0.5)
    assert n_eff == int(((y == 0) & in_band).sum())
    assert n_eff < n


def test_the_missed_fraud_rate_is_also_a_zero_one_mean() -> None:
    """The same property as the test above, for the other certified risk.

    Range and sample size alone do not pin a rate: the caught fraction is also in [0, 1] and
    leaves ``n_eff`` untouched, so an inverted sign here passed every assertion this test used
    to make while certifying the complement of the quantity ``run_conformal.py`` names.
    """
    rng = np.random.default_rng(SEED)
    n = 600
    y = (rng.random(n) < 0.3).astype(int)
    score = rng.random(n)
    in_band = (score > 0.2) & (score < 0.8)
    evaluate = missed_fraud_rate(y, score, in_band, outer_threshold=0.8)
    risk, n_eff = evaluate(0.5)
    fraud = y == 1
    caught = (score[fraud] >= 0.8) | (in_band[fraud] & (score[fraud] >= 0.5))
    assert risk == pytest.approx(float((~caught).mean()))
    assert n_eff == int(fraud.sum())


def test_the_grid_is_half_open_on_the_band() -> None:
    """A grid point at the outer threshold would flag an empty set and score zero risk.

    That is how an earlier version certified a set of vacuous configurations.
    """
    rng = np.random.default_rng(SEED)
    score = rng.random(1000)
    in_band = (score > 0.2) & (score < 0.8)
    grid = in_band_grid(score, in_band, n_points=11)
    assert grid.size == 11
    assert grid.min() >= score[in_band].min()
    # The second disjunct this line used to carry, `or grid.max() <= 0.8`, was a tautology:
    # band membership is `score < 0.8`, so every quantile of the in-band scores is below 0.8
    # whether or not the top point was dropped, and the test passed with the vacuous grid
    # restored.
    assert grid.max() < score[in_band].max(), (
        "the top quantile flags an empty set and scores zero risk regardless of the data"
    )


def test_the_abstention_rate_is_independent_of_the_threshold() -> None:
    """The band edges are frozen before the in-band threshold is chosen, so the rate is a
    constant -- which is the opposite of what this test was named after.

    The value is pinned as well as the invariance, because a stub returning a constant zero
    satisfies the invariance on its own.
    """
    rng = np.random.default_rng(SEED)
    in_band = rng.random(500) < 0.4
    evaluate = abstention_rate(in_band)
    assert evaluate(0.1)[0] == pytest.approx(float(in_band.mean()))
    assert evaluate(0.1)[0] == pytest.approx(evaluate(0.9)[0])
