# SPDX-License-Identifier: Apache-2.0
"""Exact finite-sample validation of a conformal threshold.

Independent implementation from the primary literature; see NOTICE section 2.

The mistake this module exists to prevent
-----------------------------------------
The split-conformal guarantee is a statement about an **expectation** taken over the draw
of the calibration set.  On any single finite test set the empirical error rate fluctuates
around it.  Asserting ``empirical_rate <= alpha`` therefore fails on perfectly sound
systems roughly half the time, and -- worse in practice -- *passing* such an assertion is
equally uninformative.  A study whose headline is a certified rate cannot validate that
rate with a test that is wrong half the time in both directions.

The correct object is the exact predictive distribution of the error count.  Conditional on
a calibration set of size ``n`` and a threshold taken at order statistic ``k``, the number
of errors ``E`` among ``m`` exchangeable test points is Beta-Binomial:

    E ~ BetaBinomial(m, a = n + 1 - k, b = k)

which follows from Vovk, "Conditional Validity of Inductive Conformal Predictors",
PMLR 25:475-490, 2012 (arXiv:1209.2673): the conditional coverage of an inductive conformal
predictor is ``Beta(k, n + 1 - k)`` distributed, and compounding a Binomial over that Beta
gives the Beta-Binomial above.  This is exact -- not a normal approximation, and not a
bootstrap.

So the check is: does the observed error count fall inside a central band of that exact
distribution?  Three verdicts are reported separately, because they answer different
questions and collapsing them hides the interesting case:

``expectation_ok``      the point estimate is at or below the nominal rate;
``finite_sample_ok``    the observed count lies inside the exact central band -- the
                        verdict that actually validates the implementation;
``conservative``        the count is below the band, i.e. the procedure is over-covering.
                        Not a failure, but not a success either: on this study it would
                        mean the certified abstention budget is larger than the data
                        requires, which costs step-up authentications for nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import betaln

__all__ = ["CoverageVerdict", "beta_binomial_pmf", "coverage_band", "tail_probability", "verify"]


def _require_order_index_within_calibration(n: int, k: int) -> None:
    """Reject ``k > n``, the case in which the split-conformal threshold is ``+inf``.

    Both public functions below turn ``(n, k)`` into ``BetaBinomial(m, n + 1 - k, k)``, and at
    ``k > n`` the first Beta parameter is not positive: there is no exact law, because "never
    flag" has no error distribution over the test block.  Left unchecked the call still fails,
    but inside :func:`beta_binomial_pmf` and with a message about Beta parameters that says
    nothing about the calibration set the caller actually got wrong.

    Written once because the two call sites raise the same sentence, and a guard whose wording
    drifts between two functions is one a reader stops trusting as the same guard.
    """
    if k > n:
        raise ValueError(f"order index k={k} exceeds calibration size n={n}; threshold is infinite")


def beta_binomial_pmf(m: int, a: float, b: float) -> np.ndarray:
    """Exact pmf of ``BetaBinomial(m, a, b)`` over ``e = 0 .. m``.

    Computed in log space and renormalised.  The direct form overflows for the sizes this
    study uses -- ``m`` is the number of legitimate test transactions, of order 10^5, and
    the binomial coefficient alone exceeds double precision long before that.
    """
    if m < 0:
        raise ValueError(f"m must be non-negative, got {m}")
    if a <= 0 or b <= 0:
        raise ValueError(f"Beta parameters must be positive, got a={a}, b={b}")

    e = np.arange(m + 1, dtype=float)
    # log C(m, e) = -log(m + 1) - betaln(m - e + 1, e + 1), from B(x, y) = G(x)G(y)/G(x+y).
    log_binom = -np.log(m + 1.0) - betaln(m - e + 1.0, e + 1.0)
    log_pmf = log_binom + betaln(e + a, m - e + b) - betaln(a, b)
    pmf = np.exp(log_pmf - log_pmf.max())
    return pmf / pmf.sum()


def coverage_band(n: int, k: int, m: int, level: float = 0.99) -> tuple[int, int]:
    """Central predictive interval for the error count, at the given level.

    ``n`` is the calibration size, ``k`` the order index of the threshold, ``m`` the number
    of test points.  Returns inclusive ``(low, high)`` bounds: the central (equal-tailed)
    interval containing at least ``level`` of the exact distribution, with equal tail mass
    excluded on each side.  Equal-tailed is not the shortest such interval -- at the
    ``coverage.csv`` alpha = 0.01 row this returns (973, 1266), width 293, while (972, 1263),
    width 291, already carries 0.99001 -- but it is the one the verdict is defined against.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level!r}")
    _require_order_index_within_calibration(n, k)

    pmf = beta_binomial_pmf(m, a=n + 1 - k, b=k)
    cdf = np.cumsum(pmf)
    tail = (1.0 - level) / 2.0
    low = int(np.searchsorted(cdf, tail, side="left"))
    high = int(np.searchsorted(cdf, 1.0 - tail, side="left"))
    return low, min(high, m)


def tail_probability(n: int, k: int, m: int, observed: int) -> float:
    """One-sided ``P(E >= observed)`` under the exact law.

    The one-sided form is the operationally meaningful one: over-covering wastes budget,
    but under-covering breaks the certificate, so only the upper tail is a failure.
    """
    _require_order_index_within_calibration(n, k)
    pmf = beta_binomial_pmf(m, a=n + 1 - k, b=k)
    # Clamped at both ends.  Only the upper end was, so a negative observed count sliced the
    # far upper tail from the right and returned a number near zero where P(E >= observed) is
    # exactly 1 -- the strongest possible pass reported as the strongest possible failure.
    return float(pmf[max(0, min(observed, m)) :].sum())


@dataclass(frozen=True)
class CoverageVerdict:
    """The outcome of an exact finite-sample coverage check."""

    n_calibration: int
    order_index: int
    n_test: int
    observed_errors: int
    empirical_rate: float
    nominal_rate: float
    band_low: int
    band_high: int
    band_level: float
    tail_p: float

    @property
    def expectation_ok(self) -> bool:
        return self.empirical_rate <= self.nominal_rate

    @property
    def finite_sample_ok(self) -> bool:
        return self.band_low <= self.observed_errors <= self.band_high

    @property
    def conservative(self) -> bool:
        return self.observed_errors < self.band_low

    def summary(self) -> str:
        inside = "inside" if self.finite_sample_ok else "OUTSIDE"
        note = " (conservative)" if self.conservative else ""
        return (
            f"{self.observed_errors} errors on {self.n_test:,} test points "
            f"(rate {self.empirical_rate:.5f} against nominal {self.nominal_rate:.5f}); "
            f"{inside} the exact {self.band_level:.0%} band [{self.band_low}, {self.band_high}]"
            f"{note}, one-sided tail p={self.tail_p:.4f}"
        )


def verify(
    *,
    n_calibration: int,
    order_index: int,
    n_test: int,
    observed_errors: int,
    alpha: float,
    band_level: float = 0.99,
) -> CoverageVerdict:
    """Check an observed error count against the exact predictive law."""
    if n_test <= 0:
        # Fail-open, and in the one verdict that is supposed to catch a broken split.  With
        # `n_test == 0` the Beta-Binomial pmf is `[1.0]`, so the band collapses to `(0, 0)`,
        # the tail probability is exactly 1.0, and `finite_sample_ok` evaluates `0 <= 0 <= 0`
        # and returns **True** -- a passing finite-sample verdict for a check that never ran.
        # The only trace was a `nan` rate in a column nothing asserts on.
        #
        # A mis-tagged `block`, a collapsed arm split or a truncated parquet would have been
        # reported as clean coverage at every level.  See D-132.
        raise ValueError(
            f"n_test={n_test}: an empty test block has no error distribution, so there is no "
            f"coverage claim to check. This usually means the block filter matched nothing -- "
            f"verify the `block` column of the scores file being read."
        )
    # One call, not two.  Taking [0] and [1] from separate calls evaluated the Beta-Binomial
    # pmf twice for one band, and at the sizes this study uses that pmf is the whole cost of
    # the function -- measured at 49.5 ms for `verify` against 21.2 ms for one pmf.
    band_low, band_high = coverage_band(n_calibration, order_index, n_test, band_level)
    return CoverageVerdict(
        n_calibration=n_calibration,
        order_index=order_index,
        n_test=n_test,
        observed_errors=observed_errors,
        empirical_rate=observed_errors / n_test,
        nominal_rate=alpha,
        band_low=band_low,
        band_high=band_high,
        band_level=band_level,
        tail_p=tail_probability(n_calibration, order_index, n_test, observed_errors),
    )


def split_conformal_coverage(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    *,
    alpha_grid: Sequence[float],
    band_level: float,
) -> list[dict[str, object]]:
    """One coverage verdict per level, for one calibration/test pair.

    Extracted from ``scripts/run_conformal.py``, which computed this inline for a single arm
    and seed.  Two committed tables report the same quantity across three arms and five
    seeds -- ``coverage_by_arm.csv`` and ``coverage_by_arm_seeds.csv`` -- and **no script
    wrote either of them**, so ``make reproduce`` could not regenerate the evidence behind
    the results section's central contrast.  The tests recorded that as a strict xfail rather
    than hiding it.

    Putting the computation here rather than copying it into a second script is what makes
    the two tables comparable: a producer that re-derived the order statistic independently
    could drift from the one the certificate uses, and the whole point of the by-arm table is
    that it is the same procedure applied to a different split.

    A level whose order index exceeds the calibration block is skipped rather than clamped:
    the quantile does not exist at that sample size, and reporting a clamped one would assert
    coverage the data cannot support.
    """
    legit_cal = cal_scores[cal_labels == 0]
    legit_test = test_scores[test_labels == 0]
    rows: list[dict[str, object]] = []
    for alpha in alpha_grid:
        order_index = int(np.ceil((1 - alpha) * (legit_cal.size + 1)))
        if order_index > legit_cal.size:
            continue
        threshold = float(np.sort(legit_cal)[order_index - 1])
        observed = int((legit_test >= threshold).sum())
        verdict = verify(
            n_calibration=legit_cal.size,
            order_index=order_index,
            n_test=legit_test.size,
            observed_errors=observed,
            alpha=alpha,
            band_level=band_level,
        )
        rows.append(
            {
                "alpha": alpha,
                "threshold": threshold,
                "n_cal_legit": int(legit_cal.size),
                "n_test_legit": int(legit_test.size),
                "observed_errors": observed,
                "empirical_rate": verdict.empirical_rate,
                "band_low": verdict.band_low,
                "band_high": verdict.band_high,
                "tail_p": verdict.tail_p,
                "expectation_ok": verdict.expectation_ok,
                "finite_sample_ok": verdict.finite_sample_ok,
                "conservative": verdict.conservative,
            }
        )
    return rows
