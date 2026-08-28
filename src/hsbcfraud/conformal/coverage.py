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

from dataclasses import dataclass

import numpy as np
from scipy.special import betaln

__all__ = ["CoverageVerdict", "beta_binomial_pmf", "coverage_band", "tail_probability", "verify"]


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
    of test points.  Returns inclusive ``(low, high)`` bounds: the smallest interval
    containing at least ``level`` of the exact distribution, with equal tail mass excluded
    on each side.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level!r}")
    if k > n:
        raise ValueError(f"order index k={k} exceeds calibration size n={n}; threshold is infinite")

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
    pmf = beta_binomial_pmf(m, a=n + 1 - k, b=k)
    return float(pmf[min(observed, m) :].sum())


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
    return CoverageVerdict(
        n_calibration=n_calibration,
        order_index=order_index,
        n_test=n_test,
        observed_errors=observed_errors,
        empirical_rate=observed_errors / n_test if n_test else float("nan"),
        nominal_rate=alpha,
        band_low=coverage_band(n_calibration, order_index, n_test, band_level)[0],
        band_high=coverage_band(n_calibration, order_index, n_test, band_level)[1],
        band_level=band_level,
        tail_p=tail_probability(n_calibration, order_index, n_test, observed_errors),
    )
