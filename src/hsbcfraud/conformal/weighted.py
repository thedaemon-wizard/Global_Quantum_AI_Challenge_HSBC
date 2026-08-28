# SPDX-License-Identifier: Apache-2.0
"""Non-exchangeable split conformal with fixed weights.

Independent implementation from the primary literature; see NOTICE section 2.

Source
------
Barber, Candes, Ramdas and Tibshirani, "Conformal prediction beyond exchangeability",
*Annals of Statistics* 51(2):816-845, 2023 (arXiv:2202.13415), equation (11): replace the
unweighted empirical quantile of the calibration scores with

    Qhat_{1-alpha}( sum_i wtilde_i * delta_{s_i}  +  wtilde_{n+1} * delta_{+inf} )

where ``wtilde_i = w_i / (w_1 + ... + w_n + 1)`` and ``wtilde_{n+1} = 1 / (w_1 + ... + w_n + 1)``.
The point mass at ``+inf`` is the test point's own weight and is what preserves validity
when the calibration weights are small; omitting it is the usual implementation error and
produces a threshold that is too low.

With ``w_i = 1`` this reduces exactly to standard split conformal, which the tests assert.
That reduction is the reason the method is safe to adopt: it can only help.

What this does and does not buy
-------------------------------
Theorem statement, eq. (3):

    coverage gap  <=  ( sum_i w_i * d_TV(Z, Z^i) ) / ( 1 + sum_i w_i )

Two constraints follow, and both are load-bearing:

1. **The weights must be fixed and not fitted on the calibration data.**  Tuning ``rho`` to
   whatever makes the empirical coverage look best voids the theorem.  ``rho`` is therefore
   pinned in ``docs/protocol.md`` and hashed before any experiment runs.

2. **The bound is not numerically evaluable forward.**  ``d_TV`` involves the unknown joint
   law.  The paper's two closed forms -- ``2*eps/(1-rho)`` under bounded drift and ``rho^k``
   after a changepoint -- both route through its Lemma 1, which assumes the observations are
   **independent**.  A serially correlated transaction stream is not, and falls under the
   paper's separate covariate-time-series treatment.  So this study does not quote ``rho^k``
   as though it were an evaluated number.

What is reported instead is the *inverse*: given the risk gap actually observed between
calibration and test, :func:`implied_total_variation` returns the average per-step total
variation distance that would be required to explain it.  That turns an unquantifiable
forward bound into a quantity a reader can judge -- and it is honest about direction, since
it is a necessary consequence of the observation rather than a guarantee about the future.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["geometric_weights", "implied_total_variation", "weighted_conformal_threshold"]


def geometric_weights(n: int, rho: float) -> np.ndarray:
    """Weights ``w_i = rho^(n + 1 - i)`` for ``i = 1..n``, oldest first.

    The most recent calibration point receives weight ``rho`` and the oldest ``rho^n``, so
    recency is favoured smoothly rather than by a hard window.  ``rho`` must be fixed in
    advance; see the module docstring.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"rho must lie in (0, 1], got {rho!r}")
    exponents = np.arange(n, 0, -1, dtype=float)  # n, n-1, ..., 1
    return rho**exponents


def weighted_conformal_threshold(
    scores: np.ndarray, weights: np.ndarray, alpha: float
) -> tuple[float, float]:
    """Weighted quantile of Barber et al. eq. (11).

    Returns ``(qhat, mass_at_infinity)``.  The second value is ``wtilde_{n+1}``: when it
    exceeds ``alpha`` the threshold is necessarily ``+inf``, because the test point's own
    weight alone exceeds the tail budget.  Returning it lets the caller report *why* a
    threshold is infinite rather than presenting an unexplained ``inf``.

    Scores are assumed ordered oldest-to-newest so that they align with
    :func:`geometric_weights`; the function does not sort them into time order because it
    cannot know it, and silently reordering would break the correspondence.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    s = np.asarray(scores, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if s.shape != w.shape:
        raise ValueError(f"scores {s.shape} and weights {w.shape} must have equal length")
    if s.size == 0:
        return math.inf, 1.0
    if not np.isfinite(s).all():
        raise ValueError("calibration scores contain NaN or infinity")
    if (w < 0).any():
        raise ValueError("weights must be non-negative")

    total = w.sum() + 1.0
    normalised = w / total
    mass_at_infinity = 1.0 / total

    order = np.argsort(s, kind="stable")
    cumulative = np.cumsum(normalised[order])
    # Smallest score whose cumulative normalised weight reaches 1 - alpha.  If the finite
    # atoms never reach it, the remaining mass sits on +inf and that is the threshold.
    reached = np.searchsorted(cumulative, 1.0 - alpha, side="left")
    if reached >= s.size:
        return math.inf, mass_at_infinity
    return float(s[order][reached]), mass_at_infinity


def implied_total_variation(observed_gap: float, weights: np.ndarray) -> float:
    """Average per-step ``d_TV`` that would be needed to explain an observed coverage gap.

    Inverts eq. (3) treating ``d_TV(Z, Z^i)`` as constant in ``i``:

        gap = ( sum_i w_i * d ) / ( 1 + sum_i w_i )   =>   d = gap * (1 + sum w) / sum w

    Read this as a diagnostic, not a guarantee.  It answers "how non-exchangeable would the
    stream have to be for the drift I measured to be consistent with the theory", which is a
    question a reviewer can evaluate, in contrast to a forward bound containing a nuisance
    parameter nobody can estimate.  Values at or above 1 mean the observed gap exceeds what
    the bound can explain at any drift level, which would indicate a defect rather than
    drift.
    """
    w = np.asarray(weights, dtype=float).ravel()
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights sum to zero; no calibration mass")
    return float(observed_gap * (1.0 + total) / total)
