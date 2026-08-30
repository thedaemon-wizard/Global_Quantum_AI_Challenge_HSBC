# SPDX-License-Identifier: Apache-2.0
"""Split-conformal thresholds, and the class-conditional degeneracy floor.

Independent implementation from the primary literature; see NOTICE section 2 for why no
code was ported from an existing implementation of the same procedures.

Sources
-------
The split-conformal threshold is the order statistic

    k = ceil( (1 - alpha) * (n + 1) ),    qhat = s_(k)

of the calibration nonconformity scores, with ``qhat = +inf`` when ``k > n`` -- Vovk,
Gammerman and Shafer, *Algorithmic Learning in a Random World*.  The ``n + 1`` is not a
continuity correction: it accounts for the test point itself, and dropping it produces a
threshold that under-covers by roughly ``1/n``.

The class-conditional (Mondrian) construction is Vovk, Lindsay, Nouretdinov and Gammerman,
"Mondrian Confidence Machine", Royal Holloway technical report, 2003, with the conditional
validity theorem in Vovk, "Conditional Validity of Inductive Conformal Predictors",
PMLR 25:475-490, 2012 (arXiv:1209.2673).

The degeneracy floor is stated exactly in Ding, Angelopoulos, Bates, Jordan and Tibshirani,
NeurIPS 36:64555-64576, 2023 (arXiv:2306.09335): for any class whose calibration count
satisfies ``|I^y| < (1/alpha) - 1``, the classwise quantile is ``+inf`` and that class
enters every prediction set regardless of score.  Note the exact form -- it is
``(1/alpha) - 1``, not ``ceil(1/alpha) - 1``, and the difference decides borderline cases.

An implementation note that is easy to get wrong
------------------------------------------------
Released versions of at least one widely used library compute this as a quantile of level
``((n+1)(1-alpha))/n`` with ``method="higher"``, which selects the next-higher order
statistic in some (n, alpha) combinations.  That is conservative, so it never breaks
coverage -- but here it would silently inflate the certified abstention budget, which is a
cost paid by an artefact rather than by the data.  Smoke check S8 measures the discrepancy
over this study's pinned alpha grid; this module uses the order statistic directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ClassThreshold",
    "conformal_threshold",
    "degeneracy_floor",
    "mondrian_thresholds",
]


def degeneracy_floor(alpha: float) -> float:
    """Smallest calibration count admitting a finite classwise quantile at ``alpha``.

    The bound is **strict**: a class with ``|I^y| < floor`` has ``qhat^y = +inf``, and one
    with ``|I^y| == floor`` already admits a finite quantile.  The algebra is
    ``ceil((n+1)(1-alpha)) > n  <=>  n < (1/alpha) - 1``, and an earlier version of this
    docstring said "at or below", which is off by one at exact equality.  No reported
    configuration sits there -- the smallest headroom in ``degeneracy.csv`` is 1122 rows --
    so nothing measured changes, but the difference is the whole content of the bound.

    Returned as a float because the bound is ``(1/alpha) - 1`` exactly and rounding it
    changes which borderline configurations are reported as degenerate.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    return (1.0 / alpha) - 1.0


def conformal_threshold(scores: np.ndarray, alpha: float) -> tuple[float, int, int]:
    """The split-conformal threshold, its order index, and the calibration size.

    Returns ``(qhat, k, n)``.  ``qhat`` is ``+inf`` when ``k > n``, meaning the calibration
    set is too small to certify at this ``alpha`` -- reported rather than silently clipped,
    because a threshold of ``+inf`` is a real answer ("never flag") and quietly replacing it
    with the maximum observed score would fabricate a guarantee the data cannot support.

    Scores must be finite.  A NaN would sort to the end and silently become the threshold.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    values = np.asarray(scores, dtype=float).ravel()
    if values.size == 0:
        return math.inf, 0, 0
    if not np.isfinite(values).all():
        raise ValueError("calibration scores contain NaN or infinity")

    ordered = np.sort(values)
    n = int(ordered.size)
    k = math.ceil((1.0 - alpha) * (n + 1))
    qhat = math.inf if k > n else float(ordered[k - 1])
    return qhat, k, n


@dataclass(frozen=True)
class ClassThreshold:
    """A per-class threshold together with everything needed to judge whether to trust it."""

    label: int
    threshold: float
    order_index: int
    n_calibration: int
    floor: float
    degenerate: bool

    @property
    def headroom(self) -> float:
        """Calibration points above the floor.  Negative means degenerate."""
        return self.n_calibration - self.floor


def mondrian_thresholds(
    scores: np.ndarray, labels: np.ndarray, alpha: float
) -> dict[int, ClassThreshold]:
    """Class-conditional thresholds, one per label present in the calibration set.

    Marginal conformal prediction is close to useless at this study's class balance: a
    threshold chosen to cover 99 % of a population that is 96.5 % legitimate can achieve
    that by covering the majority class alone.  Conditioning on the class is what makes the
    guarantee say something about each class separately.

    The asymmetry that follows is a design property worth stating rather than discovering:
    the guarantee this study certifies is on the **legitimate** class, which is the majority
    class, so its calibration count is large and the degeneracy floor never binds.  The
    fraud class is where degeneracy is a live risk, and each class's headroom is returned so
    the caller can report it instead of assuming it.
    """
    values = np.asarray(scores, dtype=float).ravel()
    tags = np.asarray(labels).ravel()
    if values.shape != tags.shape:
        raise ValueError(f"scores {values.shape} and labels {tags.shape} must have equal length")

    floor = degeneracy_floor(alpha)
    out: dict[int, ClassThreshold] = {}
    for label in np.unique(tags):
        member = values[tags == label]
        qhat, k, n = conformal_threshold(member, alpha)
        out[int(label)] = ClassThreshold(
            label=int(label),
            threshold=qhat,
            order_index=k,
            n_calibration=n,
            floor=floor,
            # Strict, matching degeneracy_floor's own docstring and the `headroom` property
            # beside it: at n == floor the algebra gives ceil((n+1)(1-alpha)) == n, which is a
            # finite quantile. `<=` reported a class at exactly the floor as degenerate.
            degenerate=bool(n < floor),
        )
    return out
