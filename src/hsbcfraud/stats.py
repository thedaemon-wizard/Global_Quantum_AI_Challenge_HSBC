# SPDX-License-Identifier: Apache-2.0
"""Statistical procedures, matched to the estimands this study actually compares.

Independent implementation from standard references; see NOTICE section 2.

Why the bootstrap is clustered
------------------------------
Rows in IEEE-CIS are not independent.  The label rule propagates a reported chargeback to
later transactions sharing a user account, email address or billing address, so rows are
clustered by entity; measured on this file, a median of 4 and a 95th percentile of 107
transactions share a ``card1`` value.  A row-level bootstrap resamples rows as though they
were independent draws, which they are not, so its interval does not have its nominal
coverage.  Every interval in this study resamples **entities**, not rows.

The direction of the error is deliberately not asserted here.  For a sample mean under
positive intra-cluster correlation the classical result is that ignoring clusters makes
intervals too narrow.  Average precision is a rank statistic, not a mean, and on a synthetic
fixture with cluster-level risk and 6 % prevalence the clustered interval came out
*narrower* than the row-level one (width ratio 0.89).  So the honest statement is that the
row-level interval is invalid under this dependence and the sign of its error is
statistic-dependent -- which is why the clustered form is used everywhere rather than
applied only where it was expected to matter.

Why there is more than one test
-------------------------------
Applying one test everywhere is a common and visible error.  Average precision is a
threshold-free ranking quantity and is compared by paired clustered bootstrap; a decision at
a fixed threshold is a paired binary outcome and is compared by exact McNemar.  Comparing
two models each at *its own* operating point produces a discordance table that is not
comparable at all, so McNemar here always takes decisions made at a common threshold.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats

from hsbcfraud.progress import ProgressReporter

__all__ = [
    "BootstrapResult",
    "McNemarResult",
    "TostResult",
    "clustered_bootstrap_difference",
    "exact_mcnemar",
    "holm_bonferroni",
    "minimum_detectable_effect",
    "minimum_detectable_effect_from_standard_error",
    "tost_equivalence",
]


# Resamples between progress ticks.  One tick per resample would dominate the loop at small
# sizes, where a resample costs about 1.5 ms; a hundred keeps the reporting under a percent.
REPORT_EVERY = 100


@dataclass(frozen=True)
class BootstrapResult:
    """A paired difference with a clustered percentile interval."""

    point: float
    low: float
    high: float
    level: float
    n_resamples: int
    n_clusters: int
    # Standard deviation of the resampled differences.  Carried explicitly because a power
    # calculation needs it and recovering it from the interval width would impose a normal
    # approximation that the percentile interval itself deliberately avoids.
    spread: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0

    def summary(self) -> str:
        verdict = "excludes" if self.excludes_zero else "includes"
        return (
            f"{self.point:+.4f} [{self.low:+.4f}, {self.high:+.4f}] "
            f"({self.level:.0%} clustered bootstrap over {self.n_clusters:,} entities, "
            f"{verdict} zero)"
        )


def clustered_bootstrap_difference(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    clusters: np.ndarray,
    *,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
    reporter: ProgressReporter | None = None,
) -> BootstrapResult:
    """Paired difference ``metric(a) - metric(b)`` with a cluster bootstrap interval.

    Clusters are resampled with replacement and all their rows travel together, which is
    what preserves the within-entity dependence.  The two scores are always evaluated on the
    *same* resample, so the comparison stays paired and the shared sampling noise cancels --
    an unpaired interval on this data would be dominated by which entities were drawn.

    Resamples whose draw contains a single class are skipped rather than counted as zero:
    average precision is undefined without both classes, and silently substituting a value
    would bias the interval toward whatever was substituted.

    ``reporter`` is optional and receives one tick per hundred resamples.  It exists because
    this is not a fast function at the sizes the scripts use: measured on this host, 2,000
    resamples take 3.0 s over 2,537 rows, 13.2 s over 20,000, and 87.6 s over the full
    115,534-row test block.  Ninety seconds of silence in the middle of a script reads as a
    hang, and the ticks also expose the usable-resample count as it accumulates, which is the
    quantity that decides whether the interval is trustworthy at all.
    """
    y = np.asarray(y_true).ravel()
    a = np.asarray(score_a, dtype=float).ravel()
    b = np.asarray(score_b, dtype=float).ravel()
    g = np.asarray(clusters).ravel()
    if not (y.shape == a.shape == b.shape == g.shape):
        raise ValueError("y_true, both scores and clusters must have equal length")

    unique, inverse = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(unique.size)]
    point = metric(y, a) - metric(y, b)

    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for draw in range(n_resamples):
        picked = rng.integers(0, len(members), size=len(members))
        rows = np.concatenate([members[i] for i in picked])
        if np.unique(y[rows]).size < 2:
            continue
        deltas.append(metric(y[rows], a[rows]) - metric(y[rows], b[rows]))
        if reporter is not None and (draw + 1) % REPORT_EVERY == 0:
            reporter.tick(
                draw + 1,
                usable=len(deltas),
                median=float(np.median(deltas)) if deltas else float("nan"),
            )

    if len(deltas) < 0.5 * n_resamples:
        raise ValueError(
            f"only {len(deltas)} of {n_resamples} resamples contained both classes; "
            "the cluster structure makes a bootstrap interval unreliable here"
        )
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(deltas, [tail, 1.0 - tail])
    return BootstrapResult(
        point=float(point),
        low=float(low),
        high=float(high),
        level=level,
        n_resamples=len(deltas),
        n_clusters=int(unique.size),
        spread=float(np.std(deltas, ddof=1)),
    )


@dataclass(frozen=True)
class McNemarResult:
    """Exact McNemar on a paired discordance table."""

    n01: int
    n10: int
    p_value: float

    def summary(self) -> str:
        return f"discordance {self.n01}/{self.n10}, exact McNemar p={self.p_value:.4g}"


def exact_mcnemar(decisions_a: np.ndarray, decisions_b: np.ndarray) -> McNemarResult:
    """Exact two-sided McNemar test on paired binary decisions.

    Exact rather than the chi-squared approximation because the discordant counts here are
    small -- the band holds a few percent of traffic and the two scorers agree on most of it
    -- and the approximation is unreliable below roughly 25 discordant pairs.

    Under the null the discordant pairs are ``Binomial(n01 + n10, 1/2)``; the two-sided
    p-value doubles the smaller tail and clips at 1.
    """
    a = np.asarray(decisions_a).astype(bool).ravel()
    b = np.asarray(decisions_b).astype(bool).ravel()
    if a.shape != b.shape:
        raise ValueError(f"decision arrays must have equal length, got {a.shape} and {b.shape}")

    n01 = int(np.sum(~a & b))
    n10 = int(np.sum(a & ~b))
    n = n01 + n10
    if n == 0:
        # Identical decisions everywhere. p = 1 is right, and it is worth reporting rather
        # than treating as an error: it usually means the second scorer changed nothing.
        return McNemarResult(n01=0, n10=0, p_value=1.0)
    p = min(1.0, 2.0 * stats.binom.cdf(min(n01, n10), n, 0.5))
    return McNemarResult(n01=n01, n10=n10, p_value=float(p))


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm step-down, returning which hypotheses are rejected at family-wise ``alpha``.

    Uniformly more powerful than Bonferroni and equally free of dependence assumptions.
    Monotonicity is enforced explicitly: once a hypothesis fails to be rejected, every
    later one in the sorted order is retained too, which is the step-down rule and is easy
    to omit by accident when implementing this from the definition.
    """
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: dict[str, bool] = {}
    still_rejecting = True
    for rank, (name, p) in enumerate(ordered):
        if still_rejecting and p <= alpha / (m - rank):
            out[name] = True
        else:
            still_rejecting = False
            out[name] = False
    return out


@dataclass(frozen=True)
class TostResult:
    """Two one-sided tests for equivalence within a stated margin."""

    difference: float
    margin: float
    p_value: float
    equivalent: bool
    informative: bool

    def summary(self) -> str:
        if not self.informative:
            return (
                f"equivalence test UNINFORMATIVE: margin {self.margin:.4f} is wider than the "
                f"effect it would need to exclude"
            )
        verdict = "equivalent" if self.equivalent else "not shown equivalent"
        return f"{verdict} within +/-{self.margin:.4f} (TOST p={self.p_value:.4g})"


def tost_equivalence(
    differences: np.ndarray, margin: float, *, alpha: float = 0.05, reference_effect: float = 0.023
) -> TostResult:
    """Two one-sided tests, with an honesty gate on the margin.

    Failing to reject a difference is not evidence of no difference; TOST is the test that
    makes the positive claim.  But an equivalence test whose margin is wider than the effect
    it is supposed to rule out proves nothing, and reporting one is worse than reporting
    nothing, so ``informative`` is returned alongside the verdict.

    ``reference_effect`` defaults to 0.023 average precision, the point estimate reported by
    Chaves et al. (arXiv:2603.06473) for a comparable hybrid architecture on this dataset
    family.  If the margin this study can actually support is wider than that, the honest
    statement is that the comparison cannot distinguish the two hypotheses.
    """
    d = np.asarray(differences, dtype=float).ravel()
    if d.size < 2:
        raise ValueError("TOST needs at least two paired differences")
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(d.size))
    if se == 0.0:
        return TostResult(mean, margin, 0.0, abs(mean) < margin, margin <= reference_effect)

    df = d.size - 1
    p_lower = stats.t.sf((mean + margin) / se, df)
    p_upper = stats.t.cdf((mean - margin) / se, df)
    p = float(max(p_lower, p_upper))
    return TostResult(
        difference=mean,
        margin=margin,
        p_value=p,
        equivalent=bool(p < alpha),
        informative=bool(margin <= reference_effect),
    )


def minimum_detectable_effect_from_standard_error(
    standard_error: float, *, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Smallest paired difference detectable at the given power, from a standard error.

    This is the form to use whenever the uncertainty came from a bootstrap, because a
    bootstrap standard deviation *is* a standard error: the sample size and, for a clustered
    resample, the dependence structure are already inside it.

    It is also the only correct form for average precision.  AP is not a mean of per-row
    quantities -- it is a functional of the whole ranking -- so "the standard deviation of one
    observation" does not exist for it and no division by the square root of anything is
    meaningful.  Passing a bootstrap standard error to ``minimum_detectable_effect`` instead
    understates the MDE by a factor of the square root of the sample size, which on the band
    evaluation block is about fifty and turns a gate into a formality.
    """
    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_power = stats.norm.ppf(power)
    return float((z_alpha + z_power) * standard_error)


def minimum_detectable_effect(
    sd: float, n: int, *, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Smallest paired difference detectable at the given power, for a two-sided test.

    ``sd`` is the standard deviation of a **single paired observation**, and ``n`` the number
    of pairs; the standard error is formed here as ``sd / sqrt(n)``.  If the uncertainty is
    already a standard error -- anything from a bootstrap -- call
    :func:`minimum_detectable_effect_from_standard_error` instead, or the sample size is
    counted twice.

    Computed and reported **before** the comparison runs, as a gate.  The pre-registration
    commits to declaring the quantum comparison underpowered in advance if this exceeds
    0.023 average precision, rather than running it and discovering afterwards that a null
    result was uninformative.
    """
    if n < 2:
        raise ValueError(f"need at least two observations, got {n}")
    return minimum_detectable_effect_from_standard_error(
        sd / math.sqrt(n), alpha=alpha, power=power
    )
