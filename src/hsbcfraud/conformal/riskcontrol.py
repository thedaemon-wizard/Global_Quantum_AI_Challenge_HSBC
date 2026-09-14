# SPDX-License-Identifier: Apache-2.0
"""Learn-then-Test risk control, in the PAC form a bank can put in a control document.

Independent implementation from the primary literature; see NOTICE section 2.

Source
------
Angelopoulos, Bates, Candes, Jordan and Lei, "Learn then Test: Calibrating Predictive
Algorithms to Achieve Risk Control", *Annals of Applied Statistics* 19(2):1641-1662, 2025
(arXiv:2110.01052).  The construction reframes risk control as multiple hypothesis testing:
for each candidate decision parameter ``lambda`` on a **finite, pre-registered** grid, test

    H_lambda :  R(lambda) > alpha

on the calibration set, and return the set of ``lambda`` for which H is rejected under
family-wise error control.  Any returned ``lambda`` then satisfies
``P( R(lambda) <= alpha ) >= 1 - delta``.

The Hoeffding-Bentkus p-value is the tighter of two concentration bounds for a loss in
[0, 1] (Bates, Angelopoulos, Lei, Malik and Jordan, 2021):

    p_HB = min(  exp(-n * h1(Rhat ^ alpha, alpha)),   e * P( Bin(n, alpha) <= ceil(n Rhat) )  )

with ``h1(a, b) = a log(a/b) + (1-a) log((1-a)/(1-b))``.  Hoeffding is tighter in the bulk,
Bentkus in the tail; taking the minimum is valid because each is separately a valid
p-value.

Why PAC and not the expectation
-------------------------------
Conformal Risk Control (Angelopoulos et al., ICLR 2024) bounds ``E[R]``, the expectation
over the calibration draw.  That is a statement about the average behaviour of the
procedure, not about the system that was actually deployed.  A model risk function asking
"what is the chance the control I signed off is breached" wants the high-probability form,
so LTT is the headline here and the expectation form is reported beside it.

Why both risks are certified
----------------------------
Controlling only the false-decline side would leave uncertified exactly the side PSD2 caps
and the side the loss reserve is held against.  A grid point is admissible only if **both**
null hypotheses are rejected, which is a intersection-union test: the family-wise correction
runs across the joint grid, so the joint statement inherits the same ``1 - delta``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "SELECT_RULES",
    "RiskControlResult",
    "RiskDefinition",
    "abstention_rate",
    "band_conditional_false_decline",
    "hoeffding_bentkus_p_value",
    "in_band_grid",
    "learn_then_test",
    "missed_fraud_rate",
]

# The reporting rules `learn_then_test` accepts.  Named once so the entry validation and the
# dispatch below cannot drift apart; a rule accepted by one and unknown to the other would
# raise from the middle of a certification run rather than at the call.
SELECT_RULES = frozenset({"max_recall", "min_abstention"})


def _h1(a: float, b: float) -> float:
    """KL divergence between Bernoulli(a) and Bernoulli(b), with the boundaries handled."""
    if a <= 0.0:
        return -math.log(1.0 - b) if b < 1.0 else math.inf
    if a >= 1.0:
        return -math.log(b) if b > 0.0 else math.inf
    return a * math.log(a / b) + (1.0 - a) * math.log((1.0 - a) / (1.0 - b))


def hoeffding_bentkus_p_value(empirical_risk: float, n: int, alpha: float) -> float:
    """Valid p-value for ``H: R > alpha`` given ``n`` observations of a [0, 1] loss.

    Returns 1.0 when the empirical risk already exceeds ``alpha``: there is then no evidence
    against the null at all, and returning a small number would be exactly backwards.
    """
    if n <= 0:
        return 1.0
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    if empirical_risk >= alpha:
        return 1.0

    hoeffding = math.exp(-n * _h1(min(empirical_risk, alpha), alpha))
    bentkus = math.e * float(stats.binom.cdf(math.ceil(n * empirical_risk), n, alpha))
    return float(min(1.0, hoeffding, bentkus))
    # The `math.e` above is a known, measured conservatism, not an oversight.  Both risks here
    # are means of 0/1 losses, and Bates et al. Remark 4 -- the source this bound is taken from
    # -- says that in that case "this exact binomial result gives the most precise upper
    # confidence bound and should always be used".  Dropping the factor would do exactly that.
    #
    # It is retained because the pre-registration was frozen on this estimator before any model
    # was fitted and the single `D_test` evaluation is spent; changing the p-value now is a new
    # campaign, which is the same reasoning that declined the tuned baseline in D-147.
    #
    # What it costs was measured rather than assumed: on all five certified cells the Bentkus
    # branch binds, so the penalty is exactly `e`, and the p-values are between 1e-34 and 1e-103
    # against a Holm threshold near 0.005.  Nothing there is close.  See D-163.


@dataclass(frozen=True)
class RiskDefinition:
    """A named risk, its target level, and how to evaluate it at a decision parameter.

    ``evaluate`` returns ``(empirical_risk, n_effective)``.  The second value matters and is
    not simply the number of rows: a risk conditioned on the legitimate class inside the
    band has an effective sample size equal to the number of legitimate band transactions,
    and using the full row count would make the concentration bound claim more evidence than
    the data contains.
    """

    name: str
    target: float
    evaluate: Callable[[float], tuple[float, int]]


def band_conditional_false_decline(
    y: np.ndarray, score: np.ndarray, in_band: np.ndarray, decide_high: bool = True
) -> Callable[[float], tuple[float, int]]:
    """P(decline | legitimate, in band), as a function of the in-band decision threshold.

    This is the risk the study certifies, and the conditioning on band membership is the
    point.  The unconditional false-decline rate over all legitimate traffic is dominated by
    the outer threshold and would be nearly unchanged if the in-band scorer were replaced by
    a coin -- so a certificate on the unconditional rate would not constrain the component it
    is supposed to license.
    """
    legit_band = (y == 0) & in_band
    n = int(legit_band.sum())
    band_scores = score[legit_band]

    def evaluate(lam: float) -> tuple[float, int]:
        if n == 0:
            return float("nan"), 0
        flagged = band_scores >= lam if decide_high else band_scores <= lam
        return float(flagged.mean()), n

    return evaluate


def missed_fraud_rate(
    y: np.ndarray, score: np.ndarray, in_band: np.ndarray, outer_threshold: float
) -> Callable[[float], tuple[float, int]]:
    """False-negative rate: the fraction of fraud the composite rule lets through.

    A **mean of per-observation 0/1 losses**, which is what Hoeffding-Bentkus requires.

    This replaces an earlier ``recall_shortfall`` that returned
    ``max(0, floor - recall) / floor``.  That quantity is a nonlinear transform of a mean,
    not a mean of losses, and the Hoeffding-Bentkus bound (Bates, Angelopoulos, Lei, Malik
    and Jordan, 2021) applies only to the latter.  The p-values it produced were therefore
    not valid p-values, and any certificate resting on them was not a certificate.  See
    ``docs/decisions.md`` D-024.

    Controlling the false-negative rate at ``alpha_fn`` is equivalent to placing a recall
    floor at ``1 - alpha_fn``, so nothing is lost by the reformulation -- but now the loss
    really is ``1`` when a fraudulent transaction is approved and ``0`` when it is declined,
    averaged over fraudulent transactions.

    A fraud is caught when the outer threshold declines it, or when it is inside the band and
    the in-band rule declines it.  Fraud below the band is approved by both stages and counts
    as missed, which is correct: the composite rule really does let it through.
    """
    y = np.asarray(y).ravel()
    score = np.asarray(score, dtype=float).ravel()
    in_band = np.asarray(in_band).astype(bool).ravel()

    fraud = y == 1
    n_fraud = int(fraud.sum())
    fraud_scores = score[fraud]
    fraud_in_band = in_band[fraud]
    caught_by_outer = fraud_scores >= outer_threshold

    def evaluate(lam: float) -> tuple[float, int]:
        if n_fraud == 0:
            return float("nan"), 0
        caught_in_band = fraud_in_band & (fraud_scores >= lam)
        missed = ~(caught_by_outer | caught_in_band)
        return float(missed.mean()), n_fraud

    return evaluate


def in_band_grid(score: np.ndarray, in_band: np.ndarray, n_points: int) -> np.ndarray:
    """Decision thresholds spaced over the scores actually present in the band.

    Two properties matter, and a linear grid over the band edges has neither.

    **The upper endpoint is excluded.**  Band membership is ``lo <= score < hi`` and the
    in-band rule is ``score >= lambda``, so at ``lambda = hi`` the flagged set is empty by
    construction: the false-decline risk is structurally zero regardless of the data, and it
    is the only grid point whose p-value clears the family-wise level.  Every certificate
    produced before this function existed selected exactly that point and therefore certified
    a rule that declines nobody.  A grid point whose risk cannot depend on the data is not a
    candidate decision, it is an artefact of the parameterisation.

    **Spacing follows the score distribution, not the interval.**  Scores in the band are far
    from uniform, so a linear grid concentrates most of its points where almost no
    transactions lie and resolves the operating region with two or three.  Quantile spacing
    puts the grid where the decisions change.

    Learn-then-Test needs the grid to be finite and fixed in advance, not to be evenly
    spaced; the grid is a function of the calibration scores, which are seen before any risk
    is evaluated, so this is a change of parameterisation rather than a change of the
    pre-registered commitment.
    """
    if n_points < 2:
        raise ValueError(f"need at least two grid points, got {n_points}")
    band_scores = np.asarray(score, dtype=float).ravel()[np.asarray(in_band).astype(bool).ravel()]
    if band_scores.size == 0:
        raise ValueError("no calibration rows fall inside the band")
    # Left-closed quantiles: the lowest is the most aggressive reachable rule (decline the
    # whole band), and the top quantile is dropped for the reason above.
    quantiles = np.linspace(0.0, 1.0, n_points + 1)[:-1]
    return np.unique(np.quantile(band_scores, quantiles))


def abstention_rate(in_band: np.ndarray) -> Callable[[float], tuple[float, int]]:
    """Fraction of traffic routed into the band -- the operational cost of the design.

    Independent of ``lambda`` by construction, since the band edges are frozen before the
    in-band threshold is chosen.  Reported rather than certified: it is a budget the bank
    sets, not a risk the procedure controls.
    """
    n = int(in_band.size)
    rate = float(in_band.mean()) if n else float("nan")

    def evaluate(_lam: float) -> tuple[float, int]:
        return rate, n

    return evaluate


@dataclass(frozen=True)
class RiskControlResult:
    """Admissible decision parameters, and the evidence for each."""

    admissible: np.ndarray
    selected: float
    per_lambda: list[dict[str, float]]
    delta: float
    fwer_method: str

    @property
    def any_admissible(self) -> bool:
        return self.admissible.size > 0

    def summary(self) -> str:
        if not self.any_admissible:
            return f"no grid point is certifiable at delta={self.delta}"
        return (
            f"{self.admissible.size} of {len(self.per_lambda)} grid points certifiable at "
            f"delta={self.delta} ({self.fwer_method}); selected lambda={self.selected:.6g}"
        )


def learn_then_test(
    grid: np.ndarray,
    risks: list[RiskDefinition],
    *,
    delta: float = 0.05,
    fwer_method: str = "bonferroni_holm",
    select: str = "max_recall",
) -> RiskControlResult:
    """Certify a set of decision parameters against several risks simultaneously.

    A grid point is admissible only if every risk's null is rejected at the corrected level,
    which is the intersection-union construction: the joint statement then holds with the
    same ``1 - delta``.

    ``select`` chooses among admissible points for reporting.  This is the one place where a
    heuristic is acceptable, because any admissible point already satisfies the guarantee --
    but it is named explicitly rather than left implicit, since a library default that
    silently optimises a secondary objective is easy to mistake for part of the guarantee.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie in (0, 1), got {delta!r}")
    if not risks:
        raise ValueError("at least one risk must be supplied")
    # Validated here, beside delta and risks, rather than only on the branch that consumes
    # it.  That branch is reached only when something is admissible, so on the 43 of the 48
    # committed configurations that certify nothing a misspelled rule came back as
    # "not certifiable" instead of raising -- a typo wearing the costume of a result.
    if select not in SELECT_RULES:
        raise ValueError(f"unknown select rule {select!r}; expected one of {sorted(SELECT_RULES)}")

    lambdas = np.asarray(grid, dtype=float).ravel()
    m = lambdas.size

    rows: list[dict[str, float]] = []
    p_matrix = np.ones((m, len(risks)))
    for i, lam in enumerate(lambdas):
        row: dict[str, float] = {"lambda": float(lam)}
        for j, risk in enumerate(risks):
            empirical, n_eff = risk.evaluate(float(lam))
            p = (
                1.0
                if not np.isfinite(empirical)
                else hoeffding_bentkus_p_value(empirical, n_eff, risk.target)
            )
            p_matrix[i, j] = p
            row[f"{risk.name}_risk"] = empirical
            row[f"{risk.name}_n"] = n_eff
            row[f"{risk.name}_p"] = p
        rows.append(row)

    # The family is every (grid point, risk) pair.  Correcting only across grid points would
    # ignore that each point is tested more than once.
    flat = p_matrix.ravel()
    order = np.argsort(flat)
    n_tests = flat.size
    rejected_flat = np.zeros(n_tests, dtype=bool)
    if fwer_method == "bonferroni_holm":
        for rank, idx in enumerate(order):
            if flat[idx] <= delta / (n_tests - rank):
                rejected_flat[idx] = True
            else:
                break  # step-down: once one is retained, so are all larger p-values
    elif fwer_method == "bonferroni":
        rejected_flat = flat <= delta / n_tests
    else:
        raise ValueError(f"unknown fwer_method {fwer_method!r}")

    rejected = rejected_flat.reshape(m, len(risks))
    admissible_mask = rejected.all(axis=1)
    admissible = lambdas[admissible_mask]

    for i, row in enumerate(rows):
        row["admissible"] = float(admissible_mask[i])

    if admissible.size == 0:
        selected = float("nan")
    elif select == "max_recall":
        # Lowest admissible threshold flags the most, so it maximises recall among points
        # that already satisfy both certificates.
        selected = float(admissible.min())
    elif select == "min_abstention":
        selected = float(admissible.max())
    else:  # pragma: no cover - unreachable; `select` is validated on entry
        raise ValueError(f"unknown select rule {select!r}")

    return RiskControlResult(
        admissible=admissible,
        selected=selected,
        per_lambda=rows,
        delta=delta,
        fwer_method=fwer_method,
    )
