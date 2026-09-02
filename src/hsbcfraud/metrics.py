# SPDX-License-Identifier: Apache-2.0
"""Metrics, in both the count-weighted and value-weighted forms a bank manages to.

Why value weighting appears everywhere
--------------------------------------
Every headline metric here has a value-weighted twin.  This is not decoration: the quantity
PSD2 SCA-RTS Article 19 actually regulates is *total value of fraudulent remote transactions
divided by total value of all remote transactions*, on a rolling 90-day basis.  A model
tuned on count-weighted recall can pass a count-weighted target while failing the regulated
one, because fraud value distribution is not the fraud count distribution -- measured on
IEEE-CIS, the value-weighted fraud rate runs above the count-weighted rate in most 14-day
buckets.

Why ROC-AUC is reported but not led with
----------------------------------------
At 0.172 % prevalence on ULB, ROC-AUC saturates near 0.99 for almost any competent model and
carries very little information.  The dataset's own Kaggle page recommends AUPRC for exactly
this reason: "Given the class imbalance ratio, we recommend measuring the accuracy using the
Area Under the Precision-Recall Curve (AUPRC)."  Note that the recommendation is the dataset
page's and not the originating paper's -- Dal Pozzolo et al. endorse ROC-AUC -- and both this
module and the challenge statement previously credited the paper with it.

ROC-AUC is reported for comparability with the challenge statement's
baseline table, and AUPRC is the primary metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

__all__ = [
    "ClassificationReport",
    "false_decline_rate",
    "recall_at_threshold",
    "report",
    "threshold_for_value_fraud_rate",
    "value_weighted_fraud_rate",
]


@dataclass(frozen=True)
class ClassificationReport:
    """The challenge statement's named metrics, plus their value-weighted twins.

    The named set -- AUC-ROC, AUPRC, F1, precision, recall, confusion matrix -- is reported
    in full and in one place, because a reviewer working through the brief will look for
    exactly these and should not have to assemble them from adjacent quantities.
    """

    n: int
    n_positive: int
    threshold: float
    roc_auc: float
    average_precision: float
    precision: float
    recall: float
    f1: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    value_recall: float = float("nan")
    value_fraud_rate_approved: float = float("nan")
    false_decline_rate: float = float("nan")
    extra: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, float]:
        return {
            "n": self.n,
            "n_positive": self.n_positive,
            "threshold": self.threshold,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tn": self.true_negative,
            "fp": self.false_positive,
            "fn": self.false_negative,
            "tp": self.true_positive,
            "value_recall": self.value_recall,
            "value_fraud_rate_approved": self.value_fraud_rate_approved,
            "false_decline_rate": self.false_decline_rate,
            **self.extra,
        }


def false_decline_rate(y_true: np.ndarray, decisions: np.ndarray) -> float:
    """P(decline | legitimate).  The quantity this study certifies.

    Note the conditioning: the denominator is legitimate transactions, not all
    transactions.  A rate computed over all transactions would be dominated by prevalence
    and would move when the fraud rate moved, which is not what a customer-friction control
    should do.
    """
    y = np.asarray(y_true).ravel()
    d = np.asarray(decisions).astype(bool).ravel()
    legitimate = y == 0
    if not legitimate.any():
        return float("nan")
    return float(d[legitimate].mean())


def recall_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    y = np.asarray(y_true).ravel()
    flagged = np.asarray(scores).ravel() >= threshold
    positive = y == 1
    if not positive.any():
        return float("nan")
    return float(flagged[positive].mean())


def value_weighted_fraud_rate(
    y_true: np.ndarray, amounts: np.ndarray, approved: np.ndarray | None = None
) -> float:
    """Fraudulent value over total value, optionally restricted to approved transactions.

    With ``approved`` supplied this is the PSD2 Article 19 quantity for the branch that was
    let through without step-up authentication: the rate that governs eligibility for the
    transaction-risk-analysis exemption.
    """
    y = np.asarray(y_true).ravel()
    amount = np.asarray(amounts, dtype=float).ravel()
    mask = np.ones(y.size, dtype=bool) if approved is None else np.asarray(approved).astype(bool)
    total = amount[mask].sum()
    if total <= 0:
        return float("nan")
    return float(amount[mask & (y == 1)].sum() / total)


def threshold_for_value_fraud_rate(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    target_rate: float,
    *,
    max_decline_rate: float = 1.0,
) -> tuple[float, float, float]:
    """Highest threshold whose approved branch meets a value-weighted fraud-rate ceiling.

    The Neyman-Pearson operating point the pre-registration commits to: hold the regulated
    rate at or below its ceiling and approve as much as that allows.

    **Highest**, not lowest, and the direction is the whole content of the function.  As the
    threshold falls, fewer transactions are approved and the fraud rate among the approved
    falls with it, so the *lowest* threshold satisfies any ceiling trivially -- by declining
    almost everyone.  An earlier version of this function scanned ascending and returned the
    first match; on IEEE-CIS it reported a threshold of 3.5e-06 with an achieved fraud rate
    of exactly 0.0000 % and recall 1.000 for all three PSD2 tiers, which is what "decline
    everything" looks like when it is mistaken for an operating point.

    ``max_decline_rate`` can cap how much of the population may be declined, so a solution
    that meets the ceiling only by refusing most traffic is rejected rather than reported.
    It defaults to ``1.0``, which is a no-op -- a decline share cannot exceed 1, so the guard
    never fires unless a caller sets it.  The envelope call in ``scripts/run_conformal.py``
    leaves it at the default deliberately: amendment A2 reports the required decline rate
    (94.74 %, ``results/tables/envelope.csv``) as a distance measurement rather than as an
    operating point.

    Returns ``(threshold, achieved_value_rate, recall)``, or ``(nan, nan, nan)`` when no
    threshold satisfies both constraints.
    """
    y = np.asarray(y_true).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    amount = np.asarray(amounts, dtype=float).ravel()

    # Descending over the observed scores: a coarse grid would report a rate the deployed
    # system could not reproduce.
    for threshold in np.unique(s)[::-1]:
        approved = s < threshold
        if not approved.any():
            continue
        if float((~approved).mean()) > max_decline_rate:
            continue
        rate = value_weighted_fraud_rate(y, amount, approved)
        if rate <= target_rate:
            return float(threshold), float(rate), recall_at_threshold(y, s, threshold)
    return float("nan"), float("nan"), float("nan")


def report(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    amounts: np.ndarray | None = None,
) -> ClassificationReport:
    """Every metric the challenge statement names, at one threshold, in one object."""
    y = np.asarray(y_true).ravel().astype(int)
    s = np.asarray(scores, dtype=float).ravel()
    decisions = s >= threshold

    tn, fp, fn, tp = confusion_matrix(y, decisions.astype(int), labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    value_recall = float("nan")
    value_rate_approved = float("nan")
    if amounts is not None:
        amount = np.asarray(amounts, dtype=float).ravel()
        fraud_value = amount[y == 1].sum()
        value_recall = (
            float(amount[(y == 1) & decisions].sum() / fraud_value)
            if fraud_value
            else float("nan")
        )
        value_rate_approved = value_weighted_fraud_rate(y, amount, ~decisions)

    return ClassificationReport(
        n=int(y.size),
        n_positive=int((y == 1).sum()),
        threshold=float(threshold),
        roc_auc=float(roc_auc_score(y, s)),
        average_precision=float(average_precision_score(y, s)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
        value_recall=value_recall,
        value_fraud_rate_approved=value_rate_approved,
        false_decline_rate=false_decline_rate(y, decisions),
    )
