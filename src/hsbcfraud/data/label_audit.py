# SPDX-License-Identifier: Apache-2.0
"""Testing whether IEEE-CIS labels are censored at the end of the file.

The concern, and why it is not rhetorical
-----------------------------------------
``isFraud`` is 0 only when no chargeback was reported within 120 days (competition host,
Kaggle discussion 101203).  The training file spans 182 days.  If the labels had been
frozen at collection time, every transaction after day 62 would be inside its own maturity
window, the last 20 % of the file -- exactly where the test block sits -- would be the worst
labelled part of it, and since the certified quantity is conditional on ``Y = 0``, the
certificate would be built on the least trustworthy labels in the dataset.

That is a specific, falsifiable claim about the data, so it is tested rather than hedged.

The test
--------
Under censorship the observed fraud rate must **decay toward the end**: frauds that will be
reported later have not yet been recorded, so late transactions are disproportionately
labelled 0.  The decay would be monotone in expectation across the trailing window, and it
would be visible in both count-weighted and value-weighted rates.

Two statistics, because a single eyeball on a noisy series is not evidence:

* Mann-Kendall trend over the trailing buckets.  Non-parametric, so it does not assume the
  fraud rate is otherwise stationary -- which it is not.
* a two-proportion comparison of the trailing window against the rest of the file.

A negative result here is a real finding and is reported as one: it licenses using the
final block as the test fold without a maturity buffer, which would otherwise cost 120 of
182 days.

Value-weighted rates are computed alongside count-weighted ones throughout, because the
regulatory quantity in PSD2 SCA-RTS Article 19 is value-weighted and because censorship
need not be neutral with respect to transaction size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["CensoringVerdict", "audit_label_censoring", "bucket_fraud_rates"]

# The chargeback observation window the label rule uses, in days.
LABEL_WINDOW_DAYS = 120


@dataclass(frozen=True)
class CensoringVerdict:
    """The outcome of the censoring test, in enough detail to be checked."""

    trailing_days: int
    n_buckets_trailing: int
    mann_kendall_tau: float
    mann_kendall_p: float
    trailing_rate: float
    earlier_rate: float
    two_proportion_p: float
    censoring_detected: bool

    def summary(self) -> str:
        verdict = "DETECTED" if self.censoring_detected else "not detected"
        return (
            f"terminal label censoring {verdict}: Mann-Kendall tau={self.mann_kendall_tau:+.3f} "
            f"(p={self.mann_kendall_p:.3f}) over the trailing {self.trailing_days} days; "
            f"trailing fraud rate {self.trailing_rate:.4f} vs {self.earlier_rate:.4f} earlier "
            f"(p={self.two_proportion_p:.3f})"
        )


def bucket_fraud_rates(
    frame: pd.DataFrame, *, bucket_days: int = 14, amount_column: str = "TransactionAmt"
) -> pd.DataFrame:
    """Count-weighted and value-weighted fraud rate per fixed-width bucket of days.

    Buckets rather than a rolling window because the series is reported in the appendix and
    a reader has to be able to read the numbers off it.
    """
    for required in ("day", "isFraud", amount_column):
        if required not in frame.columns:
            raise KeyError(f"bucket_fraud_rates needs a {required!r} column")

    bucket = (frame["day"] // bucket_days) * bucket_days
    fraud = frame["isFraud"].to_numpy()
    amount = frame[amount_column].to_numpy()

    work = pd.DataFrame(
        {
            "bucket": bucket.to_numpy(),
            "is_fraud": fraud,
            "amount": amount,
            "fraud_amount": np.where(fraud == 1, amount, 0.0),
        }
    )
    grouped = work.groupby("bucket", as_index=False).agg(
        n=("is_fraud", "size"),
        n_fraud=("is_fraud", "sum"),
        total_amount=("amount", "sum"),
        fraud_amount=("fraud_amount", "sum"),
    )
    grouped["count_rate"] = grouped["n_fraud"] / grouped["n"]
    grouped["value_rate"] = grouped["fraud_amount"] / grouped["total_amount"]
    grouped["bucket_end"] = grouped["bucket"] + bucket_days - 1
    return grouped


def audit_label_censoring(
    frame: pd.DataFrame,
    *,
    bucket_days: int = 14,
    trailing_days: int = LABEL_WINDOW_DAYS,
    alpha: float = 0.05,
) -> tuple[CensoringVerdict, pd.DataFrame]:
    """Test for a terminal decay in the fraud rate, the signature of label censoring.

    ``trailing_days`` defaults to the label rule's own 120-day window: that is the region
    that would be affected if the labels had been frozen at collection time.
    """
    buckets = bucket_fraud_rates(frame, bucket_days=bucket_days)
    last_day = int(frame["day"].max())
    cutoff = last_day - trailing_days

    trailing = buckets[buckets["bucket"] >= cutoff]
    earlier = buckets[buckets["bucket"] < cutoff]
    if len(trailing) < 3:
        raise ValueError(
            f"only {len(trailing)} buckets fall in the trailing {trailing_days} days; "
            "widen the window or narrow the buckets before testing for a trend"
        )

    # Mann-Kendall via Kendall's tau against bucket order.  A negative tau is the direction
    # censorship would produce; a positive one is evidence against it.
    tau, tau_p = stats.kendalltau(trailing["bucket"], trailing["count_rate"])
    tau = float(tau)
    tau_p = float(tau_p)

    trailing_fraud = int(trailing["n_fraud"].sum())
    trailing_n = int(trailing["n"].sum())
    earlier_fraud = int(earlier["n_fraud"].sum())
    earlier_n = int(earlier["n"].sum())
    trailing_rate = trailing_fraud / trailing_n
    earlier_rate = earlier_fraud / earlier_n

    # One-sided: censorship can only depress the trailing rate, so only that direction is
    # evidence for it.  Fisher's exact rather than a normal approximation because the
    # question deserves an exact answer and the table is small enough to afford one.
    table = [[trailing_fraud, trailing_n - trailing_fraud], [earlier_fraud, earlier_n - earlier_fraud]]
    two_proportion_p = float(stats.fisher_exact(table, alternative="less").pvalue)

    # Both witnesses must point the same way before censorship is declared.  A single
    # marginal p-value on a noisy, non-stationary series is not enough to justify discarding
    # two thirds of the file.
    detected = bool(tau < 0 and tau_p < alpha and two_proportion_p < alpha)

    verdict = CensoringVerdict(
        trailing_days=trailing_days,
        n_buckets_trailing=len(trailing),
        mann_kendall_tau=tau,
        mann_kendall_p=tau_p,
        trailing_rate=trailing_rate,
        earlier_rate=earlier_rate,
        two_proportion_p=two_proportion_p,
        censoring_detected=detected,
    )
    return verdict, buckets
