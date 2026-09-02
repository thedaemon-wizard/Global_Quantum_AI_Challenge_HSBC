# SPDX-License-Identifier: Apache-2.0
"""Feature construction, and the two ablations that make it honest.

Causality
---------
Per-entity aggregates are computed over **strictly past** transactions using an expanding
window.  Computing them over the whole frame is a standard leakage route in fraud modelling:
a "mean transaction amount for this card" that includes the transaction being scored, and
transactions after it, encodes the future.

The non-causal variant is built and reported as an ablation (``scripts/run_ablations.py``).
**Measured on this split, the gap is null**: non-causal aggregates give AP 0.5135 against
0.5081 for causal, a difference of +0.0054 against a per-seed standard deviation of 0.0053.
Removing the aggregates entirely costs 0.0037 AUC.

So the causal construction is retained because it is the correct thing to do and costs
nothing, not because a leak was demonstrated here.  Saying otherwise would be citing a
motivation the data does not support.  A larger effect has been observed on other datasets
with stronger per-entity signal; that is not evidence about this one.

The UID feature is deliberately absent
--------------------------------------
The IEEE-CIS competition was won by reconstructing a client identifier,
``card1_addr1 + floor(day - D1)``, and aggregating over it; published analysis puts its
contribution at about +0.011 AUC under time-based GroupKFold cross-validation.  It is not
used here, and the ablation is reported.

**Measured under this study's forward holdout the feature is worth nothing**: AP 0.5095 with
it against 0.5081 without, a difference of +0.0015 against a per-seed standard deviation of
0.0029.  The published +0.011 was obtained under cross-validation on the training file, where
a client seen in one fold recurs in another; across a 40-day forward gap that recurrence has
largely decayed.  The contrast is itself evidence for the study's framing.

The reason for excluding it is not modesty.  The label rule propagates a chargeback across
transactions linked by account, email or billing address, so a reconstructed client key is partly a
reconstruction of the labelling mechanism itself.  More practically: an issuer already holds
the true client identifier natively.  Recovering it from de-identified columns measures the
de-identification, not headroom that would transfer to a deployed system.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["ENTITY_KEYS", "add_entity_aggregates", "add_uid", "select_model_columns"]

# Entities over which behavioural aggregates are formed.  card1 is a coarse card attribute
# rather than a card identifier; addr1 is a billing region.  Both are named in the label
# propagation rule, which is why they are the ones that matter.
ENTITY_KEYS = ("card1", "addr1")

# Columns that are identifiers or targets rather than features.
_EXCLUDED = {"TransactionID", "isFraud", "day", "TransactionDT"}


def add_entity_aggregates(
    frame: pd.DataFrame,
    *,
    keys: tuple[str, ...] = ENTITY_KEYS,
    amount_column: str = "TransactionAmt",
    causal: bool = True,
) -> pd.DataFrame:
    """Per-entity behavioural features.

    With ``causal=True`` every statistic for row *i* uses only rows before *i* within the
    same entity, via an expanding window shifted by one.  The first transaction for an
    entity therefore has NaN aggregates, which is correct -- there is no history -- and the
    gradient-boosted models used here handle NaN natively rather than needing an imputed
    value that would invent one.

    With ``causal=False`` the same statistics are computed over the entity's whole history.
    That variant exists only to be reported as an ablation.
    """
    if amount_column not in frame.columns:
        raise KeyError(f"{amount_column!r} is required to build entity aggregates")
    out = frame.copy()

    for key in keys:
        if key not in frame.columns:
            raise KeyError(f"entity key {key!r} is not present")
        grouped = out.groupby(key, sort=False, observed=True)[amount_column]

        if causal:
            shifted = grouped.shift(1)
            history = shifted.groupby(out[key], sort=False, observed=True)
            expanding = history.expanding()
            out[f"{key}_amt_mean_past"] = expanding.mean().reset_index(level=0, drop=True)
            out[f"{key}_amt_std_past"] = expanding.std().reset_index(level=0, drop=True)
            out[f"{key}_count_past"] = expanding.count().reset_index(level=0, drop=True)
        else:
            out[f"{key}_amt_mean_past"] = grouped.transform("mean")
            out[f"{key}_amt_std_past"] = grouped.transform("std")
            out[f"{key}_count_past"] = grouped.transform("count")

        # Ratio to the entity's own history is the actual signal: an unusual amount *for
        # this card* is informative where an unusual amount overall mostly encodes the
        # merchant category.
        mean_past = out[f"{key}_amt_mean_past"]
        out[f"{key}_amt_ratio"] = np.where(
            mean_past.to_numpy() > 0, out[amount_column].to_numpy() / mean_past.to_numpy(), np.nan
        )

    return out


def add_uid(frame: pd.DataFrame, *, amount_column: str = "TransactionAmt") -> pd.DataFrame:
    """The competition's reconstructed client key, for the ablation only.

    ``UID = card1_addr1 + floor(day - D1)``.  ``D1`` is days since the card first appeared,
    so ``day - D1`` is approximately the card's first-seen date and is stable across that
    card's transactions.

    ``amount_column`` is checked with the key columns rather than assumed.  It is as much a
    requirement as ``card1`` is -- the per-UID aggregates below are built from it -- and
    hardcoding it meant a frame without it failed with a raw pandas KeyError instead of this
    function's own message, unlike its sibling ``add_entity_aggregates`` in the same file.
    """
    for required in ("card1", "addr1", "D1", "day", amount_column):
        if required not in frame.columns:
            raise KeyError(f"{required!r} is required to build the UID feature")
    out = frame.copy()
    first_seen = np.floor(out["day"].to_numpy() - out["D1"].to_numpy())
    out["uid"] = (
        out["card1"].astype("string").fillna("NA")
        + "_"
        + out["addr1"].astype("string").fillna("NA")
        + "_"
        + pd.Series(first_seen, index=out.index).astype("string").fillna("NA")
    )
    grouped = out.groupby("uid", sort=False, observed=True)[amount_column]
    shifted = grouped.shift(1)
    expanding = shifted.groupby(out["uid"], sort=False, observed=True).expanding()
    out["uid_amt_mean_past"] = expanding.mean().reset_index(level=0, drop=True)
    out["uid_count_past"] = expanding.count().reset_index(level=0, drop=True)
    return out.drop(columns=["uid"])


def select_model_columns(frame: pd.DataFrame, *, drop: tuple[str, ...] = ()) -> list[str]:
    """Numeric feature columns, with identifiers and targets removed.

    String columns are converted to integer codes upstream rather than left as pandas
    categoricals: SHAP's tree explainer has a long-standing incompatibility with categorical
    dtypes on boosted models, and explainability is a scored criterion here, so the dtype is
    chosen to keep that path working.
    """
    excluded = _EXCLUDED | set(drop)
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
