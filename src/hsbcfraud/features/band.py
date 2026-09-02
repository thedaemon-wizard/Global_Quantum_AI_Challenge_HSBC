# SPDX-License-Identifier: Apache-2.0
"""The in-band feature pipeline, shared by every arm that works inside the abstention band.

The band arm's inputs have to be constructed identically wherever they are used, or the arms
stop being comparable and an explanation stops describing the model it claims to explain.
This module is the single definition: the band edges, the rows that fall inside them, the
feature subset, and the scaling.

It was extracted from ``scripts/run_mps.py`` unchanged.  Five scripts now call it -- ``run_mps``,
``run_explain``, ``run_power``, ``measure_latency`` and ``run_seed_sweep`` -- rather than each
carrying a copy, which is what makes "the same rows and the same features" a fact about the code
instead of a claim in prose.  ``run_power`` was the last holdout: it carried a line-for-line
reimplementation of all four functions, so the power gate was measured on a band that only
happened to agree with the one the arms use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

# Rows drawn from the training block to estimate mutual information.  The full block is
# 356,216 rows and the estimator is O(n log n) per feature over 431 features; the subsample is
# what makes the selection affordable, and it is seeded so the choice is reproducible.
MUTUAL_INFORMATION_PROBE_ROWS = 20_000


def band_edges(
    band_scores: np.ndarray, decline_budget: float, traffic_budget: float
) -> tuple[float, float]:
    """The abstention band's lower and upper score thresholds.

    Both are quantiles of ``D_band`` and of nothing else.  Deriving them from the block that
    later certifies would make band membership a data-dependent event and break the
    exchangeability argument the guarantee rests on.
    """
    high = float(np.quantile(band_scores, 1.0 - decline_budget))
    low = float(np.quantile(band_scores, 1.0 - decline_budget - traffic_budget))
    return low, high


def rows_in_band(scores: pd.DataFrame, block: str, low: float, high: float) -> np.ndarray:
    """Original row indices for one block's scores that fall inside the band.

    Half-open on the right, matching the decision rule: a score at the upper edge is a
    decline, not an abstention.
    """
    inside = scores[scores["block"] == block]
    selected = inside[(inside["score"] >= low) & (inside["score"] < high)]
    return selected["row"].to_numpy()


def select_band_features(
    frame: pd.DataFrame,
    train_rows: np.ndarray,
    labels: np.ndarray,
    columns: list[str],
    *,
    k: int,
    seed: int,
) -> list[str]:
    """The ``k`` features with the highest mutual information with the label.

    Selected on the **training** block, never on the band or calibration blocks, so that the
    feature choice carries no information from the data the guarantee is computed on.
    """
    if k < 1:
        raise ValueError(f"need at least one feature, got {k}")
    if k > len(columns):
        raise ValueError(f"asked for {k} features from a pool of {len(columns)}")
    rng = np.random.default_rng(seed)
    probe = rng.choice(
        len(train_rows), size=min(MUTUAL_INFORMATION_PROBE_ROWS, len(train_rows)), replace=False
    )
    probe_x, _ = prepare(frame, train_rows[probe], columns, None)
    information = mutual_info_classif(probe_x, labels[train_rows[probe]], random_state=seed)
    return [columns[i] for i in np.argsort(information)[::-1][:k]]


def prepare(
    frame: pd.DataFrame,
    rows: np.ndarray,
    columns: list[str],
    scaler: MinMaxScaler | None,
) -> tuple[np.ndarray, MinMaxScaler]:
    """Numeric matrix scaled to [0, 1], which is the domain the local feature map needs.

    The scaler is fitted once, on the block passed with ``scaler=None``, and reused everywhere
    else.  Fitting it per block would leak the deployment distribution into the encoding.

    **The imputation median is not threaded the same way, and that is a known defect.**  It is
    recomputed from ``rows`` on every call, so the second call in a train-then-evaluate pair
    fills the evaluation block's missing values with the evaluation block's own medians.  The
    scaler has an object to carry it across calls and the median does not, which is why only
    one of the two got threaded.  The exposure is bounded but real: an imputed value derived
    from the evaluation block can fall outside the training range and is then silently clipped
    by ``MinMaxScaler(clip=True)``.  Fixing it means giving the median the same treatment as
    the scaler -- an in/out parameter -- and every committed number that descends from this
    function moves when it lands, so it needs a decision entry and a re-run rather than a
    quiet edit.
    """
    numeric = frame.iloc[rows][columns]
    filled = (
        numeric.fillna(numeric.median(numeric_only=True))
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    if scaler is None:
        scaler = MinMaxScaler(clip=True).fit(filled)
    return scaler.transform(filled).astype(np.float32), scaler
