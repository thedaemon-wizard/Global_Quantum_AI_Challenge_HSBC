#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""E1 -- form the four-block split and its two control arms, and audit their integrity.

Writes ``results/tables/splits.csv`` (one row per arm and block) and
``results/tables/data_integrity.csv`` (one row per integrity check).

The integrity checks exist because a split can be wrong in ways that do not raise.  Blocks
can silently overlap; a class can vanish from a block; two blocks can be distinguishable by
a classifier, which means they are not exchangeable and the conformal guarantee does not
transfer; and entities can span the boundary, which on this dataset is not a defect but a
measured property that dictates how confidence intervals must be computed.

    .venv/bin/python scripts/make_splits.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hsbcfraud.config import load_config
from hsbcfraud.data.ieee_cis import IEEE_CIS_ZIP, load_ieee_cis
from hsbcfraud.data.splits import (
    BLOCK_NAMES,
    Blocks,
    card_disjoint_blocks,
    stratified_blocks,
    temporal_blocks,
)

REPO = Path(__file__).resolve().parents[1]
COLUMNS = ["TransactionID", "TransactionAmt", "card1", "addr1", "D1", "ProductCD"]
# Features offered to the two-sample block-discrimination classifier.
TWO_SAMPLE_FEATURES = ["TransactionAmt", "card1", "addr1", "D1"]


def block_rows(frame: pd.DataFrame, blocks: Blocks) -> list[dict[str, object]]:
    """One summary row per block: size, span, class balance, and both fraud rates."""
    rows: list[dict[str, object]] = []
    for name in BLOCK_NAMES:
        part = frame.iloc[blocks[name]]
        fraud = part["isFraud"].to_numpy()
        amount = part["TransactionAmt"].to_numpy()
        fraud_amount = float(amount[fraud == 1].sum())
        total_amount = float(amount.sum())
        rows.append(
            {
                "arm": blocks.arm,
                "block": name,
                "n_rows": int(part.shape[0]),
                "n_fraud": int(fraud.sum()),
                "n_legit": int((fraud == 0).sum()),
                "day_first": int(part["day"].min()),
                "day_last": int(part["day"].max()),
                "count_fraud_rate": float(fraud.mean()),
                "value_fraud_rate": fraud_amount / total_amount,
                "total_amount": total_amount,
            }
        )
    return rows


def two_sample_auc(
    frame: pd.DataFrame, left: np.ndarray, right: np.ndarray, seed: int, features: list[str]
) -> float:
    """Can a classifier tell two blocks apart from their features alone?

    This is the operational form of "are these exchangeable".  An AUC near 0.5 means the
    blocks are indistinguishable; an AUC well above it means a model can tell which block a
    transaction came from, which is precisely the distribution shift the conformal guarantee
    has to survive.  The number is reported rather than used as a pass/fail gate, because on
    time-ordered payments data a shift is expected -- the point is to quantify it.

    ``features`` is a parameter, and must be, because including the column a split was
    partitioned on makes the test circular.  Measured here: with ``card1`` in the feature
    set the card-disjoint arm scores 0.9685, which says nothing except that the split did
    what it was asked to.  Excluding the split key is what makes the three arms comparable.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    x = pd.concat([frame.iloc[left][features], frame.iloc[right][features]], axis=0)
    y = np.concatenate([np.zeros(left.size), np.ones(right.size)])

    # Cap for runtime; the estimate is stable well below the full block size.
    cap = 60_000
    if len(x) > cap:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(x), size=cap, replace=False)
        x, y = x.iloc[keep], y[keep]

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=seed, stratify=y)
    model = HistGradientBoostingClassifier(max_iter=120, random_state=seed)
    model.fit(x_tr, y_tr)
    return float(roc_auc_score(y_te, model.predict_proba(x_te)[:, 1]))


def entity_overlap(frame: pd.DataFrame, blocks: Blocks, key: str) -> dict[str, float]:
    """How much of the test block's entity population also appears in training.

    High overlap is not a bug to be fixed here -- it is a property of the data that follows
    from the label-propagation rule, and it is the reason confidence intervals downstream
    are card-level block bootstrap rather than row-level.
    """
    train = set(pd.unique(frame.iloc[blocks["train"]][key].dropna()))
    test_values = pd.unique(frame.iloc[blocks["test"]][key].dropna())
    shared = sum(1 for value in test_values if value in train)
    return {
        "n_entities_train": float(len(train)),
        "n_entities_test": float(len(test_values)),
        "n_shared": float(shared),
        "fraction_test_entities_seen_in_train": (
            shared / len(test_values) if len(test_values) else 0.0
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--zip", type=Path, default=REPO / "datasets" / IEEE_CIS_ZIP)
    parser.add_argument("--out", type=Path, default=REPO / "results" / "tables")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    loaded = load_ieee_cis(args.zip, COLUMNS)
    frame = loaded.frame
    seed = cfg.split.seeds[0]

    print(
        f"IEEE-CIS: {loaded.n_rows:,} rows, {loaded.n_frauds:,} frauds "
        f"({loaded.fraud_rate:.4%}), span {loaded.span_days:.2f} days"
    )

    arms: dict[str, Blocks] = {
        "temporal": temporal_blocks(frame["day"], cfg.split),
        "stratified": stratified_blocks(frame["isFraud"], cfg.split, seed),
        "card_disjoint": card_disjoint_blocks(frame[cfg.split.entity_key], cfg.split, seed),
    }

    summary_rows: list[dict[str, object]] = []
    integrity_rows: list[dict[str, object]] = []
    for name, blocks in arms.items():
        blocks.assert_disjoint(loaded.n_rows)
        summary_rows.extend(block_rows(frame, blocks))

        for block in BLOCK_NAMES:
            part = frame.iloc[blocks[block]]
            if int(part["isFraud"].sum()) == 0:
                raise SystemExit(f"arm {name!r} block {block!r} contains no frauds; split unusable")

        # Two feature sets. The neutral one omits the entity key so the three arms are
        # comparable; the full one includes it and is reported because the gap between them
        # is itself the measurement of how much block identity is carried by the entity.
        neutral = [f for f in TWO_SAMPLE_FEATURES if f != cfg.split.entity_key]
        auc_neutral = two_sample_auc(frame, blocks["cal"], blocks["test"], seed, neutral)
        auc_full = two_sample_auc(frame, blocks["cal"], blocks["test"], seed, TWO_SAMPLE_FEATURES)
        overlap = entity_overlap(frame, blocks, cfg.split.entity_key)
        integrity_rows.append(
            {
                "arm": name,
                "cal_vs_test_two_sample_auc": auc_neutral,
                "cal_vs_test_two_sample_auc_with_entity": auc_full,
                "entity_key": cfg.split.entity_key,
                **overlap,
            }
        )
        print(
            f"  {name:14s} cal-vs-test AUC {auc_neutral:.4f} (without {cfg.split.entity_key}), "
            f"{auc_full:.4f} (with) | "
            f"{overlap['fraction_test_entities_seen_in_train']:.1%} of test entities in train"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(args.out / "splits.csv", index=False)
    pd.DataFrame(integrity_rows).to_csv(args.out / "data_integrity.csv", index=False)
    print(f"\nWrote {args.out / 'splits.csv'} and {args.out / 'data_integrity.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
