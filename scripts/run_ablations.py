#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""E3b -- the two leakage ablations that make the classical baseline honest.

Writes ``results/tables/ablations.csv``.

This study's whole comparison rests on the classical arm being tuned at least as hard as the
quantum arm.  An artificially weak baseline is the commonest way a quantum comparison is
tilted, and the two most available ways to inflate a fraud baseline on IEEE-CIS are both
forms of leakage.  Rather than assert that neither was used, both are run and reported.

**Causal aggregates.**  Per-entity behavioural features computed over an entity's whole
history include the transaction being scored and every transaction after it.  The causal
version uses an expanding window shifted by one, so only strictly past rows contribute.

Measured here, the gap is null: +0.0054 AP for the non-causal variant against a per-seed
standard deviation of 0.0053.  That is the reported result, not a disappointment.  The causal
construction is kept because it is correct and costs nothing.

**The UID feature.**  The competition was won by reconstructing a client key,
``card1_addr1 + floor(day - D1)``, and aggregating over it.  It is not used in the reported
baseline.  The reason is not modesty: the label rule propagates a chargeback across
transactions linked by account, email or billing address, so a reconstructed client key is
partly a reconstruction of the labelling mechanism.  An issuer holds the true identifier
natively, so recovering it from de-identified columns measures the de-identification rather
than transferable headroom.

Measured here it is worth +0.0015 AP, inside noise, against a published +0.011 AUC obtained
under time-based cross-validation.  The difference is mechanical -- cross-validation lets a
client recur across folds and a 40-day forward gap does not -- and is independent evidence for
this study's framing.

    .venv/bin/python scripts/run_ablations.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from hsbcfraud.config import load_config
from hsbcfraud.data.ieee_cis import IEEE_CIS_ZIP, load_ieee_cis
from hsbcfraud.data.splits import temporal_blocks
from hsbcfraud.features.engineering import add_entity_aggregates, add_uid, select_model_columns
from hsbcfraud.paths import display_path
from hsbcfraud.progress import SweepTimer, run_log

REPO = Path(__file__).resolve().parents[1]


def encode_strings(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype == "object" or str(out[column].dtype) == "str":
            out[column] = out[column].astype("category").cat.codes.astype("int32")
    return out


def fit_and_score(frame: pd.DataFrame, features: list[str], blocks, seed: int):
    import xgboost as xgb

    y = frame["isFraud"].to_numpy()
    x = frame[features]
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.5,
        min_child_weight=4,
        reg_lambda=1.0,
        tree_method="hist",
        device="cuda",
        eval_metric="aucpr",
        random_state=seed,
        verbosity=0,
    )
    start = time.perf_counter()
    model.fit(x.iloc[blocks["train"]], y[blocks["train"]])
    scores = model.predict_proba(x.iloc[blocks["test"]])[:, 1]
    elapsed = time.perf_counter() - start
    y_test = y[blocks["test"]]
    return {
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "average_precision": float(average_precision_score(y_test, scores)),
        "n_features": len(features),
        "fit_seconds": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--zip", type=Path, default=REPO / "datasets" / IEEE_CIS_ZIP)
    parser.add_argument("--out", type=Path, default=REPO / "results" / "tables")
    parser.add_argument("--runs", type=Path, default=REPO / "results" / "runs")
    # The first three of cfg.split.seeds, which is what produced the committed 12-row table.
    # Not cfg.split.seeds itself: that list is five, `make ablations` passes no flags, and the
    # fallback therefore rebuilt this arm at 20 rows -- a different experiment under the same
    # filename, disagreeing with the three seeds docs/decisions.md records for it.
    parser.add_argument("--seeds", type=int, nargs="*", default=[20260828, 20260829, 20260830])
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seeds = args.seeds
    if not seeds:
        # `--seeds` with no values used to fall through to the config.  Silently substituting a
        # different seed set is how this table drifted in the first place, so it is an error.
        raise SystemExit("--seeds needs at least one value; omit the flag for the reported three")

    loaded = load_ieee_cis(args.zip, None, with_identity=True)
    base = encode_strings(loaded.frame)
    blocks = temporal_blocks(base["day"], cfg.split)

    # Four arms.  "reported" is the configuration every other table in this study uses.
    variants = {
        "reported_causal_no_uid": lambda f: add_entity_aggregates(f, causal=True),
        "noncausal_aggregates": lambda f: add_entity_aggregates(f, causal=False),
        "causal_plus_uid": lambda f: add_uid(add_entity_aggregates(f, causal=True)),
        "no_aggregates": lambda f: f,
    }

    rows: list[dict] = []
    sweep = SweepTimer(len(variants) * len(seeds))
    with run_log("ablations", directory=args.runs) as run:
        run.info(f"{len(variants)} variants x {len(seeds)} seeds on the temporal arm")
        for name, build in variants.items():
            frame = build(base)
            features = select_model_columns(frame)
            for seed in seeds:
                started = time.perf_counter()
                result = fit_and_score(frame, features, blocks, seed)
                sweep.record(time.perf_counter() - started)
                rows.append({"variant": name, "seed": seed, "arm": "temporal", **result})
                run.info(
                    f"  {name:24s} seed {seed} features {result['n_features']:3d} "
                    f"AUC {result['roc_auc']:.4f} AP {result['average_precision']:.4f} "
                    f"-- {sweep.summary()}"
                )

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "ablations.csv", index=False)

    print("\nMean over seeds, and the gap against the reported configuration:")
    summary = table.groupby("variant")[["roc_auc", "average_precision"]].agg(["mean", "std"])
    reference = summary.loc["reported_causal_no_uid"]
    for variant in summary.index:
        row = summary.loc[variant]
        d_auc = row[("roc_auc", "mean")] - reference[("roc_auc", "mean")]
        d_ap = row[("average_precision", "mean")] - reference[("average_precision", "mean")]
        marker = "  (reported)" if variant == "reported_causal_no_uid" else ""
        print(
            f"  {variant:24s} AUC {row[('roc_auc', 'mean')]:.4f} "
            f"(sd {row[('roc_auc', 'std')]:.4f}, delta {d_auc:+.4f})  "
            f"AP {row[('average_precision', 'mean')]:.4f} "
            f"(sd {row[('average_precision', 'std')]:.4f}, delta {d_ap:+.4f}){marker}"
        )
    print(f"\nWrote {display_path(args.out / 'ablations.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
