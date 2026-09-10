#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""E3 -- classical baselines on the temporal split, with two leakage ablations.

Writes ``results/tables/baselines.csv`` (one row per model, arm and seed) and the frozen
score interface ``results/runs/scores_<arm>_<seed>.parquet`` that every downstream stage
consumes.  The leakage ablations live in ``scripts/run_ablations.py``.

The score interface is what keeps the conformal layer independent of how a score was
produced: it carries the row index, the block, the label, the amount and the entity key,
so the risk-control stage never needs to know whether the score came from a GBDT, a quantum
kernel or a coin.

    .venv/bin/python scripts/run_baselines.py --seeds 20260828
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hsbcfraud import metrics
from hsbcfraud.config import load_config
from hsbcfraud.data.ieee_cis import IEEE_CIS_ZIP, load_ieee_cis
from hsbcfraud.data.splits import Blocks, card_disjoint_blocks, stratified_blocks, temporal_blocks
from hsbcfraud.features.engineering import add_entity_aggregates, select_model_columns
from hsbcfraud.paths import display_path
from hsbcfraud.progress import SweepTimer, run_log

REPO = Path(__file__).resolve().parents[1]

# All 394 transaction columns plus the 40 identity columns, left-joined.  Hand-picking a
# subset is itself a modelling decision made with knowledge of the dataset, and an
# artificially weak baseline is the most common way a comparison gets tilted -- this study's
# argument requires the classical arm to be as strong as the quantum arm is given room to be.
#
# On the word "tuned", which this project used to use here and no longer does: there is no
# hyperparameter search in this repository -- no grid, no optuna, no cross-validated sweep.
# The seven XGBoost settings below are fixed at values the IEEE-CIS public solutions converged
# on, and what *was* compared is the feature set, measured immediately below.  Calling that
# "tuned" overstates the effort spent on the arm the whole comparison is measured against,
# which is the direction that flatters the quantum result.  See D-140.
# Measured: 40 hand-picked columns give AUC 0.8830 / AP 0.4823; the full set gives
# 0.8805 / 0.5083, i.e. the same AUC and materially better average precision.
BASE_COLUMNS = None


def encode_strings(frame: pd.DataFrame) -> pd.DataFrame:
    """String columns to integer codes.

    Integer codes rather than pandas categoricals, deliberately: SHAP's tree explainer has a
    long-standing incompatibility with categorical dtypes on boosted models, and
    explainability is a scored criterion, so the encoding is chosen to keep that path open.
    """
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype == "object" or str(out[column].dtype) == "str":
            out[column] = out[column].astype("category").cat.codes.astype("int32")
    return out


# The deployed full-traffic scorer.  A module constant rather than literals inside `fit_xgboost`,
# because two other scripts need to describe *this* model and were each re-typing it:
# `measure_latency.py` priced a 400-tree depth-6 stand-in while its comment said the timing
# "prices the deployed model rather than a stand-in", and `tune_baseline.py` kept a third copy as
# the incumbent to search around.  One dict, imported, so the three cannot disagree again.
XGBOOST_PARAMS = dict(
    n_estimators=1000,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.5,
    min_child_weight=4,
    reg_lambda=1.0,
    tree_method="hist",
    eval_metric="aucpr",
    verbosity=0,
)


def fit_xgboost(x_train, y_train, x_eval, seed: int, device: str = "cuda"):
    import xgboost as xgb

    model = xgb.XGBClassifier(
        **XGBOOST_PARAMS,
        device=device,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model, model.predict_proba(x_eval)[:, 1]


def fit_lightgbm(x_train, y_train, x_eval, seed: int):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=600,
        num_leaves=96,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_samples=40,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    return model, model.predict_proba(x_eval)[:, 1]


def fit_logistic(x_train, y_train, x_eval, seed: int):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    model.fit(x_train, y_train)
    return model, model.predict_proba(x_eval)[:, 1]


FITTERS = {"xgboost": fit_xgboost, "lightgbm": fit_lightgbm, "logistic": fit_logistic}

# The arm the study reports; the controls come from the config so the two cannot drift apart.
REPORTED_ARM = "temporal"


def evaluate(frame: pd.DataFrame, blocks: Blocks, features: list[str], model_name: str, seed: int):
    """Fit on train, score band/cal/test, and report at a provisional 0.5 threshold.

    The threshold here is provisional and exists only so the confusion matrix is defined;
    the operating point that matters is chosen in E4 from the decline budget, and the
    certified one in E5.  Reporting a metric at 0.5 and calling it the result would be
    reporting a number nobody would deploy.
    """
    y = frame["isFraud"].to_numpy()
    x = frame[features]
    fitter = FITTERS[model_name]

    train_idx = blocks["train"]
    other_idx = np.concatenate([blocks["band"], blocks["cal"], blocks["test"]])
    start = time.perf_counter()
    _, scores_other = fitter(x.iloc[train_idx], y[train_idx], x.iloc[other_idx], seed)
    fit_seconds = time.perf_counter() - start

    scores = np.full(len(frame), np.nan)
    scores[other_idx] = scores_other

    rows = []
    for block in ("band", "cal", "test"):
        idx = blocks[block]
        rep = metrics.report(
            y[idx], scores[idx], 0.5, amounts=frame["TransactionAmt"].to_numpy()[idx]
        )
        rows.append(
            {
                "model": model_name,
                "arm": blocks.arm,
                "seed": seed,
                "block": block,
                "fit_seconds": fit_seconds,
                **rep.as_row(),
            }
        )
    return rows, scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--zip", type=Path, default=REPO / "datasets" / IEEE_CIS_ZIP)
    parser.add_argument("--out", type=Path, default=REPO / "results" / "tables")
    parser.add_argument("--runs", type=Path, default=REPO / "results" / "runs")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    # The defaults must reproduce the committed table, because `make reproduce` passes no
    # flags.  They did not: the committed baselines.csv is xgboost over the temporal arm and
    # both controls, and these defaulted to all three fitters over the temporal arm alone.
    # `make reproduce` therefore overwrote the table with a different shape and the next
    # target, summarise_split_arms.py, halted on the two missing arms -- so the repository's
    # own reproduction command could not rebuild the table behind its largest reported effect.
    # lightgbm and logistic remain available; they are simply not what the study reports.
    parser.add_argument("--models", nargs="*", default=["xgboost"])
    parser.add_argument("--arms", nargs="*", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seeds = args.seeds or cfg.split.seeds
    arms = args.arms or [REPORTED_ARM, *cfg.split.control_arms]

    loaded = load_ieee_cis(args.zip, BASE_COLUMNS, with_identity=True)
    frame = encode_strings(loaded.frame)
    causal = add_entity_aggregates(frame, causal=True)
    features = select_model_columns(causal)
    print(f"{loaded.n_rows:,} rows, {len(features)} features, seeds {seeds}, arms {arms}")

    arm_blocks = {
        "temporal": lambda s: temporal_blocks(frame["day"], cfg.split),
        "stratified": lambda s: stratified_blocks(frame["isFraud"], cfg.split, s),
        "card_disjoint": lambda s: card_disjoint_blocks(frame[cfg.split.entity_key], cfg.split, s),
    }

    rows: list[dict] = []
    # Counted rather than computed as len(seeds) * len(arms): the score interface is written
    # only on the xgboost pass below, so `--models lightgbm` wrote no parquet at all while the
    # closing message still promised one per arm and seed -- pointing a reader at a stage that
    # had not run when the next one failed to find its input.
    written = 0
    args.runs.mkdir(parents=True, exist_ok=True)
    fits = [(arm, seed, model) for arm in arms for seed in seeds for model in args.models]
    # Three model families over the requested arms and seeds is about thirteen minutes of
    # fitting with nothing on stdout between the first line and the last. The jobs are not
    # equal-cost -- logistic regression is far cheaper than either boosted family -- so the
    # estimate is reported together with the spread it comes from, which is what
    # SweepTimer.summary prints, rather than as a deadline.
    sweep = SweepTimer(len(fits))
    blocks_for: dict[tuple[str, int], Blocks] = {}

    with run_log("baselines", directory=args.runs) as run:
        run.info(f"{len(fits)} fits: arms {arms}, seeds {seeds}, models {args.models}")
        for index, (arm, seed, model_name) in enumerate(fits, start=1):
            # Cached rather than rebuilt per model: the split is the same for all three, and
            # rebuilding it would be work done only to be thrown away.
            blocks = blocks_for.setdefault((arm, seed), arm_blocks[arm](seed))
            model_rows, scores = evaluate(causal, blocks, features, model_name, seed)
            rows.extend(model_rows)
            test_row = next(r for r in model_rows if r["block"] == "test")
            sweep.record(float(test_row["fit_seconds"]))
            run.info(
                f"  fit {index}/{len(fits)}  {arm:13s} seed {seed} {model_name:9s} "
                f"test AUC {test_row['roc_auc']:.4f} AP {test_row['average_precision']:.4f} "
                f"({test_row['fit_seconds']:.0f}s) -- {sweep.summary()}"
            )
            if model_name == "xgboost":
                # The frozen score interface. Everything downstream reads this, not the
                # model, so a stage can be re-run without refitting and a different scorer
                # can be substituted without touching the conformal code.
                block_of = np.full(len(causal), "train", dtype=object)
                for name in ("band", "cal", "test"):
                    block_of[blocks[name]] = name
                pd.DataFrame(
                    {
                        "row": np.arange(len(causal)),
                        "block": block_of,
                        "y": causal["isFraud"].to_numpy(),
                        "score": scores,
                        "amount": causal["TransactionAmt"].to_numpy(),
                        "entity": causal[cfg.split.entity_key].to_numpy(),
                        "day": causal["day"].to_numpy(),
                    }
                ).to_parquet(args.runs / f"scores_{arm}_{seed}.parquet", index=False)
                written += 1

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "baselines.csv", index=False)
    print(f"\nWrote {display_path(args.out / 'baselines.csv')} and {written} score file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
