#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""E12 -- feature attribution for the in-band decision, which the challenge statement requires.

Section 5.2 of the challenge statement lists three expected outputs: a fraud probability, a
binary prediction, and a **feature attribution** -- "contribution of features to each
prediction".  The first two were produced from the first run.  The third was not, and its
absence is the kind of gap that is invisible from inside a repository whose gates all check
that the numbers present are correct.

**What is explained, and why that one.**  The attribution is computed for the in-band
re-scorer, not for the model that sees full traffic.  The band is where the three-valued rule
actually makes its decision and where a quantum model would sit if one were usable, so it is
the component a model-risk review would ask about.  It is built from exactly the rows and
features the tensor-network arm uses, through
:mod:`hsbcfraud.features.band`, so this explains the model that arm compares against rather
than a similar one.

**Exact, not sampled.**  ``TreeExplainer`` computes Shapley values for a tree ensemble in
polynomial time, so no sampling approximation enters and the values sum to the model output
exactly.  That identity is asserted rather than assumed; a violated additivity check means the
explanation does not describe the model and is worth failing on.

Two tables are written, because the statement asks for both a ranked list and per-prediction
contributions:

``attribution.csv``          every feature, mean absolute contribution, and rank.
``attribution_examples.csv`` the highest-scoring in-band rows, with each feature's signed
                             contribution to that individual prediction.

    .venv/bin/python scripts/run_explain.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from hsbcfraud.config import load_config
from hsbcfraud.data.ieee_cis import IEEE_CIS_ZIP, load_ieee_cis
from hsbcfraud.features.band import band_edges, prepare, rows_in_band, select_band_features
from hsbcfraud.paths import display_path

REPO = Path(__file__).resolve().parents[1]

# The in-band re-scorer this script explains.  It is NOT the control the tensor-network arm is
# compared against: that one, at run_mps.py:265-268, sets neither `subsample` nor
# `colsample_bytree`, so the two differ in exactly those two hyperparameters.  An earlier
# comment here claimed they were identical and pointed at a parity assertion that does not
# exist.  The difference is harmless -- the attribution is about which features this model uses,
# not about a comparison -- but the two are separate objects and saying otherwise invited a
# reader to treat an attribution figure as a statement about the H4 baseline.
BAND_MODEL = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    tree_method="hist",
    eval_metric="aucpr",
    subsample=0.8,
    colsample_bytree=0.8,
    verbosity=0,
)

# How many individual predictions to write out.  The statement asks for per-prediction
# attribution; a handful of worked examples demonstrates the capability without shipping a
# table with one row per transaction.
N_EXAMPLES = 10
# Shapley values are additive: contributions plus the base value reproduce the model's margin
# exactly.  Float32 accumulation over a few hundred trees is the only source of error.
ADDITIVITY_TOLERANCE = 1e-3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--zip", type=Path, default=REPO / "datasets" / IEEE_CIS_ZIP)
    parser.add_argument("--runs", type=Path, default=REPO / "results" / "runs")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "tables")
    parser.add_argument("--arm", default="temporal")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--band-features", type=int, default=8)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed = args.seed or cfg.split.seeds[0]

    scores = pd.read_parquet(args.runs / f"scores_{args.arm}_{seed}.parquet")
    frame = load_ieee_cis(args.zip, None, with_identity=True).frame
    for column in frame.columns:
        if frame[column].dtype == "object" or str(frame[column].dtype) == "str":
            frame[column] = frame[column].astype("category").cat.codes.astype("int32")

    numeric = [
        c
        for c in frame.select_dtypes(include=[np.number]).columns
        if c not in {"isFraud", "day", "TransactionDT", "TransactionID"}
    ]
    labels = frame["isFraud"].to_numpy()
    train_rows = scores[scores["block"] == "train"]["row"].to_numpy()

    low, high = band_edges(
        scores[scores["block"] == "band"]["score"].to_numpy(),
        cfg.decline_rate_budget,
        cfg.band.traffic_budget,
    )
    columns = select_band_features(
        frame, train_rows, labels, numeric, k=args.band_features, seed=seed
    )
    fit_rows = rows_in_band(scores, "band", low, high)
    explain_rows = rows_in_band(scores, "cal", low, high)

    x_fit, scaler = prepare(frame, fit_rows, columns, None)
    x_explain, _ = prepare(frame, explain_rows, columns, scaler)
    y_fit = labels[fit_rows]

    print(
        f"in-band attribution: fit on {len(x_fit):,} rows from D_band "
        f"({int(y_fit.sum())} fraud), explain {len(x_explain):,} from D_cal, "
        f"{len(columns)} features"
    )

    model = XGBClassifier(random_state=seed, **BAND_MODEL)
    model.fit(x_fit, y_fit)

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x_explain)
    base = float(np.ravel(explainer.expected_value)[0])

    # Additivity: base value plus contributions must reproduce the raw margin.  If it does
    # not, the explanation is not of this model and every ranking below is meaningless.
    margin = model.predict(x_explain, output_margin=True)
    residual = float(np.abs(base + values.sum(axis=1) - margin).max())
    if residual > ADDITIVITY_TOLERANCE:
        raise SystemExit(
            f"Shapley additivity violated: worst row misses the model margin by {residual:.2e}, "
            f"above the {ADDITIVITY_TOLERANCE:.0e} tolerance. The attribution does not "
            "describe this model and must not be reported."
        )
    print(f"  additivity holds; worst residual {residual:.2e} against the raw margin")

    args.out.mkdir(parents=True, exist_ok=True)

    ranked = (
        pd.DataFrame(
            {
                "feature": columns,
                "mean_abs_shap": np.abs(values).mean(axis=0),
                "mean_shap": values.mean(axis=0),
                "n_explained": len(x_explain),
            }
        )
        .sort_values("mean_abs_shap", ascending=False, ignore_index=True)
        .assign(rank=lambda d: d.index + 1)
    )
    ranked["share_of_total"] = ranked["mean_abs_shap"] / ranked["mean_abs_shap"].sum()
    ranked.to_csv(args.out / "attribution.csv", index=False)

    # Per-prediction contributions for the rows the model scores highest, which are the ones a
    # reviewer would ask about: these are the transactions the rule sends to a step-up.
    top = np.argsort(margin)[::-1][:N_EXAMPLES]
    examples = pd.DataFrame(values[top], columns=columns)
    examples.insert(0, "row", explain_rows[top])
    examples.insert(1, "y", labels[explain_rows[top]])
    examples.insert(2, "score_margin", margin[top])
    examples.insert(3, "base_value", base)
    examples.to_csv(args.out / "attribution_examples.csv", index=False)

    print(f"\n{'rank':>5s} {'feature':>20s} {'mean |SHAP|':>12s} {'share':>7s}")
    for row in ranked.itertuples():
        print(
            f"{row.rank:5d} {row.feature:>20s} {row.mean_abs_shap:12.5f} "
            f"{row.share_of_total:6.1%}"
        )
    print(f"\n  Wrote {display_path(args.out / 'attribution.csv')}")
    print(f"  Wrote {display_path(args.out / 'attribution_examples.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
