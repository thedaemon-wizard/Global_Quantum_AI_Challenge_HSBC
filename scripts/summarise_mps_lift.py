#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Compare the tensor network to the baseline in a way the base rate does not confound.

Section 4 reported the arm as a percentage of the baseline's average precision at two scales:
about 88 % on the eight-feature in-band problem and about 49 % at full scale, and concluded
that the deficit widens with dimension.

**That conclusion was an artefact of the evaluation sets.**  Average precision floors at the
positive rate -- a classifier that ranks at random scores the base rate, not zero -- and the two
sets do not share one.  The in-band evaluation block is roughly 11 % positive because the band
is where the scorer already concentrates fraud; the held-out block is roughly 3 %.  So the
in-band ratio starts from a much higher floor and flatters whatever sits on it.

Dividing lift by lift removes the floor from both sides, and the two settings then agree to
within a point.  ROC AUC, whose floor is a fixed 0.5 and needs no correction, is reported
alongside as an independent check; it moves in the *opposite* direction to the raw shares,
which is the clearest evidence that the raw comparison was measuring the base rate.

Nothing is fitted here.  Every value is arithmetic over committed tables, so this runs in
milliseconds and cannot disagree with the tables the proposal quotes.

    .venv/bin/python scripts/summarise_mps_lift.py

Writes ``results/tables/mps_lift.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hsbcfraud.paths import display_path

REPO = Path(__file__).resolve().parents[1]

REPORTED_ARM = "temporal"

# ROC AUC's floor is a property of the metric, not of the data: a random ranking scores 0.5
# whatever the base rate.  Naming it here rather than inlining 0.5 is what lets the two rows
# below share one code path.
RANDOM_RANKING_AUC = 0.5


def lift_share(model: float, baseline: float, floor: float) -> float:
    """The model's share of the baseline's improvement over what random ranking scores.

    Both numerator and denominator are measured from the same floor, so the result is
    comparable across evaluation sets with different positive rates -- which the raw ratio
    ``model / baseline`` is not.
    """
    return (model - floor) / (baseline - floor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, default=REPO / "results" / "tables")
    args = parser.parse_args(argv)

    band = pd.read_csv(args.tables / "mps_band.csv")
    sweep = pd.read_csv(args.tables / "mps_seed_sweep.csv")
    baselines = pd.read_csv(args.tables / "baselines.csv")
    splits = pd.read_csv(args.tables / "splits.csv")
    riskcontrol = pd.read_csv(args.tables / "riskcontrol.csv")

    # The in-band positive rate is not in any table as a column; it is implied by the two
    # counts riskcontrol.csv records for the configuration run_mps.py evaluated on.
    n_band = int(band["n_eval"].iloc[0])
    row = riskcontrol[riskcontrol["n_band_cal"] == n_band]
    if row.empty:
        raise SystemExit(
            f"no riskcontrol.csv configuration has n_band_cal == {n_band}; the in-band "
            f"positive rate cannot be derived, so the floor correction would be guesswork"
        )
    band_positive_rate = float(
        (row["n_band_cal"].iloc[0] - row["n_legit_band_cal"].iloc[0]) / n_band
    )

    test = baselines[(baselines["arm"] == REPORTED_ARM) & (baselines["block"] == "test")]
    full_positive_rate = float(
        splits[(splits["arm"] == REPORTED_ARM) & (splits["block"] == "test")]
        ["count_fraud_rate"].iloc[0]
    )

    rows = [
        {
            "setting": "band",
            "n_features": int(band["n_features"].iloc[0]),
            "n_eval": n_band,
            "positive_rate": band_positive_rate,
            "mps_average_precision": float(band[band["model"] == "mps"]["average_precision"].max()),
            "baseline_average_precision": float(
                band[band["model"] == "xgboost"]["average_precision"].iloc[0]
            ),
            "mps_roc_auc": float(band[band["model"] == "mps"]["roc_auc"].max()),
            "baseline_roc_auc": float(band[band["model"] == "xgboost"]["roc_auc"].iloc[0]),
        },
        {
            "setting": "full",
            "n_features": int(sweep["n_features"].iloc[0]),
            "n_eval": int(sweep["n_eval"].iloc[0]),
            "positive_rate": full_positive_rate,
            "mps_average_precision": float(sweep["average_precision"].max()),
            "baseline_average_precision": float(test["average_precision"].max()),
            "mps_roc_auc": float(sweep["roc_auc"].max()),
            "baseline_roc_auc": float(test["roc_auc"].max()),
        },
    ]

    frame = pd.DataFrame(rows)
    frame["raw_ap_share"] = (
        100 * frame["mps_average_precision"] / frame["baseline_average_precision"]
    )
    frame["ap_lift_share"] = 100 * frame.apply(
        lambda r: lift_share(
            r["mps_average_precision"], r["baseline_average_precision"], r["positive_rate"]
        ),
        axis=1,
    )
    frame["auc_lift_share"] = 100 * frame.apply(
        lambda r: lift_share(r["mps_roc_auc"], r["baseline_roc_auc"], RANDOM_RANKING_AUC),
        axis=1,
    )

    args.tables.mkdir(parents=True, exist_ok=True)
    target = args.tables / "mps_lift.csv"
    frame.to_csv(target, index=False)

    for r in frame.itertuples():
        print(
            f"  {r.setting:<5s} {r.n_features:>3d} features, "
            f"{100 * r.positive_rate:5.2f} % positive  "
            f"raw AP share {r.raw_ap_share:5.1f} %  ->  lift share {r.ap_lift_share:5.1f} %  "
            f"(AUC lift share {r.auc_lift_share:5.1f} %)"
        )
    print(f"\nWrote {display_path(target)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
