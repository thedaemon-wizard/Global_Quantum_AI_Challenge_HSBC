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
alongside as a second reading on the same fit; it moves in the *opposite* direction to the raw
shares, which is the clearest evidence that the raw comparison was measuring the base rate.

**Both arms are represented by their best fit, and that favours the quantum arm.**  The tensor
network is selected over sixteen fits and the baseline over five seeds, so the selection is
wider on the side this study reports as *losing*.  That asymmetry is deliberate and is stated
rather than corrected: the conclusion is that the arm loses, and it is worth more if it loses
at its own best.  Averaging instead would lower the tensor network's full-scale average
precision from 0.2512 to about 0.1997 and widen the deficit.

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


# The metric that selects the representative fit.  Average precision, because it is the metric
# the comparison is *about*; ROC AUC then describes the same model rather than a different one.
SELECTION_METRIC = "average_precision"


def best_fit(frame: pd.DataFrame) -> pd.Series:
    """The single best fit by :data:`SELECTION_METRIC`, with every metric read from that one row.

    This used to take an independent maximum per column, which is wrong in a way that is easy to
    miss: ``frame["average_precision"].max()`` and ``frame["roc_auc"].max()`` need not come from
    the same fit, and here they do not.  In ``mps_band.csv`` the best average precision is at bond
    dimension 32 and the best ROC AUC at bond dimension 4; in ``mps_seed_sweep.csv`` they are at
    two different seeds.  The published row therefore paired one model's average precision with a
    *different* model's ROC AUC, while the module docstring offered the AUC column as an
    independent check on the AP column -- a check that only means something if both describe the
    same classifier.

    Selecting one row costs the AUC column its status as a free-standing best case, which is the
    honest trade: it is now the same-model companion to the AP figure, and that is what makes the
    two comparable at all.
    """
    return frame.loc[frame[SELECTION_METRIC].idxmax()]


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

    mps_band_best = best_fit(band[band["model"] == "mps"])
    mps_full_best = best_fit(sweep)
    baseline_full_best = best_fit(test)
    baseline_band = band[band["model"] == "xgboost"].iloc[0]

    rows = [
        {
            "setting": "band",
            "n_features": int(band["n_features"].iloc[0]),
            "n_eval": n_band,
            "positive_rate": band_positive_rate,
            "mps_average_precision": float(mps_band_best["average_precision"]),
            "baseline_average_precision": float(baseline_band["average_precision"]),
            "mps_roc_auc": float(mps_band_best["roc_auc"]),
            "baseline_roc_auc": float(baseline_band["roc_auc"]),
        },
        {
            "setting": "full",
            "n_features": int(sweep["n_features"].iloc[0]),
            "n_eval": int(sweep["n_eval"].iloc[0]),
            "positive_rate": full_positive_rate,
            "mps_average_precision": float(mps_full_best["average_precision"]),
            "baseline_average_precision": float(baseline_full_best["average_precision"]),
            "mps_roc_auc": float(mps_full_best["roc_auc"]),
            "baseline_roc_auc": float(baseline_full_best["roc_auc"]),
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
