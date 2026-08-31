#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Write the per-transaction output the challenge statement asks for as its first outcome.

The statement's Expected Outcomes open with "fraud probability scores (float [0,1]) and binary
predictions for each transaction".  Both existed here and neither was reachable from the
submission: the scores lived in ``results/runs/scores_*.parquet``, which the portal does not
accept as a format, and the binary decision existed only as a count in the appendix.  A
requirement met in the repository and invisible in the upload is scored as unmet.

This writes one row per transaction in the held-out block, under the certified configuration at
the tightest level that certifies -- the probability, the three-valued decision, and the binary
decline that decision implies.  It computes nothing new: the score comes from the frozen scorer
output and the three thresholds come from the certificate, so the file cannot disagree with the
tables the proposal quotes.

    .venv/bin/python scripts/export_predictions.py

Writes ``results/tables/predictions.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hsbcfraud.config import load_config
from hsbcfraud.paths import display_path

REPO = Path(__file__).resolve().parents[1]

# The block the certificate is validated on, and the only one whose decisions are reportable:
# the other three build the rule.
REPORTED_BLOCK = "test"

# Column order, chosen so the two things the statement names come first and the provenance of
# the decision follows.  Named for a reader who has the CSV and not the code.
COLUMNS = (
    "transaction_row",
    "fraud_probability",
    "decision",
    "predicted_fraud",
    "day",
    "band_low",
    "band_high",
    "in_band_threshold",
)


def certified_configuration(riskcontrol: pd.DataFrame) -> pd.Series:
    """The certified row at the tightest alpha, which is the one the documents foreground.

    Ties on alpha are broken by the widest band budget, because that is the configuration with
    the largest calibration block behind it and therefore the one a reader should be shown.
    """
    certified = riskcontrol[riskcontrol["certified"].astype(bool)]
    if certified.empty:
        raise SystemExit("no certified configuration in riskcontrol.csv; nothing to export")
    ordered = certified.sort_values(["alpha", "budget"], ascending=[True, False])
    return ordered.iloc[0]


def decide(scores: pd.Series, low: float, high: float, threshold: float) -> pd.Series:
    """The three-valued rule, with the same boundaries the implementation uses.

    Half-open on the right: a score at the upper edge declines rather than abstaining, matching
    ``features.band.rows_in_band``.  Inside the band the rule declines at or above the certified
    threshold -- which, in the configuration certified here, is a third threshold on the same
    score rather than a second model.  See docs/guarantee.md.
    """
    decision = pd.Series("approve", index=scores.index, dtype="object")
    in_band = (scores >= low) & (scores < high)
    decision[in_band] = "step-up"
    decision[in_band & (scores >= threshold)] = "step-up-declined"
    decision[scores >= high] = "decline"
    return decision


def build(scores: pd.DataFrame, configuration: pd.Series) -> pd.DataFrame:
    block = scores[scores["block"] == REPORTED_BLOCK]
    decision = decide(
        block["score"],
        configuration["band_lo"],
        configuration["band_hi"],
        configuration["selected_lambda"],
    )
    # A step-up that the in-band threshold then declines is a decline for the purpose the
    # statement asks about, so both terminal-decline branches count toward the binary output.
    declined = decision.isin(["decline", "step-up-declined"]).astype(int)
    frame = pd.DataFrame(
        {
            "transaction_row": block["row"].to_numpy(),
            "fraud_probability": block["score"].to_numpy(),
            "decision": decision.to_numpy(),
            "predicted_fraud": declined.to_numpy(),
            "day": block["day"].to_numpy(),
            "band_low": configuration["band_lo"],
            "band_high": configuration["band_hi"],
            "in_band_threshold": configuration["selected_lambda"],
        }
    )
    return frame[list(COLUMNS)].sort_values("transaction_row", ignore_index=True)


def operating_point(frame: pd.DataFrame, configuration: pd.Series) -> pd.DataFrame:
    """What the rule does to a day of traffic, as shares of the held-out block.

    The proposal argues that a three-valued rule is what makes the problem tractable, and this
    is the sentence that argument needs: how much traffic each branch actually takes. It is an
    aggregate of the per-transaction file rather than a new measurement.
    """
    counts = frame["decision"].value_counts()
    total = len(frame)
    rows = [
        {
            "decision": name,
            "n": int(counts.get(name, 0)),
            "share": counts.get(name, 0) / total,
        }
        for name in ("approve", "step-up", "step-up-declined", "decline")
    ]
    rows.append({"decision": "declined overall", "n": int(frame["predicted_fraud"].sum()),
                 "share": float(frame["predicted_fraud"].mean())})
    rows.append({"decision": "stepped up overall", "n": int(counts.get("step-up", 0)
                                                            + counts.get("step-up-declined", 0)),
                 "share": float((counts.get("step-up", 0)
                                 + counts.get("step-up-declined", 0)) / total)})
    summary = pd.DataFrame(rows)
    summary["n_transactions"] = total
    summary["alpha"] = configuration["alpha"]
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--runs", type=Path, default=REPO / "results" / "runs")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "tables")
    parser.add_argument("--arm", default="temporal")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed = cfg.split.seeds[0]
    scores = pd.read_parquet(args.runs / f"scores_{args.arm}_{seed}.parquet")
    configuration = certified_configuration(pd.read_csv(args.out / "riskcontrol.csv"))

    frame = build(scores, configuration)
    summary = operating_point(frame, configuration)
    summary.to_csv(args.out / "operating_point.csv", index=False)
    target = args.out / "predictions.csv"
    # No rounding. A reader must be able to recompute `decision` from `fraud_probability` and
    # the three thresholds in the same row; six decimal places broke that at the boundary and
    # left the file disagreeing with its own columns.
    frame.to_csv(target, index=False)

    counts = frame["decision"].value_counts()
    print(
        f"{len(frame):,} transactions in the {REPORTED_BLOCK} block, "
        f"certified at alpha {configuration['alpha']:g}, band budget {configuration['budget']:g}"
    )
    for name in ("approve", "step-up", "step-up-declined", "decline"):
        share = counts.get(name, 0) / len(frame)
        print(f"  {name:<18s} {counts.get(name, 0):>8,}  {share:6.2%}")
    print(f"  declined overall   {frame['predicted_fraud'].sum():>8,}")
    print(f"\nWrote {display_path(target)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
