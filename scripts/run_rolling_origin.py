#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Re-run the coverage check at five rolling calibration origins over the frozen score file.

This is the measurement that narrowed the study's central temporal claim, recorded as
amendment-adjacent decision D-025: the deviation between realised and nominal false-decline
rate is **origin-dependent**, not a fixed temporal penalty.  It breaches the exact band at two
of five origins, reaches 1.41 times nominal at the worst, and reverses to 0.68 at the earliest.

``rolling_origin.csv`` backs Table 3 of the proposal and four bound claims, and no script in the
repository wrote it.  ``make reproduce`` carried it forward and ``freeze.py --check`` passed it
trivially, because a file nothing rewrites cannot differ from its own hash.

The procedure is the one D-025 states: a 20-day calibration window and a 20-day test window
stepped forward in 10-day increments over the frozen scores, with the model unchanged, judged
against the exact Beta-Binomial predictive interval at the 99 % level.  Only the split moves --
nothing is refitted, which is the point: five refits of one partition are not five samples of
the temporal deviation.

    .venv/bin/python scripts/run_rolling_origin.py

Writes ``results/tables/rolling_origin.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hsbcfraud.config import load_config
from hsbcfraud.conformal.coverage import coverage_band
from hsbcfraud.conformal.split import conformal_threshold
from hsbcfraud.paths import display_path

REPO = Path(__file__).resolve().parents[1]

# The sweep, exactly as D-025 defines it.  Windows are in days because the split is snapped to
# day boundaries; a row-count window would cut a day in half and put the same card on both
# sides of the calibration boundary.
CALIBRATION_DAYS = 20
TEST_DAYS = 20
STEP_DAYS = 10
FIRST_ORIGIN = 101
N_ORIGINS = 5

# The level the certificate is judged at here.  This is the unconditional alpha grid's loosest
# entry, not the band-conditional scale: the question is whether split-conformal coverage holds
# under a temporal split at all, which is prior to any band.
ALPHA = 1e-2

# The predictive interval the verdict is read against.  Two-sided, because the question is
# whether the realised count is consistent with the law, not whether it is safe.
BAND_LEVEL = 0.99


def origins() -> list[int]:
    return [FIRST_ORIGIN + step * STEP_DAYS for step in range(N_ORIGINS)]


def evaluate(scores: pd.DataFrame, cal_start: int) -> dict[str, object]:
    """One origin: calibrate on its window, read the next, judge against the exact law."""
    cal_end = cal_start + CALIBRATION_DAYS
    test_end = cal_end + TEST_DAYS

    legitimate = scores[scores["y"] == 0]
    calibration = legitimate[legitimate["day"].between(cal_start, cal_end - 1)]
    evaluation = legitimate[legitimate["day"].between(cal_end, test_end - 1)]

    threshold, order_index, n_calibration = conformal_threshold(
        calibration["score"].to_numpy(), ALPHA
    )
    m = len(evaluation)
    observed = int((evaluation["score"].to_numpy() >= threshold).sum())
    low, high = coverage_band(n=n_calibration, k=order_index, m=m, level=BAND_LEVEL)

    # "BREACHED" in capitals because `docs/claims.yaml` selects on that exact string to count
    # breaches, and the LaTeX table lowercases it for display. Emphasis in a data file is not
    # a habit worth spreading, but changing it here would move a bound claim for no gain a
    # reader sees.
    if observed < low:
        verdict = "conservative"
    elif observed > high:
        verdict = "BREACHED"
    else:
        verdict = "inside"

    return {
        "cal_start": cal_start,
        "ratio": (observed / m) / ALPHA if m else float("nan"),
        "verdict": verdict,
        "observed": observed,
        "band_low": low,
        "band_high": high,
    }


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

    frame = pd.DataFrame([evaluate(scores, start) for start in origins()])
    target = args.out / "rolling_origin.csv"
    frame.to_csv(target, index=False)

    breached = int((frame["verdict"] == "breached").sum())
    print(
        f"{N_ORIGINS} origins, {CALIBRATION_DAYS}-day calibration and {TEST_DAYS}-day test "
        f"windows stepped by {STEP_DAYS}, alpha {ALPHA:g}"
    )
    for row in frame.itertuples():
        print(
            f"  cal {row.cal_start:>3d}-{row.cal_start + CALIBRATION_DAYS - 1:<3d} "
            f"ratio {row.ratio:.3f}  observed {row.observed:>4d} "
            f"against [{row.band_low}, {row.band_high}]  {row.verdict}"
        )
    print(
        f"\n{breached} of {N_ORIGINS} breach; ratios span "
        f"{frame['ratio'].min():.3f} to {frame['ratio'].max():.3f}"
    )
    print(f"Wrote {display_path(target)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
