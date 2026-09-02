#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Write the two by-arm coverage tables, which were committed and which nothing produced.

``coverage_by_arm.csv`` and ``coverage_by_arm_seeds.csv`` carry the contrast the three-arm
design exists to produce: the temporal split breaches its coverage interval at the loose
levels, the stratified split does not, and the card-disjoint split sits between them.  The
results section leads on that contrast and `docs/claims.yaml` binds four claims to it.

**No script wrote either table.**  They were committed by hand, which meant
``scripts/freeze.py --check`` passed them trivially -- a file nothing rewrites can never differ
from its own hash -- and ``make reproduce`` could not regenerate them.
``tests/test_repo_hygiene.py`` recorded the gap as a strict xfail rather than hiding it, and
this script is what clears it.

Nothing new is measured.  Every row re-derives from the per-seed score files that
``run_baselines.py`` already writes, through the same
:func:`hsbcfraud.conformal.coverage.split_conformal_coverage` the certificate itself uses.
Sharing that function is the point: a producer that re-implemented the order statistic could
drift from the one under test, and then the by-arm table would no longer be the same procedure
applied to a different split, which is the only thing that makes the comparison mean anything.

    .venv/bin/python scripts/run_coverage_arms.py

Writes ``results/tables/coverage_by_arm.csv`` and
``results/tables/coverage_by_arm_seeds.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hsbcfraud.config import load_config
from hsbcfraud.conformal.coverage import split_conformal_coverage
from hsbcfraud.paths import display_path, require_run_artefact

REPO = Path(__file__).resolve().parents[1]

# The arm the study reports first, then its two controls.  Taken from the config rather than
# typed so this cannot drift from `run_baselines.py`, which produces the score files below.
REPORTED_ARM = "temporal"

# `coverage_by_arm.csv` reports one seed per arm.  It is the first configured seed, not the
# best one: D-020 retracted a maximum-across-seeds figure that had been quoted as a centre,
# and `summarise_coverage.py` exists to report the five-seed distribution the single split
# oversold.  Anything other than "the first seed" would reintroduce a selection.
HEADLINE_SEED_INDEX = 0

BY_ARM_COLUMNS = (
    "arm", "alpha", "nominal", "empirical", "ratio", "band_low", "band_high",
    "observed_errors", "n_test_legit", "tail_p", "finite_sample_ok", "conservative",
)
BY_SEED_COLUMNS = (
    "arm", "seed", "alpha", "ratio", "empirical", "inside", "conservative", "tail_p",
)


def coverage_for(scores: pd.DataFrame, alpha_grid, band_level: float) -> list[dict]:
    """Coverage verdicts for one (arm, seed) score file, calibrating on cal and reading test."""
    cal = scores[scores["block"] == "cal"]
    test = scores[scores["block"] == "test"]
    return split_conformal_coverage(
        cal["score"].to_numpy(),
        cal["y"].to_numpy(),
        test["score"].to_numpy(),
        test["y"].to_numpy(),
        alpha_grid=alpha_grid,
        band_level=band_level,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--runs", type=Path, default=REPO / "results" / "runs")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "tables")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    arms = [REPORTED_ARM, *cfg.split.control_arms]
    seeds = cfg.split.seeds
    headline_seed = seeds[HEADLINE_SEED_INDEX]

    by_arm: list[dict] = []
    by_seed: list[dict] = []
    for arm in arms:
        for seed in seeds:
            scores = pd.read_parquet(
                require_run_artefact(
                    args.runs / f"scores_{arm}_{seed}.parquet", produced_by="baseline"
                )
            )
            for row in coverage_for(scores, cfg.risk.alpha_grid, cfg.risk.coverage_band_level):
                inside = row["band_low"] <= row["observed_errors"] <= row["band_high"]
                by_seed.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "alpha": row["alpha"],
                        "ratio": row["empirical_rate"] / row["alpha"],
                        "empirical": row["empirical_rate"],
                        "inside": bool(inside),
                        "conservative": row["conservative"],
                        "tail_p": row["tail_p"],
                    }
                )
                if seed == headline_seed:
                    by_arm.append(
                        {
                            "arm": arm,
                            "alpha": row["alpha"],
                            "nominal": row["alpha"],
                            "empirical": row["empirical_rate"],
                            "ratio": row["empirical_rate"] / row["alpha"],
                            "band_low": row["band_low"],
                            "band_high": row["band_high"],
                            "observed_errors": row["observed_errors"],
                            "n_test_legit": row["n_test_legit"],
                            "tail_p": row["tail_p"],
                            "finite_sample_ok": row["finite_sample_ok"],
                            "conservative": row["conservative"],
                        }
                    )

    args.out.mkdir(parents=True, exist_ok=True)
    arm_frame = pd.DataFrame(by_arm)[list(BY_ARM_COLUMNS)]
    seed_frame = pd.DataFrame(by_seed)[list(BY_SEED_COLUMNS)]
    arm_frame.to_csv(args.out / "coverage_by_arm.csv", index=False)
    seed_frame.to_csv(args.out / "coverage_by_arm_seeds.csv", index=False)

    for arm in arms:
        rows = seed_frame[seed_frame["arm"] == arm]
        print(f"  {arm:<14s} {int((~rows['inside']).sum())} of {len(rows)} outside the interval")
    print(
        f"\nWrote {display_path(args.out / 'coverage_by_arm.csv')} "
        f"({len(arm_frame)} rows, seed {headline_seed}) and "
        f"{display_path(args.out / 'coverage_by_arm_seeds.csv')} ({len(seed_frame)} rows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
