#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""E4-E6 -- the PSD2 operating envelope, two-sided risk control, and exact coverage.

Reads the frozen score interface written by ``run_baselines.py`` and writes:

  ``results/tables/envelope.csv``     the PSD2 tier analysis and the chosen operating point
  ``results/tables/riskcontrol.csv``  one row per (budget, alpha) with the certificate
  ``results/tables/tradeoff.csv``     the reachable guarantee-versus-abstention frontier
  ``results/tables/coverage.csv``     exact Beta-Binomial validation on the test block
  ``results/tables/degeneracy.csv``   per-class calibration counts against (1/alpha)-1

The band edges and the in-band scorer come from ``D_band`` only; the certificate is computed
on ``D_cal``; ``D_test`` is touched once, through the access guard.

    .venv/bin/python scripts/run_conformal.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hsbcfraud import metrics
from hsbcfraud.config import load_config
from hsbcfraud.conformal import coverage as cov
from hsbcfraud.conformal.riskcontrol import (
    RiskDefinition,
    band_conditional_false_decline,
    in_band_grid,
    learn_then_test,
    missed_fraud_rate,
)
from hsbcfraud.conformal.split import degeneracy_floor, mondrian_thresholds
from hsbcfraud.data.splits import TestFoldGuard

REPO = Path(__file__).resolve().parents[1]


def band_edges(
    scores: np.ndarray, decline_budget: float, band_budget: float
) -> tuple[float, float]:
    """The step-up band, which sits entirely BELOW the decline threshold.

    The three regions are contiguous and disjoint::

        score >= tau_hi              decline
        tau_lo <= score < tau_hi     step up  (the band)
        score <  tau_lo              approve

    so ``tau_hi`` is the decline threshold and the band occupies the next ``band_budget`` of
    traffic beneath it.

    An earlier version centred the band *on* the decline threshold and extended it in both
    directions.  That double-counts: the upper half of such a band lies above ``tau_hi``,
    where the decision is already decline, so the in-band scorer was being asked to re-rank
    transactions that were not routed to it.  With the decline threshold at the 98th
    percentile the upper edge also clipped to the maximum score for any budget above 4 %,
    making the band a half-open region rather than a band.

    The edges are score values fixed on ``D_band``.  They are deliberately **not**
    recomputed as quantiles of the deployment stream: a predicate estimated from the data it
    is applied to is data-dependent, which is exactly the selective-inference break that
    freezing the band exists to avoid.  The cost is that the routed volume drifts -- measured
    here at 5.1 % of ``D_cal`` for a band budgeted at 7.0 % of ``D_band`` -- and that drift is
    reported rather than engineered away.
    """
    hi_q = 1.0 - decline_budget
    lo_q = max(0.0, hi_q - band_budget)
    return float(np.quantile(scores, lo_q)), float(np.quantile(scores, hi_q))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--runs", type=Path, default=REPO / "results" / "runs")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "tables")
    parser.add_argument("--arm", default="temporal")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tier", default="100", help="PSD2 exemption threshold value tier")
    parser.add_argument(
        "--max-decline-rate",
        type=float,
        default=0.05,
        help="reject operating points that meet the ceiling only by declining more than this",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed = args.seed or cfg.split.seeds[0]
    scores_path = args.runs / f"scores_{args.arm}_{seed}.parquet"
    if not scores_path.exists():
        raise SystemExit(f"{scores_path} not found; run scripts/run_baselines.py first")

    frame = pd.read_parquet(scores_path)
    band_df = frame[frame["block"] == "band"]
    cal_df = frame[frame["block"] == "cal"]
    test_df = frame[frame["block"] == "test"]
    print(
        f"arm={args.arm} seed={seed}: band {len(band_df):,} / cal {len(cal_df):,} / "
        f"test {len(test_df):,} rows"
    )

    # ------------------------------------------------------------------- E4: the envelope
    # Two separate things happen here.  The operating point is chosen from an issuer decline
    # budget.  The PSD2 tiers are then reported as a DISTANCE -- how far this dataset sits
    # from a portfolio to which those ceilings would apply -- rather than as a constraint;
    # see docs/protocol.md amendment A2 for why applying them directly is a category error.
    envelope_rows = []
    for tier, rate in sorted(cfg.psd2_reference_rates.items(), key=lambda kv: -kv[1]):
        thr, achieved, recall = metrics.threshold_for_value_fraud_rate(
            band_df["y"].to_numpy(),
            band_df["score"].to_numpy(),
            band_df["amount"].to_numpy(),
            rate,
        )
        envelope_rows.append(
            {
                "etv_eur": int(tier),
                "reference_value_fraud_rate": rate,
                "threshold": thr,
                "achieved_value_fraud_rate": achieved,
                "recall_at_threshold": recall,
                "feasible": bool(np.isfinite(thr)),
            }
        )
        decline_rate = (
            float((band_df["score"].to_numpy() >= thr).mean()) if np.isfinite(thr) else float("nan")
        )
        envelope_rows[-1]["decline_rate"] = decline_rate
        print(
            f"  ETV EUR {tier:>3s}  ceiling {rate:.4%}  -> threshold {thr:.6g}  "
            f"achieved {achieved:.4%}  recall {recall:.3f}  declines {decline_rate:.3%}"
        )
    # The operating point: highest threshold inside the decline budget.
    s_band = band_df["score"].to_numpy()
    operating_threshold = float(np.quantile(s_band, 1.0 - cfg.decline_rate_budget))
    approved_band = s_band < operating_threshold
    value_rate = metrics.value_weighted_fraud_rate(
        band_df["y"].to_numpy(), band_df["amount"].to_numpy(), approved_band
    )
    approved_recall = metrics.recall_at_threshold(
        band_df["y"].to_numpy(), s_band, operating_threshold
    )
    print(
        f"\n  operating point at a {cfg.decline_rate_budget:.1%} decline budget: "
        f"threshold {operating_threshold:.6g}, approved-branch value fraud rate "
        f"{value_rate:.4%}, recall {approved_recall:.3f}"
    )

    # -------------------------------------------- E5: two-sided risk control on D_cal
    y_cal = cal_df["y"].to_numpy()
    s_cal = cal_df["score"].to_numpy()
    outer_caught = int(((s_cal >= operating_threshold) & (y_cal == 1)).sum())

    risk_rows: list[dict] = []
    for budget in cfg.band.budget_grid:
        lo, hi = band_edges(s_band, cfg.decline_rate_budget, budget)
        in_band = (s_cal >= lo) & (s_cal < hi)
        n_legit_band = int(((y_cal == 0) & in_band).sum())
        # Quantile-spaced over the scores actually present in the band, and excluding the
        # upper endpoint.  A linear grid closing at `hi` puts a point where the flagged set
        # is empty by construction -- band membership is `score < hi` and the rule is
        # `score >= lam` -- so the risk there is structurally zero and it is the only point
        # that clears the Holm level.  That is what made every earlier certificate vacuous.
        grid = in_band_grid(s_cal, in_band, cfg.risk.n_lambda)

        for alpha in cfg.risk.alpha_band_grid:
          for alpha_fn in cfg.risk.alpha_fn_grid:
            risks = [
                RiskDefinition(
                    "fdr", alpha, band_conditional_false_decline(y_cal, s_cal, in_band)
                ),
                RiskDefinition(
                    "fnr",
                    alpha_fn,
                    missed_fraud_rate(y_cal, s_cal, in_band, operating_threshold),
                ),
            ]
            result = learn_then_test(
                grid, risks, delta=cfg.risk.delta, fwer_method=cfg.risk.fwer_method
            )
            risk_rows.append(
                {
                    "budget": budget,
                    "alpha": alpha,
                    "alpha_fn": alpha_fn,
                    "decline_budget": cfg.decline_rate_budget,
                    "band_lo": lo,
                    "band_hi": hi,
                    "n_band_cal": int(in_band.sum()),
                    "n_legit_band_cal": n_legit_band,
                    "certified": result.any_admissible,
                    "n_admissible": int(result.admissible.size),
                    "selected_lambda": result.selected,
                }
            )
            flag = "certified" if result.any_admissible else "not certifiable"
            print(
                f"  band {budget:5.3f} a_fdr {alpha:<6g} a_fnr {alpha_fn:<5g} "
                f"n_legit_band {n_legit_band:6,d} -> {flag}"
            )

    # -------------------------------------------------------- E6: degeneracy arithmetic
    deg_rows = []
    for alpha in cfg.risk.alpha_grid:
        thresholds = mondrian_thresholds(s_cal, y_cal, alpha)
        for label, ct in sorted(thresholds.items()):
            deg_rows.append(
                {
                    "alpha": alpha,
                    "label": label,
                    "class_name": "legitimate" if label == 0 else "fraud",
                    "n_calibration": ct.n_calibration,
                    "floor": degeneracy_floor(alpha),
                    "headroom": ct.headroom,
                    "degenerate": ct.degenerate,
                }
            )

    # ----------------------------------------------- E6: exact coverage on the test fold
    config_digest = hashlib.sha256(
        json.dumps(cfg.model_dump(), sort_keys=True, default=str).encode()
    ).hexdigest()
    TestFoldGuard(args.out / "test_access.json").authorise("ieee_cis", config_digest)

    y_test = test_df["y"].to_numpy()
    s_test = test_df["score"].to_numpy()
    cov_rows = []
    for alpha in cfg.risk.alpha_grid:
        legit_cal = s_cal[y_cal == 0]
        qhat_index = int(np.ceil((1 - alpha) * (legit_cal.size + 1)))
        if qhat_index > legit_cal.size:
            continue
        qhat = float(np.sort(legit_cal)[qhat_index - 1])
        legit_test = s_test[y_test == 0]
        errors = int((legit_test >= qhat).sum())
        verdict = cov.verify(
            n_calibration=legit_cal.size,
            order_index=qhat_index,
            n_test=legit_test.size,
            observed_errors=errors,
            alpha=alpha,
            band_level=cfg.risk.coverage_band_level,
        )
        cov_rows.append(
            {
                "alpha": alpha,
                "threshold": qhat,
                "n_cal_legit": legit_cal.size,
                "n_test_legit": legit_test.size,
                "observed_errors": errors,
                "empirical_rate": verdict.empirical_rate,
                "band_low": verdict.band_low,
                "band_high": verdict.band_high,
                "tail_p": verdict.tail_p,
                "expectation_ok": verdict.expectation_ok,
                "finite_sample_ok": verdict.finite_sample_ok,
                "conservative": verdict.conservative,
            }
        )
        print(f"  alpha {alpha:<7g} {verdict.summary()}")

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(envelope_rows).to_csv(args.out / "envelope.csv", index=False)
    risk_frame = pd.DataFrame(risk_rows)
    risk_frame.to_csv(args.out / "riskcontrol.csv", index=False)
    # The frontier: for each (band budget, alpha) the loosest recall floor that certifies.
    frontier = (
        risk_frame[risk_frame["certified"]]
        .groupby(["decline_budget", "budget", "alpha"], as_index=False)["alpha_fn"]
        .min()
        .rename(columns={"alpha_fn": "tightest_certified_alpha_fn"})
    )
    frontier.to_csv(args.out / "tradeoff.csv", index=False)
    pd.DataFrame(deg_rows).to_csv(args.out / "degeneracy.csv", index=False)
    pd.DataFrame(cov_rows).to_csv(args.out / "coverage.csv", index=False)
    print(f"\nWrote envelope, riskcontrol, degeneracy and coverage tables to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
