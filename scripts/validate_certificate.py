#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""H5: does the certified rule hold its risk on the held-out block?

The study's headline deliverable is a Learn-then-Test certificate on the band-conditional
false-decline rate.  Until now it was never checked on ``D_test``.  ``riskcontrol.csv`` reports
calibration-side quantities only, and ``coverage_by_arm.csv`` covers the *unconditional*
split-conformal threshold, which protocol section 1 states explicitly is not the certified
estimand.  So the one number the submission is built on had no held-out evidence behind it.

**The pre-registered test does not match the estimand, and this reports both.**

Protocol section 8 pre-registers H5 as an "exact Beta-Binomial tail probability".  That law is
the right one for split conformal, where the threshold is the ``k``-th order statistic of the
calibration scores and the coverage is therefore ``Beta(n+1-k, k)`` distributed, making the
test-error count Beta-Binomial.  Learn-then-Test does not select ``lambda`` that way: it tests
a null per grid point under family-wise error control and returns the admissible set.  The
guarantee is ``P(R(lambda_hat) > alpha) <= delta``, and conditional on ``lambda_hat`` -- which
depends only on ``D_cal`` -- the declines among in-band legitimate test rows are independent
Bernoulli draws.  The count is therefore **Binomial**, and under the certificate its rate is at
most ``alpha``, so ``P(Bin(m, alpha) >= observed)`` is a valid and conservative test.

Both are computed.  The binomial tail is the operative one; the Beta-Binomial is reported
because it was pre-registered and dropping it silently would hide that the pre-registration
specified a test for the wrong mechanism.  Amendment A5 records this.

    .venv/bin/python scripts/validate_certificate.py

Writes ``results/tables/h5_validation.csv``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from hsbcfraud.config import load_config
from hsbcfraud.conformal.coverage import tail_probability
from hsbcfraud.data.splits import TestFoldGuard

REPO = Path(__file__).resolve().parents[1]

# Pre-registered rejection level for H5 (protocol section 8).
REJECT_AT = 0.01


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
    certified = pd.read_csv(args.out / "riskcontrol.csv")
    certified = certified[certified["certified"].astype(bool)]
    if certified.empty:
        raise SystemExit("no certified configuration in riskcontrol.csv; nothing to validate")

    # The single-evaluation rule applies: this is a D_test read.  It authorises with the
    # SAME digest scripts/run_conformal.py uses, because H5 does not introduce a new
    # configuration -- it validates the one already certified.  Passing a different string
    # would make the guard refuse, and rightly: two distinct configurations touching the test
    # fold is exactly what it exists to stop.  Verified by it doing so when this script first
    # used its own label.
    config_digest = hashlib.sha256(
        json.dumps(cfg.model_dump(), sort_keys=True, default=str).encode()
    ).hexdigest()
    guard = TestFoldGuard(args.out / "test_access.json")
    guard.authorise("ieee_cis", config_digest)

    test = scores[scores["block"] == "test"]
    y_test = test["y"].to_numpy()
    s_test = test["score"].to_numpy()

    rows = []
    for row in certified.itertuples():
        # The band edges were frozen on D_band and are carried in the certificate row; the
        # rule is applied to D_test unchanged. Re-deriving them here would be a second look.
        in_band = (s_test >= row.band_lo) & (s_test < row.band_hi)
        legitimate = in_band & (y_test == 0)
        m = int(legitimate.sum())
        declined = int(((s_test >= row.selected_lambda) & legitimate).sum())
        realised = declined / m if m else float("nan")

        binomial_tail = float(stats.binom.sf(declined - 1, m, row.alpha)) if m else float("nan")
        # The pre-registered statistic, reported for the comparison. n and k are the
        # calibration size and order index the split-conformal law would use.
        n_cal = int(row.n_legit_band_cal)
        k = int(np.ceil((n_cal + 1) * (1 - row.alpha)))
        beta_binomial_tail = (
            float(tail_probability(n=n_cal, k=min(k, n_cal), m=m, observed=declined))
            if m and n_cal
            else float("nan")
        )

        rows.append(
            {
                "hypothesis": "H5",
                "band_budget": row.budget,
                "alpha": row.alpha,
                "alpha_fn": row.alpha_fn,
                "selected_lambda": row.selected_lambda,
                "n_legit_band_test": m,
                "declined": declined,
                "realised_risk": realised,
                "binomial_tail_p": binomial_tail,
                "beta_binomial_tail_p": beta_binomial_tail,
                "holds_at_alpha": bool(realised <= row.alpha) if m else False,
                "rejected_at_0p01": bool(binomial_tail < REJECT_AT) if m else False,
                "seed": seed,
                "arm": args.arm,
            }
        )

    frame = pd.DataFrame(rows)
    target = args.out / "h5_validation.csv"
    frame.to_csv(target, index=False)

    print(f"H5 on the {args.arm} arm, {len(frame)} certified configurations\n")
    print(
        f"{'budget':>7s} {'alpha':>6s} {'n':>7s} {'declined':>9s} {'realised':>9s} "
        f"{'binom p':>9s} {'betabin p':>10s}  verdict"
    )
    for row in frame.itertuples():
        verdict = "holds" if row.realised_risk <= row.alpha else "EXCEEDS alpha"
        if row.rejected_at_0p01:
            verdict += ", rejected at 0.01"
        print(
            f"{row.band_budget:7.3f} {row.alpha:6.2f} {row.n_legit_band_test:7d} "
            f"{row.declined:9d} {row.realised_risk:9.4f} {row.binomial_tail_p:9.4f} "
            f"{row.beta_binomial_tail_p:10.4f}  {verdict}"
        )
    held = int((frame["realised_risk"] <= frame["alpha"]).sum())
    print(f"\n  {held} of {len(frame)} certified configurations hold their risk on D_test.")
    print(f"  Wrote {target.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
