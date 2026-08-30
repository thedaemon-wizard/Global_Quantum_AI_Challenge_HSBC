#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""A reviewer's walkthrough of the certificate, from band edges to held-out validation.

This traces the study's headline deliverable end to end against the committed tables, and it
recomputes rather than reprints: every figure it shows is derived here from
``results/tables/*.csv`` and asserted equal to what the submission claims. If a table changes
and a document does not, this fails.

It is a **repository artefact, not a submission artefact.** The portal accepts no ``.ipynb``
and all five slots are full, so this exists for a reviewer reading the repository. It is a
paired script rather than a notebook so that it can be linted, tested and diffed like the rest
of the code; ``make walkthrough`` runs it, and ``jupytext`` will open it as a notebook if that
is preferred.

Deliberately cheap: it reads tables and does arithmetic. Nothing is refitted, so it takes
about a second and needs no GPU. What it verifies is that the *reported* chain is internally
consistent, not that the models retrain -- ``make reproduce`` is what does that.

    .venv/bin/python notebooks/walkthrough.py
"""

# %%
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "results" / "tables"

pd.set_option("display.width", 100)
pd.set_option("display.max_columns", 20)

# %% [markdown]
# ## 1. The four blocks
#
# Contiguous, snapped to day boundaries, and disjoint. `D_band` is the only block permitted to
# set the band edges; `D_cal` certifies; `D_test` is read once.

# %%
splits = pd.read_csv(TABLES / "splits.csv")
temporal = splits[splits["arm"] == "temporal"].set_index("block")
print(temporal[["n_rows", "n_fraud", "n_legit", "day_first", "day_last"]])

total = int(temporal["n_rows"].sum())
print(f"\ntotal rows: {total:,}")
assert total == 590_540, "the four blocks must partition the dataset exactly"

# Blocks must not overlap in time, which is the whole point of the temporal arm.
ordered = temporal.loc[["train", "band", "cal", "test"]]
assert (ordered["day_first"].to_numpy()[1:] > ordered["day_last"].to_numpy()[:-1]).all(), (
    "blocks overlap in time; the temporal split is not a forward holdout"
)
print("blocks are contiguous and strictly ordered in time")

# %% [markdown]
# ## 2. Which configurations certify
#
# 48 pre-registered `(band budget, alpha, alpha_FN)` cells. A cell certifies when Learn-then-Test
# finds an admissible threshold under Holm correction across the grid.

# %%
risk = pd.read_csv(TABLES / "riskcontrol.csv")
certified = risk[risk["certified"].astype(bool)]
print(f"{len(certified)} of {len(risk)} configurations certify\n")
print(
    certified[
        ["budget", "alpha", "alpha_fn", "band_lo", "band_hi", "n_legit_band_cal", "selected_lambda"]
    ].to_string(index=False)
)

# %% [markdown]
# **The certificate must not be vacuous.** A selected threshold sitting at the upper band edge
# flags nothing, so its risk is structurally zero and the guarantee says nothing. An earlier
# version of this table was exactly that: 32 cells certified, every one at the edge.

# %%
margin = (certified["band_hi"] - certified["selected_lambda"]).min()
print(f"smallest margin below the upper band edge: {margin:.4f}")
assert margin > 0, "a threshold at the band edge certifies an empty flagged set"

# %% [markdown]
# **What limits the reach.** Nothing certifies at the tightest band budget. The mechanism is
# sample size acting through the concentration bound, not the class-conditional degeneracy
# floor -- the floor is `(1/alpha) - 1`, which is two orders of magnitude below the rows
# available and cannot bind anywhere on this grid.

# %%
tightest = risk[risk["budget"] == risk["budget"].min()]
rows_available = int(tightest["n_legit_band_cal"].min())
floor_at_five_pct = (1.0 / 0.05) - 1.0
print(f"tightest band budget: {tightest['budget'].iloc[0]}")
print(f"  certifies:            {int(tightest['certified'].astype(bool).sum())} of {len(tightest)}")
print(f"  in-band legit rows:   {rows_available}")
print(f"  degeneracy floor:     {floor_at_five_pct:.0f}")
assert floor_at_five_pct < rows_available, (
    "if the floor exceeded the rows available it would be the binding constraint"
)
print("  -> the floor cannot bind; the concentration bound is what does")

# %% [markdown]
# ## 3. Does the certificate hold on data it never saw?
#
# This is H5. The certified cells are applied to `D_test` unchanged -- same band edges, same
# lambda, no recalibration -- and the realised band-conditional false-decline rate is compared
# with the alpha each was certified at.

# %%
h5 = pd.read_csv(TABLES / "h5_validation.csv")
view = h5[["band_budget", "alpha", "n_legit_band_test", "declined", "realised_risk"]].copy()
view["risk / alpha"] = (h5["realised_risk"] / h5["alpha"]).round(3)
view["holds"] = h5["holds_at_alpha"]
print(view.to_string(index=False))

# The realised risk is recomputed from the counts rather than read, so a wrong risk column
# would surface here instead of propagating.
recomputed = h5["declined"] / h5["n_legit_band_test"]
assert (recomputed - h5["realised_risk"]).abs().max() < 1e-9, (
    "realised_risk does not equal declined / n_legit_band_test"
)

worst = (h5["realised_risk"] / h5["alpha"]).max()
print(f"\nall {len(h5)} hold; the tightest uses {worst:.3f} of its budget")
assert bool(h5["holds_at_alpha"].all()), "a certified configuration failed on held-out data"

# %% [markdown]
# **Read it for what it is.** Holding a ceiling of 0.10 to 0.25 is a real check that the
# machinery transfers, and a weak one, because those ceilings are loose. Amendment A1 predicted
# that from the band's sample size before the arm ran.

# %% [markdown]
# ## 4. Where the guarantee breaks: time
#
# Split-conformal coverage is judged against the exact Beta-Binomial law for the sample size,
# not against alpha. Checking `empirical <= alpha` is a one-sided test against the wrong null:
# on a correctly calibrated system it passes about half the time.

# %%
seeds = pd.read_csv(TABLES / "coverage_by_arm_seeds.csv")
summary = (
    seeds.groupby(["arm", "alpha"])
    .agg(mean_ratio=("ratio", "mean"), outside=("inside", lambda c: int((~c.astype(bool)).sum())))
    .reset_index()
)
print(summary.to_string(index=False))

# %% [markdown]
# The temporal arm falls outside on every seed at the two loosest levels; the stratified arm on
# none, at any level. The card-disjoint arm is **not clean, and the two levels fail
# differently** -- which is the distinction an earlier draft of the proposal got wrong.

# %%
card = seeds[(seeds["arm"] == "card_disjoint") & (~seeds["inside"].astype(bool))]
print(card[["seed", "alpha", "ratio", "conservative"]].to_string(index=False))

over = card[card["conservative"].astype(bool)]
under = card[~card["conservative"].astype(bool)]
print(f"\nover-covering (conservative, guarantee intact): {len(over)}")
print(f"under-covering (the direction that breaches):   {len(under)}")
assert len(under) == 1 and (under["ratio"] > 1).all(), (
    "the anti-conservative cell is what stops this arm being reported as a pass"
)
print("-> one seed at the tightest level runs the other way, so the arm is not a pass")

# %% [markdown]
# ## 5. And the size of the temporal breach depends on where you calibrate
#
# Re-running at five rolling calibration origins. The deviation is not a fixed factor, and the
# earliest origin reverses direction entirely -- which is why the study reports no single
# inflation number.

# %%
rolling = pd.read_csv(TABLES / "rolling_origin.csv")
print(rolling[["cal_start", "ratio", "verdict", "band_low", "band_high"]].to_string(index=False))
print(f"\nratio ranges {rolling['ratio'].min():.3f} to {rolling['ratio'].max():.3f}")
assert rolling["ratio"].min() < 1.0 < rolling["ratio"].max(), (
    "origins must straddle 1.0 for the direction-reversal claim to hold"
)

# %% [markdown]
# ## 6. The quantum arms, for completeness
#
# Both are negative and both are reported in the body. The kernel never reached the task: the
# a-priori screens rejected every configuration before it was run.

# %%
screens = pd.read_csv(TABLES / "screens.csv")
both = screens["passes_conditioning"].astype(bool) & screens["passes_distinctness"].astype(bool)
passed_both = int(both.sum())
print(f"{len(screens)} configurations screened")
print(f"  pass conditioning:  {int(screens['passes_conditioning'].astype(bool).sum())}")
print(f"  pass both:          {passed_both}")
print(f"  max RBF correlation: {screens['rbf_correlation'].max():.4f}")
assert passed_both == 0, "the kernel arm was stopped by its own pre-registered screens"

# %% [markdown]
# The tensor network ran. Every interval contains zero, and the comparison is underpowered
# against its own pre-registered ceiling -- so what survives is a non-superiority bound, not a
# null result and not evidence that the tensor network is worse.

# %%
mps = pd.read_csv(TABLES / "mps_h4.csv")
print(mps[["bond_dimension", "ap_difference", "ci_low", "ci_high"]].to_string(index=False))
assert ((mps["ci_low"] < 0) & (mps["ci_high"] > 0)).all(), "an interval excludes zero"
print(f"\nevery interval contains zero; upper bound on any gain: +{mps['ci_high'].max():.4f} AP")


# %%
def main() -> int:
    """Entry point so the walkthrough can be run as a script and wired into `make`."""
    print("\nWalkthrough complete: every assertion above holds against the committed tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
