# Reference to implementation

Which reference each piece of the implementation realises, where the document states it, which
table reports the outcome, and — the column that matters — whether the code does what the
reference says.

[`REFERENCES.md`](REFERENCES.md) records what is cited.
[`REFERENCE_CROSSCHECK.md`](REFERENCE_CROSSCHECK.md) records, automatically, which files reach
each entry. Neither answers the question a reviewer actually has, which is whether the
mathematics in the documents is the mathematics in the code.

That question is not rhetorical here. Checking it in this round found three formulas that the
documents printed and the code did not compute: the effective-rank definition, the RBF screen,
and the tensor-network contraction ([D-083](decisions.md)). It also found the documents
describing a two-model decision rule against a one-model implementation
([D-086](decisions.md)). All four are corrected; this file exists so the check is repeatable
rather than remembered.

**Coverage.** 18 of the 56 numbered entries reach an implementation file. The other 38 are cited
for context, for prior art, for regulation or for a software version, and are not implemented by
anything here — that is not a defect, and `REFERENCE_CROSSCHECK.md` already reports where each
is reached from.

**Two rows were deleted rather than corrected**, because `make_crosscheck.py` reaches an entry by
first-author surname and a surname is not unique. CP-10 was reached from `conformal/split.py`
only because Tibshirani appears in CP-4's author list, and FR-6 from `quantum/featuremaps.py`
only because Wang appears in QM-2's. Neither is implemented anywhere in this repository. A
crosscheck reporting an entry as reached is evidence of a string match, not of an
implementation, and this table is where that distinction has to be drawn by hand.

---

## 1. The five the guarantee rests on

These were verified line by line against the cited source, because a defect in any of them
would invalidate a headline claim rather than a sentence.

| Reference | Implements | Stated at | Reported in | Verified |
|---|---|---|---|---|
| **CP-7** Angelopoulos, Bates, Candès, Jordan & Lei, *AoAS* 19(2):1641–1662, 2025 — Learn-then-Test | [`conformal/riskcontrol.py`](../src/hsbcfraud/conformal/riskcontrol.py) `hoeffding_bentkus_p_value`, `learn_then_test` | README §4.2; proposal §2 | [`riskcontrol.csv`](../results/tables/riskcontrol.csv) | **Matches.** The code computes `min(exp(-n·h₁(R̂∧α, α)), e·P(Bin(n,α) ≤ ⌈nR̂⌉))`, which is the printed form term for term. It additionally clips at 1 and returns 1 when `R̂ ≥ α`; both are conservative and neither appears in the printed formula because neither changes the bound |
| **CP-1** Vovk, Gammerman & Shafer, *Algorithmic Learning in a Random World*, 2005 — exact coverage law | [`conformal/coverage.py`](../src/hsbcfraud/conformal/coverage.py) `beta_binomial_pmf`, `coverage_band` | README §4.3 | [`coverage.csv`](../results/tables/coverage.csv), [`coverage_by_arm.csv`](../results/tables/coverage_by_arm.csv) | **Matches.** Called as `beta_binomial_pmf(m, a=n+1−k, b=k)` at both call sites, which is `E ~ BetaBinomial(m, n+1−k, k)` as printed. Computed in log space because the binomial coefficient overflows at m ≈ 10⁵ |
| **CP-4** Ding, Angelopoulos, Bates, Jordan & Tibshirani, *NeurIPS* 2023 — class-conditional degeneracy | [`conformal/split.py`](../src/hsbcfraud/conformal/split.py) `degeneracy_floor`, `mondrian_thresholds` | README §4.4; [`guarantee.md`](guarantee.md) §5 | [`degeneracy.csv`](../results/tables/degeneracy.csv) | **Matches, after a correction.** The floor is `(1/α) − 1` exactly, not `⌈1/α⌉ − 1`. The bound is strict: at `n = floor` the quantile is finite. `mondrian_thresholds` read `n <= floor` and now reads `n < floor` ([D-084](decisions.md)) |
| **QM-12** Stoudenmire & Schwab, *NeurIPS* 29:4799, 2016 — MPS classification | [`quantum/mps.py`](../src/hsbcfraud/quantum/mps.py) `MPSClassifier` | proposal §4; [`RESULTS.md`](RESULTS.md) | [`mps_h4.csv`](../results/tables/mps_h4.csv), [`mps_seed_sweep.csv`](../results/tables/mps_seed_sweep.csv) | **Matches, after a correction.** The written contraction had every bond index contracted and so no free output index — a scalar, not a classifier. The model carries a (χ, 2) head that closes the final bond into two class logits, and both statements of the formula now show it ([D-083](decisions.md)) |
| **QM-1** Huang et al., *Nature Communications* 12:2631, 2021 — geometric difference | [`quantum/screens.py`](../src/hsbcfraud/quantum/screens.py) `geometric_difference`, `effective_rank_ratio` | proposal §4; [`RESULTS.md`](RESULTS.md) | [`screens.csv`](../results/tables/screens.csv) | **Computed, not gated** ([amendment A6](protocol.md)). Separately, the effective-rank definition the documents printed was the participation ratio while the code computes the exponential of the spectral entropy; the two agree only on a flat spectrum and differ by 31 % on a geometric one ([D-083](decisions.md)) |

## 2. The rest of the implemented set

Reached from an implementation file, checked for attribution rather than line by line.

| Reference | Implements | Reported in |
|---|---|---|
| **CP-2** Vovk, 2012 — conditional validity | [`conformal/coverage.py`](../src/hsbcfraud/conformal/coverage.py) | `coverage_by_arm.csv` |
| **CP-3** Vovk et al. — Mondrian confidence machine | [`conformal/split.py`](../src/hsbcfraud/conformal/split.py) `mondrian_thresholds` | `degeneracy.csv` |
| **CP-5** Barber, Candès, Ramdas & Tibshirani — beyond exchangeability | [`conformal/weighted.py`](../src/hsbcfraud/conformal/weighted.py) `geometric_weights` | not certified; the weighted arm is reported as not carrying a numeric penalty |
| **CP-9** Angelopoulos et al. — conformal risk control | [`conformal/riskcontrol.py`](../src/hsbcfraud/conformal/riskcontrol.py) | `riskcontrol.csv` |
| **QM-2** Thanasilp et al. — exponential concentration | [`quantum/featuremaps.py`](../src/hsbcfraud/quantum/featuremaps.py) | `screens.csv` |
| **QM-4**, **QM-5**, **QM-7** — kernel benchmarking and bandwidth | [`quantum/featuremaps.py`](../src/hsbcfraud/quantum/featuremaps.py) | `screens.csv` |
| **QM-6** Kakavand et al. — quantum kernels on fraud | [`quantum/screens.py`](../src/hsbcfraud/quantum/screens.py) | `screens.csv` |
| **QM-8** — hybrid mixture of experts | [`stats.py`](../src/hsbcfraud/stats.py) | `mps_h4.csv` |
| **QM-15** — tensor-network kernel machines | [`quantum/mps.py`](../src/hsbcfraud/quantum/mps.py) | `mps_full.csv` |
| **DS-3** Dal Pozzolo et al. — calibration under undersampling | [`metrics.py`](../src/hsbcfraud/metrics.py) | `baselines.csv` |
| **SW-7** SHAP 0.52.0 | [`scripts/run_explain.py`](../scripts/run_explain.py) | `attribution.csv`, `attribution_examples.csv` |

## 3. How to repeat this check

The mechanical half is generated:

```
.venv/bin/python scripts/make_crosscheck.py    # which files reach each entry
```

The judgement half is not, and cannot be. For each entry in section 1, open the cited source and
the named function together and read the formula in both. That is how the three mismatches in
[D-083](decisions.md) were found, and none of them would have been caught by any gate in this
repository: every one of those numbers agreed with its table, and the table agreed with the code.
The formula printed beside the number was what disagreed.

**要確認.** Section 2 is checked for attribution — that the entry is cited where it is
implemented — and not line by line against each paper. A reader who needs that assurance for a
specific entry should treat it as unverified here.
