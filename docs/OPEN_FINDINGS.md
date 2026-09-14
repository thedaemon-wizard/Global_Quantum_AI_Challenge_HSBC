# Open findings

What a full audit of this repository turned up beyond what the decision log records, with the
current status of each. It exists because a submission that lists only what it repaired invites
the reader to assume the list is complete.

**Where it stands on 2026-09-11.** Of the ten findings opened here, **seven are closed**, each
with the tables it touches reproducing byte-identically; one will not be fixed and says why; one
is deferred; one is reproduced but unexplained. Two further findings were reproduced and then
**withdrawn**, and are kept at the foot so the same ground is not covered twice.

Every entry was reproduced before being written down. The closed ones keep their original
description rather than being deleted, because a list of defects that quietly loses its entries
as they are fixed is no more trustworthy than one that never had them.

**Status values.** *Closed* means fixed and verified. *Latent* means the defect cannot occur with
the shipped configuration and needs a config a reviewer would have to write by hand. *Deferred*
means it is real under the shipped configuration and was judged not worth the change this close
to the deadline, with the risk stated. *Will not fix* means acting on it would change results
rather than tidy code. *Needs confirmation* means the behaviour is reproduced but not explained.

---

## MEDIUM

### M1 -- CLOSED 2026-09-11: four implementations of the split-conformal order statistic

`conformal/split.py:96` is the validated one. `run_conformal.py:266` and
`validate_certificate.py:104` each re-derive it with `np.ceil` and `np.sort`.

**Closed.** All four now share one definition. `run_conformal.py` calls `conformal_threshold`;
`validate_certificate.py` calls a new `order_index(n, alpha)` beside it, because it holds only a
count from a committed table and not the scores. Its `min(k, n_cal)` clamp is gone -- it defeated
the guard that raises on `k > n`, returning the tail for a *different* order index under the
original one's name. `riskcontrol.csv`, `coverage.csv`, `envelope.csv` and `h5_validation.csv`
all reproduce byte-identically. The fourth copy, in `conformal/coverage.py`, was the one that
mattered and was fixed earlier: it accepted NaN calibration scores that `conformal_threshold` refuses, and certified a
corrupted block as clean coverage ([D-148](decisions.md)). The two remaining copies sit in
scripts that read score files this pipeline produces, so the NaN path is not reachable from
`make reproduce`. Consolidating them means touching the certificate producer, and the
certificate is the deliverable.

**Fix:** call `conformal_threshold(scores, alpha)` at both sites and use the returned
`(qhat, k, n)`. One line each, plus a check that `riskcontrol.csv` reproduces byte-identically.

### M2 -- CLOSED 2026-09-11: `run_conformal.py` kept a private `band_edges`

Four scripts import `hsbcfraud.features.band.band_edges`, whose docstring calls itself "the
single definition". `run_conformal.py:52-82` -- the script that writes `band_lo` and `band_hi`
into `riskcontrol.csv` -- has its own.

**Closed.** The private copy is deleted and its (better) documentation moved into the library,
which keeps the raise: a clamp that turns an impossible budget into a plausible band is the kind
of fallback this project removes rather than documents. `riskcontrol.csv`, `coverage.csv` and
`envelope.csv` reproduce byte-identically.

### M3 -- `select_model_columns` and `encode_strings` are copied across six scripts

`features/engineering.py` is imported by `run_baselines.py` and `run_ablations.py`. Six other
scripts inline the same selection, and the copies use `select_dtypes(include=[np.number])` where
the library uses `is_numeric_dtype` -- which differ on boolean columns. `encode_strings` exists
in six copies and in no library module.

**Two of the copies were real defects and both are fixed.** `tune_baseline.py` searched 400 raw
columns instead of the 439 the scorer ships with, so the tuning study answered its question about
a model nobody deploys ([D-147](decisions.md)); `measure_latency.py` priced the deployed
hyperparameters on the same wrong matrix ([D-153](decisions.md)). Both now import the pipeline.

**Status: will not fix in the remaining two, and the reason changed on 2026-09-11.** The
remaining copies are in `run_mps.py` and `run_seed_sweep.py`, which fit 431 columns where
`run_baselines.py` fits 439. That difference is **not** duplication drift to be tidied away -- it
is the feature asymmetry [D-155](decisions.md) measured at 0.0031 AP, one part in eighty of the
gap, and disclosed in proposal section 4. Changing the selection would refit the tensor network on
a different matrix, which is new science rather than a refactor.

**And it would discard a verification that cost twelve GPU-hours.** The full sweep was
regenerated from a fresh clone on an idle GPU and reproduces ROC AUC, average precision and final
loss to the last digit on all sixteen fits ([D-158](decisions.md)). That result is a statement
about *this* code. Editing the two scripts it exercised would invalidate it and require the
twelve hours again, to change a headline number by a quantity already measured as immaterial.

**If it is ever done**, the honest form is not a refactor but an experiment: refit the arm on the
matched 439 columns, report both, and say which is the pre-registered one.

### M4 -- CLOSED 2026-09-11: `reps` was read from the config at one call site of five

`screen_kernels.py:171` passes `reps=cfg.quantum.reps`. `report_circuits.py`, `plot_circuits.py`,
`measure_latency.py` and `check_parity.py` take `build_feature_map`'s default of 2.

**Closed, and it was worse than filed.** Neither `circuits.csv` nor `screens.csv` *recorded*
which repetition count produced them, so the drift would have been invisible in the artefacts as
well as in the call sites. All five sites now pass it, and `circuits.csv` carries a `reps`
column; every pre-existing column is unchanged.

### M5 -- CLOSED 2026-09-11: the `TestFoldGuard` configuration digest was hand-copied

`run_conformal.py:256` and `validate_certificate.py:80` each compute the digest that keys the
single-evaluation ledger. They must agree or `authorise` refuses.

**Closed.** `configuration_digest(cfg)` now sits beside `TestFoldGuard` in `data/splits.py` and
both scripts call it. The digest is unchanged -- `e9fc0d4c...`, the same key the committed ledger
holds, checked before and after -- and `riskcontrol.csv`, `coverage.csv`, `envelope.csv` and
`h5_validation.csv` all reproduce byte-identically.

### M6 -- CLOSED 2026-09-11: the latency table now prices the shipped feature matrix

`measure_latency.py` timed the deployed hyperparameters on 431 raw numeric columns against the
439 `run_baselines.py` fits. It now imports `BASE_COLUMNS`, `encode_strings`,
`add_entity_aggregates` and `select_model_columns` from the script that owns the model, and the
table was retaken on an idle host once the clean-room reproduction released the machine.

The scorer's 1-thread p50 moved 0.154 to 0.162 ms -- so the eight columns were worth about 5 %,
which is inside this measurement's own noise floor and was worth establishing rather than
assuming. `LatencyKernelVersusScorer` is 568x and `LatencyKernelBudgetShare` 66.8 %. See
[D-153](decisions.md).

---

## LOW

### L1 -- CLOSED 2026-09-11: `summarise_seed_sweep.py` divided by a zero spread

`ap_across_chi_spread` is 0.0 when the sweep has one bond dimension, and the division that
reports "seed noise is N times the capacity signal" is unguarded. The output file is written
*before* the division, so `make summaries` aborts on a table that is already correct.

**Closed.** Guarded, with the single-bond-dimension case reported rather than raised.

### L2 -- `run_conformal.py` writes no run log

It is one of four scripts with no progress destination. The existing gate keys on
`load_ieee_cis`, and this script reads parquet score files instead, so it is outside the check.

**Status: deferred, and the priority is lower than it first looked.** It runs in about a minute,
so it is not the silent-for-hours case `run_log` exists for. It is still the certificate
producer, and a durable log of the run that produces the deliverable is worth having.

**Fix:** wrap `main` in `run_log("conformal", directory=args.runs)`.

### L3 -- CLOSED 2026-09-11: `export_predictions.py` broke ties by row order

Ties on `(alpha, budget)` across `alpha_fn` are resolved by whichever row pandas returns first.
Exactly one `alpha_fn` certifies today, so nothing is ambiguous; a wider grid would silently
change `selected_lambda` and every exported decision.

**Closed.** `alpha_fn` is now the third sort key, ascending, so the tightest false-negative
constraint wins. `predictions.csv` reproduces byte-identically.

---

## Needs confirmation

### N1 -- Which configuration pays the OpenMP barrier is not reproducible

Three measurements of identical code, all-core p50 at batch 1:

| | 2026-08-30 | 2026-09-06 | 2026-09-11 |
|---|---|---|---|
| Classical scorer | 19.854 ms | 19.061 ms | **0.315 ms** |
| In-band re-scorer | 19.154 ms | **0.075 ms** | 19.051 ms |

The ~19 ms penalty appears in every run and lands on a **different component each time**. The
earlier framing of this entry -- that one component's figure had moved and needed explaining --
was wrong: the first run, where both paid it, was the coincidence. What decides whether XGBoost
parallelises a given payload has not been established here.

**Why it is recorded rather than resolved.** It reaches no result. All six bound latency claims
select `profile: 1-thread CPU`, which is the per-request serving shape and which reproduces to
within 5 % across the two runs that priced the right model. The all-core rows stay in the table
as the trap [D-063](decisions.md) wrote them to document, and no figure from them is quoted as a
quantity -- the docstring's "236x and 261x" is withdrawn for that reason.

---

## Outside this repository

### X1 -- The portfolio and section 8 name different Qiskit credentials

Section 8 links the portfolio, which is the proposal's only verification URL. Section 8 names the
IBM certification; the portfolio still lists only the Qiskit Advocate. A reviewer opening the
link finds a credential the proposal does not name, and vice versa.

**Author action.** Recorded in [VERIFICATION_CHECKLIST.md](VERIFICATION_CHECKLIST.md) section 5c
with the two other portfolio divergences. Nothing in this repository can fix it.

---

## Closed 2026-09-14 -- literature pass

Four items from a sweep of the 2026 conformal and tensor-network literature, all resolved.

* **CP-18's stated limitation was added, after the check that nearly removed it.** SCRC's closing
  paragraph concedes it gives no guarantee for rejected samples, which "should be handled by a
  downstream fallback mechanism such as human review" -- the region certified here. Two
  verification passes reported the sentence absent before a third found it: **it is in v2 and
  both passes had read v1**, the version our own entry was dated to. Quoted in section 1, entry
  pinned to v2, method recorded in [D-164](decisions.md).
* **The p-value's factor of e is a conservatism its own source recommends against.** Both risks
  are 0/1 losses and Bates et al. Remark 4 says the exact binomial "should always be used" there.
  Measured at exactly `e` on all five certified cells, where the p-values run 1e-34 to 1e-103
  against a Holm threshold near 0.005, so nothing flips. Not changed -- the pre-registration is
  frozen and the single `D_test` read is spent. [D-163](decisions.md).
* **Holm rather than LTT's recommended fixed-sequence procedure.** Two opposing risks on a
  two-dimensional grid, so no ordering can be committed in advance without prejudging which
  binds. Chosen for validity under arbitrary dependence, at a cost this data cannot show.
  [D-165](decisions.md).
* **Three citation errors corrected against primary records.** The no-free-lunch paper was
  attributed to "Yu, Xu and colleagues" -- it is Wu, Ye, Deng and Yu, naming the last author
  first and an "Xu" who is not among them. QM-15's "about three times faster" is 2.8x, computed
  here from runtimes the paper reports without stating any multiple, and 1.4x against its
  full-batch variant. FR-5's datasets are YelpChi, S-FFSD, FTFD and BankSim, not IEEE-CIS.

---

## Refuted

Two findings from the same audit were reproduced and then **withdrawn**, and are listed so the
same ground is not covered twice.

* **`validate_certificate.py` clamps an order index** with `min(k, n_cal)`, defeating a guard
  that raises on `k > n`. The mechanism is real. Reaching it needs `n_legit_band_cal < 19`, and
  the shipped budget grid gives 847, 1572, 2259 and 4831 -- a traffic budget forty times below
  the smallest shipped value, inside a field `check_protocol.py` hashes.
* **`run_power.py` hardcodes the cluster key** as `"card1"` while the config defines
  `entity_key`. Both shipped configurations set `entity_key = "card1"`, and the hardcoded column
  is identical to the parquet's `entity` column over all 590,540 rows, reproducing
  `n_clusters = 689` exactly. Latent coupling, not a defect in the shipped pipeline.
