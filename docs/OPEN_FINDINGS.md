# Open findings

What a full audit of this repository turned up and **did not fix**, with the reason in each case.
Everything confirmed and fixed is in [decisions.md](decisions.md); this file is the residue.

It exists because a submission that lists only what it repaired invites the reader to assume the
list is complete. Each row below was reproduced before being written down, and each says what
would change if it were acted on.

**Status values.** *Latent* means the defect cannot occur with the shipped configuration and
needs a config a reviewer would have to write by hand. *Deferred* means it is real under the
shipped configuration and was judged not worth the change this close to the deadline, with the
risk stated. *Needs confirmation* means the behaviour is reproduced but not explained.

---

## MEDIUM

### M1 -- Four implementations of the split-conformal order statistic

`conformal/split.py:96` is the validated one. `run_conformal.py:266` and
`validate_certificate.py:104` each re-derive it with `np.ceil` and `np.sort`.

**Status: deferred.** The fourth copy, in `conformal/coverage.py`, was the one that mattered and
is fixed: it accepted NaN calibration scores that `conformal_threshold` refuses, and certified a
corrupted block as clean coverage ([D-148](decisions.md)). The two remaining copies sit in
scripts that read score files this pipeline produces, so the NaN path is not reachable from
`make reproduce`. Consolidating them means touching the certificate producer, and the
certificate is the deliverable.

**Fix:** call `conformal_threshold(scores, alpha)` at both sites and use the returned
`(qhat, k, n)`. One line each, plus a check that `riskcontrol.csv` reproduces byte-identically.

### M2 -- `run_conformal.py` keeps a private `band_edges`

Four scripts import `hsbcfraud.features.band.band_edges`, whose docstring calls itself "the
single definition". `run_conformal.py:52-82` -- the script that writes `band_lo` and `band_hi`
into `riskcontrol.csv` -- has its own.

**Status: latent.** Both return `(0.900, 0.950)` on the shipped configuration. They diverge only
where the library raises and the copy clamps: with `decline_rate_budget = 0.05` and a budget of
`0.98`, the library refuses a quantile outside `[0, 1]` and the private copy silently returns
`(0.0, 0.950)`. No shipped configuration reaches it.

**Fix:** delete the private copy and import the library one. The clamp-versus-raise question then
gets decided once, in the place that documents itself as the single definition.

### M3 -- `select_model_columns` and `encode_strings` are copied across six scripts

`features/engineering.py` is imported by `run_baselines.py` and `run_ablations.py`. Six other
scripts inline the same selection, and the copies use `select_dtypes(include=[np.number])` where
the library uses `is_numeric_dtype` -- which differ on boolean columns. `encode_strings` exists
in six copies and in no library module.

**Status: deferred, and one instance of it was a real defect.** `tune_baseline.py` used a copy
and therefore searched 400 raw columns instead of the 439 the scorer ships with, so the tuning
study answered its question about a model nobody deploys ([D-147](decisions.md)). That one is
fixed by importing. The rest describe models whose committed tables already reproduce, so the
divergence is currently latent rather than active.

**Fix:** move `encode_strings` into `features/engineering.py` and have every script import both.
Then re-run `make reproduce` and confirm every science column is unchanged.

### M4 -- `reps` is read from the config at one call site of five

`screen_kernels.py:171` passes `reps=cfg.quantum.reps`. `report_circuits.py`, `plot_circuits.py`,
`measure_latency.py` and `check_parity.py` take `build_feature_map`'s default of 2.

**Status: latent.** `cfg.quantum.reps` is 2, so every site agrees today. A config setting `reps:
3` would make `screens.csv` describe three-repetition circuits while `circuits.csv` reports
two-repetition depths, silently -- and `report_circuits.py`'s claim that it "cannot drift from the
arm it describes" would become false.

**Fix:** pass `reps=cfg.quantum.reps` at all five sites, or drop the parameter's default so the
caller must choose.

### M5 -- The `TestFoldGuard` configuration digest is hand-copied

`run_conformal.py:256` and `validate_certificate.py:80` each compute the digest that keys the
single-evaluation ledger. They must agree or `authorise` refuses.

**Status: deferred.** They do agree. If they stopped agreeing the failure is loud and
self-describing -- "a second evaluation was requested for Y" -- rather than silent, which is why
this is not urgent.

**Fix:** a `configuration_digest(cfg)` helper beside `TestFoldGuard` in `data/splits.py`.

---

## LOW

### L1 -- `summarise_seed_sweep.py` divides by a zero spread

`ap_across_chi_spread` is 0.0 when the sweep has one bond dimension, and the division that
reports "seed noise is N times the capacity signal" is unguarded. The output file is written
*before* the division, so `make summaries` aborts on a table that is already correct.

**Status: latent.** Reachable only via `run_seed_sweep.py --bonds 4`; the shipped sweep has four.

**Fix:** guard on `across > 0` and print "single bond dimension, no capacity signal".

### L2 -- `run_conformal.py` writes no run log

It is one of four scripts with no progress destination. The existing gate keys on
`load_ieee_cis`, and this script reads parquet score files instead, so it is outside the check.

**Status: deferred, and the priority is lower than it first looked.** It runs in about a minute,
so it is not the silent-for-hours case `run_log` exists for. It is still the certificate
producer, and a durable log of the run that produces the deliverable is worth having.

**Fix:** wrap `main` in `run_log("conformal", directory=args.runs)`.

### L3 -- `export_predictions.py` breaks ties by row order

Ties on `(alpha, budget)` across `alpha_fn` are resolved by whichever row pandas returns first.
Exactly one `alpha_fn` certifies today, so nothing is ambiguous; a wider grid would silently
change `selected_lambda` and every exported decision.

**Fix:** make the tie-break explicit and record it in the protocol.

---

## Needs confirmation

### N1 -- The all-core latency figure moved two orders of magnitude between runs

The in-band re-scorer's all-core p50 was 19.154 ms on 2026-08-30 and 0.075 ms on 2026-09-06, on
**identical** model parameters (400 trees, depth 6, eight features). The classical scorer's
all-core figure stayed near 19 ms across both. What decides whether XGBoost parallelises a
payload this small has not been established.

**Why it is recorded rather than resolved.** No bound claim reads an all-core row -- every
latency claim selects the 1-thread profile, which is the per-request serving shape and is stable
to about a fifth across the two runs. The all-core rows are kept in the table as the documented
trap they were always meant to be ([D-063](decisions.md)), and the docstring now states the
instability instead of quoting one run's ratio as a property.

---

## Outside this repository

### X1 -- The portfolio and section 8 name different Qiskit credentials

Section 8 links the portfolio, which is the proposal's only verification URL. Section 8 names the
IBM certification; the portfolio still lists only the Qiskit Advocate. A reviewer opening the
link finds a credential the proposal does not name, and vice versa.

**Author action.** Recorded in [VERIFICATION_CHECKLIST.md](VERIFICATION_CHECKLIST.md) section 5c
with the two other portfolio divergences. Nothing in this repository can fix it.

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
