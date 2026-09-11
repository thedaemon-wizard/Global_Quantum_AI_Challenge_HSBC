# Reproducing from an empty directory

What a third party has to do to get from a clone to a verified certificate, what each step was
measured to cost, and — the part that matters more — **which steps were actually exercised and
which were not**.

Executed 2026-08-30 and again on 2026-08-31 on the machine in
[ENVIRONMENT.md](ENVIRONMENT.md), from a copy of the working tree with `.venv`, `.git`,
`datasets/` and every cache excluded, so nothing from the development environment could leak
into it. Section 2b covers the second run, which exists because four producers were written
after the first.

---

## 1. The procedure

```bash
git clone <repository> hsbcfraud && cd hsbcfraud

# The dataset is never committed. Place ieee-fraud-detection.zip in datasets/ yourself;
# the Kaggle terms have to be accepted by the person downloading it, not by this repository.
mkdir -p datasets && cp /path/to/ieee-fraud-detection.zip datasets/

make venv         # python3.12, torch cu130 first, then the project and its dev extras
make venv-gpu     # OPTIONAL, and see below: without it two parity claims cannot hold
make smoke        # S0-S8 environment assertions; any failure stops the build
make walkthrough  # trace the certificate against the committed tables (seconds, no GPU)
make reproduce    # refit everything except the 13-GPU-hour sweep, then write the manifest
make check        # rebuild both PDFs, then the claim, citation, protocol and manifest gates
```

**Two steps may report that they skipped, and the run is still correct.** Both were found by
running this procedure on 2026-09-05, when it failed on nine claims; both now say what they did
instead of quietly producing a worse answer.

**Latency is not re-measured unless you ask.** `latency.csv` is a record of *one machine*, and
six bound claims quote it. Any other CPU produces different numbers, so re-measuring during a
reproduction would fail those six for every reviewer -- which is exactly what happened to the
third pass, where a loaded host produced timings two to four times the committed ones and took
the kernel's share of the authorisation budget from 75.9 % to 177.8 %, past the point where the
conclusion inverts. (Both figures are as they stood then. The committed baseline has since been
re-measured to **61.3 %** on a quiet host, after [D-147](decisions.md) corrected the model the
scorer rows priced; the point the episode makes is unchanged and the ratio between a quiet and a
loaded reading is if anything larger.)

`measure_latency.py` therefore preserves the committed table by default and says so. Pass
`--measure` to take your own reading; it will still refuse above 0.25 load per core, because a
timing taken while the machine is busy describes the other work rather than the model. Section 5
of the proposal names the workstation the committed figures belong to, so they are a scoped
measurement rather than a portable one.

**Parity is skipped without the optional extra.** `qiskit-aer` and its GPU wheel live in
`gpu-crosscheck`, so a default environment cross-checks the fidelity kernel against Braket
alone: 4 comparisons against the committed 12. Those four still run and still have to agree.
`check_parity.py` prints the reason and **does not overwrite the wider committed table**. The
extra is optional by choice --- it pulls `cuquantum-cu11` under NVIDIA's proprietary licence
([NOTICE](../NOTICE) section 4) and no scientific figure depends on it --- so install it to
regenerate those rows, or read the skip message and proceed.

**Why skipping rather than failing.** The first version of the load guard exited non-zero, which
aborted `make reproduce` partway. That is worse for a reviewer than the wrong numbers it was
written to prevent: they get no run at all, instead of a run with one table they can be told to
distrust. Both guards now report and continue, `make check` passes, and the two messages say
exactly what was not re-verified. A green build that names what it skipped is more useful than a
red one nobody can interpret ([D-135](decisions.md)).

`make venv` is the step with a real cost: torch from the CUDA 13.0 index is a large download.
Everything after it is fast except `make reproduce`.

## 2. What this run established

| Step | Result |
|---|---|
| `python3.12 -m venv` from the system interpreter | Python 3.12.11 |
| `pip install torch==2.13.0` from the cu130 index | installed; `torch.cuda.is_available()` **True** |
| `pip install -e ".[dev]"` | clean, no resolution conflicts |
| `make smoke` | **9 of 9 accounted for**, 6 pass and 3 skip in the fresh environment: S1 and S5 both need `qiskit-aer`, which lives in the `gpu-crosscheck` extra, and S7 needs ULB; the committed environment reports 8 and 1 |
| `make walkthrough` | every assertion holds against the committed tables |

The three skips are declared, not silent:

* **S1, Aer GPU on sm_120** — the same missing `qiskit-aer`. It additionally needs a GPU device
  and `nvidia-smi`, so it is the one of the three that passes on the committed environment.
* **S5, Aer against Qiskit 2.5** — `qiskit-aer` is in the optional `gpu-crosscheck` extra, which
  `make venv` deliberately does not install because it pulls NVIDIA's proprietary cuStateVec
  binary (see [NOTICE](../NOTICE)). Only the E11 parity cross-check needs it, and no scientific
  figure depends on that path. Install it with `make venv-gpu`.
* **S7, ULB temporal split feasibility** — the ULB dataset is not present and **nothing in this
  repository fetches it**. That is also why E15 was not run; see
  [PROVENANCE.md](PROVENANCE.md) §1.2.

**This is the honest scope of the check.** It establishes that the environment builds from
nothing on a machine with this GPU, that the committed source imports and runs, and that the
reported certificate chain is internally consistent. It does **not**, by itself, establish that
the models refit to the same numbers — that is `make reproduce`, whose cost is in
[ENVIRONMENT.md](ENVIRONMENT.md) §4 and §7.

## 2a. What the clean room found, and what it got wrong

`make_splits.py` — the first script `make reproduce` invokes — appeared to hang for over ten
minutes. It does not: on an idle machine it completes in **about two seconds** after the data
load. The stall was contention from unrelated processes saturating the cores, and the
measurement was of the machine rather than of the code. [D-066](decisions.md) records the
misdiagnosis in full, including the two successive wrong explanations the instrument itself
disproved.

What survives is narrower and still worth having: the script printed nothing between the dataset
line and its results, so on a busy machine it was indistinguishable from a wedged process. It
now reports twelve units with elapsed and remaining time. Only the scripts that import
`hsbcfraud.progress` report anything; the rest of `scripts/` is still silent. This one is
instrumented because it is the first thing a reviewer runs.

## 2b. Second run, 2026-08-31: the four producers written after the first run

The first run predates `summarise_split_arms.py`, `summarise_seed_sweep.py`,
`run_rolling_origin.py` and `export_predictions.py`, which is four of the repository's producers
and five of its committed tables. A reproduction record that predates the scripts it is meant to
cover is not a record, so the procedure was repeated from `git archive HEAD` extracted into an
empty directory.

**Every derived table reproduces byte-identically.** `split_arm_baselines.csv`,
`mps_seed_sweep_summary.csv`, `rolling_origin.csv`, `predictions.csv` and `operating_point.csv`
are identical to the committed files, from a checkout containing no results of their own beyond
the frozen scorer output the producers read.

**It found one defect, which is the reason to do this.** `run_rolling_origin.py` printed
"0 of 5 breach" against a table containing two. The verdict string had been changed to
`BREACHED` in the row builder to match the selector in `claims.yaml`, and the summary line two
functions below still counted `breached`. The CSV was correct throughout and every gate passed,
because nothing in this repository compares a console line to the file it summarises. The three
verdicts are now named constants read in both places.

**Three tests cannot run from an archive extraction, and should not be expected to.**

| Test | Why it cannot run here | Correct behaviour |
|---|---|---|
| `test_the_built_proposal_has_no_clipped_lines` | needs `submission/proposal.log`, which only a LaTeX build produces | fails rather than passes, which is the rule stated in `check_pdf.py`: silently passing when the evidence is absent is the worst thing a gate can do |
| `test_every_source_file_is_tracked` | calls `git ls-files`; an extracted archive is not a repository | same |
| `test_results_tables_are_tracked` | same | same |

Every other collected test either passes or is a recorded expected failure. At the last
measurement, **2026-09-06**, the tree collected 213: these three, **no strict `xfail`s at all**,
212 that pass and one skipped. The three named in the table above are counted among the 212,
because in a clone with `make pdf` run first they pass; the table says what makes them fail
otherwise. The one skip is a parametrised case with an empty argument set, which is what an
empty defect registry looks like. The five `xfail`s the previous measurement recorded on 2026-09-02
have all cleared, and they cleared by the defects being fixed rather than by the entries being
deleted: `scripts/run_coverage_arms.py` now produces the two producerless tables, and
`audit_labels.py`, `run_explain.py` and `run_power.py` each open a `run_log`. Both registries --
`TABLES_WHOSE_PRODUCER_IS_MISSING` and `SILENT_WITHOUT_A_DESTINATION` -- are now empty dicts, and
they are kept rather than deleted so the next such defect has a place to be recorded instead of
argued about. The composition is stated with
its date rather than as a bare "N of M", because the suite grows and both halves of such a
figure go stale silently: this sentence read "172 of 175" against a tree that collected 179,
and then "184 of 187" against a tree in which five of those 187 had since become expected
failures, so the pass count was five too high. Run `make pdf` first, or clone rather than
extract, and all three run.

## 3. Two levels of verification, and what each answers

| Command | Cost | Answers |
|---|---|---|
| `make walkthrough` | ~1 s, no GPU | Is what the submission reports internally consistent? Recomputes the block partition, the certified set, the held-out validation, the coverage directions and both quantum arms from `results/tables/` and asserts each |
| `make check` | seconds | Does every number in the documents still resolve to its table, does every citation resolve, is the protocol lock intact, does the manifest match? |
| `make reproduce` | minutes plus the data load | Do the tables regenerate from the source? |
| `make seedsweep` | **12.1 GPU-hours** | Does the full-scale tensor-network arm regenerate? Excluded from `reproduce` deliberately — a reviewer checking the certificate should not have to spend a day re-deriving a result the frozen manifest already covers |

A reviewer who wants to check the **certificate** rather than the tensor-network arm can stop
after `make walkthrough` and `make check`.

## 4. Why the dataset is not committed

IEEE-CIS is redistributed under Kaggle's competition terms, which bind the person who downloads
it. Committing the zip would redistribute it under this repository's licence instead, which the
terms do not permit. `.gitignore` excludes `datasets/` for that reason, and
`tests/test_repo_hygiene.py::test_every_source_file_is_tracked` exists because an earlier,
unanchored version of that same pattern silently excluded `src/hsbcfraud/data/` from three
pushed commits — the loader the first Makefile target imports was simply absent from any clone.

## 5. Determinism

**Checked, not asserted.** Regenerating the split tables into a scratch directory reproduced
both committed copies **bit-identically**:

```
.venv/bin/python scripts/make_splits.py --out <scratch>
diff results/tables/splits.csv          <scratch>/splits.csv           # identical
diff results/tables/data_integrity.csv  <scratch>/data_integrity.csv   # identical
```

`--out` points elsewhere on purpose, so the check that verifies the committed tables can never
overwrite them.

Seeds are fixed in [`configs/default.yaml`](../configs/default.yaml) and the PDFs build under
`SOURCE_DATE_EPOCH` so they hash reproducibly. `scripts/freeze.py --check` compares a later run
against `MANIFEST.sha256.json` and reports any artefact whose hash moved.

Two things are **not** bit-reproducible across machines and are not claimed to be: GPU
floating-point reductions in XGBoost and PyTorch depend on the device and driver, and the
latency table in [ENVIRONMENT.md](ENVIRONMENT.md) §5 is a wall-clock measurement of this
workstation. Both are stated with the hardware they were measured on.

## 2c. Third run, 2026-09-05: what a clean room is actually for

The record above described a tree that no longer existed. Roughly twenty scripts and several
modules had changed since 2026-08-31 -- three new producers, a rewritten data loader, a
reclassified manifest -- so the second pass was asserting reproduction of code nobody was
shipping. That alone justified a third run.

It reproduced the study and **found two defects that no gate in this repository could have
caught**, because both are properties of the environment rather than of the tree.

**Everything scientific reproduced, and the measurement is stronger than "the log looked
right".** `make check` stops at the claims gate, so it never reached the manifest comparison;
the table below is a direct digest-and-column diff of all 34 committed tables against the clean
room's, run afterwards.

| | Tables | What differs |
|---|---|---|
| Byte-identical | **26** | nothing |
| Identical in every measured value | **5** | one wall-clock column only: `fit_seconds` in `ablations`, `baselines`, `mps_band`, `mps_full`, and `gram_seconds` in `screens` |
| Did not reproduce | **2** | `latency.csv`, whose every measured column *is* a timing; `parity.csv`, 4 rows against 12 |
| Bookkeeping | **1** | `decision_log.csv` -- the clone predates the entries written since |

The middle row is the result worth reading twice. **The full-scale tensor-network fits reproduce
every metric exactly** -- `mps_band.csv` and `mps_full.csv` agree to the last digit on ROC AUC
and average precision across a fresh virtual environment, a rebuilt CUDA stack and a separate
process; only the seconds they took differ. So does the 120-configuration screen, and so do all
15 baseline fits.

Of the 103 bound claims, **94 passed and 9 failed**, and every one of the 9 belongs to the two
tables in the "did not reproduce" row.

**Defect one: a latency benchmark measures the host.** Six timing claims failed. The scorer
tail moved from 0.32 ms to 1.27, the kernel tail from 129 ms to 302, and
`LatencyKernelBudgetShare` from **75.9 % to 177.8 %** -- past the point where the claim's own
note, "under one, so it fits", inverts into "does not fit". (75.9 % was the committed value at
the time; it is **61.3 %** since the 2026-09-06 re-measurement, [D-147](decisions.md).) Nothing was wrong with the code.
The host was busy, at a load average of 33 across 20 cores, **and the busiest thing on it was
the work of auditing this submission**. `measure_latency.py` now refuses above 0.25 load per
core, because a number that silently changes by a factor of four is worse than a run that
stops.

**Defect two: the documented procedure could not reproduce two of the claims it verifies.**
`qiskit-aer` sits in the optional `gpu-crosscheck` extra, so `make venv` cross-checks the
fidelity kernel against Braket alone -- 4 comparisons where the committed table has 12.
`check_parity.py` reports this precisely and by name; `make check` then fails on
`ParityComparisons` and `ParityWorstDifference` with nothing linking the two. The tooling was
honest and the instructions were not. Section 1 now states both conditions.

**What this run establishes about the previous two.** They passed. They were also run on a quiet
machine by someone who had just built the tree, which is the one reader a reproduction check
does not need to satisfy. Running it under adversarial conditions -- stale code, contended
host, default environment -- is what turned up anything, and both findings are things a
reviewer would have hit first.


## 2d. Fourth run, 2026-09-06: the prediction, tested

The third pass ended with `make check` failing on nine claims and a diagnosis that the two
tables which could not reproduce anywhere were being regenerated and then failed against. The
fix -- latency re-measurement made opt-in, parity refusing to overwrite a wider result -- made a
prediction: **a fourth pass should come back green.** This is that pass.

`make reproduce` exit 0. `make check` **exit 0**, and it reached the manifest comparison rather
than halting at the claims gate:

```
All 102 claims agree with their tables.
every citation resolves to a reference entry
Protocol unchanged. Guarantee hash a8275e8fc7666cb3, amendments A1-A9
All 51 scientific artefacts match the manifest.
```

**What actually reproduced**, measured by diffing all 34 committed tables against the clean
room's own regenerated ones rather than by reading the run log:

| | Tables |
|---|---|
| Byte-identical | **28** |
| Identical in every measured value, differing only in a wall-clock column | **5** |
| Differing in a measured value | **1**, the decision-entry count |

The last is `decision_log.csv`, a projection of `docs/decisions.md` that moves whenever an entry
is added, and which [D-128](decisions.md) reclassified as specification for exactly that reason.
So **every scientific value in every table reproduced**, including the full-scale
tensor-network fits, across a fresh virtual environment and a rebuilt CUDA stack.

Twenty-eight byte-identical against the third pass's twenty-six. The two that moved into the
column are latency and parity, which now preserve rather than overwrite.

**Why the quoted artefact count is 51 and the manifest now says 52.** The extra file is
`results/tables/baseline_tuning.csv`, the hyperparameter sweep, which was committed at 19:15 --
after this pass had already reached its manifest comparison. The count moved because the study
gained a table, not because anything in the clean room disagreed with it. Read the block above as
the output of a run against the tree as it stood, which is what it is.

**One correction to how the third pass was reported.** After fixing the last failing claim, a
`git checkout` in the clean room reverted its regenerated tables to the committed ones, and the
claim check that followed was therefore committed-claims against committed-tables -- trivially
true, and briefly written up as though it meant something. It did not. This pass was re-run
from scratch for that reason, and the table above is a direct file diff, not an inference from
a log.

## 2e. Fifth run, 2026-09-11: a fresh clone of HEAD, start to finish

The fourth pass ran against a working tree. This one starts from `git clone` of `origin/main` at
commit `c196644`, stages only the dataset -- which the licence forbids committing -- and then
follows section 7 of the README verbatim: `make venv`, `make smoke`, `make walkthrough`,
`make reproduce`, `make check`. Three hours thirty-two minutes, unattended.

```
=== EXIT-VENV 0 ===        1m28s, torch cu130 first
=== EXIT-SMOKE 0 ===       S0-S8, all 9 checks accounted for
=== EXIT-WALKTHROUGH 0 === every assertion holds against the committed tables
=== EXIT-REPRODUCE 0 ===   3h31m
=== EXIT-CHECK 0 ===       All 102 claims agree; 51 scientific artefacts match the manifest
=== EXIT-PYTEST 0 ===      212 passed, 1 skipped
```

**Zero errors or tracebacks** across the whole reproduction log.

**What reproduced, diffed directly rather than read off the log.** All 35 committed tables were
compared column by column against the clean room's own, separating measured values from
wall-clock ones:

| | Tables |
|---|---|
| Byte-identical | **30** |
| Identical in every measured value, differing only in a timing column | **5** |
| **Differing in a measured value** | **0** |

**Thirty-one of the thirty-five were actually rewritten**, which matters more than the totals: a
file that is never touched cannot fail a comparison. The four the run did not regenerate are the
four it is designed not to, each for a reason a reviewer can check:

| Table | Why it was not rewritten |
|---|---|
| `latency.csv` | opt-in by design; the run printed "NOT re-measuring latency" and said why |
| `parity.csv` | the `gpu-crosscheck` extra is absent, so 4 comparisons ran against 12 committed, and the wider table is kept |
| `mps_seed_sweep.csv` | the 13-GPU-hour sweep, excluded from `make reproduce` and run by `make seedsweep` |
| `mps_seed_spread.csv` | the documented producerless exemption ([PROVENANCE.md](PROVENANCE.md) section 1.4) |

**The built documents are bit-reproducible.** `proposal.pdf`, `appendix.pdf`, `method.png` and
`certified_region.png` rebuilt to SHA-256 digests identical to the committed ones, at 6 and 3
pages. A reviewer following the README gets the same bytes, not merely the same numbers.

**The parity gate was exercised on the path that used to be wrong.** `check_parity.py` took its
skip branch -- four Braket rows against twelve committed -- and printed "Every comparison that did
run agreed with the reference." That sentence is now true by construction: [D-148](decisions.md)
moved the agreement check above the skip, so the branch can no longer return 0 on a disagreement.
This run is the first to exercise it in the environment the documented procedure produces.

