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

**Two conditions this procedure did not state, both found by running it (2026-09-05).**

**Run it on an otherwise idle machine.** `make reproduce` measures inference latency, and a
latency benchmark measures the host as much as the model. The third clean-room pass was run on
a busy workstation and produced timings two to four times the committed ones, failing `make
check` on six claims with nothing to connect the failure to its cause. `measure_latency.py` now
refuses above 0.25 load per core rather than producing numbers that are not comparable.

**`make venv` alone cannot reproduce the parity claims.** `qiskit-aer` and its GPU wheel live in
the optional `gpu-crosscheck` extra, so a default environment cross-checks the fidelity kernel
against Braket only: **4 comparisons, not the committed 12**. `check_parity.py` says so plainly
--- *"aer\_cpu needs the gpu-crosscheck extra"* --- but `make check` then fails on
`ParityComparisons` and `ParityWorstDifference` without pointing back at it. The extra is
optional by choice: it pulls `cuquantum-cu11` under NVIDIA's proprietary licence
([NOTICE](../NOTICE) section 4), and no scientific figure depends on it. Install it to reproduce
those two claims; skip it and expect exactly those two to differ.

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
measurement, 2026-09-02, the tree collected 187: these three, five strict `xfail`s that record
defects elsewhere in the repository — two producerless tables and three scripts that read the
full dataset behind no progress destination — and 179 that pass. The composition is stated with
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
| `make seedsweep` | **13 GPU-hours** | Does the full-scale tensor-network arm regenerate? Excluded from `reproduce` deliberately — a reviewer checking the certificate should not have to spend a day re-deriving a result the frozen manifest already covers |

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

**Everything scientific reproduced.** All 15 baseline fits match `baselines.csv` to within
5e-5, read out of `results/runs/baselines.log` rather than taken on trust. The conformal stage
reproduced its coverage verdicts exactly -- 1,667 errors on 111,592 test points, rate 0.014938,
tail p below 1e-17, outside the 99 % band at the two loosest levels. The screens rejected the
same 120 configurations for the same reasons. `make walkthrough` passed every assertion.

**Defect one: a latency benchmark measures the host.** Six timing claims failed. The scorer
tail moved from 0.32 ms to 1.27, the kernel tail from 129 ms to 302, and
`LatencyKernelBudgetShare` from **75.9 % to 177.8 %** -- past the point where the claim's own
note, "under one, so it fits", inverts into "does not fit". Nothing was wrong with the code.
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
