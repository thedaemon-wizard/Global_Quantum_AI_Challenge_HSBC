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
make smoke        # S0-S8 environment assertions; any failure stops the build
make walkthrough  # trace the certificate against the committed tables (seconds, no GPU)
make reproduce    # refit everything except the 13-GPU-hour sweep, then write the manifest
make check        # rebuild both PDFs, then the claim, citation, protocol and manifest gates
```

`make venv` is the step with a real cost: torch from the CUDA 13.0 index is a large download.
Everything after it is fast except `make reproduce`.

## 2. What this run established

| Step | Result |
|---|---|
| `python3.12 -m venv` from the system interpreter | Python 3.12.11 |
| `pip install torch==2.13.0` from the cu130 index | installed; `torch.cuda.is_available()` **True** |
| `pip install -e ".[dev]"` | clean, no resolution conflicts |
| `make smoke` | **9 of 9 accounted for**, 7 pass and 2 skip in the fresh environment, where the Aer GPU cross-check skips as well as the ULB check; the committed environment reports 8 and 1 |
| `make walkthrough` | every assertion holds against the committed tables |

The two skips are declared, not silent:

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
now reports twelve units with elapsed and remaining time. Twenty-three of the twenty-six scripts
in `scripts/` still have no progress reporting; this one is instrumented because it is the first
thing a reviewer runs.

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

172 of 175 pass. Run `make pdf` first, or clone rather than extract, and all three run.

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
