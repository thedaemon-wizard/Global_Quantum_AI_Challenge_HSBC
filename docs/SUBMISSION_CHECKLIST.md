# Submission checklist

Every item is a **verifiable assertion**, not a reminder. Each carries the command that
decides it and the file that records the answer. An item is complete only when its command
has been run on the machine that will produce the submission and its output recorded here.

Deadline: **2026-09-15**. Track: HSBC, Global Quantum + AI Challenge 2026.

Legend: `[x]` verified by a command that passed; `[ ]` not yet verified; `[!]` verified and
**failing**, with the failure recorded rather than hidden.

---

## A. Scientific integrity

These are the items whose failure would make the submission wrong rather than incomplete.

| | Item | Command | Recorded in |
|---|---|---|---|
| `[x]` | Pre-registration hash matches, or differs only with a dated amendment | `.venv/bin/python scripts/check_protocol.py` | exit 0 or 2, never 1 |
| `[x]` | The H4 power gate was computed **before** the comparison it governs | `scripts/run_mps.py` refuses to start without it | `results/tables/power.csv` |
| `[x]` | The power gate's noise estimate reflects the comparison it governs | fixed: the cross-family proxy gives SE 0.0164 against the 0.0199 the comparison produced, where the within-family figure was 0.0025 | [D-030](decisions.md) |
| `[x]` | H4 is reported as underpowered: the effect the comparison could resolve is 0.0557, against a ceiling of 0.023 and a pre-registered estimate of 0.0461 | `scripts/run_power.py` prints FAIL and `run_mps.py` echoes it | `power.csv`, `mps_h4.csv` |
| `[ ]` | Coverage is judged against the exact Beta-Binomial law, not against `rate <= alpha` | **no command decides it.** `make conformal` was named here in error: `run_conformal.py` writes `coverage.csv`, a four-row table with no `arm` column, and nothing in the tree writes `coverage_by_arm.csv` or `coverage_by_arm_seeds.csv`. The property holds of the committed files; what is missing is the producer that would let a reviewer rebuild them. `tests/test_repo_hygiene.py` carries both in `TABLES_WHOSE_PRODUCER_IS_MISSING` as strict xfails, so the entry clears itself the moment one is written | `coverage_by_arm.csv` column `finite_sample_ok` |
| `[x]` | Every risk certificate rests on a mean of per-observation 0/1 losses, as Hoeffding-Bentkus requires | `pytest tests/test_riskcontrol.py` asserts the risk equals the mean of the indicator, exactly | [D-024](decisions.md) |
| `[x]` | A fit that goes non-finite stops with recoverable parameters and names the step | `pytest tests/test_mps_guards.py` | [D-035](decisions.md) |
| `[x]` | Long runs report progress, and the log is readable while the run continues | `pytest tests/test_progress.py` | [D-033](decisions.md) |
| `[x]` | An exploratory run cannot overwrite a pre-registered result table | non-default arguments divert to `results/runs/exploratory/` | [D-034](decisions.md) |
| `[x]` | No certified configuration sits at the half-open band boundary, where the flagged set is empty by construction | inspect `selected_lambda` against `band_hi` | `riskcontrol.csv` |
| `[x]` | `D_test` was never used for selection: one authorised configuration, and the guard raises on a second distinct one | `test_access.json` records `evaluations: 1` and one configuration digest; amendment A8 lists the scripts that read the block for reporting outside the guard | `results/tables/test_access.json` |
| `[x]` | Every number in the proposal prose resolves to a table | `make claims` | `docs/claims.yaml` |
| `[x]` | No claim is defined and used nowhere | `scripts/check_claims.py --unused` | `docs/claims.yaml` |
| `[x]` | Every citation in the documents resolves to an entry in [REFERENCES.md](REFERENCES.md) | `scripts/check_claims.py --citations` | [REFERENCE_CROSSCHECK.md](REFERENCE_CROSSCHECK.md) |

### A.1 Claims that must NOT appear

Checked by reading, because no script can catch a sentence that was never measured.

| | Prohibited claim | Why |
|---|---|---|
| `[x]` | Any form of quantum advantage | Neither arm produced one. The kernel was rejected by its screens; the MPS lost to a tuned GBDT. |
| `[x]` | A portfolio-level PSD2 compliance rate | Category error: the SCA-RTS ceilings govern exemption eligibility on a whole portfolio, not a decline threshold on a fraud-enriched benchmark. [Amendment A2](protocol.md). |
| `[x]` | "Temporal splits inflate the false-decline rate by 1.4×" as a fixed quantity | Origin-dependent: 0.684 / 0.893 / 1.411 / 1.198 / 1.126 across five origins. [D-025](decisions.md). |
| `[x]` | That the certificate binds the **unconditional** false-decline rate | It binds the band-conditional rate. The unconditional rate is dominated by `tau_hi`. |
| `[x]` | A broad claim about MPS versus GBDT "on tabular payment-fraud data" | `mps_full.csv` now exists, so the full-scale comparison is reported. The claim is still narrowed: a 2026-08-28 literature check found published MPS work on tabular data to be largely generative, and "we did not find a comparable benchmark" is what the body says rather than "none exists". |

---

## B. Reproducibility

| | Item | Command |
|---|---|---|
| `[x]` | Environment assertions S0–S8 pass on the submission machine | `make smoke` |
| `[x]` | Aer reports a real GPU device, not a silent CPU fallback | `make venv-gpu` asserts it; smoke S1 reads `nvidia-smi` |
| `[x]` | No source file is excluded from git by an unanchored ignore pattern | `pytest tests/test_repo_hygiene.py` |
| `[x]` | Every committed result table is tracked | same |
| `[x]` | Nothing invokes a bare `python3` (3.9 on this host) | same |
| `[x]` | No chained pandas assignment (a silent no-op under copy-on-write) | same |
| `[x]` | Every script the Makefile references exists | `test_makefile_scripts_exist`, passing |
| `[ ]` | A fresh clone reproduces the tables | clone to a temp dir, `make reproduce`, `make check` |
| `[ ]` | The SHA-256 manifest matches after a second run | `make check` |

---

## C. Deliverable format

| | Item | How verified |
|---|---|---|
| `[x]` | Body PDF is at most **6** pages, A4, minimum 10 pt, per the Phase 1 guidelines section 5 | `scripts/check_pdf.py --max-pages 6 --paper a4 --min-font 10` |
| `[x]` | Appendix PDF is at most 3 pages, same constraints | same, `--max-pages 3` |
| `[ ]` | PDFs build deterministically (`SOURCE_DATE_EPOCH` fixed) so they can be hashed | `make pdf` twice, compare hashes |
| `[x]` | Figures are generated from tables, not drawn by hand | `make figures` |
| `[x]` | The staged set carries only formats the portal accepts, in at most 5 slots | `assemble_submission.py` refuses any other format; verified against the live form's `accept` attribute |
| `[x]` | Every uploaded file's format is confirmed against the live portal form, per slot | five slots; PDF, PDF, CSV, PY, PNG. Re-verified in a browser 2026-08-31 against the HSBC submission panel, which reports "0 uploaded, 5 slots left" and lists PDF, PNG, JPG, WEBP, GIF, PY, JSON, JS, XLS, XLSX, CSV, DOC, DOCX |
| `[x]` | Each of the four Expected Outcomes on the portal's challenge panel is answered by a named artefact a reviewer can open | per-transaction probabilities and binary predictions: `HSBC-predictions.csv`; feature attribution: proposal §5 and `attribution_examples.csv`; classical-baseline comparison: §3; quantum encoding and circuit design: §4 and §7. Checked against the objective and outcomes shown on the portal's own challenge panel, 2026-08-31 |
| `[x]` | The staged set is under the stated size cap on either reading | 12 MB total, largest `HSBC-predictions.csv` at 11 MB; the guidelines say "File size must not exceed 20 MB" without specifying per file or total, so both readings are satisfied. Checked by `du -sb submission/portal/`, not by a gate |
| `[x]` | README carries the development environment and the measured benchmark timings | README §1 states the machine and a six-row stage-cost table inline, so a reader sees both without following a link; [ENVIRONMENT.md](ENVIRONMENT.md) carries the pinned versions and the per-stage detail. Previously this row was satisfied by the link alone, which is not what "carries" means |
| `[x]` | README reports results as figures as well as tables | `coverage_by_arm.png`, `certified_region.png`, `mps_h4.png`, all generated by `make figures` from the tables |
| `[x]` | Every claim checked this round is recorded with its source and verdict | [FACTCHECK_LOG.md](FACTCHECK_LOG.md) |
| `[ ]` | **Repository is public before review** --- the proposal's title block prints its URL, which 404s while the repo is private | flip on 2026-09-15 with the upload |
| `[x]` | No citation in either PDF requires a file that is not uploaded | instruments named in full (`SR 26-2`, `EU AI Act`); zero bare `[XX-N]` keys in either PDF |
| `[x]` | Every dataset named in the pre-registration was either used or formally withdrawn | ULB withdrawn by amendment A9 |
| `[ ]` | Uploaded | `make submission`, then upload manually |

---

## D. Licensing and data handling

| | Item | Status |
|---|---|---|
| `[x]` | Code is Apache-2.0 with an SPDX header on every source file | enforced by `tests/test_repo_hygiene.py` |
| `[x]` | Raw data is never committed | `.gitignore` anchored `/datasets/` and `/data/` |
| `[x]` | IEEE-CIS is used under Kaggle competition rules, not redistributed | [NOTICE](../NOTICE) |
| `[x]` | ULB is two-layer, ODbL on the database and DbCL v1.0 on its contents, and it is not redistributed | [NOTICE](../NOTICE) |
| `[x]` | `cuquantum-cu11` (NVIDIA proprietary SLA) is not installed by default | separate `make venv-gpu` target, NOTICE §4 |
| `[x]` | Conformal code was written from the literature, not ported from an unlicensed repository | [D-010](decisions.md) |
| `[ ]` | Repository is private until 2026-09-15, then public | manual |

---

## E. Known incomplete

Recorded here rather than omitted, so the gap is visible to whoever picks this up.

| Item | Blocking? |
|---|---|
| ~~Conformal, risk-control and coverage tests~~ | Done. `tests/test_conformal.py` and `tests/test_riskcontrol.py`; writing them found an off-by-one in the degeneracy-floor docstring |
| ~~Makefile-referenced scripts not yet written~~ | Done. Every referenced script exists and `make reproduce` completes; the experiments that were not run are named in a comment at the head of the targets |
| ~~`REGULATORY_SOURCES.md`, `guarantee.md`, `REFERENCE_CROSSCHECK.md`~~ | Done. All three exist; the cross-check is generated by `scripts/make_crosscheck.py` |
| `configs/default.yaml` frozen hash re-verified after every amendment | Yes, before the final freeze |
| Portal URLs re-verified in a browser immediately before submission | Yes |

---

## F. Final sequence

Run in this order on the submission machine. Do not reorder: `check` gates against the
built PDFs, so building them afterwards validates nothing.

```
make smoke
make test
make reproduce
make pdf
make check
make submission
```

Then, and only then, re-verify the portal URLs in a browser and upload.
