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
| `[x]` | H4 is reported as underpowered, since the corrected MDE (0.0461) exceeds the ceiling (0.023) | `scripts/run_power.py` prints FAIL and `run_mps.py` echoes it | `power.csv` |
| `[x]` | Coverage is judged against the exact Beta-Binomial law, not against `rate <= alpha` | `make conformal` | `coverage_by_arm.csv` column `finite_sample_ok` |
| `[!]` | Every risk certificate rests on a mean of per-observation 0/1 losses, as Hoeffding-Bentkus requires | **no test exists**; verified only by reading `missed_fraud_rate` | [D-024](decisions.md) |
| `[x]` | A fit that goes non-finite stops with recoverable parameters and names the step | `pytest tests/test_mps_guards.py` | [D-035](decisions.md) |
| `[x]` | Long runs report progress, and the log is readable while the run continues | `pytest tests/test_progress.py` | [D-033](decisions.md) |
| `[x]` | An exploratory run cannot overwrite a pre-registered result table | non-default arguments divert to `results/runs/exploratory/` | [D-034](decisions.md) |
| `[x]` | No certified configuration sits at the half-open band boundary, where the flagged set is empty by construction | inspect `selected_lambda` against `band_hi` | `riskcontrol.csv` |
| `[ ]` | `D_test` was evaluated once, for one configuration | read the counter | `results/tables/test_access.json` |
| `[ ]` | Every number in the proposal prose resolves to a table | `make claims` | `docs/claims.yaml` |
| `[ ]` | Every citation in the proposal resolves to an entry in [REFERENCES.md](REFERENCES.md) | `scripts/check_claims.py` | — |

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
| `[ ]` | Every script the Makefile references exists | same — currently **xfail**, see §E |
| `[ ]` | A fresh clone reproduces the tables | clone to a temp dir, `make reproduce`, `make check` |
| `[ ]` | The SHA-256 manifest matches after a second run | `make check` |

---

## C. Deliverable format

| | Item | How verified |
|---|---|---|
| `[ ]` | Body PDF is at most 5 pages, A4, minimum 10 pt | `scripts/check_pdf.py --max-pages 5 --paper a4 --min-font 10` |
| `[ ]` | Appendix PDF is at most 3 pages, same constraints | same, `--max-pages 3` |
| `[ ]` | PDFs build deterministically (`SOURCE_DATE_EPOCH` fixed) so they can be hashed | `make pdf` twice, compare hashes |
| `[ ]` | Figures are generated from tables, not drawn by hand | `make figures` |
| `[ ]` | Portal accepts the assembled file set | `make submission`, then upload |

---

## D. Licensing and data handling

| | Item | Status |
|---|---|---|
| `[x]` | Code is Apache-2.0 with an SPDX header on every source file | enforced by `tests/test_repo_hygiene.py` |
| `[x]` | Raw data is never committed | `.gitignore` anchored `/datasets/` and `/data/` |
| `[x]` | IEEE-CIS is used under Kaggle competition rules, not redistributed | [NOTICE](../NOTICE) |
| `[x]` | ULB ODbL share-alike applies to the database, and it is not redistributed | [NOTICE](../NOTICE) |
| `[x]` | `cuquantum-cu11` (NVIDIA proprietary SLA) is not installed by default | separate `make venv-gpu` target, NOTICE §4 |
| `[x]` | Conformal code was written from the literature, not ported from an unlicensed repository | [D-010](decisions.md) |
| `[ ]` | Repository is private until 2026-09-15, then public | manual |

---

## E. Known incomplete

Recorded here rather than omitted, so the gap is visible to whoever picks this up.

| Item | Blocking? |
|---|---|
| The proposal LaTeX body and appendix — the scored artifact | **Yes** |
| 15 Makefile-referenced scripts not yet written (`test_makefile_scripts_exist` xfails with the list) | **Yes** for `make reproduce` end to end |
| `PROVENANCE.md`, `REGULATORY_SOURCES.md`, `guarantee.md`, `claims.yaml`, `tables.yaml`, `REFERENCE_CROSSCHECK.md` | Needed for `make claims` and `make check` |
| `configs/default.yaml` frozen hash re-verified after the A1–A4 amendments | Yes, before the final freeze |
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
