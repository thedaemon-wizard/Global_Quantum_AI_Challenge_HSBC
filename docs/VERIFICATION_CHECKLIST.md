# Verification checklist

What has been checked against what, and what has not. Distinct from the other two checklists:
[`COMPLIANCE_CHECKLIST.md`](COMPLIANCE_CHECKLIST.md) maps the challenge statement's requirements
to where each is met, and [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md) is the list to
walk immediately before uploading. This file records the **verification work itself** -- which
official documents were read against the submission, which external artefacts were reconciled,
and which checks are still open.

A row is ticked only when the check was actually run, not when it was planned.

---

## 1. Official documents, read against the submission

| | Document | Checked | Result |
|---|---|---|---|
| `[x]` | HSBC Challenge Statement (vFinalRevised, 16 pp.) | 2026-08-31, all seven sections | The statement itself has no "Expected Outcomes" heading: §5.2 is "Expected Outputs" over three rows, with four separate "Reporting Considerations". The four Expected Outcomes are the portal challenge panel's wording, and each is answered by a named artefact. Two named requirements were unmet and are now met (class-imbalance handling, the sponsor's own cited comparators) |
| `[x]` | Challenge Statement, revised against original | 2026-08-31, word-by-word diff; re-run at character level 2026-09-02 | The revision **removes** the parenthetical "(~24,000 rows)" and re-paginates from 18 pages to 16. **No URL is added and no word is added** -- after whitespace normalisation the only content opcode is that one deletion, and the apparent URL differences are line-wrap points moving under re-pagination. No requirement changed. An earlier automated diff reported the two as textually identical, which a character-level diff does not support |
| `[x]` | Phase 1 Submission Guidelines | 2026-08-31 | 6 pages + 3 appendix, A4, 10 pt floor, five slots -- all asserted by `scripts/check_pdf.py` and `scripts/assemble_submission.py` rather than by eye. §5 says "File size must not exceed 20 MB" without specifying per file or total; **no script checks it**, and the staged set is 12 MB total with an 11 MB largest file, so it is under the cap on either reading |
| `[x]` | Assessment Criteria | 2026-08-31; findings re-verified 2026-09-02 | An adversarial pass over 10 criteria findings **confirmed none of them** -- each was either already satisfied elsewhere in the submission or rested on a stale copy of a file. Two proposed fixes carried factual errors that would have contradicted the claim ledger. §4 below is about the Terms, not these criteria |
| `[x]` | Terms and Conditions | 2026-08-31; full clause pass 2026-09-02 | All 13 sections, see §4 below |

## 2. Live surfaces, re-verified in a browser

| | Surface | Checked | Result |
|---|---|---|---|
| `[x]` | Portal dashboard and challenges list | re-read in a browser 2026-09-02, signed in | Five challenges, all still "Not submitted"; "My Files: No files uploaded yet" |
| `[x]` | HSBC challenge submission panel | re-read in a browser 2026-09-02 | "0 uploaded, 5 slots left"; formats PDF, PNG, JPG, WEBP, GIF, PY, JSON, JS, XLS, XLSX, CSV, DOC, DOCX -- unchanged across three checks. All five staged files are within this list |
| `[x]` | Which challenge-statement version is operative | 2026-09-02 | **Settled.** The portal links `vFinalRevised` from an **August** upload path; the other three documents sit under April. The 18-page `vF-1` is superseded. This had been carried as an open question |
| `[x]` | IEEE-CIS Kaggle rules, sections 7.A and 7.B | read in full in a browser 2026-09-02 | 7.A permits use "for non-commercial purposes only ... and for academic research and education". 7.B forbids transmitting, duplicating, publishing or redistributing the Competition Data. Both are consistent with what `NOTICE` and `PROVENANCE.md` state, and with section 5 of the proposal saying a commercial PoC cannot reuse this file |
| `[x]` | Whether the uploaded CSV is a redistribution of Competition Data | 2026-09-02, tested against the archive | **It is not, and this was checked rather than argued.** Not one IEEE-CIS column name appears in `predictions.csv`; its `transaction_row` is a positional index (475,006 upward) and **not** the dataset's `TransactionID` (2,987,000 upward). The file carries model output and thresholds only |
| `[x]` | Programme roadmap | 2026-08-31 | Phase I closes 15 Sep 2026; review 16 Sep - 14 Nov; Phase II PoC 17 Nov 2026 - 28 Feb 2027; winners 30 Apr 2027 |
| `[x]` | GitHub rendering of every markdown document | 2026-08-31; re-checked 2026-09-02 | All spans render; enforced by `scripts/check_markdown_math.py` against GitHub's own `POST /markdown` and MathJax. The Mermaid pipeline diagram was watched rendering in a browser on 2026-09-02 |
| `[ ]` | In-browser re-check of math rendering, 2026-09-02 | **inconclusive, and the reason matters** | The README's formulas displayed as raw LaTeX in the browser used for this check. That is **not** a defect in this repository: GitHub's *own* documentation page for mathematical expressions displayed `$\sqrt{3x-1}+(1+x)^2$` as raw text in the same session, with zero math nodes. GitHub renders math client-side, and that renderer was not executing. The control experiment is what stopped this being filed as a defect against the submission. Re-check from a different browser before 2026-09-15 |
| `[ ]` | The repository URL printed in the proposal title block | **manual, before upload** | Confirmed 404 to an unauthenticated request on 2026-09-02, which is correct while private. Must be public on 2026-09-15 or the title block points at nothing |
| `[x]` | The portfolio, which is the proposal's only verification URL | fetched and read 2026-09-02 | HTTP 200. Four claims agree with §8 exactly: the sole proprietorship "filed 1 Aug 2026", Team MerQury, three merged pull requests with two bounty issues (#2242 and #749) for USD 200, and QIntern 2026. A fifth agreed until 2026-09-06, when §8 replaced the Qiskit Advocate line with a certification that has a public badge; the portfolio still lists only the Advocate. **Two now differ, and in both the proposal is the one to trust** -- see §5c |
| `[ ]` | The upload itself, and the lead-contact field | **manual, before upload** | Cannot be automated |

## 2b. The portal, read in a browser on 2026-09-06

Read while signed in at
`https://quantumaiportal.thequantuminsider.com/user/.../#challenges`. What the live form says,
rather than what this repository assumes about it:

| | Portal | Consequence here |
|---|---|---|
| HSBC challenge status | **Not submitted**, `0 uploaded - 5 slots left`; **My Files** reads "No files uploaded yet" | The five-file plan fits exactly, with nothing to remove first |
| Allowed formats | `PDF, PNG, JPG, WEBP, GIF, PY, JSON, JS, XLS, XLSX, CSV, DOC, DOCX` | The staged set is `.pdf .pdf .csv .py .png` -- **every one allowed** |
| Operative statement | the download link resolves under **`/uploads/2026/08/`**, `HSBC-Challenge-Statement-vFinalRevised.pdf` | Confirms the August revision is the live document; the April `vF-1` is superseded |
| Resource documents | Submission guidelines, Assessment Criteria and Terms all under `/uploads/2026/04/` | The three governing documents have not moved since April |
| Submission form fields | a file input and nothing else -- no team-name, title or description field | Whatever the account's registration holds is what the organisers see; the form offers no place to state a trading name at upload time |

**Re-checked 2026-09-12, and one thing on the page changed.** The HSBC panel is unchanged --
`0 uploaded - 5 slots left`, the same allowed-format list, the upload control present, and the
statement still resolving under `/uploads/2026/08/`. What changed is a **different track**: the
Airbus challenge now reads **Submitted**, with its button changed from `START` to `VIEW`. That is
a separate entry and does not touch this submission, but it is worth recording for two reasons --
it confirms the portal renders a filed track distinguishably, so the HSBC panel's `Not submitted`
is a positive statement rather than a default; and it establishes that one account may hold
submissions to several problem statements, which the Submission Guidelines section 4.2 permits
("A team may submit to multiple problem statements, but each must be a separate submission").

**The four Expected Outcomes the portal lists for this challenge, against what ships:**

| Portal wording | Where it is answered |
|---|---|
| "Fraud probability scores (float [0,1]) and binary predictions for each transaction" | `predictions.csv`, 115,534 rows: `fraud_probability` spanning [0.0000, 1.0000], `predicted_fraud`, and `decision` carrying the three-valued rule |
| "Feature attribution analysis explaining individual predictions" | `attribution.csv` and `attribution_examples.csv`, from `scripts/run_explain.py` |
| "Quantitative comparison with at least one classical baseline (XGBoost, LightGBM, or CatBoost)" | `baselines.csv` -- a gradient-boosted baseline over five seeds and three splits. **One model, not two:** this cell said "XGBoost **and** LightGBM" until 2026-09-12, which is the falsehood [D-008](decisions.md) and [D-092](decisions.md) already removed from the README -- `baselines.csv` has 45 rows and every one is `xgboost`. The statement asks for "at least one classical baseline", which one satisfies |
| "Documentation of quantum encoding strategy, circuit design choices, and conditions where quantum methods..." | proposal section 4 and `circuits.csv`; `results/figures/circuits.png` draws the screened encodings |

**Still manual, and still open.** The upload itself, and confirming the repository URL in the
title block resolves once the repository goes public on 2026-09-15. Neither can be automated
from here.

## 2c. The full-scale seed sweep

`make reproduce` deliberately excludes `make seedsweep` -- sixteen full-scale tensor-network fits,
about **12.1 GPU-hours** -- so `mps_seed_sweep.csv` is the one committed table no clean-room pass
had ever regenerated. It backs `SweepBestAp` directly and, through
`mps_seed_sweep_summary.csv`, the seed-spread and capacity figures section 4 argues from.

| | Check | Run | Result |
|---|---|---|---|
| `[x]` | Sixteen full-scale fits reproduce their measured columns from a fresh clone | clean-room clone at `41c52f7`, idle GPU, 2026-09-11 09:13 to 21:19 | **Exact on every fit**: `max \|delta\| = 0.000e+00` for ROC AUC, average precision and final loss across all 16. Summary table byte-identical. `gpu_contended = False`, so the timings are quotable |
| `[x]` | The same code path reproduces at smaller scale | clean-room pass of 2026-09-11 | `mps_full.csv` -- four full-scale fits, 431 sites -- reproduced every measured column exactly. The sweep is the same path at four seeds |

**Does the sweep verification still cover the shipped code?** It was run against `41c52f7` and
the tree has moved since, so the question is not rhetorical. Two files the sweep touches changed:

| File | Change | Can it alter the sweep's output? |
|---|---|---|
| `scripts/run_seed_sweep.py` | text inside the `raise SystemExit` on the GPU guard's refusal path | **No.** That path is not taken on an idle GPU, which is the condition the run was made under |
| `src/hsbcfraud/data/splits.py` | **19 insertions, 0 deletions** -- `configuration_digest` added | **No.** Purely additive; no existing function changed |

`git diff --stat 41c52f7..HEAD -- src/hsbcfraud/data/splits.py` gives `19 +++++++++++++++++++`
with no deletions, and the `run_seed_sweep.py` diff is confined to the message string. So the
16-of-16 result stands for the submitted code and **the twelve hours are not repeated**: a
re-run would exercise the same code on the same inputs and could only reproduce what is already
recorded.

**Two things the attempt established before it ran.**

The guard refused the first launch: `1 process(es) hold the GPU`. It was **Firefox**, holding a
284 MiB compositing context. The guard is right to fire -- it cannot tell a browser from a
training job, and a contended run reports `fit_seconds` wrong by a factor of three -- but its
message named no way forward. It now names `--allow-shared-gpu` and states the cost, which is
that every row is stamped `gpu_contended=True` and its timings must not be quoted.

**The committed table predates its own producer.** It has no `gpu_contended` column; the current
`run_seed_sweep.py` always writes one. So the committed sweep could not have been reproduced by
the shipped code under *any* GPU condition -- a schema difference, not a measurement one, and
invisible until something tried to regenerate it. This run is what closes that.

## 3. Reproduction

| | Check | Run | Result |
|---|---|---|---|
| `[x]` | Clean-room, first pass | 2026-08-30 | [`CLEANROOM.md`](CLEANROOM.md) §2 |
| `[x]` | Clean-room, second pass covering the four producers written afterwards | 2026-08-31 | All five derived tables byte-identical; found one defect, a verdict literal that had drifted between a row builder and its summary. [`CLEANROOM.md`](CLEANROOM.md) §2b |
| `[x]` | Every committed table has a producer | continuous, `tests/test_repo_hygiene.py` | **One table has none, and it is the documented exemption.** `mps_seed_spread.csv` ([`PROVENANCE.md`](PROVENANCE.md) §1.4). This row read "one documented exemption" when the true figure was three, because the gate behind it searched for the file name anywhere in `scripts/` and `src/` and so counted the three scripts that *read* `coverage_by_arm.csv` as producing it. The gate was corrected, which exposed `coverage_by_arm.csv` and `coverage_by_arm_seeds.csv` as producerless; both were carried as strict xfails until `scripts/run_coverage_arms.py` was written, and `TABLES_WHOSE_PRODUCER_IS_MISSING` is now empty. Both tables reproduce byte-identically ([D-132](decisions.md)) |
| `[x]` | Walkthrough against the committed tables | `make walkthrough` | Every assertion holds |
| `[x]` | **Seventh clean-room pass, the final tree, under a lock** | clone at `66d4fad`, 2026-09-12 12:07 to 15:22, one execution enforced by an atomic lock ([D-160](decisions.md)) | **All stages exit 0, zero errors or tracebacks.** 29 tables byte-identical, 5 differing only in a wall-clock column, **0 differing in a measured value**; 31 of 35 rewritten and the four preserved each announce themselves. Both PDFs differ by **exactly one character** -- `159` against `160`, the decision count from the single post-clone commit. `circuits.pdf` reproduces across a re-run of its producer, testing the [D-159](decisions.md) fix rather than asserting it |
| `[x]` | Full-scale sweep re-run | `make seedsweep` in the clean-room clone, 2026-09-11, idle GPU | **All 16 fits reproduce ROC AUC, average precision and final loss to the last digit** -- `max \|delta\| = 0.000e+00` on each -- and the derived summary table is byte-identical. 12.1 GPU-hours |

## 4. The two governing documents, and how thinly they had been checked

An earlier draft of this file said the Assessment Criteria and the Terms and Conditions had
"never been read against this submission". That was wrong, and correcting it is the point of the
section. [`COMPLIANCE_CHECKLIST.md`](COMPLIANCE_CHECKLIST.md) section D already carried five
rows against the Terms.

What is true is thinner and more useful: **five rows against thirteen sections**, and at least
one of those rows was satisfied by evidence that did not cover the clause. D3 ticks "no
confidential or proprietary third-party information" and reasons entirely about *datasets* --
"both datasets are public, and neither is redistributed". It was blind to two teammates' names
and personal email addresses sitting in `CREDENTIALS.md`, which is exactly the third-party
information that clause is about, and which was found and removed on 2026-08-31 by looking for
something else.

That is the same failure as the predictions row, which ticked an Expected Outcome while citing a
file format the portal rejects: **a checklist row is only as good as the evidence column, and a
tick with the wrong evidence is worse than an unticked row**, because it stops anyone looking
again.

**The clause-by-clause pass has since been run** (2026-09-02). `COMPLIANCE_CHECKLIST.md`
section D went from 5 rows to 22, at least one per clause and one per lettered sub-clause of §4,
and three of the five originals were restated because their reasoning did not reach the clause
they ticked. The first attempt at that pass wrote 19 rows and described them as "one per
clause" while still omitting §1, §4.5 and §13. Two rows are **Needs confirmation and belong
to the author, not to this repository**: whether any current engagement makes him a "Contractor
of ... Challenge enterprise sponsors" under §2, and whether to register a trading name, since
§4.2 permits Resonance to publicise the entry by team name and withholds personal names.

Null clauses now carry a row saying they are null. The reason is the failure below: an absent
row and a considered-and-null row are indistinguishable to the next reader.

## 5. External artefacts reconciled against the submission

| | Artefact | Result |
|---|---|---|
| `[x]` | IBM `ibm_fez` hardware run | VQC and QSVM, June 2026, in a public repository. Restored to the team section after a compression round dropped it |
| `[x]` | QIntern 2026 Project 12 role | Programme and role named; the supporting repository is **private**, so no locator into it is cited. Third-party names and contact addresses removed |
| `[x]` | Yale Peaked Hackathon placement and score | 13th of 549 at 450 of 550. Two separate corrections. **The placement is a team's** -- the leaderboard row and the CV both read *MerQury*, and section 8 of the proposal presented it without a qualifier two lines under the words "Single-person team". Now attributed. Separately, the section claimed the solved count follows from the score, which is arithmetically false -- ten subsets of the weights sum to the missing 100 -- and paired the 69-qubit figure with a method the repository's own verifier reports as failing on it ([D-102](decisions.md), [D-105](decisions.md)) |
| `[x]` | unitaryHACK merges | **Three** pull requests, not the two the proposal implied: `cuda-quantum#4693`, `QuEST#783`, `cuda-quantum#4751`. Merge dates and branches read from the GitHub API on 2026-09-02. Two closed bounty issues, USD 200; the third is a follow-on and is not a bounty ([D-102](decisions.md)) |
| `[x]` | CV sole-proprietorship date | **Settled 2026-09-02.** The filing records a business start date of **1 August 2026**, which is what the submission and the portfolio use. The CV's "Jul 2026 - Present" is the error, and it is outside this repository. **The CV should be corrected before it is sent anywhere alongside this submission**, since §8 links the portfolio and a reviewer who opens both sees the discrepancy |

## 5c. Where the portfolio and the proposal disagree

§8 links the portfolio, so a reviewer can open both. Four differences are worth the author's
attention, and **none of them is a defect in this submission**. All four live in files outside
this repository and are recorded here because the submission points at them.

| | Portfolio says | The submission says | Which is right |
|---|---|---|---|
| Yale field size | "#13 / **550 teams** (team score 450)" | "13th of **549** at 450 of 550" | **The submission.** [D-061](decisions.md) settled this from the organiser's own leaderboard: the pagination control reads `11-20 of 549`, so 549 is the field, and 550 is the maximum score. The portfolio uses 550 for both, which is the exact conflation D-061 was written to remove |
| Yale method reach | "9/10 peaked-circuit challenges **up to 69 qubits**" | "matrix-product-state runs are exact to **60** qubits and degrade above it" | **Both, about different things.** Nine of ten were solved and the largest was 69 qubits; separately, the *saved tensor-network* runs are exact to 60. [D-059](decisions.md) established that no artefact attributes the 69-qubit answer to a tensor network, which is why §8 makes the narrower claim. Not a contradiction, but a reviewer reading them together may not see that |
| QPoland title | "Quantum Graph Kernels for Molecular Classification" | "Quantum kernels" | **Neither is wrong; the submission is deliberately narrower.** [D-114](decisions.md) reverted a "graph kernels" label because [D-057](decisions.md) removed that attribution for want of any checkable implementation. The CV uses a third form, "Quantum-Inspired Graph Kernels" |
| Qiskit credential | "Qiskit Advocate (Feb 2026)" | "IBM Certified Quantum Computation using Qiskit v2.X Developer - Associate" | **Both are held; §8 prints the one a reviewer can check.** The Advocate programme issues no badge, so it has no URL; the certification has one, recorded in [CREDENTIALS.md](CREDENTIALS.md) §8. §8 has room for a single Qiskit line and spends it on the verifiable credential. **Author action, outside this repository:** add the certification to the portfolio, or a reviewer opening the link finds a credential the proposal does not name and vice versa |

**Needs confirmation, outside this repository.** The portfolio describes the graph-kernel work as
"10-fold stratified cross-validation" and the CV as "nested 5-fold". One of the two is wrong, or
both were run and neither says so. The submission cites neither figure, so nothing here depends
on it.


## 5b. The seven open questions this study started with

The earliest planning round listed seven things that could not be settled at the time and fixed
how each would be handled. All seven are now closed, and they are listed here because a reader
should be able to see that the questions the work opened with were answered rather than
forgotten.

| # | The question | How it closed |
|---|---|---|
| 1 | Whether a local simulator satisfies "use Amazon Braket" | Settled by the statement itself: hardware execution is "not expected nor required" and entries that do not use it "are not penalized". Section 7 states that both arms are simulator-only and that no number comes from a QPU |
| 2 | Whether the transaction stream is beta-mixing | Not claimed, as planned. The empirical proxies that were to stand in for it -- a block permutation test and a martingale wealth curve -- were **not run**, and amendment A7 discloses that rather than leaving the plan's intention to imply they were |
| 3 | Whether a temporal split of ULB leaves fraud in every fold | Moot: ULB was withdrawn by amendment A9, and the statement permits using one or two of its three datasets |
| 4 | Whether a label-maturity buffer would shrink calibration below the degeneracy floor | Measured and closed early: no trailing censoring was detected, and the floor never binds -- the smallest headroom in `degeneracy.csv` is 1,122 rows |
| 5 | Whether the GPU wheel for a particular quantum framework carries `sm_120` | The plan's rule was to claim no GPU acceleration until it was shown to work. It was never shown and is never claimed: that framework appears nowhere in this repository |
| 6 | A published AUPRC table that a paywall blocked, to be verified by hand before citing | The strongest form of closure: it was never cited. The rule was "verify before quoting"; nothing quotes it, so nothing rests on it |
| 7 | The issuer-side cost of a false decline, which is not public | Parameterised, as planned. Section 1 uses the public figures the challenge statement supplies and leaves the issuer-side multiplier to the bank |

Item 2 is the one worth reading twice. The plan's answer was to substitute empirical proxies for
an unprovable assumption; the proxies were then not executed. What makes that acceptable is that
the appendix says so by name, rather than the plan's intention being allowed to read as a
result.

## 6. Standing limits

Three things this apparatus cannot check, stated so they are not mistaken for covered:

* **A figure is checked by eye.** No gate catches a label collision, an unreadable axis or a
  misleading scale. Every figure in the submission has been looked at; that is the whole
  assurance.
* **A formula against the literature is a judgement.** [`REFERENCE_IMPLEMENTATION.md`](REFERENCE_IMPLEMENTATION.md)
  records which five were read line by line against their cited source and which thirteen were
  checked only for attribution.
* **The upload is manual.** Everything up to `submission/portal/` is verified; what is actually
  uploaded, and to which slot, is not.
