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
| `[x]` | HSBC Challenge Statement (vFinalRevised, 16 pp.) | 2026-08-31, all seven sections | Four Expected Outcomes each answered by a named artefact; two named requirements were unmet and are now met (class-imbalance handling, the sponsor's own cited comparators) |
| `[x]` | Challenge Statement, revised against original | 2026-08-31, word-by-word diff | The revision **adds URLs** and three words. No requirement changed. An audit agent reported the two as textually identical, which the diff does not support |
| `[x]` | Phase 1 Submission Guidelines | 2026-08-31 | 6 pages + 3 appendix, A4, 10 pt floor, 20 MB per file, five slots -- all asserted by `scripts/check_pdf.py` and `scripts/assemble_submission.py` rather than by eye |
| `[x]` | Assessment Criteria | 2026-08-31 | See §4 below |
| `[x]` | Terms and Conditions | 2026-08-31 | See §4 below |

## 2. Live surfaces, re-verified in a browser

| | Surface | Checked | Result |
|---|---|---|---|
| `[x]` | Portal dashboard and challenges list | 2026-08-31, signed in | Five challenges, all "Not submitted" |
| `[x]` | HSBC challenge submission panel | 2026-08-31 | "0 uploaded, 5 slots left"; formats PDF, PNG, JPG, WEBP, GIF, PY, JSON, JS, XLS, XLSX, CSV, DOC, DOCX -- unchanged from the previous check |
| `[x]` | Programme roadmap | 2026-08-31 | Phase I closes 15 Sep 2026; review 16 Sep - 14 Nov; Phase II PoC 17 Nov 2026 - 28 Feb 2027; winners 30 Apr 2027 |
| `[x]` | GitHub rendering of every markdown document | 2026-08-31 | All spans render; enforced by `scripts/check_markdown_math.py` against GitHub's own `POST /markdown` and MathJax |
| `[ ]` | The repository URL printed in the proposal title block | **manual, before upload** | 404s until the repository is made public on 2026-09-15 |
| `[ ]` | The upload itself, and the lead-contact field | **manual, before upload** | Cannot be automated |

## 3. Reproduction

| | Check | Run | Result |
|---|---|---|---|
| `[x]` | Clean-room, first pass | 2026-08-30 | [`CLEANROOM.md`](CLEANROOM.md) §2 |
| `[x]` | Clean-room, second pass covering the four producers written afterwards | 2026-08-31 | All five derived tables byte-identical; found one defect, a verdict literal that had drifted between a row builder and its summary. [`CLEANROOM.md`](CLEANROOM.md) §2b |
| `[x]` | Every committed table has a producer | continuous, `tests/test_repo_hygiene.py` | One documented exemption, recorded in [`PROVENANCE.md`](PROVENANCE.md) §1.4 |
| `[x]` | Walkthrough against the committed tables | `make walkthrough` | Every assertion holds |
| `[ ]` | Full-scale sweep re-run | not re-run | 13 GPU-hours; the frozen manifest covers it and no claim depends on re-deriving it |

## 4. What the two previously unread documents established

Recorded here rather than in the compliance checklist because the finding is about the
*verification*, not about a requirement mapping.

**要確認 until the audit completes.** The Assessment Criteria and the Terms and Conditions had
never been read against this submission before 2026-08-31, through five audit rounds. That is
itself the finding worth recording: the submission was checked exhaustively against its own
internal consistency and against the challenge statement, and the two documents that govern how
it is *scored* and what the entrant *undertakes* were not in the loop. Results are folded into
[`COMPLIANCE_CHECKLIST.md`](COMPLIANCE_CHECKLIST.md) and [`FACTCHECK_LOG.md`](FACTCHECK_LOG.md)
as they are confirmed.

## 5. External artefacts reconciled against the submission

| | Artefact | Result |
|---|---|---|
| `[x]` | IBM `ibm_fez` hardware run | VQC and QSVM, June 2026, in a public repository. Restored to the team section after a compression round dropped it |
| `[x]` | QIntern 2026 Project 12 role | Programme and role named; the supporting repository is **private**, so no locator into it is cited. Third-party names and contact addresses removed |
| `[x]` | Yale Peaked Hackathon placement and score | 13th of 549 at 450 of 550 |
| `[x]` | unitaryHACK merges | CUDA-Q and QuEST |
| `[ ]` | CV sole-proprietorship date | **要確認**: the CV reads "Jul 2026 - Present"; the portfolio and the filing say 1 August 2026. The submission uses 1 August. A reviewer comparing the two would see the discrepancy, and it is the author's to settle |

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
  records which five were read line by line against their cited source and which fifteen were
  checked only for attribution.
* **The upload is manual.** Everything up to `submission/portal/` is verified; what is actually
  uploaded, and to which slot, is not.
