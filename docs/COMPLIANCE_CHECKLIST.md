# Compliance checklist — 2026 Global Quantum + AI Challenge, HSBC track

Every requirement stated in the four official documents, with where it is satisfied and how it
was checked. Verified 2026-08-30 against the PDFs downloaded from the portal that day.

Sources, with the locator each was retrieved from. The URLs were read off the live portal in a
browser on 2026-09-02, not reconstructed:

| Short name | Document | Locator |
|---|---|---|
| **GUIDE** | `2026-04-06-Phase-1-Submission-Guidelines-VF.pdf` | `/wp-content/uploads/2026/04/` |
| **CRIT** | `2026-04-06-Assessment-Criteria-VF.pdf` | `/wp-content/uploads/2026/04/` |
| **TERMS** | `2026-04-06-Terms-and-Conditions-VF.pdf` | `/wp-content/uploads/2026/04/` |

**The challenge statement's canonical version is settled.** Two files exist locally --- an
18-page `HSBC-Challenge-Statement-vF-1.pdf` and the 16-page `vFinalRevised` --- and which one
the organisers regard as operative could not be established from inside this repository. The
portal answers it: the HSBC challenge card links
`/wp-content/uploads/2026/08/HSBC-Challenge-Statement-vFinalRevised.pdf`, under an **August**
upload path where the other three documents sit under April. **`vFinalRevised` is the operative
document**, which is the one this checklist and the submission are written against.
| **STMT** | `HSBC-Challenge-Statement-vFinalRevised.pdf` |

Status values are **met**, **not met**, or **要確認** — the last meaning the requirement is
understood but satisfying it depends on something outside this repository.

---

## A. Format and packaging (GUIDE §5)

| | Requirement | Status | How it is checked |
|---|---|---|---|
| A1 | Concept proposal at most **6 pages** | met, 6/6 | `check_pdf.py --max-pages 6`, run by `make pdf` |
| A2 | Appendices at most **3 additional pages** | met, 3/3 | same, `--max-pages 3` |
| A3 | A4 or US Letter | met, A4 | `check_pdf.py` reads the media box of every page |
| A4 | PDF format | met | both artefacts are PDF |
| A5 | Minimum **10 pt** font | met, body 10.00 pt | `check_pdf.py` reads rendered size from the text matrix, not the declared size |
| A6 | All text in English | met | no CJK codepoint in either PDF's extracted text |
| A7 | File size at most **20 MB** | met, 0.58 MB total | `du` over `submission/portal/` |
| A8 | Uploaded through the Challenge portal | 要確認, and the only one left | the upload is manual. The live form was read on 2026-08-30 and reads "0 uploaded, 5 slots left" with allowed formats PDF, PNG, JPG, WEBP, GIF, PY, JSON, JS, XLS, XLSX, CSV, DOC, DOCX — exactly what `assemble_submission.py` enforces |

## B. Required content (GUIDE §4)

| | Requirement | Where | Status |
|---|---|---|---|
| B1 | §4.1 Team profile: name, lead contact, member description, prior quantum experience | proposal §8, plus the portal's own form fields | met in the document; the contact detail is a portal field |
| B2 | §4.2 Problem statement identified | title block and §1 | met |
| B3 | §4.3.1 Problem framing, and why quantum is credible | §1 | met |
| B4 | §4.3.2 Technical approach, with paradigm named | §2, §4 — quantum kernel (gate-based) and matrix product state (quantum-inspired), both stated | met |
| B5 | §4.3.3 Feasibility and resource requirements, assumptions and constraints stated | §5 | met — added this round, see [D-045](decisions.md) |
| B6 | §4.3.4 Expected impact, with quantitative targets | §1 | met — added this round |
| B7 | §4.3.5 Validation plan, metrics defining success | §6 | met — added this round |
| B8 | §4.3.6 Hybrid / cross-domain integration and its rationale | §7 | met |
| B9 | §4.3.7 Team capability | §8 | met |
| B10 | §4.4 Link to a public code repository | §8 names the repository | met |

## C. Challenge statement requirements (STMT)

| | Requirement | Status | Note |
|---|---|---|---|
| C1 | §4.1 Report **ROC AUC** | met | proposal §3, appendix §A4 |
| C2 | §4.1 Report **AUPRC** | met | proposal §3, appendix §A4 |
| C3 | §4.1 Report **F1** | met | appendix §A4 — computed since the first run and unreported until now, [D-046](decisions.md) |
| C4 | §4.1 Report **precision** | met | appendix §A4 |
| C5 | §4.1 Report **recall** | met | appendix §A4 |
| C6 | §4.1 Report a **confusion matrix** | met | appendix §A4 |
| C7 | §4.1 Benchmark against published results and report comparison methodology | met | proposal §3 gives the 1st-place IEEE-CIS solution both ways -- ROC AUC 0.9459 on the private leaderboard and about 0.9363 in time-based cross-validation over the training file -- against this study's strict forward holdout of that training file, and names the difference in task |
| C8 | §5.2 Output: fraud probability | met | continuous score in [0, 1], one row per held-out transaction, uploaded as `HSBC-predictions.csv` from [`predictions.csv`](../results/tables/predictions.csv). Previously evidenced only by `scores_*.parquet`, which the portal does not accept as a format |
| C9 | §5.2 Output: binary prediction | met | the `predicted_fraud` column of the same file, with the three-valued `decision` it derives from. Previously evidenced only by a count in appendix §A4 |
| C10 | §5.2 Output: **feature attribution** | met | `attribution.csv` and `attribution_examples.csv`, TreeSHAP with an asserted additivity check; proposal §5 reports the finding — [D-048](decisions.md) |
| C11 | §5.3 Include a classical baseline | met | tuned gradient-boosted model, five seeds |
| C11a | The portal's OBJECTIVE line says "using Amazon Braket" | met, and the statement is softer than the portal summary | The statement asks participants to use Braket but puts **"Any quantum or quantum-inspired framework (Qiskit, PennyLane, Cirq, cuQuantum, ITensor, Braket, etc.)" in scope**, and states that "full end-to-end model training or inference on quantum hardware is not expected nor required". Braket's `LocalSimulator` is the reference implementation the fidelity kernel is checked against ([`parity.csv`](../results/tables/parity.csv), proposal §7); the tensor-network arm is PyTorch, which the in-scope list permits. Recorded because the portal's one-line summary reads as a harder requirement than the statement it summarises |
| C12 | §5.3 Document class-imbalance handling | met | protocol §2; threshold and band budgets rather than resampling |
| C13 | §5.3 Feature selection for quantum approaches | met | eight features in the band arm, stated in §4 |
| C14 | §4.2 State the **sample count used for quantum execution** | met | proposal §7 |
| C15 | §4.2 Subsampling must be **stratified** | met | `screen_kernels.py` stratifies on the label |
| C16 | §4.2 Robustness under distribution shift | met | this is the study's central result, §3 |
| C17 | §4.2 Describe encoding strategy and circuit design | met | §4 |
| C18 | §4.2 Hardware execution | not applicable | simulator only; STMT §5.4 states participants who do not execute on hardware are not penalised, and §7 says so |
| C19 | §5.3 "Good to have": inference latency | met | E13 measured: `results/tables/latency.csv`, reported in proposal §5 and [ENVIRONMENT.md](ENVIRONMENT.md) §5 |
| C20 | §5.3 "Good to have": training time comparison | met | §5 reports fit times for both arms |
| C21 | §5.3 "Good to have": qubit count and circuit depth | met | `circuits.csv`, measured by decomposing every screened encoding to a portable basis; proposal §7 |

## C-bis. Repository presentation (standing requirements for this project)

Not imposed by the official documents; recorded here because they are conditions on the
artefact a reviewer receives and nothing else checks them.

| | Requirement | Status | How it is checked |
|---|---|---|---|
| P1 | README opens with a Deliverables section | met | `README.md` §1, ahead of the claims table |
| P2 | Deliverables names each uploaded file, its format and its contents | met | §1 table, five rows |
| P3 | Supporting documents are linked rather than inlined, so the README stays readable | met | every `.md` in `docs/` is linked from README §1 — asserted as a property rather than a count, because the count drifted once already |
| P4 | Development environment and measured benchmark cost are recorded | met | [ENVIRONMENT.md](ENVIRONMENT.md): hardware, pinned versions, wall clock per stage, circuit depth |
| P5 | Results are shown as tables and figures, with LaTeX/MathJax notation for the mathematics | met | README §2 carries the overview figure, §4 the certificate and §5 the three result figures, with every table in [RESULTS.md](RESULTS.md); `$R(\lambda)$`, `$\alpha_{\mathrm{FN}}$` and the Beta-Binomial law render as mathematics |
| P6 | No emoji or decorative characters in any document or source file | met | scanned across `*.md`, `*.tex`, `*.py`, `*.yaml`, `Makefile` |
| P7 | No mention of an AI assistant where the project does not require it | met | D-037 was rewritten as a methodological entry; nothing else in the tree mentions one |
| P8 | Git history carries no assistant attribution | met | two human authors, no `Co-authored-by` trailer anywhere in the history (32 commits when checked, 2026-08-30) |
| P9 | Every long run reports progress and writes a readable log | met | `src/hsbcfraud/progress.py`, `pytest tests/test_progress.py` |
| P10 | No content, filename or citation from the private planning folder appears in the repository | met | scanned across every tracked file. `.gitignore` retains a protective rule naming the directory so that one cannot be committed by accident; nothing else names it |
| P11 | Biographical credentials are checked against the artefacts, not only fenced | met | [CREDENTIALS.md](CREDENTIALS.md) records each claim and its backing. Three unsupported claims removed, one ownership and one method attribution corrected, one rank removed and then restored against the organiser's own leaderboard, and one hardware claim withdrawn — [D-057](decisions.md), [D-059](decisions.md), [D-060](decisions.md), [D-061](decisions.md), [D-062](decisions.md) |
| P12 | The QIntern credential names its benchmark and does not extrapolate to payments | met | proposal §8 states CIC-IoT2023 and that what transfers is the machinery, not a number. Regression found and fixed, [D-056](decisions.md) |
| P13 | Hardware statements are scoped to this study rather than to the author | met | proposal §7 says no number here comes from a QPU, which is what the study supports. The earlier wording, "the author has not used a QPU", was refuted by his own public benchmark repository (VQC and QSVM on `ibm_fez`, 2026-06-19) — corrected, and the hardware run now appears in §8 where it scores |
| P14 | The challenge statement's internal inconsistency is resolved rather than guessed | met | the "~24,000 rows" figure appears in `HSBC-Challenge-Statement-vF-1.pdf` and not in the revised final; the organisers removed it, so no query is needed |
| P15 | Credentials that cannot be verified externally are marked as such rather than dropped or asserted | met | Qiskit Advocate is confirmed by an email the author holds; the programme is at Tier 0 and no badge exists, so no URL is offered — [CREDENTIALS.md](CREDENTIALS.md) §1 |
| P16 | Supporting material not cited in the proposal is inventoried | met | CV, project addendum, portfolio and repositories listed in [CREDENTIALS.md](CREDENTIALS.md) §5, so brevity is a choice rather than an absence |

## D. Terms and conditions (TERMS)

**This section had five rows against thirteen clauses**, and two of them were satisfied by
reasoning that did not reach the clause. It is now clause by clause. Where a clause imposes
nothing on the entrant, the row says so rather than being omitted, because an absent row and a
null row are indistinguishable to the next reader — which is how D3 stayed ticked through three
separate findings ([D-104](decisions.md)).

| | Requirement | Status |
|---|---|---|
| D1 | §2 Eligibility — the enumerated categories are "startups, research teams, universities, and industry teams", which does not name individuals | met, but not by the phrase the earlier row used. "Single-person team permitted" appears nowhere in the Terms. What covers the entry is the second bullet, "the legal authority to enter into these Terms **on behalf of yourself** or your team/organization", which contemplates an individual entrant; and the registered sole proprietorship is in any case an organisation. §8 records the registration |
| D1a | §2 "Not be subject to any applicable sanctions, export controls, or trade restrictions" | met. Japan is not a sanctioned jurisdiction; the submission carries no controlled technology, ships no cryptographic implementation, and uses public datasets and open-source software throughout. The quantum work is simulator-only (§7), so no controlled hardware access is transferred |
| D1b | §2 "Employees and Contractors of Resonance or its Challenge enterprise sponsors may not submit proposals as Participants" — a disqualification bar with no row before now | **要確認**, and the answer is the author's alone. The bar reaches sponsors as well as the organiser, and the entrant is a sole proprietor who takes inbound consulting work, so "Contractor of a Challenge enterprise sponsor" is a live category rather than an academic one. Nothing in this repository can settle it. **To confirm before upload: no current or recent engagement with Resonance Alliance Inc. / The Quantum Insider, HSBC, or any other listed Challenge sponsor** |
| D2 | §3 Original work, no third-party IP infringement | met; all dependencies are listed in `NOTICE` with their licences, and the conformal implementation was written from the published papers rather than adapted from the unlicensed team repository. §8 said "which this submission reuses", which read as the opposite and contradicted the appendix; corrected ([D-106](decisions.md)) |
| D3 | §3 No confidential or proprietary third-party information **disclosed without permission** | met **now**. The earlier tick reasoned only about datasets — "both datasets are public, and neither is redistributed" — which is true and was blind to the actual exposure twice: two teammates' names and email addresses ([D-094](decisions.md)), then their unpublished prose quoted at length from a private repository ([D-104](decisions.md)). Both are removed. The datasets reasoning still holds and is now one leg of three |
| D3a | §3 "You have the right to submit the materials provided" | met. Every artefact in the five portal files is the author's own or derived from the two datasets under their stated terms; nothing originates in the private team repository |
| D3b | §3 "Your submission complies with all applicable laws and regulations" | met. Regulation is cited structurally and never as a compliance assertion — `REGULATORY_SOURCES.md` states this explicitly, and the submission makes no high-risk classification claim under the EU AI Act |
| D4 | §4.1 Participant retains all IP | null for the entrant; nothing to do. Worth recording that it is what makes the Apache-2.0 release on 2026-09-15 permissible: §4.1 reserves the right to "license, or otherwise exploit your work", and §4.3 confirms sponsors acquire no ownership by access |
| D5 | §4.2 Non-exclusive licence to Resonance for assessment and promotion | null for the entrant. Note the direction of the two halves: the licence Resonance takes is limited to administration, evaluation and judging, and it expressly does **not** extend to commercial exploitation of the underlying IP |
| D5a | §4.2 "Resonance will not share the full content ... unless it is already publicly available" | **switched off deliberately, and the trade is recorded.** The repository goes public on 2026-09-15, the same day as the upload, so this protection lapses by its own terms from that moment. That is the intended choice — the proposal's verifiability rests on a reader being able to open the repository — but it is a choice, not an oversight ([D-107](decisions.md)) |
| D5b | §4.2 "Reference your **team name**, submission title, and a summary description ... Personal names of individual team members will not be disclosed without your prior written consent" | **要確認, author decision.** The entry has no team name: both title blocks carry the Challenge's own programme name. For a single-person entry, the two things Resonance may publicise are a team name that does not exist and a title; and the one identifier that does exist, the author's personal name, is the one §4.2 withholds absent written consent. **If the author wants the entry publicised, a trading name should go in the portal's team-name field** |
| D6 | §4.3 Sponsor access, review-only, no ownership acquired | null for the entrant |
| D7 | §4.4 Post-Challenge engagement governed by a separate bilateral agreement | null for Phase I. Relevant to Phase II: §5 already records that the IEEE-CIS competition rules permit non-commercial research use only, so a commercial PoC cannot reuse this file. The protocol ports; the data does not |
| D8 | §5 Evaluation criteria may be adjusted; all decisions final | null. Section C tracks the criteria as published |
| D9 | §6 Prizes may be modified; taxes are the recipient's | null for Phase I |
| D10 | §7 Confidentiality of sponsor material, **conditioned on being selected as a finalist** | not yet attached. §1 calls the Phase I statements "published", so quoting the statement here is unobjectionable. **It attaches at Phase II, and this repository is public** — any non-public sponsor material received as a finalist must not enter it. Recorded here because the constraint arrives with the selection, not with the work |
| D11 | §8 Data and privacy — Resonance's handling of the entrant's data | null for the entrant. Note the onward direction the Terms are silent on: the upload transfers 115,534 rows to a company in Ontario. See E1a |
| D12 | §9–§11 Disclaimers, code of conduct, modification and cancellation | null |
| D13 | §12 Governing law: Ontario, Canada | null; recorded so a later reader does not assume Japanese law governs |

## E. Data licensing

| | Requirement | Status |
|---|---|---|
| E1 | IEEE-CIS: competition licence, not redistributed | met; `scripts/` load from a local copy and the file is not in the repository |
| E2 | ULB: database under **ODbL**, contents under **DbCL v1.0** | met | `NOTICE` always stated both; `README.md`, `PROVENANCE.md`, `A3-reproduction.tex` and `SUBMISSION_CHECKLIST.md` now do too |
| E3 | Third-party software licences recorded | met; `NOTICE` |
| E4 | cuQuantum proprietary surface disclosed and scoped | met; installed only by the optional `make venv-gpu`, used only by E11's parity cross-check, no scientific figure depends on it — [D-050](decisions.md) |
| E1a | What the uploaded `predictions.csv` actually contains, against both the IEEE-CIS competition licence (E1) and TERMS §8 | met. The file carries a row index, a model score, the decision it implies, the day offset and the three thresholds — **no feature values and no dataset field**. It is model output over a public benchmark's row ordering, not a redistribution of the Competition Data, and it carries nothing that could identify a cardholder. Both the licence question and the privacy question are answered by the same fact about the column set |

---

## Open items, by priority

| Priority | Item | Proposed fix |
|---|---|---|
| ~~HIGH~~ | ~~C10 feature attribution absent~~ | Done. `scripts/run_explain.py` computes it over the calibration band; the result is that 75.8 % of the in-band model is one anonymised card identifier, reported in proposal §5 |
| ~~MEDIUM~~ | ~~C21 qubit count and circuit depth are not stated~~ | Done. `scripts/report_circuits.py` measures both for every screened encoding; proposal §7 reports the maxima |
| ~~MEDIUM~~ | ~~E2 ULB licence stated incompletely in four documents~~ | Done. `NOTICE` was already correct; the four prose statements now match it |
| ~~LOW~~ | ~~C19 inference latency~~ | Done 2026-08-30. Classical scorer 0.084 ms p50 per authorisation, quantum kernel 96.577 ms; the kernel tail takes 0.759 of the residual issuer-side budget, so the band is what makes it affordable. The first run measured an OpenMP thread-barrier default rather than model cost and was discarded ([D-063](decisions.md)) |
| 要確認 | A8, B1 | The upload and the lead-contact field are portal actions, not repository state |
