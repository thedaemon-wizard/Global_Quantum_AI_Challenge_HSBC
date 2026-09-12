# Compliance checklist — 2026 Global Quantum + AI Challenge, HSBC track

Every requirement stated in the four official documents, with where it is satisfied and how it
was checked. Verified 2026-08-30 against the PDFs downloaded from the portal that day, and
revised 2026-09-02 (portal locators, the canonical-version question, and section D clause by
clause).

Sources, with the locator each was retrieved from. The URLs were read off the live portal in a
browser on 2026-09-02, not reconstructed:

| Short name | Document | Locator |
|---|---|---|
| **GUIDE** | `2026-04-06-Phase-1-Submission-Guidelines-VF.pdf` | `/wp-content/uploads/2026/04/` |
| **CRIT** | `2026-04-06-Assessment-Criteria-VF.pdf` | `/wp-content/uploads/2026/04/` |
| **TERMS** | `2026-04-06-Terms-and-Conditions-VF.pdf` | `/wp-content/uploads/2026/04/` |
| **STMT** | `HSBC-Challenge-Statement-vFinalRevised.pdf` | `/wp-content/uploads/2026/08/` |

**The challenge statement's canonical version is settled.** Two files exist locally --- an
18-page `HSBC-Challenge-Statement-vF-1.pdf` and the 16-page `vFinalRevised` --- and which one
the organisers regard as operative could not be established from inside this repository. The
portal answers it: the HSBC challenge card links
`/wp-content/uploads/2026/08/HSBC-Challenge-Statement-vFinalRevised.pdf`, under an **August**
upload path where the other three documents sit under April. **`vFinalRevised` is the operative
document**, which is the one this checklist and the submission are written against.

Status values are **met**, **not met**, or **Needs confirmation** — the last meaning the requirement is
understood but satisfying it depends on something outside this repository.

---

## A. Format and packaging (GUIDE §5)

| | Requirement | Status | How it is checked |
|---|---|---|---|
| A1 | Concept proposal at most **6 pages** | met, 6/6 | `check_pdf.py --max-pages 6`, run by `make pdf` |
| A2 | Appendices at most **3 additional pages** | met, 3/3 | same, `--max-pages 3` |
| A3 | A4 or US Letter | met, A4 | `check_pdf.py` reads the media box of every page |
| A4 | PDF format | met | both artefacts are PDF |
| A5 | Minimum **10 pt** font | met, body 10.00 pt; 2.8 % of proposal characters and 0.4 % of appendix characters are mathematical sub/superscripts below it, smallest 4.98 pt, which the checker exempts. GUIDE §5 states no exemption, so whether a reviewer counts scripts against a body floor is **Needs confirmation** | `check_pdf.py` reads rendered size from the text matrix, not the declared size, and prints the sub-floor fraction rather than hiding it |
| A6 | All text in English | met | no CJK codepoint in either PDF's extracted text |
| A7 | File size at most **20 MB** | met, about 12 MB total, largest file about 11 MB | `du -sb submission/portal/` and `ls -S submission/portal/ | head -1`. Stated to two significant figures on purpose: the exact byte count moves every time the PDFs are rebuilt, and an exact figure here was already stale by 19 KB. The margin against the cap is a factor of 1.7 on the total and 1.8 on the largest file, so no rebuild can put it in doubt. GUIDE §5 does not say whether the cap is per file or total; both readings are satisfied |
| A8 | Uploaded through the Challenge portal | Needs confirmation, and the only one left | the upload is manual. The live form was read on 2026-08-30 and reads "0 uploaded, 5 slots left" with allowed formats PDF, PNG, JPG, WEBP, GIF, PY, JSON, JS, XLS, XLSX, CSV, DOC, DOCX — exactly what `assemble_submission.py` enforces |

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
| B10 | §4.4 Link to a public code repository — **optional**, "if relevant" | §8 names the repository | provided, not required |

## C. Challenge statement requirements (STMT)

**On the numbering.** Two different rows both carried the identifier `C18` -- §4.1 on published
quantum results and §4.2 on hardware execution -- so the section had 22 rows and 21 usable
identifiers, and the two resolved items at the foot of this file pointed one row off their
intended target. The rows after the duplicate were renumbered on 2026-09-06 and those two
pointers were moved with them: what this file called `C19 inference latency` is now `C20`, and
`C21 qubit count and circuit depth` is now `C22`. The rows themselves and their verdicts are
unchanged.

| | Requirement | Status | Note |
|---|---|---|---|
| C1 | §4.1 Report **ROC AUC** | met | proposal §3, appendix §A4 |
| C2 | §4.1 Report **AUPRC** | met | proposal §3, appendix §A4 |
| C3 | §4.1 Report **F1** | met | appendix §A4 — computed since the first run and unreported until now, [D-046](decisions.md) |
| C4 | §4.1 Report **precision** | met | appendix §A4 |
| C5 | §4.1 Report **recall** | met | appendix §A4 |
| C6 | §4.1 Report a **confusion matrix** | met | appendix §A4 |
| C7 | §4.1 Benchmark against published results and report comparison methodology | met | §3 gives the 1st-place IEEE-CIS solution both ways -- ROC AUC 0.9459 on the private leaderboard, and about 0.9363 in time-based cross-validation over the training file **with their UID feature removed, which they report as worth about +0.011 AUC** -- against this study's strict forward holdout of that training file, and names the difference in task. The UID qualifier was in `REFERENCES.md` (DS-5) and in this row, and had dropped out of the shipped PDF; restored 2026-09-02 ([D-125](decisions.md)) |
| C8 | §5.2 Output: fraud probability | met | continuous score in [0, 1], one row per held-out transaction, uploaded as `HSBC-predictions.csv` from [`predictions.csv`](../results/tables/predictions.csv). Previously evidenced only by `scores_*.parquet`, which the portal does not accept as a format |
| C9 | §5.2 Output: binary prediction | met | the `predicted_fraud` column of the same file, with the three-valued `decision` it derives from. Previously evidenced only by a count in appendix §A4 |
| C10 | §5.2 Output: **feature attribution** | met | `attribution.csv` and `attribution_examples.csv`, TreeSHAP with an asserted additivity check; proposal §5 reports the finding — [D-048](decisions.md) |
| C11 | §5.3 Include a classical baseline | met | gradient-boosted model, five seeds |
| C11a | The portal's OBJECTIVE line says "using Amazon Braket" | met, and the statement is softer than the portal summary | The statement asks participants to use Braket but puts **"Any quantum or quantum-inspired framework (Qiskit, PennyLane, Cirq, cuQuantum, ITensor, Braket, etc.)" in scope**, and states that "full end-to-end model training or inference on quantum hardware is not expected nor required". Braket's `LocalSimulator` is the reference implementation the fidelity kernel is checked against ([`parity.csv`](../results/tables/parity.csv), proposal §7); the tensor-network arm is PyTorch, which the in-scope list permits. Recorded because the portal's one-line summary reads as a harder requirement than the statement it summarises |
| C12 | §5.3 Document class-imbalance handling | met | protocol §2; threshold and band budgets rather than resampling |
| C13 | §5.3 Feature selection for quantum approaches | met | eight features in the band arm, stated in §4 |
| C14 | §4.2 State the **sample count used for quantum execution** | met | proposal §7 |
| C15 | §4.2 Subsampling must be **stratified** | met | `screen_kernels.py` stratifies on the label |
| C16 | §4.2 Robustness under distribution shift | met | this is the study's central result, §3 |
| C17 | §4.2 and §5.2 Describe the quantum approach, encoding strategy and circuit design | met **now** | The earlier tick pointed at §4, which described the screen dimensions ("encoding x qubits x bandwidth x entanglement") without naming a single feature map. The three the screens actually ran -- `z`, `zz` and dense-angle -- and the two entanglement patterns are now named in §4, and the transpiled depths and two-qubit counts are in §7 ([D-117](decisions.md)) |
| C18 | §4.1 The two published quantum results the statement itself cites are engaged on their own terms | met **now**, and one ground changed. §4 previously dismissed the Deloitte/AWS precision as quoted "with no recall or base rate". The blog post was read on 2026-09-02 and **the base rate is reported**: ULB, 284,807 transactions, 492 frauds, 0.172 % — twenty times below this task's. The objection is now the accurate and stronger one, that a thresholded precision is not comparable across base rates differing by that factor. The post also gives a threshold sweep (0.87/0.89/0.92) and a classical arm (0.83/0.84/0.86) that the statement's summary omits ([D-129](decisions.md)) |
| C19 | §4.2 Hardware execution | not applicable | simulator only; STMT §5.4 states participants who do not execute on hardware are not penalised, and §7 says so |
| C20 | §5.3 "Good to have": inference latency | met | E13 measured: `results/tables/latency.csv`, reported in proposal §5 and [ENVIRONMENT.md](ENVIRONMENT.md) §5 |
| C21 | §5.3 "Good to have": training time comparison | met | §5 reports fit times for both arms |
| C22 | §5.3 "Good to have": qubit count and circuit depth | met | `circuits.csv`, measured by decomposing every screened encoding to a portable basis; proposal §7 |
| C23 | §5.2 Report "any observed quantum improvement and under what conditions" | met | **None observed, and both halves are stated.** §1 says "No quantum advantage of any kind"; §4's *What would change the quantum conclusion* names the conditions — an encoding demonstrably outside the RBF family at a bandwidth that does not concentrate would pass the same screens, and the tensor-network comparison needs an evaluation block powered for cross-family noise. Added 2026-09-12: the requirement had no row, though it was answered |
| C24 | §5.3 Hardware: error mitigation, and DM1 for noise-aware prototyping | not applicable, and adjudicated rather than skipped | The error-mitigation bullet sits in a list introduced by "Teams using hardware are encouraged to", so it is conditioned on hardware use; DM1 is offered to "prototype noise-aware circuits **before** hardware execution", and there is none here to precede. A noise study would price the noise sensitivity of a method the screens rejected on **noiseless** grounds before it reached the task, which cannot change a conclusion — [D-097](decisions.md), *No hardware run, and the statement is the reason*. The one mandatory clause in that block is the sample count, which is C14 |
| C25 | §5.3 Hardware: native gate sets and connectivity, to minimise transpilation | met for the gate set, deliberately not for connectivity | `report_circuits.py` transpiles every screened encoding to `rz, sx, x, cx`, which every superconducting Braket backend decomposes close to. Connectivity is **deliberately** not modelled: a depth quoted against one machine's coupling map would imply a hardware run this study did not make, and the script says so at its `PORTABLE_BASIS` definition |
| C26 | §5.3 "Good to have": simulator versus hardware comparison | not applicable | No hardware run, so there is nothing to compare against. §5.4 does not penalise it. What is offered instead is a **four-way simulator** cross-check — exact statevector, Braket `LocalSimulator`, Aer CPU and Aer GPU, agreeing to 2.887e-15 over 12 comparisons (`parity.csv`) — which is this study's stated substitute |
| C27 | §5.4 Scope: binary classification (fraud vs legitimate) | met, and extended | The statement's binary output is delivered exactly: `predicted_fraud` is an integer in {0, 1} and `fraud_probability` spans [0, 1], both per held-out transaction. The three-valued rule is an **extension** of that scope, not a substitute — §1 opens by arguing an issuer has three actions rather than two, and the binary column is derived from the same decision |

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
| P6 | No emoji or decorative characters, and no Japanese, in any document or source file | met, **and now enforced** | The evidence here used to read "scanned", which is a statement about one afternoon rather than a property of the tree — and both conventions had in fact drifted: 10 occurrences of the warning glyph U+26A0 and 28 of the status marker written in Japanese. Both are swept, and `tests/test_repo_hygiene.py` now fails on CJK, kana, fullwidth forms, dingbats and the emoji planes across every tracked text file. The ranges are deliberately narrow: em dashes, section signs, Greek letters, ceilings and accented surnames are legitimate and non-ASCII, and a gate that rejected them would be deleted rather than obeyed |
| P7 | No mention of an AI assistant where the project does not require it | **met**, 2026-09-02. `git grep -nwiE 'agent|assistant|claude|anthropic|copilot'` over every tracked file now returns only this row. The four that remained were `.gitignore:11` and `:14`, where the ignore rule moved to `.git/info/exclude` so the path is still ignored but not published, and two lines of `decisions.md` prose. The earlier evidence read "nothing else in the tree mentions one", which the scan refuted; it was recorded as unmet rather than ticked, and is ticked now because the scan passes. The row itself is the single remaining occurrence in the tree, and is exempt by necessity: a requirement cannot state what it forbids without naming it |
| P8 | Git history carries no assistant attribution, and reads as the sole author it claims | **met**, re-derived 2026-09-02. No `Co-authored-by` trailer anywhere in the history. `git log` carries **three author strings, all the same person** (confirmed by the author): two name variants on `amon.koike@daemons.jp`, one a typo for the GitHub handle, plus `a-koike <amon06251994@gmail.com>`. A `.mailmap` collapses them to one canonical identity, so `git shortlog -sne` reports a single author while the raw history is untouched, and `tests/test_repo_hygiene.py` pins that. **GitHub was already correct**: its contributor view matches by email, both addresses are registered to the same account, and the API reports one contributor. Commit counts are deliberately not written here -- an earlier version of this row carried two that were stale within a day ([D-125](decisions.md)) |
| P9 | Every long run reports progress and writes a readable log | **met** | `src/hsbcfraud/progress.py`, `pytest tests/test_progress.py`. The evidence cell used to be the test alone, which read `met` only because the list of long runs was a hand-written tuple that had gone stale. It is now **derived** — every script whose AST calls `load_ieee_cis` is a long run — and that derivation immediately exposed three that were not: `audit_labels.py`, `run_explain.py` and `run_power.py` each loaded all 590,540 rows behind no progress destination. They were carried in `SILENT_WITHOUT_A_DESTINATION` as strict xfails rather than allowlisted, so an entry could not outlive the defect, and all three have since been wrapped in `run_log`. The registry is now an empty dict, kept rather than deleted so the next such script has somewhere to be recorded. `screen_kernels.py` was the fourth and cleared the same way |
| P10 | No content, filename or citation from the private planning folder appears in the repository | met, **and now enforced** | The evidence here read "scanned across every tracked file" and named a protective rule in `.gitignore` — and **half of that had gone stale**: [D-125](decisions.md) moved the rule to `.git/info/exclude` so the path would stop appearing in a published file, and this row still cited the old location. `tests/test_repo_hygiene.py` now fails on any tracked file referencing the directory. It checks the *path*, not the note filenames: naming those here would put the strings the rule protects into a public repository, so the gate would leak what it guards. The notes also sit outside this working tree in a parent that is not a git repository, so a clone cannot draw them in ([D-131](decisions.md)) |
| P11 | Biographical credentials are checked against the artefacts, not only fenced | met | [CREDENTIALS.md](CREDENTIALS.md) records each claim and its backing. Three unsupported claims removed, one ownership and one method attribution corrected, one rank removed and then restored against the organiser's own leaderboard, and one hardware claim withdrawn — [D-057](decisions.md), [D-059](decisions.md), [D-060](decisions.md), [D-061](decisions.md), [D-062](decisions.md) |
| P12 | The QIntern credential names its benchmark and does not extrapolate to payments | met | proposal §8 states CIC-IoT2023 and that what transfers is the machinery, not a number. Regression found and fixed, [D-056](decisions.md) |
| P13 | Hardware statements are scoped to this study rather than to the author | met | proposal §7 says no number here comes from a QPU, which is what the study supports. The earlier wording, "the author has not used a QPU", was refuted by his own public benchmark repository (VQC and QSVM on `ibm_fez`, 2026-06-19) — corrected, and the hardware run now appears in §8 where it scores |
| P14 | The challenge statement's internal inconsistency is resolved rather than guessed | met | the "~24,000 rows" figure appears in `HSBC-Challenge-Statement-vF-1.pdf` and not in the revised final; the organisers removed it, so no query is needed |
| P15 | Credentials that cannot be verified externally are marked as such rather than dropped or asserted | met | Every §8 row carries its verification route in [CREDENTIALS.md](CREDENTIALS.md) §1. The Qiskit row is the worked case, and it turned twice: a static fetch returned only the badge **title**, so the holder and date were first recorded as unverified and §8 printed no year. Opening the same URL in a **browser** rendered all of them — issued to Amon Koike by IBM Professional Certification on 6 September 2026 — so the limit was the tool rather than the evidence, the row now carries every field, and §8 prints the year ([§8](CREDENTIALS.md#8-c3--the-ibm-qiskit-credential)) |
| P16 | Supporting material not cited in the proposal is inventoried | met | CV, project addendum, portfolio and repositories listed in [CREDENTIALS.md](CREDENTIALS.md) §5, so brevity is a choice rather than an absence |
| P17 | The README opens with a diagram of the method, and the diagram's numbers come from the results tables rather than being typed | met | `pytest tests/test_repo_hygiene.py -k "method_figure or mermaid"`. Two figures, two gates: `test_the_method_figure_is_generated_and_reachable` fails if `method.png` is absent from the build, missing from `make_figures.main`'s builder tuple, or no longer embedded in the README and the proposal; `test_mermaid_node_numbers_agree_with_the_split_table` fails if the pipeline diagram is deleted, and re-derives every block size it prints from `splits.csv`. Neither cell records that someone looked |
| P18 | Acceptance of the Kaggle competition rules is recorded, and recorded at the strength the evidence supports | met | `pytest tests/test_repo_hygiene.py -k kaggle_rules`. [PROVENANCE.md](PROVENANCE.md) §1.1 carries a **Rules accepted** row marked *attested by the author, not verifiable from this repository*. The gate pins the qualifier, not just the row: the failure it guards is the attestation quietly firming into a claim of verification during a later edit, next to two rule quotations that **are** independently checkable at a URL |
| P19 | The README stays short enough to be read end to end | met | `pytest tests/test_repo_hygiene.py -k readme_stays_short`. 469 lines against a 600-line ceiling. P3 already requires supporting documents to be linked rather than inlined, but that is a property of the `docs/` tree and held while the README itself grew past four hundred lines — nothing measured the length the requirement is about. The ceiling is a tripwire with deliberate headroom, not a measured optimum, and the remedy when it fires is to move a section into `docs/` rather than to compress prose |

## F. Assessment criteria (CRIT), and what the page budget actually spends on each

Sections A to E track what the submission must *contain*. This one tracks what it is *scored*
on, which no other section did -- the criteria were mapped in prose in the README and in D8's
note, and never as a checklist.

Share of body text measured from `proposal.pdf` by locating each section heading in the extracted
text. It is a proxy for pages, not a page count. **Re-measured 2026-09-11**, after section 4
gained the feature-asymmetry and in-band-floor disclosures and section 8 lost the Qiskit
certification; the 2026-09-02 figures are in the second column so the drift is visible.

| | Criterion | Weight | Answered in | 2026-09-02 | **2026-09-11** | Gap |
|---|---|---|---|---|---|---|
| F1 | Problem Relevance and Impact | **25 %** | §1 | 16.5 % | **16.5 %** | **−8.5** |
| F2 | Technical Approach and Innovation | **25 %** | §2 and §4 | 30.9 % | **32.1 %** | +7.1 |
| F3 | Feasibility | **20 %** | §5 | 14.4 % | **14.8 %** | −5.2 |
| F4 | Validation Plan | **15 %** | §3 and §6 | 27.4 % | **27.0 %** | **+12.0** |
| F5 | Hybrid / Cross-Domain | **5 %** | §7 | 5.2 % | **5.1 %** | +0.1 |
| F6 | Team Capability | **10 %** | §8 | 4.6 % | **4.5 %** | −5.5 |

**Read the gaps with one caveat, which is large.** The mapping is not a partition. §3 reports
the results, and those results are simultaneously the evidence for Technical (the two quantum
arms), for Problem and Impact (what the rule does to a day of traffic) and for Validation (the
held-out read). Attributing all of §3 to Validation is what produces the +12.4, and a reviewer
scoring Technical will read §3 too. The honest statement is that the *reported* allocation
over-weights the sections that carry measurements and under-weights the two that carry argument,
§1 and §5.

**What is worth acting on, and what is not.** F1 at 16.5 % against the joint-highest weight was
carried here as the one real finding. **It was re-read line by line on 2026-09-11 and the finding
does not survive the reading.** Section 1 already does all of this, with no sentence spare:

* frames the decision as three actions rather than two, and says why the middle one is what makes
  the problem tractable at issuer volume;
* engages four conformal papers *specifically* -- Singh, Zhu, Nayak and Bushara, Chen -- and says
  what each does and does not condition on, rather than gesturing at a literature;
* states what a successful PoC would demonstrate, and admits this study meets it only at loose
  levels ($\alpha \ge 0.10$);
* prices a day of traffic: **86.72 % approved, 9.35 % stepped up, 3.93 % declined**, and names
  the middle number as the whole argument;
* gives three impact consequences, **none of which needs a quantum result**;
* quotes the statement's own cost figures and then **attaches none of its own**, because the
  issuer-side cost is not public;
* says both quantum arms failed, and how they failed differently;
* closes with what is *not* claimed.

The low share is a property of density, not of absence -- the section is the most compressed in
the document. Adding words to reach 25 % would dilute an argument that currently has no filler,
and the mapping is not a partition anyway: a reviewer scoring Impact reads the traffic split in
§1 **and** the held-out results in §3.

F6 at 4.5 % is likewise not a finding: §8 is a credentials list, every line bound to a verified
artefact in [`CREDENTIALS.md`](CREDENTIALS.md), and it got *shorter* on 2026-09-11 when a
Foundational-level certificate was removed for arguing less than the results beside it
([D-156](decisions.md)). Padding it would add words rather than evidence. F5 is at its weight.
F2's surplus already narrowed once, when a paragraph duplicated between §4 and §5 was removed
([D-112](decisions.md)), and widened again on 2026-09-11 for two disclosures that had to be made
([D-155](decisions.md), [D-157](decisions.md)) -- surplus spent on honesty rather than on
argument.

**Both PDFs are at their limits**, 6/6 and 3/3, and neither may grow. Any move toward F1 has
to displace something, and the project's rule is that displacement goes to
[`RESULTS.md`](RESULTS.md) rather than to a smaller font, which the format gate rejects anyway.

## D. Terms and conditions (TERMS)

**This section had five rows against thirteen clauses**, and two of them were satisfied by
reasoning that did not reach the clause. It is now clause by clause, one row per section and
one per lettered sub-clause of §4 — 22 rows. The first pass wrote 19 and still omitted §1,
§4.5 and §13, which is the same failure one iteration smaller. Where a clause imposes
nothing on the entrant, the row says so rather than being omitted, because an absent row and a
null row are indistinguishable to the next reader — which is how D3 stayed ticked through three
separate findings ([D-104](decisions.md)).

| | Requirement | Status |
|---|---|---|
| D0 | §1 The Challenge — programme structure and Resonance's right to modify it | null for the entrant. Recorded because D10 argues from it: §1 describes Phase 1 as responding to "published enterprise problem statements", which is what makes quoting the statement here unobjectionable |
| D1 | §2 Eligibility — the enumerated categories are "startups, research teams, universities, and industry teams", which does not name individuals | met, but not by the phrase the earlier row used. "Single-person team permitted" appears nowhere in the Terms. What covers the entry is the second bullet, "the legal authority to enter into these Terms **on behalf of yourself** or your team/organization", which contemplates an individual entrant; and the registered sole proprietorship is in any case an organisation. §8 records the registration |
| D1a | §2 "Not be subject to any applicable sanctions, export controls, or trade restrictions" | met. Japan is not a sanctioned jurisdiction; the submission carries no controlled technology, ships no cryptographic implementation, and uses public datasets and open-source software throughout. The quantum work is simulator-only (§7), so no controlled hardware access is transferred |
| D1b | §2 "Employees and Contractors of Resonance or its Challenge enterprise sponsors may not submit proposals as Participants" | **met, confirmed by the author 2026-09-02**: no current contractual relationship with Resonance Alliance Inc. / The Quantum Insider or with HSBC. The bar is a disqualification criterion and the entrant is a sole proprietor who takes inbound work, so it was carried as an open question rather than assumed; it is now answered by the only party who could answer it |
| D2 | §3 Original work, no third-party IP infringement | met; all dependencies are listed in `NOTICE` with their licences, and the conformal implementation was written from the published papers rather than adapted from the unlicensed team repository. §8 said "which this submission reuses", which read as the opposite and contradicted the appendix; corrected ([D-106](decisions.md)) |
| D3 | §3 No confidential or proprietary third-party information **disclosed without permission** | met **now**. The earlier tick reasoned only about datasets — "both datasets are public, and neither is redistributed" — which is true and was blind to the actual exposure twice: two teammates' names and email addresses ([D-094](decisions.md)), then their unpublished prose quoted at length from a private repository ([D-104](decisions.md)). Both are removed. The datasets reasoning still holds and is now one leg of three |
| D3a | §3 "You have the right to submit the materials provided" | met. Every artefact in the five portal files is the author's own or derived from the two datasets under their stated terms; nothing originates in the private team repository |
| D3b | §3 "Your submission complies with all applicable laws and regulations" | met. Regulation is cited structurally and never as a compliance assertion — `REGULATORY_SOURCES.md` states this explicitly, and the submission makes no high-risk classification claim under the EU AI Act |
| D4 | §4.1 Participant retains all IP | null for the entrant; nothing to do. Worth recording that it is what makes the Apache-2.0 release on 2026-09-15 permissible: §4.1 reserves the right to "license, or otherwise exploit your work", and §4.3 confirms sponsors acquire no ownership by access |
| D5 | §4.2 Non-exclusive licence to Resonance for assessment and promotion | null for the entrant. Note the direction of the two halves: the licence Resonance takes is limited to administration, evaluation and judging, and it expressly does **not** extend to commercial exploitation of the underlying IP |
| D5a | §4.2 "Resonance will not share the full content ... unless it is already publicly available" | **switched off deliberately, and the trade is recorded.** The repository goes public on 2026-09-15, the same day as the upload, so this protection lapses by its own terms from that moment. That is the intended choice — the proposal's verifiability rests on a reader being able to open the repository — but it is a choice, not an oversight ([D-107](decisions.md)) |
| D5b | §4.2 "Reference your **team name**, submission title, and a summary description ... Personal names of individual team members will not be disclosed without your prior written consent" | **resolved 2026-09-02.** The entry now carries the trading name **Quantum Daemons** in both title blocks and in §8, tied to the sole-proprietor registration. **The portal has no team-name field on the submission form** — it is four hidden inputs, a file picker and three buttons, read live on 2026-09-02. The organisation name is an *account-level* field set at registration (`reg_company_type`, `reg_company_name`), which is what §4.2's "organizational affiliations ... company, or institution" reaches. **Needs confirmation, and the author's to check while signed in: whether that registration field already holds the trading name.** The document is under our control and now states it; the account field is not |
| D6 | §4.3 Sponsor access, review-only, no ownership acquired | null for the entrant |
| D7 | §4.4 Post-Challenge engagement governed by a separate bilateral agreement | null for Phase I. Relevant to Phase II: §5 already records that the IEEE-CIS competition rules permit non-commercial research use only, so a commercial PoC cannot reuse this file. The protocol ports; the data does not |
| D7a | §4.5 Resonance Program IP — platform, evaluation frameworks, schemas, templates, branding | null for the entrant; nothing in this submission derives from any of them |
| D8 | §5 Evaluation criteria may be adjusted; all decisions final | null for the entrant. The published criteria are CRIT's six weighted headings (Problem Relevance & Impact 25 %, Technical Approach & Innovation 25 %, Feasibility 20 %, Validation Plan 15 %, Hybrid / Cross-Domain 5 %, Team Capability 10 %); section B maps the proposal sections that answer them via GUIDE §4.3, which is their near-image. Section C tracks STMT, not CRIT. Nothing here can bind Resonance's right to adjust them |
| D9 | §6 Prizes may be modified; taxes are the recipient's | null for Phase I |
| D10 | §7 Confidentiality of sponsor material, **conditioned on being selected as a finalist** | not yet attached. §1 calls the Phase I statements "published", so quoting the statement here is unobjectionable. **It attaches at Phase II, and this repository is public** — any non-public sponsor material received as a finalist must not enter it. Recorded here because the constraint arrives with the selection, not with the work |
| D11 | §8 Data and privacy — Resonance's handling of the entrant's data | null for the entrant. Note the onward direction the Terms are silent on: the upload transfers 115,534 rows to a company in Ontario. See E1a |
| D12 | §9–§11 Disclaimers, code of conduct, modification and cancellation | null |
| D13 | §12 Governing law: Ontario, Canada | null; recorded so a later reader does not assume Japanese law governs |
| D14 | §13 Contact | null; `hello@resonance.holdings` and the portal |

## E. Data licensing

| | Requirement | Status |
|---|---|---|
| E1 | IEEE-CIS: competition licence, not redistributed | met | `scripts/` load from a local copy and the file is not in the repository; `.gitignore` carries an anchored `/datasets/` pattern. Competition rules §7.B forbid redistribution, so the archive is **never** attached to a submission or committed. **Its SHA-256 is**, per member, verified on every load — a digest redistributes nothing and is the strongest identity statement the licence permits ([D-127](decisions.md)) |
| E2 | ULB: database under **ODbL**, contents under **DbCL v1.0** | met; `NOTICE` always stated both, and `README.md`, `PROVENANCE.md`, `A3-reproduction.tex` and `SUBMISSION_CHECKLIST.md` now do too |
| E3 | Third-party software licences recorded | met; `NOTICE` |
| E4 | cuQuantum proprietary surface disclosed and scoped | met; installed only by the optional `make venv-gpu`, used only by E11's parity cross-check, no scientific figure depends on it — [D-050](decisions.md) |
| E1a | What the uploaded `predictions.csv` actually contains, against both the IEEE-CIS competition licence (E1) and TERMS §8 | met. The file carries a row index, a model score, the decision it implies, the binary label that decision derives, the day offset and the three thresholds — eight columns, and **no feature values and no dataset field**. It is model output over a public benchmark's row ordering, not a redistribution of the Competition Data, and it carries nothing that could identify a cardholder. Both the licence question and the privacy question are answered by the same fact about the column set |

---

## Open items, by priority

| Priority | Item | Proposed fix |
|---|---|---|
| ~~HIGH~~ | ~~C10 feature attribution absent~~ | Done. `scripts/run_explain.py` computes it over the calibration band; the result is that 75.8 % of the in-band model is one anonymised card identifier, reported in proposal §5 |
| ~~MEDIUM~~ | ~~C22 qubit count and circuit depth are not stated~~ | Done. `scripts/report_circuits.py` measures both for every screened encoding; proposal §7 reports the maxima |
| ~~MEDIUM~~ | ~~E2 ULB licence stated incompletely in four documents~~ | Done. `NOTICE` was already correct; the four prose statements now match it |
| ~~LOW~~ | ~~C20 inference latency~~ | Done 2026-08-30, **re-measured 2026-09-06** ([D-147](decisions.md)): classical scorer 0.154 ms p50 per authorisation, quantum kernel 76.843 ms. The figures below are the 2026-08-30 reading, kept as written; the kernel tail takes 0.759 of the residual issuer-side budget, so the band is what makes it affordable. The first run measured an OpenMP thread-barrier default rather than model cost and was discarded ([D-063](decisions.md)) |
| Needs confirmation | A8, B1 | The upload and the lead-contact field are portal actions, not repository state |
