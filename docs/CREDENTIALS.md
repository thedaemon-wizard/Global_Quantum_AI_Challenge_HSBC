# Credential provenance

Every biographical claim in the proposal's team section, with what backs it and how it was
checked. This file exists because of a structural gap: those claims are wrapped in a
`\Record{}` macro, which declares a figure as the author's own record rather than a measurement
from this study — and therefore **exempts it from every numeric gate in this repository**. The
exemption is correct. It also meant nothing checked them until 2026-08-30, and when they were
checked, six needed changing.

Verified 2026-08-30; C3 re-verified and replaced 2026-09-06 (§8). Where a claim cannot be settled
from an artefact it is marked **Needs confirmation** rather than asserted or quietly dropped — and
**no Needs confirmation remains open in this file**. The three that were carried here were all closed
on 2026-08-30 by going to the artefact:

| Item | Outcome |
|---|---|
| `conformal_calibrate.py` authorship "not independently confirmable" | **Wrong.** A teammate's own handoff attributes it and records reproducing its numbers to six decimals (§2) |
| Yale peak bitstrings for problems 3–9 "differ from the published set" | **Wrong.** Same nine answers, opposite bit order: P3–P9 are exact string reversals, P1 is a palindrome, P2 is the one row transcribed un-reversed (§3) |
| Sole-proprietor filing date | **Settled: 1 August 2026** (§1, and §7 for a CV that says otherwise) |

Two of the three were errors in this file's own earlier auditing, not in the credential. Both
made the author's record look weaker than the evidence supports.

**Why §8 is not a fourth open marker.** The Qiskit row records that the badge holder and issue
date could not be read from Credly's static HTML. That is a statement about *this file's reach*,
not an unsettled question about the credential: a reviewer settles it by clicking the badge URL,
which is precisely what a **Needs confirmation** marker exists to flag as *impossible* for the
reader. So it is written as a standing caveat with the route attached, on the same footing as C1,
C2 and C6, rather than as a marker that will never close.

---

## 1. What the proposal claims

| # | Claim | Status | Backed by |
|---|---|---|---|
| C1 | Registered sole proprietor in Japan | **date confirmed by the author, filing not checkable here** | Filed **1 August 2026**, confirmed 2026-08-30. The proposal states the status without a date, so nothing in the submission turns on it; see §7 for a discrepancy in a document outside this repository |
| C2 | B.S. Physics, Tokyo Denki University | not checkable here | diploma, held by the author |
| C3 | IBM Certified Quantum Computation using Qiskit v2.X Developer - Associate | **badge title verified; holder and issue date not verifiable from outside the browser** | see [§8](#8-c3--the-ibm-qiskit-credential) |
| C4 | QIntern 2026 Project 12 (QWorld), Team A — split-conformal calibration and results freeze | **verified, and independently attributed by a teammate** | see §2. Both the Day-15 calibration module and the Days 26–27 freeze are named as the author's in handoff documents committed by a teammate, who re-ran each and reproduced the numbers |
| C5 | Yale Peaked Hackathon 2026, rank 13 of 549 entries at 450 of 550 points, team MerQury; MPS runs exact to 60 qubits | **rank and score verified on the organiser's leaderboard; the solved count and the method boundary are the author's own record** | see §3. The two have different sources and are no longer stated as though they had one |
| C6 | QPoland 2025 runner-up, team The Cats Cradle | **team confirmed by the author, placement not checkable here** | no artefact on this machine; the placement stands with no technical detail attached. The team name is the author's own record, confirmed 2026-09-02 ([D-111](decisions.md)); this row carried the placement bare for two days after the submission attributed it, which is the same gap that left the Yale placement unattributed |
| C7 | unitaryHACK 2026, three pull requests merged into NVIDIA CUDA-Q and QuEST, two closing bounty issues | **verified by merged pull request; bounty status verified against the maintainers' label** | see §4 |

## 2. C4 — QIntern 2026 Project 12

Source: the Team A working repository, which is **private** and not this one. A reviewer
cannot follow any locator into it, so the paragraphs below record what was checked and by whom
rather than pointing at commits nobody outside the team can open. Commit identifiers and the
local paths they were read from have been removed for that reason; the checks themselves were
run and are described.

**What this section does and does not reproduce.** The Terms and Conditions warrant that a
submission contains no "confidential or proprietary information of any third party disclosed
without permission" (§3), and this repository becomes public on 2026-09-15. The source
repository is private and its handoff documents are two teammates' unpublished writing. So the
attributions below are **described, not quoted**, and the teammates are identified by role
rather than by name or contact address. Nothing is lost by it: a verbatim quotation from a
repository a reviewer cannot open carries no more weight than a description of the same
document, because neither can be checked from outside. An earlier version of this section
asserted that no third party's information was reproduced here while carrying four block
quotations of exactly that kind ([D-104](decisions.md)).

The one exception is the team's own four-word caveat on the provisional status of its numbers,
which the proposal also prints. It is retained because it *limits* the author's claim rather
than supporting it, and removing a third party's disclaimer while keeping the credit it
qualifies would be the wrong trade.

**What is the author's own.** The split-conformal calibration module and the results freeze.
The freeze commit (2026-07-30) first introduces four Week-5 scripts under his authorship ---
disentanglement, the results table, the coverage figure and the freeze itself --- and it is the
last of these that computes the SHA-256 manifest rows.

**The calibration module is independently attributed, and the earlier Needs confirmation was wrong.**
the calibration module does arrive in a bulk "Initial commit", so
per-file provenance is absent — but that commit was made by a **teammate**, on whose account
the repository is hosted, not by the author, and three days later the same teammate committed
a Days 16--17 handoff document that attributes exactly this artefact to the author. That
document, in the teammate's own commit, does three things: it names the Day-15 calibration
module as the author's and builds on it; it records a full-suite re-run on the teammate's own
machine, 35 of 35 passing, itemised by which contributor wrote which test; and it records
re-running `conformal_calibrate.py` and reproducing the author's α = 0.05 figures --- the
CIC quantile, the coverage and the false-zero-day rate --- as an exact match, with the other two
corpora agreeing to the sixth decimal.

That is stronger than an attribution: a teammate re-ran the module and reproduced its numbers to
six decimals, in a commit he signed. An earlier version of this file called the authorship
"not independently confirmable here", which was wrong — it was confirmable, in a file two
directories away that had not been read.

**What is the team's.** The paper's conformal-coverage proposition belongs to **Team A**, a
three-person team, and two of the weekly READMEs say so directly: Team A, not any individual,
owns the paper's **Proposition 3 (conformal coverage)**. (An earlier week's README uses the
looser phrase "statistical-guarantee centrepiece", and an earlier version of this file cited
that one for the stronger reading.) The Beta-Binomial coverage harness and the McNemar/Holm
protocol are a teammate's Days 16--17: git puts both scripts in a commit by another
contributor. The proposal previously read as though both were the author's; §8 now attributes
them to the team.

**The 135 checks are exactly 135, at the freeze, and the count is the teammate's.** Counted at
the freeze commit. The repository is at 182 today by its own Week-5 README --- 183 is the raw `def test_` count,
which over-counts --- so the proposal says "green at that freeze".
The freeze itself is independently verified in a second teammate's handoff document,
a Days 28--30 handoff document, which records that every number verifies: the full suite at 135
passing plus 8 subtests, all 39 scalars returned by `freeze_results.py --reproduce` to 1e-9, and
the whole reproduced on a *different* environment --- Python 3.10 with scikit-learn 1.7.2 ---
to identical bytes and a manifest byte-identical to the author's v1.0.

**The numbers are provisional and the proposal says so.** All eleven Week-5 artefacts carry a
**run-level** source stamp reading *dummy*, pointing at the placeholder score interface. Six
**row-level** entries in two of them read *real* — the XGBoost classical baseline, one per
dataset, which is genuinely measured; only the quantum arm is a placeholder. An earlier version
of this file said "every artefact", which understated the work.

The team's own README says the quantum side is still on the dummy interface and that every
QS-Net number that week is therefore provisional, in the four words the proposal reuses:
*"protocol-final, numbers-provisional."* The coverage result therefore
verifies the **harness**, over five seeds and three intrusion-detection corpora, not a model.
The proposal uses the team's own phrase and claims the machinery rather than a result.

**Domain, per the project's own rule.** The benchmark is intrusion detection, not payments, and
the proposal says so. Nothing is extrapolated to card fraud.

## 3. C5 — Yale Peaked Hackathon 2026

Settled against the organiser's own leaderboard rather than against any repository.

**Primary source: the BlueQubit leaderboard**, `app.bluequbit.io/hackathons/wSvCWg8f38spoXX3`,
page 2, scope **World view**, read 2026-08-30 and re-read the same day against a screenshot.
The event's own title on that page is **Yale Peaked Hackathon 2026**.

| Rank | Team | Score | Time penalty | Submissions |
|---|---|---|---|---|
| **13** | **MerQury** | **450 / 550** | 28.36 h | 35 |

**Two denominators exist, they are one apart, and conflating them is the trap.**

* **550 is the maximum score.** Every row reads `450/550` in a column headed SCORE, and the ten
  problems carry weights 10, 20, ... 100, which sum to 550.

  An earlier version of this bullet added that "a score of 450 is every problem but the last, so
  nine of ten follows from the score itself." **That inference is false.** The missing 100 points
  are made up by ten different subsets of the weights --- $\lbrace 100 \rbrace$,
  $\lbrace 10, 90 \rbrace$, ... , $\lbrace 10, 20, 30, 40 \rbrace$ --- so 450 is consistent
  with six, seven, eight or nine solved. The count of nine is true, but it comes from the
  author's own repository, not from the leaderboard, and the two documents that claimed it was
  derivable were claiming an independence they did not have ([D-105](decisions.md)).
* **549 is the field.** The pagination control reads `11–20 of 549`, so the leaderboard holds
  549 ranked entries and the rank of 13 is **13 of 549**.

The near-coincidence of 549 and 550 is why this took three passes to settle. An earlier draft
wrote "#13 of 550 teams", conflating the two. The correction that followed established that 550
is the score maximum — correct — but concluded from that the rank had no citable denominator,
which was wrong: 549 was on the same screen, in the pagination. **Rank 13 of 549 entries,
scoring 450 of 550 points** is the full and verified statement, and it is what the proposal now
says.

Entries, not teams: some rows carry team names (`buQeyes`, `The Logical Qubit`) and others
personal names, so the field is counted in leaderboard entries.

Ranks 11 to 20 are all tied at 450 and separated only by time penalty, which is why the
author's own repository can say "Top 10 (all tied at 450 pts)" while the rank is 13. Both are
true. The scope selector offers **World view** and **Yale View**; the cited rank is World view,
the wider of the two.

**The method is the author's own contribution, and this is checkable.** He is a contributor to
`github.com/roman-bagdasarian/Peaked-Circuits` (two contributions, authored as
`thedamon-wizard <amon.koike@daemons.jp>` and `Amon K.`). Pull request #1, from branch
`marginal_attack_by_amon`, merged 2026-04-10, adds `marginal_attack.py` (132 lines) and a
`data/Yale_Quantum_2026/P5_soft_rise.qasm.txt` fixture. The file's own description reads
*"MPS marginal attack: determine peak bitstring from single-qubit Z expectation values"* and it
drives `AerSimulator(method="matrix_product_state")` with a bond-dimension argument. So the
matrix-product-state marginal attack named in the proposal is his, and an earlier audit that
called that attribution unsupported was wrong.

**Certificate.** A Certificate of Completion naming Amon Koike, dated 2026, issued by Yale in
collaboration with YQuantum, is held by the author and is not in this repository. It certifies completion; the placement is certified by the leaderboard
above.

**The bitstring discrepancy is closed: it is an endianness artefact, and it runs in the
author's favour.** An earlier version of this file carried a Needs confirmation saying the author's working
repository recorded different peak bitstrings for problems 3 to 9 than the third-party
repository. The two records are the **same nine answers in opposite bit order**:

| Problem | Qubits | Public vs private |
|---|---|---|
| 1 | 4 | identical — `1001` is a palindrome |
| 2 | 12 | identical — the one row the third-party README transcribed un-reversed |
| 3 – 9 | 30, 40, 50, 60, 42, 58, 69 | **exact string reversals**, character for character |

Not one bit differs beyond the reversal. The clinching artefact is one the author committed
himself: `data/Yale_Quantum_2026/P5_soft_rise.qasm.txt` is the output of a single run and prints
both lines,

```
Little-Endian: 00011011001101000001010110110100101010011000011001
Big-Endian   : 10011000011001010100101101101010000010110011011000
```

the first of which is the third-party README's P5 row and the second the author's. Both
"sets" are two lines of one simulation, because the tooling emits `peak` and `peak[::-1]`
by construction.

The author's copy is the **submitted** orientation: his `solve_all.py` marks big-endian,
$q_{n-1}$-first, as correct, and it records that an endianness error cost six failed P5
submissions before it was fixed. So the earlier Needs confirmation had the direction backwards — his
repository is the correctly oriented one, and the coincidence that P1 is a palindrome while P2
happens to be transcribed the other way is exactly what made a raw comparison look like
"agrees on 1 and 2, differs on 3 to 9".

The same private table independently corroborates the field size, recording problem 10 as
"unsolved by all **549** participants".

The team is registered as **MerQury** on the leaderboard; **PeakQubit** is a label in the
author's own working repository only.

## 4. C7 — unitaryHACK 2026

| Contribution | Pull request | State |
|---|---|---|
| Recursive Quantum Shannon decomposition, MLIR unitary synthesis | `NVIDIA/cuda-quantum#4693` | merged into `main` 2026-06-15, carrying the maintainers' own `unitaryhack-accepted` label. **Bounty** |
| GPU backend change, projector and small-allocation path | `QuEST-Kit/QuEST#783` | merged by the project lead into `devel` 2026-06-22. **Bounty** |
| Unitary synthesis driven through the dialect-conversion framework | `NVIDIA/cuda-quantum#4751` | merged into `main` 2026-07-02. A follow-on to #4693. **Not a bounty**, and titled without the `[unitaryhack]` prefix the other two carry |

All three are merged upstream and independently checkable by pull request; the merge dates and
target branches above were read from the GitHub API on 2026-09-02 rather than from the author's
own record. The two bounties paid **USD 200** in total. The decomposition is genuinely
self-recursive rather than a name reused for something flatter.

The third is listed because the proposal's earlier text said only "merged into CUDA-Q and
QuEST", and this table said only two. **An understatement in a credential is a defect in the
same way an overstatement is** -- a reviewer who checks the author's public record and finds
more than was claimed learns that the claims were not carefully made, which is the same lesson
as finding less.

---

## 5. Supporting material, not cited in the proposal

Held by the author and available if a reviewer asks. Listed so that the proposal's brevity is a
choice rather than an absence.

| Item | What it covers |
|---|---|
| `Resume_Amon_Koike2026_CV` | Full employment and education history |
| `Resume_Addendum_Projects.pdf` | Project detail beyond the proposal's four lines |
| <https://thedaemon-wizard.github.io/portfolio-quantum-research-engineer/> | Portfolio, cited in the proposal |
| <https://github.com/thedaemon-wizard> | Public repositories |

The proposal cites the portfolio URL only. The GitHub handle was removed for space and is
reachable from the portfolio; the handle was separately confirmed consistent across every
repository the author owns.

## 6. Deliberately not claimed

* **Quantum hardware in this study.** No number in this proposal comes from a QPU; both quantum
  arms are simulator-only. The proposal states this rather than leaving it to be inferred, since
  the challenge statement says participants who do not run on hardware are not penalised.

  **Corrected 2026-08-30.** This bullet previously read "the author has not used one", which was
  false and publicly checkable. `github.com/thedaemon-wizard/qml_benckmark_qiskit_v2` records
  VQC and QSVM trained on IBM `ibm_fez` on 2026-06-19 (`results/benchmark.log:129-137`,
  `results/summary.csv`: VQC 0.600, QSVM 0.675, on an 8-sample subset in job mode on the open
  plan). The error was an **under**statement — it made the team weaker than the record supports
  — but the project's rule is that a claim is wrong in either direction. The scoped version is
  what the study needs, and the hardware run now appears in the team section where it counts.
* **Graph kernels, Weisfeiler–Lehman baselines, nested cross-validation.** An earlier draft
  attached these to the QPoland placement. No implementation of any of the three exists on this
  machine, so all three were removed and the placement stands alone.

## 7. A discrepancy outside this repository

`Resume_Amon_Koike2026_CV` dates the sole proprietorship **"Jul 2026 - Present"**. The filing was
**1 August 2026**, which the author confirmed on 2026-08-30 and which the portfolio also states.
The CV is the document that is wrong.

Nothing in this submission depends on it: proposal §8 says "registered sole proprietor in Japan"
with no date, which is true either way. It is recorded here because the CV and the portfolio are
both reachable from the portfolio URL the proposal cites, and a reviewer who opens both would see
two different dates for the same fact.

**Action, and it is the author's, not this repository's:** correct the CV to 1 August 2026 before
circulating it further. No change is required in `submission/`.

## 8. C3 — the IBM Qiskit credential

**What changed.** Until 2026-09-06 this row read *IBM Qiskit Advocate (2026)*, backed by an
acceptance email the author holds and which is not in this repository. That was the only item in
proposal section 8 with **no reviewer-side verification route at all**: the Advocate programme
sits at Tier 0, issues no badge, and therefore has no URL. An audit flagged that a credential a
reviewer cannot check reads worse than one that is simply absent, and the author has since
obtained a certification that does have a public record. Section 8 has room for one Qiskit line,
so the verifiable one replaces the unverifiable one.

**The exact title, and how it was established.** The proposal prints

> IBM Certified Quantum Computation using Qiskit v2.X Developer - Associate

quoted character for character from the badge page title, including the plain hyphen before
*Associate*. The LaTeX source carries a comment saying so, because `--` is better typography and
would be the wrong name.

| | |
|---|---|
| Badge | <https://www.credly.com/badges/0beb5e13-8116-4b3c-9090-85da23ee134d> |
| Badge title | IBM Certified Quantum Computation using Qiskit v2.X Developer - Associate |
| Issued to | **Amon Koike** |
| Issued by | **IBM Professional Certification** |
| Date issued | **6 September 2026** |
| Underlying exam | Fundamentals of Quantum Computing Using Qiskit v2.X Developer, C1000-179 |
| Read | in a browser, 2026-09-06, from the rendered page |

**An earlier version of this section said the last four rows could not be confirmed, and that was
wrong.** A plain HTTP fetch of the badge URL returns a navigation shell, because Credly renders
badge detail client-side, and the conclusion drawn from that was that the holder and the date
were unverifiable from here. They are not: opening the same URL in a browser renders all of them.
**The limit was the tool, not the source**, and reporting a tool limit as a property of the
evidence is the same error in miniature that [D-133](decisions.md) and [D-137](decisions.md)
record -- concluding from "I did not see it" that "it is not there". The fetch was repeated in a
browser and every field above came back.

Two consequences:

* **Section 8 now prints the year.** It printed no date while the date was unconfirmed, which was
  right at the time; every other dated item in that section carries a date this file could
  confirm, and this one now does too.
* **The level is kept.** *Associate* is part of the title the issuer publishes. Dropping it would
  name a bigger credential than the one held, and section 8 has already been corrected four times
  for exactly that kind of quiet inflation. Note the badge's own metadata separately grades it
  **Foundational**, so nothing here should be read as a senior certification.

**One thing this file still cannot do**, and it is worth stating precisely: it confirms that the
badge at that URL names Amon Koike. It cannot confirm that the person operating this repository
is that Amon Koike. No document can; a reviewer resolves it the same way they resolve C1 and C2,
by taking the author's identity as given.
