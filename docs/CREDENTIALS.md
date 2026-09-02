# Credential provenance

Every biographical claim in the proposal's team section, with what backs it and how it was
checked. This file exists because of a structural gap: those claims are wrapped in a
`\Record{}` macro, which declares a figure as the author's own record rather than a measurement
from this study — and therefore **exempts it from every numeric gate in this repository**. The
exemption is correct. It also meant nothing checked them until 2026-08-30, and when they were
checked, six needed changing.

Verified 2026-08-30. Where a claim cannot be settled from an artefact it is marked **要確認**
rather than asserted or quietly dropped — and **no 要確認 remains open in this file**. The three
that were carried here were all closed on 2026-08-30 by going to the artefact:

| Item | Outcome |
|---|---|
| `conformal_calibrate.py` authorship "not independently confirmable" | **Wrong.** A teammate's own handoff attributes it and records reproducing its numbers to six decimals (§2) |
| Yale peak bitstrings for problems 3–9 "differ from the published set" | **Wrong.** Same nine answers, opposite bit order: P3–P9 are exact string reversals, P1 is a palindrome, P2 is the one row transcribed un-reversed (§3) |
| Sole-proprietor filing date | **Settled: 1 August 2026** (§1, and §7 for a CV that says otherwise) |

Two of the three were errors in this file's own earlier auditing, not in the credential. Both
made the author's record look weaker than the evidence supports.

---

## 1. What the proposal claims

| # | Claim | Status | Backed by |
|---|---|---|---|
| C1 | Registered sole proprietor in Japan | **date confirmed by the author, filing not checkable here** | Filed **1 August 2026**, confirmed 2026-08-30. The proposal states the status without a date, so nothing in the submission turns on it; see §7 for a discrepancy in a document outside this repository |
| C2 | B.S. Physics, Tokyo Denki University | not checkable here | diploma, held by the author |
| C3 | IBM Qiskit Advocate (2026) | **confirmed, not publicly verifiable** | acceptance email held by the author (`Congratulations and welcome to the Qiskit advocate program!.pdf`, not in this repository); the programme is at Tier 0 and no badge has been issued, so there is no URL to cite |
| C4 | QIntern 2026 Project 12 (QWorld), Team A — split-conformal calibration and results freeze | **verified, and independently attributed by a teammate** | see §2. Both the Day-15 calibration module and the Days 26–27 freeze are named as the author's in handoff documents committed by a teammate, who re-ran each and reproduced the numbers |
| C5 | Yale Peaked Hackathon 2026, rank 13 of 549 entries at 450 of 550 points, nine of ten circuits to 69 qubits by an MPS marginal attack | **verified on the organiser's leaderboard, method verified by merged pull request** | see §3 |
| C6 | QPoland 2025 runner-up | not checkable here | no artefact on this machine; the placement stands with no technical detail attached |
| C7 | unitaryHACK 2026, merged into NVIDIA CUDA-Q and QuEST | **verified by merged pull request** | see §4 |

## 2. C4 — QIntern 2026 Project 12

Source: the Team A working repository, which is **private** and not this one. A reviewer
cannot follow any locator into it, so the paragraphs below record what was checked and by whom
rather than pointing at commits nobody outside the team can open. Commit identifiers and the
local paths they were read from have been removed for that reason; the checks themselves were
run and are described.

Per the Phase I submission guidelines, no third party's information is reproduced here: the
teammates who made the attributions below are identified by role, not by name or contact
address.

**What is the author's own.** The split-conformal calibration module and the results freeze.
The freeze commit (2026-07-30) first commits
`week5/scripts/{disentanglement,table_a,figure2_coverage,freeze_results}.py` under his
authorship; `freeze_results.py` computes the SHA-256 manifest rows.

**The calibration module is independently attributed, and the earlier 要確認 was wrong.**
`week3/scripts/conformal_calibrate.py` does arrive in a bulk "Initial commit", so
per-file provenance is absent — but that commit was made by a **teammate**, on whose account
the repository is hosted, not by the author, and three days later the same teammate committed
`week3/OWAIS_TASK16_17_HANDOFF.md`, which is third-party attribution of exactly this artefact:

> "built on **AK's Day-15 CQ-ZDR module** and Iwo's Day-13/14 package. Also records the
> independent verification of AK's Day-15 numbers he requested." (lines 4–5)
>
> "Full suite re-run on this machine: **35/35 pass** (24 Iwo + 11 AK)." (line 9)
>
> "`conformal_calibrate.py` re-run, trio @ α = 0.05: q(CIC) = 0.300551, coverage 0.9514,
> FZR 0.0486 — **exact match**, BoT/UNSW previews match to the 6th decimal." (lines 10–11)

That is stronger than an attribution: a teammate re-ran the module and reproduced its numbers to
six decimals, in a commit he signed. An earlier version of this file called the authorship
"not independently confirmable here", which was wrong — it was confirmable, in a file two
directories away that had not been read.

**What is the team's.** The paper's conformal-coverage proposition belongs to **Team A**, a
three-person team — `week4/README.md` and `week5/README.md` state it as "Team A owns the
paper's **Proposition 3 (conformal coverage)**". (`week3/README.md` says "statistical-guarantee
centrepiece"; an earlier version of this file cited it for the stronger wording.) The
Beta-Binomial harness and the McNemar/Holm protocol are a teammate's Days 16–17 — git puts
`week3/scripts/coverage_harness.py` and `stats_protocol.py` in a commit by another
contributor. The proposal previously read as though both were the author's; §8 now attributes
them to the team.

**The 135 checks are exactly 135, at the freeze, and the count is the teammate's.** Counted at
the freeze commit. The repository is at 183 today, so the proposal says "green at that freeze".
The freeze itself is independently verified in a second teammate's handoff document,
`week5/OWAIS_TASK28_30_HANDOFF.md`: "**Every number verifies.** Full suite 135 passed + 8
subtests … `freeze_results.py --reproduce` returns all 39 scalars to 1e-9 … Reproduced on a
*different* environment (Python 3.10, scikit-learn 1.7.2) with identical bytes", and
"byte-identical to AK's v1.0 manifest".

**The numbers are provisional and the proposal says so.** All eleven Week-5 artefacts carry a
**run-level** `"source_kind": "dummy"` and `"scores_root": "week2/interface/dummy_scores"`.
Six **row-level** entries in two of them read `"real"` — the XGBoost classical baseline, one per
dataset, which is genuinely measured; only the quantum arm is a placeholder. An earlier version
of this file said "every artefact", which understated the work.

The team's own README states: *"Everything quantum is still on the dummy interface, so every
QS-Net number this week is PROVISIONAL … protocol-final, numbers-provisional."* The coverage result therefore
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
  problems carry weights 10, 20, ... 100, which sum to 550. A score of 450 is every problem but
  the last, so **nine of ten** follows from the score itself.
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
author's favour.** An earlier version of this file carried a 要確認 saying the author's working
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
submissions before it was fixed. So the earlier 要確認 had the direction backwards — his
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
| Recursive Quantum Shannon decomposition, MLIR unitary synthesis | `NVIDIA/cuda-quantum#4693` | merged into the default branch, carrying the maintainers' own `unitaryhack-accepted` label |
| GPU backend change | `QuEST-Kit/QuEST#783` | merged by the project lead into `devel` |

Both are merged upstream and independently checkable by pull request. The decomposition is
genuinely self-recursive rather than a name reused for something flatter.

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
