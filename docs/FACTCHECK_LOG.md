# Fact-check log

What was checked, against what, on what date, and what the check changed.

The three reference documents in this folder record *what is cited*
([REFERENCES.md](REFERENCES.md)), *what reaches each entry*
([REFERENCE_CROSSCHECK.md](REFERENCE_CROSSCHECK.md)) and *where each regulatory instrument
lives* ([REGULATORY_SOURCES.md](REGULATORY_SOURCES.md)). None of them records the act of
checking: which claims were verified, against which artefact, and which ones the check
overturned. That is this file.

**Why it exists as a separate document.** Every gate in this repository compares a document to
a table. The failures below are the ones no gate can see: a number whose table was never
regenerated, a sentence never bound to a claim, a fact that lives outside the repository
altogether, or a characterisation of someone else's paper. Each was found by reading a source,
and each is recorded here so the next reader can re-run the check rather than trust it.

Verdicts are **supported**, **not supported** (the claim was wrong and was changed), or
**要確認** (could not be settled from an artefact available here).

---

## 1. Claims checked in this round — 2026-08-30

| # | Claim as written | Source consulted | Verdict | What changed |
|---|---|---|---|---|
| 1 | §3: the card-disjoint arm "in every case *over*-covers … without breaching the one-sided guarantee" | [`coverage_by_arm_seeds.csv`](../results/tables/coverage_by_arm_seeds.csv) | **not supported** | Seed 20260831 at $\alpha=0.001$ has ratio 1.5245 with `conservative=False` — it *under*-covers. §3 now states the two levels separately and the arm is not reported as a pass |
| 2 | Both PDFs: "61 decision entries" | `grep -c '^### D-' docs/decisions.md` = **63** | **not supported** | Table regenerated, claim rebound, `summarise_decisions.py` wired into `make claims`, and a test now recounts from the document ([D-064](decisions.md)) |
| 3 | Appendix: "Amendments. **Four** were made, each derived from block sizes or from $D_{\mathrm{band}}$ alone" | `scripts/check_protocol.py` → `['A1'…'A7']`; `docs/protocol.md` | **not supported** | Seven exist. All seven are now listed, and A5/A6/A7 are described as what they are — two post-hoc corrections and one disclosure of non-execution. Count bound as `ProtocolAmendments` |
| 4 | §6: "Every step was executed in this study" | `docs/protocol.md` §9 disposition and amendment A7 | **not supported** | H1–H3 were never tested and three of five drift measurements never ran. Scoped to the certification pipeline, with the exceptions named |
| 5 | §7 and CREDENTIALS: "the author has not used a QPU" | `qml_benckmark_qiskit_v2/results/benchmark.log:129–137`, `results/summary.csv` | **not supported** | VQC and QSVM were trained on IBM `ibm_fez` on 2026-06-19 (8-sample subset, job mode, open plan), in a public repository. An **under**statement, corrected in both directions: §7 now scopes the claim to this study, and §8 records the hardware run |
| 6 | ENVIRONMENT §3: temporal-arm block sizes 363,817 / 59,054 / 60,464 / 118,108 | [`splits.csv`](../results/tables/splits.csv) | **not supported** | Three of four were from other arms; the four summed to 601,443 against a dataset of 590,540. Corrected to 356,216 / 58,326 / 60,464 / 115,534, which sums exactly |
| 7 | §1: prior conformal work "certifies a *marginal* error rate" | [FR-1] abstract; [FR-3] abstract and DOI `10.1016/j.dss.2026.114717` | **not supported** | Neither is marginal. FR-1 is Mondrian **class-conditional** with cost-controlled abstention; FR-3 certifies a **false-negative rate**. Calling them marginal understated the prior work and so overstated this submission's novelty. §1 now distinguishes on the two things that are actually new here: conditioning on the abstention region, and a split that orders time |
| 8 | README: "Feature attribution … is not provided", citing D-046 | [`attribution.csv`](../results/tables/attribution.csv); `scripts/run_explain.py`; `decisions.md` | **not supported** | It is provided and reported in proposal §5. The citation was also wrong — D-046 is about primary metrics; attribution is D-048 |
| 9 | README: "six dated amendments (A1–A6)" at line 419 vs "seven" at line 42 | `docs/protocol.md` | **not supported** | Self-contradiction within one file. Both now say seven |
| 10 | README: reference ranges CP-1…13, QM-1…10, SW-1…9 | [REFERENCES.md](REFERENCES.md) ID scan | **not supported** | Actual maxima are CP-17, QM-15, SW-10. The count was 59 when this was written and is 49 now: ten entries reached by nothing were removed, and the README figure is derived from the same crosscheck rather than restated |
| 11 | PROVENANCE: "`scripts/fetch_data.py` reconstructs it" | filesystem | **not supported** | No such script. The entry now says ULB is not reconstructible here, which is also why E15 did not run |
| 12 | COMPLIANCE: "§1 links ten documents"; "34 commits" | README §1; `git log` | **not supported** | Eleven documents; 32 commits. The commit count is now dated rather than asserted as live |
| 13 | REGULATORY_SOURCES: 要確認 on the SS1/23 applicability sentence | [REFERENCES.md](REFERENCES.md) RG-5 | **stale** | The defect had already been corrected in REFERENCES.md; the open marker was left behind. Closed |
| 14 | RG-1: "Article 21 (monitoring)" among provisions relied on | repository-wide search | **not supported** | Nothing cited it. Removed from the entry; the obligation belongs to a deployment |
| 15 | CREDENTIALS: source path `QIntern2026/qi26_12/` | `git -C … rev-parse 9b29505` | **not supported** | The freeze commit is one level down, in `qi26_12/qintern-project-12-team-A/`. The cited path contains documentation copies only, with no `scripts/`, `tests/` or `RESULTS_FROZEN/` |
| 16 | CREDENTIALS: "**Every** Week-5 artefact carries `source_kind: dummy`" | `grep -rho '"source_kind"…' week5/` → 21 dummy, 7 real | **overstated** | True of all eleven run-level stamps; six row-level entries read `"real"` (the XGBoost baseline). The distinction favours the work — the classical arm was measured — and is now stated |
| 17 | §8: the author wrote "an exact Beta-Binomial coverage harness … exact McNemar under Holm" | `git log --follow week3/scripts/coverage_harness.py`, `stats_protocol.py` | **not supported** | Both land in commit `f2c2fb3` by a teammate. `CREDENTIALS.md` already said so, so the two documents contradicted each other; §8 now attributes them to the team |
| 18 | Yale credential: rank denominator | BlueQubit leaderboard, World view, page 2 | **supported, and completed** | `450/550` is the score maximum; `11–20 of 549` in the pagination is the field. Both are now stated: rank 13 of 549 entries at 450 of 550 points ([D-062](decisions.md)) |
| 19 | QIntern: "135 checks green at that freeze" | `git grep -h "def test_" 9b29505 -- '*/tests/*.py' \| wc -l` = 135; at `HEAD` = 183 | **supported** | Exact. No `parametrize` in the suite, so defined equals collected. The proposal's "at that freeze" hedge is correct |
| 20 | QIntern: five seeds, three intrusion-detection corpora | `week5/RESULTS_FROZEN/results_manifest_v1.0.json`; `SEEDS = [42…46]` | **supported** | `['CICIoT2023', 'BoT-IoT', 'UNSW-NB15']`. No occurrence of "card fraud" or "payment" anywhere in the repository, so the no-extrapolation claim holds |
| 21 | unitaryHACK: merged into CUDA-Q and QuEST | `NVIDIA/cuda-quantum#4693`, `QuEST-Kit/QuEST#783` | **supported** | Both merged upstream and checkable by pull request |
| 22 | Latency: the first measurement's figures | direct `nthread` sweep on a fitted booster | **not supported** | The first run measured an OpenMP barrier (19.33 ms at the default thread count against 0.051 ms single-threaded), not model cost. Discarded and re-measured ([D-063](decisions.md)) |
| 23 | "`make_splits.py` is silent for thirteen minutes" — my own finding during the clean-room run | re-run on an idle machine | **not supported** | The script completes in about **two seconds**. The stall was contention from orphaned processes I had left running. Two successive explanations for it were also wrong, both disproved by the instrument added alongside them ([D-066](decisions.md)) |
| 24 | Split tables are deterministic | `make_splits.py --out <scratch>`, then `diff` | **supported** | `splits.csv` and `data_integrity.csv` both reproduce **bit-identically** against the committed copies |
| 25 | Portal: accepted formats and slot count | live submission form, read 2026-08-30 | **supported** | The form itself reads "0 uploaded • 5 slots left" and "Allowed formats: PDF, PNG, JPG, WEBP, GIF, PY, JSON, JS, XLS, XLSX, CSV, DOC, DOCX" — exactly what `assemble_submission.py` enforces |
| 26 | Portal OBJECTIVE: "using Amazon Braket" | `HSBC-Challenge-Statement-vFinalRevised.pdf`, in-scope list | **supported, and the portal overstates the statement** | The statement puts "Any quantum or quantum-inspired framework (Qiskit, PennyLane, Cirq, cuQuantum, ITensor, Braket, etc.)" in scope and says hardware execution is "not expected nor required". Braket's `LocalSimulator` is the kernel's reference implementation. Recorded as C11a because nothing had answered the question |
| 28 | "conformal_calibrate.py authorship is not independently confirmable" — my own 要確認 | `week3/OWAIS_TASK16_17_HANDOFF.md`, committed by the teammate in `f2c2fb3` | **not supported; the 要確認 is lifted** | The teammate's own handoff calls it "AK's Day-15 CQ-ZDR module" and records re-running it to a six-decimal match. The bulk "Initial commit" was also **the teammate's**, not the author's, so the premise that only self-attestation remained was wrong twice. The Days 26–27 freeze is verified the same way ("byte-identical to AK's v1.0 manifest") |
| 29 | "`qi26_12/week3/` is a documentation-only copy with no scripts or tests" — my own statement | `ls qi26_12/week3/` | **not supported** | True of `week5/`, false of `week3/`, which has `scripts/` (including `conformal_calibrate.py`) and `tests/`. Corrected |
| 30 | "The author's working repo records different peak bitstrings for P3–P9 than the published set" — my own 要確認 | both README tables, compared programmatically; `data/Yale_Quantum_2026/P5_soft_rise.qasm.txt` | **not supported; it is an endianness artefact** | Same nine answers in opposite bit order. P3–P9 are **exact string reversals**; P1 is a palindrome; P2 is the one row the third-party README transcribed un-reversed. The author's own committed fixture prints both orientations from one run. His copy is the submitted orientation, so the doubt ran the wrong way |
| 32 | Protocol section 6.1: "roughly 98 calibration frauds against the 99 required" on ULB | arithmetic against `configs/default.yaml` split fractions | **not supported; withdrawn** | 98 is 20 % of ULB's 492 frauds, the **test** fraction; `split.cal` is 10 %, giving roughly 49. The conclusion survives and strengthens (49 against a floor of 99 is more degenerate), which is why the error was invisible from downstream. ULB was never obtained or measured -- amendment A9, [D-070](decisions.md) |
| 33 | Kaggle rules section 7.A | <https://www.kaggle.com/competitions/ieee-fraud-detection/rules>, read 2026-08-30 | **supported** | "non-commercial purposes only … academic research and education". Quoted verbatim in [PROVENANCE.md](PROVENANCE.md); proposal §5 and appendix A3 now give it as a reason the Phase II sprint must move to the issuer's stream |
| 31 | My first attempt at row 30: "the two records are different problem sets, 2025 vs 2026" | third-party `README.md:174-185` | **also not supported** | That README carries a Yale-2026 table with all ten problems, which I did not read — I had looked only at the `data/*.txt` files. Recorded because it is the second wrong explanation for the same item |
| 27 | Sole-proprietor filing date | confirmed by the author, 2026-08-30 | **settled: 1 August 2026** | The submission states the status without a date, so nothing in it turns on this. `Resume_Amon_Koike2026_CV` says "Jul 2026" and is the document that is wrong — see [CREDENTIALS.md](CREDENTIALS.md) §7 |


## 1b. Claims checked in this round — 2026-08-31

Checked against the live portal, the official PDFs and the repository itself. The portal rows
were read in a browser from the HSBC challenge panel while signed in.

| # | Claim as written | Source consulted | Verdict | What changed |
|---|---|---|---|---|
| 15 | COMPLIANCE C8/C9: fraud probability and binary prediction "met", evidenced by `scores_*.parquet` and an appendix count | HSBC challenge panel, upload control; [Phase 1 Submission Guidelines](https://quantumaiportal.thequantuminsider.com/wp-content/uploads/2026/04/2026-04-06-Phase-1-Submission-Guidelines-VF.pdf) | **not supported for the upload** | Parquet is not among the accepted formats, so the first Expected Outcome had no artefact a reviewer could open. `predictions.csv` now carries one row per held-out transaction and takes a portal slot ([D-088](decisions.md)) |
| 16 | The certified rule applies an in-band re-scorer `g` at threshold λ | `scripts/run_conformal.py:181`, `scripts/validate_certificate.py:90-93`, `conformal/riskcontrol.py:108` | **not supported as written** | λ is selected over, and applied to, the same full-traffic score. For the reported configuration λ = 0.0582 inside a band of [0.0320, 0.0718). Nothing is wrong mathematically; no sentence said `g = f` in this run ([D-086](decisions.md)) |
| 17 | Appendix A8: "three scripts read `D_test` without the guard" | `grep -l TestFoldGuard scripts/*.py` against the scripts that read the block | **not supported** | Six read it and four are unguarded; the protocol table also listed `run_mps.py` as authorising when it never imports the guard. A test now verifies the table against the tree ([D-089](decisions.md)) |
| 18 | Appendix: the first four amendments never used `D_cal` risks | `docs/protocol.md` A3 ("Measured on `D_band` and `D_cal`") and A4 ("Measured on `D_cal` at a 5 % band") | **not supported** | A3 and A4 both used calibration-set quantities. The sentence now names them and A6 is reclassified from correction to disclosure |
| 19 | `make reproduce` regenerates the committed results | `grep` for each table name across `scripts/` and `src/` | **not supported for four tables** | `split_arm_baselines.csv`, `rolling_origin.csv`, `mps_seed_sweep_summary.csv` and `mps_seed_spread.csv` had no producer. Three now have one and reproduce the committed values; the fourth is disclosed ([D-089](decisions.md)) |
| 20 | Proposal §4: "the maximum 1.0" RBF correlation | [`screens.csv`](../results/tables/screens.csv) | **not supported as printed** | The measurement is 0.99999975. `claims.yaml` recorded `1.0000` and YAML parsing dropped the trailing zeros, so the page asserted exactly one in a sentence about values indistinguishable to floating point |
| 21 | `test_makefile_scripts_exist` enforces that referenced scripts exist | `tests/test_repo_hygiene.py` | **not supported** | It called `pytest.xfail`, which the runner counts as a pass, so it could not fail. Now an assertion |
| 22 | Phase II runs from 17 November 2026 | [Program roadmap](https://quantumai.thequantuminsider.com/program/) | **supported** | Phase I closes 15 Sep 2026; review 16 Sep – 14 Nov; Phase II PoC 17 Nov 2026 – 28 Feb 2027 with AWS credits and Classiq tooling; winners announced 30 Apr 2027 |
| 23 | The portal accepts five uploads and a fixed format list | HSBC challenge panel, 2026-08-31 | **supported** | "0 uploaded, 5 slots left"; PDF, PNG, JPG, WEBP, GIF, PY, JSON, JS, XLS, XLSX, CSV, DOC, DOCX. The guidelines add a 20 MB per-file cap; the largest staged file is 11 MB |

## 2. Literature checked, post-2026-08

Searched for work published since August 2026 that would change the positioning in §1.

| Source | What it does | Effect here |
|---|---|---|
| **[FR-3]** Zhu et al., *DISCO*, *Decision Support Systems* 208 (2026), DOI `10.1016/j.dss.2026.114717` | Deep metric learning decoupled from risk control; conformal risk control with a formal guarantee on the **false-negative rate**, on a real card-fraud dataset | Already cited. Independently re-verified this round, including the DOI. It is the closest prior work and the reason §1 no longer describes prior conformal fraud work as "marginal" |
| **[CP-17]** Aldirawi, Li & Guo, arXiv:2604.01502 | Risk control over a finite grid can fail for non-monotone losses | Already cited. Checked rather than assumed: both risks are strictly monotone in $\lambda$ over the pre-registered grid, so the counterexample does not apply |
| **[FR-6]** Wang et al., NCPNET, *KDD* 2025, arXiv:2507.02151 | Non-exchangeable conformal prediction for temporal graphs | Already cited. Directly relevant to this study's central finding, that time breaks exchangeability |
| **[FR-5]** Chen et al., ProtoCP, *KDD* 2026, arXiv:2608.15768 | Temporal-graph prototype-conditioned conformal prediction for fraud | Already cited |
| Faryad, arXiv:2608.15718 (2026) | Controlled benchmark on real card-transaction data; no robust quantum advantage, and one apparent advantage attributed to unequal search budgets | Already cited in §4 as the closest published comparator |

**Nothing found that supersedes the positioning.** The distinguishing claims — a guarantee
conditional on the abstention region, and evaluation under a split that orders time — are
still not covered by the work above. Where prior work is closer than an earlier draft
admitted, §1 was corrected rather than the prior work minimised (row 7).

## 3. What remains 要確認

| Item | Why it cannot be settled here |
|---|---|
| Portal upload and the lead-contact field | Manual actions. The portal was re-checked in a browser on 2026-08-30: the HSBC challenge reads *Not submitted*, five slots, accepted-format list unchanged |
| Yale problems 3–9 peak bitstrings | The author's private working repository records different bitstrings than the published set. The organiser's score of 450 is itself the record of nine correct submissions, so this reads as a working copy diverging from what was submitted. Nothing in the proposal depends on it |
| `conformal_calibrate.py` authorship | The QIntern repository's bulk "Initial commit" destroys per-file provenance. Supported only by a self-authored handoff and the file's own header. Not falsified, not independently confirmable |
| CV date of the sole proprietorship | `Resume_Amon_Koike2026_CV` reads "Jul 2026 – Present"; the portfolio and the filing say 1 August 2026. A reviewer comparing the two would see it |
| Sole proprietorship, degree, QPoland placement | Documents held by the author, not present in this repository |

## 4. How to re-run these checks

```
make claims                              # every bound number, citation and the protocol lock
.venv/bin/python -m pytest -q            # includes the decision- and amendment-count recount
grep -c '^### D-' docs/decisions.md      # against results/tables/decision_log.csv
.venv/bin/python scripts/check_protocol.py
```

The rows in §1 that concern artefacts outside this repository name the exact command or file
used, so each can be repeated against the same source rather than taken on trust.
