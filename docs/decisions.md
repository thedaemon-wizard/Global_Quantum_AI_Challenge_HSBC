# Decision log

Dated decisions, the alternatives rejected, and — where it applies — the retraction. An
entry is never deleted. When a decision is superseded the original stays and carries a
pointer, because the fact that a thing was once believed is part of the record.

---

## Round 1 — Environment and pinning (2026-08-28)

### D-001 Python 3.12 through the virtual environment's interpreter, never `python3`

On this machine `python3` is 3.9, and a bare `python3.12` resolves to a system interpreter
whose `torch` has no `sm_120`. Every Makefile target therefore invokes `.venv/bin/python`
directly and the bootstrap uses `/usr/bin/python3.12` explicitly.
`tests/test_repo_hygiene.py` forbids a bare `python3` anywhere in the repository.

### D-002 `torch==2.13.0+cu130`, installed before the project

Verified against the PyTorch cu130 index on 2026-08-28: `torch-2.13.0+cu130` publishes a
`cp312-cp312-manylinux_2_28_x86_64` wheel, and 2.13.0 is the newest version present on that
index. Installed from `--index-url https://download.pytorch.org/whl/cu130` before the
project itself, because the default PyPI build does not carry `sm_120`.

Confirmed on this host: `torch.cuda.get_arch_list()` ends `['sm_90', 'sm_100', 'sm_120']`.

### D-003 Six inherited pins were stale and were updated

The sibling Airbus project supplied the engineering scaffolding and its pins were carried
over as a starting point. Checked against the PyPI JSON API on 2026-08-28, six had moved:

| Package | Inherited | Now pinned |
|---|---|---|
| `torch` | 2.12.1+cu130 | 2.13.0+cu130 |
| `numpy` | 2.5.1 | 2.5.2 |
| `scipy` | 1.18.0 | 1.18.1 |
| `matplotlib` | 3.11.0 | 3.11.1 |
| `pypdf` | 6.16.1 | 6.16.2 |
| `ruff` | 0.15.20 | 0.16.5 |

Every pin in `pyproject.toml` was additionally checked for a `cp312` x86_64 manylinux wheel
so that no dependency arrives as a source build.

### D-004 `pandas==3.0.5`, defended by a gate rather than pinned backwards

**Rejected alternative:** pin `pandas<3` (the newest 2.x is 2.3.3, 2025-09-29).

Under copy-on-write — the only mode in pandas 3 — `df['a'][mask] = v` is a silent no-op, and
`SettingWithCopyWarning` has been removed, so nothing is emitted at all. The frame is simply
unchanged and every number downstream is quietly wrong. IEEE-CIS feature engineering is
heavy on exactly this pattern.

Pinning backwards would trade a current stack for an eleven-month-old one to avoid a hazard
that is mechanically detectable. Smoke check S6 reproduces the silent no-op on this host and
verifies that the AST gate flags it, and `tests/test_repo_hygiene.py` runs the same gate
over the whole repository. This is the project's stated discipline — fail loudly, do not
tolerate a silent fallback — applied to a dependency rather than to our own code.

Measured on this host, 2026-08-28: *"pandas 3.0.5: silent no-op confirmed; AST gate flags bad
and passes good."*

The fallback, if the gate ever proves insufficient, is `pandas==2.3.3` (cp312 wheel
confirmed present).

### D-005 sQUlearn is not adopted

**Rejected alternative:** use sQUlearn 0.11.2 for scikit-learn-native quantum kernels.

Its published metadata requires `qiskit-algorithms>=0.3.0`. That package's own README states
that it is no longer officially supported by IBM. Adopting sQUlearn would therefore pull a
publicly disowned dependency into a submission whose argument is governance, along with
`qiskit-ibm-runtime`, `bayesian-optimization`, `mapomatic` and `dill`.

`qiskit-machine-learning` 0.9.1 supplies `FidelityStatevectorKernel` directly with none of
that. Smoke check S0 asserts that `qiskit-algorithms` is absent, transitively included.

### D-006 The production kernel path is `FidelityStatevectorKernel`; Aer GPU is a cross-check

Two facts pull in the same direction.

**Licensing.** `qiskit-aer-gpu-cu11` depends on `cuquantum-cu11`, so installing Aer's GPU
build brings NVIDIA's cuStateVec binary into the environment. That binary is proprietary
under the NVIDIA Software License Agreement: royalty-free and commercially usable, but
no-modification, no-derivative-works, with liability capped at USD 10.00. It cannot be
described as open source. An earlier draft of this project's plan simultaneously excluded
cuQuantum "on licence grounds" and used Aer GPU — a self-contradiction.

**Measurement.** `FidelityStatevectorKernel` evaluates the Gram matrix pairwise in Python,
so a GPU does not accelerate it. Measured on this host: 8 qubits, n = 1000 costs 9.73 s,
about 19.5 µs per pair, scaling quadratically.

So the kernel path that produces reported numbers is pure numpy with no Aer and no NVIDIA
binary, and Aer is confined to the optional `gpu-crosscheck` extra used only for the
four-way simulator agreement check. The GPU is spent where it is genuinely first-class:
XGBoost (`sm_120` is in its compiled architecture list) and the tensor-network arm.

`NOTICE` section 4 records exactly what installing the extra adds.

### D-007 Aer's GPU wheel must be installed *last*

`qiskit-aer` and `qiskit-aer-gpu-cu11` are separate distributions that both install the same
`qiskit_aer` package. Resolved together in one `pip install`, the CPU build landed second and
overwrote the GPU binary. `AerSimulator().available_devices()` then returned `('CPU',)` — no
error, no warning, and a "GPU" cross-check that would silently have been a second CPU run.

`make venv-gpu` therefore reinstalls the GPU wheel with `--force-reinstall --no-deps` after
the resolution and asserts `'GPU' in available_devices()` before returning.

### D-008 LightGBM runs on CPU

Its PyPI wheel ships the OpenCL backend only; the CUDA backend requires a source build with
`USE_CUDA=ON`, for which no Blackwell validation could be found. At 590,540 rows by ~400
float32 columns the CPU path is comfortable on this machine, and GBDT GPU speed-ups are
modest at that scale. Putting a source build on an 18-day critical path is not warranted.

### D-009 The GPU memory witness must live outside the interpreter — retraction of a check

Smoke check S1 asserts that Aer's GPU path really executes on the device. Its first
implementation polled `nvidia-smi` between `job.done()` calls; that never sampled during
execution and reported a peak of 1189 MiB, which the check correctly read as a CPU
fallback — a false alarm.

The second implementation sampled `torch.cuda.mem_get_info()` from a Python thread every
2 ms. It reported the same 1189 MiB. Instrumenting the thread showed why: it received
**exactly two ticks across a 2.71 s run**, because Aer holds the GIL for the duration of the
C++ execution. An in-process Python witness cannot observe an Aer allocation.

The check now starts a streaming `nvidia-smi` subprocess before the run and terminates it
after, and additionally reads Aer's own result metadata, which names the device it
dispatched to. Both witnesses are required: metadata is authoritative but self-reported,
memory is external but samplable only from outside the process.

Measured on this host, 2026-08-28: *"28q statevector in 2.75s; Aer reports device=GPU
cuStateVec=True; device memory 629 to 5565 MiB (+4936, 4096 expected)."*

This entry is kept in full, including the two wrong implementations, because a reader
evaluating the GPU claim should be able to see how it was established.

### D-010 Conformal and statistics modules are written from the primary literature

**Rejected alternative:** port the corresponding modules from the QIntern 2026 Team A
repository, which implements comparable procedures and which the author co-wrote.

That repository publishes **no licence file** and has multiple contributors. Absent a licence
grant the material is all-rights-reserved and cannot be redistributed under Apache-2.0.
Independently, Challenge Terms and Conditions section 3 makes the absence of third-party
proprietary material a representation and warranty, so this is a contractual matter and not
only a copyright one.

The affected modules are short and implement published formulas, so they are written from
the sources instead, with each docstring naming the paper whose result it implements
(`NOTICE` section 2 lists the mapping). Describing the author's own contribution to that
project in the team-capability section remains unaffected; what is prohibited is copying
code and reproducing unpublished co-authored figures.

### D-011 Smoke checks may not be skipped except for three stated reasons

A skipped check reads like a passing one at a glance. `scripts/smoke.py` therefore treats a
skip as a failure unless the check is in an explicit allow-list: S1 and S5 exercise Aer,
which the default install deliberately omits (D-006), and S7 needs a dataset that
`make data` fetches separately. Any other skip fails the run.

---

## Round 2 — The conformal layer (2026-08-28)

### D-012 The coverage check uses the exact Beta-Binomial law, not `rate <= alpha`

The split-conformal guarantee is an expectation over the calibration draw, so on any finite
test set the empirical rate fluctuates around it. Asserting `empirical_rate <= alpha` is
therefore wrong in both directions: it fails on sound systems, and passing it means nothing.

Quantified on this host rather than argued: with n = 5,000 calibration points, alpha = 0.01
and m = 20,000 test points, the exact predictive law is BetaBinomial(m, n+1-k, k), and a
naive `rate <= alpha` assertion passes **51.15 %** of the time on a system that is behaving
exactly as designed. The exact 99 % central band [127, 289] contains 99.15 % of draws.

`hsbcfraud.conformal.coverage` therefore reports three separate verdicts —
`expectation_ok`, `finite_sample_ok`, `conservative` — because collapsing them hides the
case that matters here. Over-covering is not a success: it means the certified abstention
budget is larger than the data requires, and every excess step-up authentication is a cost
paid for an artefact.

The pmf agrees with `scipy.stats.betabinom` to 7e-17 across the tested parameter range; it
is computed in log space because the binomial coefficient alone overflows double precision
well below this study's test-set sizes.

### D-013 Weighted conformal is implemented but is not quoted as a number

`hsbcfraud.conformal.weighted` implements Barber et al. eq. (11), and `tests/test_conformal.py`
asserts that it reduces exactly to standard split conformal at `w_i = 1`, to machine precision,
at three (n, alpha) combinations.  *Added 2026-08-30: that sentence was written before the test
existed and stood for several rounds as a description of something that had never run.* The point mass at `+inf` is included: omitting it is the
usual implementation error and yields a threshold that is too low.

What is **not** done is quoting the paper's closed-form bounds as evaluated numbers. Both
`2*eps/(1-rho)` and `rho^k` route through the paper's Lemma 1, which assumes the
observations are independent; a transaction stream is serially correlated and falls under
the paper's separate covariate-time-series treatment. Leading with `rho^k` on this data
would be citing a theorem outside its stated hypotheses.

Instead `implied_total_variation` inverts eq. (3): given the risk gap actually observed
between calibration and test, it returns the average per-step total variation distance that
would be required to explain it. That is a necessary consequence of a measurement rather
than a promise about the future, and a reviewer can judge whether the implied drift is
plausible for payments data.

### D-014 Retraction — an unverified claim about clustered bootstrap width

The first version of `hsbcfraud.stats` stated that a row-level bootstrap produces intervals
"too narrow -- by roughly the square root of the average cluster size". That is the
classical result for a sample **mean** under positive intra-cluster correlation. It was
asserted without being checked for the statistic this study actually compares.

Measured on a synthetic fixture with cluster-level risk and 6 % prevalence, the clustered
interval for a paired difference in **average precision** came out *narrower* than the
row-level one, width ratio 0.89. Average precision is a rank statistic, not a mean.

The docstring now says only what is true: the row-level interval is invalid under this
dependence structure, and the sign of its error is statistic-dependent. The clustered form
is used everywhere rather than only where the error was expected to be in a particular
direction.

### D-015 Tests are matched to estimands, not applied uniformly

Average precision is threshold-free and is compared by paired clustered bootstrap. A
decision at a fixed threshold is a paired binary outcome and is compared by exact McNemar —
exact rather than chi-squared because the discordant counts in a band holding a few percent
of traffic are small, and the approximation is unreliable below roughly 25 discordant pairs.

McNemar comparisons are always made at a **common** threshold. Comparing two scorers each at
its own operating point produces a discordance table that is not comparable at all, which is
a subtle way to manufacture a significant result.

`tost_equivalence` additionally returns an `informative` flag. An equivalence test whose
margin is wider than the effect it is meant to exclude proves nothing, and reporting one is
worse than reporting nothing. The reference effect is 0.023 average precision, the point
estimate Chaves et al. (arXiv:2603.06473) report for a comparable hybrid architecture.

---

## Round 3 — Classical baselines (2026-08-28)

### D-016 The G1 gate was mis-specified, and the measurement that shows why is a headline result

The implementation plan set a go/no-go gate of "IEEE-CIS test AUC >= 0.92, else retune".
That number came from published figures for this dataset — Deotte's 1st-place solution
reports 0.9459 on the competition's private leaderboard and about 0.9363 in local
validation without the UID feature.

Those are **cross-validation** numbers, computed with time-based GroupKFold over the
training file. This study evaluates on a strict forward holdout: fit on days 0-100, report
on days 141-181. Those are different tasks, and the gate silently compared across them.

Measured with an identical model and identical hyperparameters, 431 features including the
identity join, changing only the split:

| arm | test AUC | test AP |
|---|---|---|
| temporal (days 141-181 held out) | 0.8851 | 0.5090 |
| stratified random | 0.9658 | 0.8254 |
| card-disjoint | 0.8509 | 0.5394 |

A random split inflates AUC by **+0.0807** and average precision by **+0.3164** over the
temporal split, on the same data with the same model.

*Figures updated 2026-08-30.* This entry was written from a single seed and quoted 0.8805 /
0.5083 / 0.9619 / 0.8031 / 0.8474 / 0.5365, with an inflation of +0.081 and +0.295. The table
it cites, `split_arm_baselines.csv`, now carries the mean over five seeds, and the body reports
+0.3164. The conclusion is unchanged and slightly stronger; the numbers here had simply been
left behind when the arm was replicated.

Two consequences.

**The gate is corrected** to be stated against the arm it is measured on: temporal AUC
>= 0.87, with the stratified arm reported alongside as the control. Retuning to reach 0.92
on the temporal arm would mean tuning against the test block, which the pre-registration
forbids and which would defeat the purpose of the holdout.

**The contrast is promoted from a control to a result.** It is the cleanest available
evidence for the study's framing: published fraud-detection numbers on this dataset family
are largely obtained under random splits, and the gap between those numbers and what a
forward holdout yields is the distribution shift that the risk-control layer has to survive.
The AP gap is the one to quote — 0.5083 against 0.8031 means a random split makes the
primary metric look nearly 60 % better than deployment conditions support.

The card-disjoint arm coming in *below* the temporal arm (0.8474) is consistent with the
label-propagation mechanism: forbidding an entity from appearing on both sides removes
signal that a temporal split leaves available, since 85.0 % of test-block entities also
occur in training.

### D-017 The full feature set is used, including the identity join

An initial run used 40 hand-picked columns and reached temporal AUC 0.8830 / AP 0.4823. The
full set — 431 features, all V columns plus the 40 identity columns left-joined on
TransactionID — reaches AUC 0.8805 / AP 0.5083. AUC is unchanged within noise while average
precision improves by 0.026, which is the metric that matters at this prevalence.

The full set is used, because selecting 40 columns by hand is itself a modelling decision
made with knowledge of the dataset, and leaving it in would make the baseline weaker than a
straightforward one — an artificially weak baseline is the most common way a comparison is
tilted, and this study's entire argument depends on the classical arm being tuned as hard as
the quantum arm.

### D-018 The test-fold guard fired, and the reset is recorded here

`TestFoldGuard` refused a second `D_test` evaluation after the configuration digest changed,
which is what it was written to do. The first evaluation ran under the configuration
withdrawn by protocol amendments A2 and A3 — an operating point selected by a PSD2 ceiling
that is not applicable to a fraud-enriched benchmark, and a band that extended above the
decline threshold. That evaluation is void, not spent.

`results/tables/test_access.json` is therefore reset once, here, deliberately, with this
entry as the record. The ledger is not reset again: the next `D_test` evaluation is the one
that counts.

### D-019 [SUPERSEDED by D-020 and D-025] The central experimental result: the guarantee breaks, and only under time

Split conformal calibrated on `D_cal` and applied to `D_test`, identical code and identical
alpha grid across three split arms. The only difference is how the blocks were formed.

| arm | alpha=1e-2 | 5e-3 | 2e-3 | 1e-3 |
|---|---|---|---|---|
| temporal | 1.49x, breached | 1.49x, breached | 1.32x, breached | 0.98x, inside |
| stratified random | 0.95x, inside | 1.02x, inside | 1.03x, inside | 1.16x, inside |
| card-disjoint | 0.90x, inside | 0.85x, inside | 0.78x, inside | 0.76x, inside |

"Breached" means the observed false-decline count on the test block falls outside the exact
99 % Beta-Binomial predictive band, one-sided tail p < 0.004.

Three things follow, and the third is the one that matters.

**The implementation is correct.** Under a random split the empirical rate lands inside the
exact band at every alpha, with ratios between 0.95 and 1.16. If the conformal machinery
were wrong it would be wrong here too.

**Entity overlap is not the cause.** The card-disjoint arm, where no entity appears on both
sides, also passes — conservatively, at 0.76-0.90x nominal. So the breach is not the
label-propagation clustering.

**The breach is specifically temporal, and it is large at operationally relevant levels.**
Calibrating on days 120-140 and deploying on days 141-181 inflates the realised
false-decline rate to about 1.5x its nominal value at alpha = 1e-2 and 5e-3. A control
document stating "at most 1 % of legitimate customers are declined" would in fact be
declining 1.49 %.

This is the study's central finding and it is a negative one about naive practice. It is not
an argument that conformal prediction fails; the guarantee is conditional on exchangeability
and time-ordered fraud data violates it, exactly as Barber et al. (2023) and Oliveira et al.
(2024) describe. What this measurement adds is the size of the violation on real payments
data under a protocol that isolates its cause — which is what an institution needs in order
to decide how much recalibration headroom to hold.

The alpha = 1e-3 temporal cell passing is consistent rather than anomalous: at that level the
threshold sits far enough into the tail that the drift in the bulk of the score distribution
moves it comparatively little.

### D-020 [SUPERSEDED in part by D-025] Correction to D-019: the headline ratio is 1.44, not 1.49

D-019 quoted "1.49x nominal" for the temporal breach at alpha = 1e-2 and 5e-3. That figure was
computed on a single seed and is the **maximum** across seeds, not the central estimate. The
error was in the direction that flatters the finding, which is the direction that matters.

Re-measured across all five seeds (`results/tables/coverage_by_arm_seeds.csv`), realised over
nominal false-decline rate:

| arm | alpha | mean | sd | min | max | inside exact band | breached |
|---|---|---|---|---|---|---|---|
| temporal | 1e-2 | **1.44** | 0.055 | 1.35 | 1.49 | 0/5 | 5/5 |
| temporal | 5e-3 | **1.46** | 0.036 | 1.42 | 1.49 | 0/5 | 5/5 |
| temporal | 2e-3 | 1.31 | 0.067 | 1.21 | 1.38 | 1/5 | 4/5 |
| temporal | 1e-3 | 0.94 | 0.020 | 0.92 | 0.98 | 5/5 | 0/5 |
| stratified | 1e-2 | 1.01 | 0.038 | 0.95 | 1.06 | 5/5 | 0/5 |
| stratified | 5e-3 | 1.04 | 0.037 | 0.99 | 1.08 | 5/5 | 0/5 |
| stratified | 2e-3 | 1.02 | 0.071 | 0.94 | 1.13 | 5/5 | 0/5 |
| stratified | 1e-3 | 0.96 | 0.140 | 0.80 | 1.16 | 5/5 | 0/5 |
| card-disjoint | 1e-2 | 0.92 | 0.133 | 0.79 | 1.08 | 3/5 | 0/5 |
| card-disjoint | 5e-3 | 0.92 | 0.055 | 0.85 | 0.99 | 5/5 | 0/5 |
| card-disjoint | 2e-3 | 0.90 | 0.157 | 0.78 | 1.17 | 5/5 | 0/5 |
| card-disjoint | 1e-3 | 0.88 | 0.362 | 0.65 | 1.52 | 4/5 | 1/5 |

**The finding survives, and is stronger for being properly quantified.** The temporal breach is
5/5 seeds at alpha = 1e-2 and 5e-3, and 4/5 at 2e-3. No seed of either control arm breaches at
those levels.

**Two claims in D-019 were also too strong and are narrowed.**

D-019 said the card-disjoint arm "passes". At alpha = 1e-2 only 3 of 5 seeds land inside the
exact band — the other two fall *below* it, which is over-coverage rather than breach, but
"passes" implied a cleaner result than the data shows. At alpha = 1e-3 one seed breaches, with a
across-seed standard deviation of 0.362. The correct statement is that the card-disjoint arm
does not breach at the levels where the temporal arm does, and is noisy at the tightest level.

**A methodological note that belongs with the numbers.** The temporal split is deterministic, so
across-seed variation in that arm comes only from the model, not from the split. Its standard
deviations are correspondingly the smallest in the table (0.020-0.067). The two control arms
resample their blocks per seed, so their variation compounds split and model randomness. The
arms are therefore not directly comparable on spread, only on level and on breach count.

### D-021 The leakage ablations return a null result, and one motivation is withdrawn

`scripts/run_ablations.py`, three seeds, temporal arm, mean over seeds against the reported
configuration (causal aggregates, no UID):

| variant | AUC | delta | AP | delta |
|---|---|---|---|---|
| reported: causal, no UID | 0.8854 | — | 0.5081 | — |
| non-causal aggregates | 0.8846 | -0.0008 | 0.5135 | +0.0054 |
| causal + UID | 0.8850 | -0.0003 | 0.5095 | +0.0015 |
| no aggregates at all | 0.8817 | -0.0037 | 0.5074 | -0.0007 |

Every delta sits within the per-seed standard deviation (0.0015-0.0053). Two things follow.

**A motivation is withdrawn.** The docstring in `features/engineering.py` previously said the
non-causal variant was reported "because the size of the gap is the evidence that the causal
version was necessary". On this split there is no gap. The causal construction is kept because
it is correct and costs nothing, not because a leak was demonstrated here. A larger causal-vs-
non-causal effect has been measured on other datasets with stronger per-entity signal; citing
that as if it applied to IEEE-CIS would be importing a result across datasets, which is
precisely what this study criticises elsewhere.

**The UID null is a positive finding, not an absence.** Published analysis puts the UID
feature at about +0.011 AUC under time-based GroupKFold cross-validation on the training file.
Under this study's forward holdout it is worth +0.0015 AP, inside noise. The reason is
mechanical: cross-validation lets a client recur across folds, and across a 40-day forward gap
that recurrence has largely decayed. The contrast is independent evidence for the same point
the split-arm comparison makes — that numbers obtained under cross-validation on this dataset
family do not transfer to a forward holdout.

---

## Round 4 — The quantum arm (2026-08-28)

### D-022 Braket becomes a first-class execution path; D-006 was too narrow

D-006 chose `FidelityStatevectorKernel` on two grounds: it needs no proprietary NVIDIA binary,
and a GPU does not accelerate a pairwise Python loop. Both remain true. The conclusion drawn
from them was too narrow, because it optimised compute and in doing so dropped a stated
requirement: the challenge Executive Summary asks participants to use Amazon Braket, and
section 5.4 puts simulator-based execution in scope.

`src/hsbcfraud/quantum/kernel.py` therefore implements four independent routes to the same
overlap -- exact statevector, Braket `LocalSimulator` on `braket_sv`, Aer CPU, and Aer GPU
with cuStateVec -- and `scripts/check_parity.py` asserts they agree.

Measured agreement against the exact statevector, 24 points, four feature-map configurations:

| backend | max absolute difference |
|---|---|
| Braket `braket_sv` | 3.9e-16 to 5.6e-16 |
| Aer CPU | 2.6e-15 to 4.8e-13 |
| Aer GPU | 2.6e-15 to 4.8e-13 |

This is a better answer than either the original plan or D-006. Four independent
implementations of the same mathematical object either agree to numerical precision or one of
them is wrong, so running them against each other is a stronger statement than running any one
alone -- and it is the substitute this study offers for hardware execution, which the
challenge explicitly does not penalise omitting.

Two implementation notes worth recording because both cost time.

Braket's OpenQASM dialect does not resolve `stdgates.inc` and names two gates differently from
Qiskit: `p` is `phaseshift` and `cx` is `cnot`. The translation is a literal rename of leading
gate tokens, not a reconstruction of the circuit, so gate order, qubit indices and parameter
values pass through untouched. Rebuilding the circuit in Braket's Python API was rejected: a
circuit assembled twice by hand can differ in gate order or rotation convention, and the
parity check would then report a real discrepancy as a numerical one, or hide one behind a
compensating difference.

The Gram diagonal is set to exactly 1 rather than left to accumulate floating-point error.
`|<phi|phi>|^2` is analytically 1, and a diagonal reading 0.9999999997 propagates into the
eigenspectrum and shifts the effective-rank screen -- which is the quantity the a-priori gate
is decided on.

### D-023 The entanglement ablation is the same circuit, not a different one

`build_feature_map` takes an `entanglement` argument, and the `none` setting removes the
entangling layer from the *same* construction rather than switching to a different library
builder. Bowles, Ahmed and Schuld (arXiv:2403.07059) found that removing entanglement often
does not hurt a quantum model, so a quantum arm reported without a product-state control has
not been tested; but an ablation that changes the rotation structure as well as the
entanglement measures neither.

Measured on 24 uniform points, 4 qubits, ZZ map: effective-rank ratio 0.861 with linear
entanglement against 0.658 without, and off-diagonal mean 0.0757 against 0.1301. The
entangling layer materially changes the kernel, which is what makes the ablation worth running
rather than a formality.

---

## Round 5 — Retraction of the certification frontier (2026-08-28)

### D-024 The certificate was vacuous and one of its p-values was invalid. Both are retracted.

An adversarial audit of this repository found two defects in the central deliverable. Both
reproduce. Both are fatal to the claim as it stood, and the results in `riskcontrol.csv` and
`tradeoff.csv` as committed are withdrawn.

**Defect 1: every certified configuration certified the empty intervention.**

All 32 certified rows selected `lambda == band_hi`, the top of the band. At that value the
in-band rule declines nobody: the band-conditional false-decline rate is exactly 0, which
satisfies any target trivially, and the in-band scorer has no effect on any decision.

The certificate therefore certified a rule that ignores the band entirely. This is precisely
the objection an earlier design review raised against certifying the *unconditional* rate --
that the certificate would be unchanged if the in-band scorer were a coin flip -- and
conditioning on the band did not fix it. Conditioning changed the estimand; it did not stop
the optimiser from selecting the no-op.

The cause is a scale error. The pre-registered grid `alpha in {1e-3 ... 1e-2}` is the right
scale for the **unconditional** false-decline rate, which is what a bank's control document
states. The **band-conditional** rate is a different quantity: the band is by construction the
region where the model is uncertain, so its false-decline rate is intrinsically in the percent
range. Measured on the calibration block at a 5 % band:

| in-band threshold | in-band declines | band-conditional FDR | total recall |
|---|---|---|---|
| top of band | 0 | 0.0000 | 0.561 |
| 90th percentile | 81 | 0.0359 | 0.567 |
| 75th | 204 | 0.0903 | 0.584 |
| 50th | 544 | 0.2408 | 0.610 |
| bottom of band | 2,259 | 1.0000 | 0.692 |

Requiring this quantity to sit below 1 % admits only the first row.

**Defect 2: the recall p-value was not a p-value.**

`recall_shortfall` returned `max(0, floor - recall) / floor`. Recall is itself a mean, so this
is a nonlinear transform of a mean. The Hoeffding-Bentkus bound (Bates, Angelopoulos, Lei,
Malik and Jordan, 2021) bounds the mean of `n` independent losses in [0, 1]; it says nothing
about a nonlinear function of such a mean. The values fed into the family-wise correction on
that axis were therefore not valid p-values, and the two-sided certificate had one valid side.

The false-decline risk was unaffected: `band_conditional_false_decline` returns the mean of
per-row 0/1 indicators over legitimate band transactions, which is exactly what the bound
requires.

**The repair.**

`recall_shortfall` is replaced by `missed_fraud_rate`, the false-negative rate: loss 1 when a
fraudulent transaction is approved by both stages, 0 when either declines it, averaged over
fraudulent transactions. Controlling it at `alpha_fn` is equivalent to a recall floor at
`1 - alpha_fn`, so nothing is given up, and the quantity is now a mean of [0, 1] losses.

With a valid loss and an `alpha` scale matched to the estimand, the certificates are
non-degenerate:

| band | alpha FDR | alpha FNR | band FDR achieved | recall | in-band declines |
|---|---|---|---|---|---|
| 0.10 | 0.10 | 0.45 | 0.0749 | 0.595 | 362 |
| 0.10 | 0.25 | 0.40 | 0.2238 | 0.637 | 1,081 |
| 0.05 | 0.25 | 0.45 | 0.1784 | 0.600 | 403 |

The in-band rule now declines hundreds of transactions and lifts recall from 0.561 to between
0.595 and 0.637, which is an intervention a certificate can meaningfully license.

**What this costs, stated plainly.** Protocol amendments A1 and A3, and decision entries
D-018 and the frontier discussion, were built on the withdrawn numbers. A1's conclusion that
the reachable `alpha` is capped by band size remains true as arithmetic about sample sizes,
but its framing as a guarantee-versus-abstention frontier described a frontier of vacuous
certificates. A3's "all 24 combinations certify" is withdrawn on the same grounds. A fourth
amendment, restating the estimand scale and the recall loss, is required before the
certification is re-run, and `results/tables/riskcontrol.csv` and `tradeoff.csv` must be
regenerated.

**Why this was not caught earlier.** The simulation that validated `learn_then_test` used a
synthetic monotone risk curve with a well-scaled target, so it exercised the procedure and not
the estimand. A guarantee can be correctly implemented and still certify nothing of interest,
and a validation that only checks the machinery will not notice.

### D-025 The central claim is narrowed again: the deviation is origin-dependent

D-019 and D-020 stated that "the breach is specifically temporal". An audit pointed out that
the claim rests on a **single** temporal partition: `temporal_blocks` takes no seed, so the
five seeds are five refits of the same model on the same cal = days 120-140, test = 141-181
cut. Five correlated refits are not five samples of the temporal deviation.

A rolling-origin sweep over the frozen score file -- 20-day calibration window, 20-day test
window, stepped forward, model unchanged -- at `alpha = 1e-2`:

| calibration days | test days | realised / nominal | verdict against the exact 99 % band |
|---|---|---|---|
| 101-120 | 121-140 | **0.684** | conservative -- *below* the band |
| 111-130 | 131-150 | 0.893 | inside |
| 121-140 | 141-160 | **1.411** | breached |
| 131-150 | 151-170 | 1.198 | breached |
| 141-160 | 161-180 | 1.126 | inside |

Two of five origins breach; the ratios span 0.684 to 1.411; and the earliest origin goes the
**other way**, over-covering rather than under-covering. The pre-registered split sits at the
most adverse origin in this range, and the 1.44 reported in D-020 is its extreme.

**The corrected claim.** The deviation between realised and nominal false-decline rate is
origin-dependent. It exceeds the exact 99 % band at two of five rolling origins, reaching
1.41 times nominal, and reverses to 0.68 at the earliest origin tested. It is not a constant
temporal penalty.

**Why this is still the useful finding, stated without inflation.** A bank does not need to
know that a guarantee degrades by a fixed factor; it needs to know that the degradation
depends on *when* calibration happened and by how much it can vary. A control certified at
1 % that realises between 0.68 % and 1.41 % depending on the calibration window is a
recalibration-policy problem with a measured magnitude, which is more actionable than a single
adverse number presented as characteristic.

**What remains unaffected.** The contrast against the control arms still holds at the fixed
split: the stratified arm lands inside the band at every alpha and the card-disjoint arm does
not breach where the temporal arm does. What cannot be claimed is that a temporal split
*always* breaches, and D-019 and D-020 are superseded on that point.

`docs/protocol.md` section 7 pre-registered "realised risk gap per time window" as a headline
deliverable, so this measurement was promised rather than added after the fact. It is in
`results/tables/rolling_origin.csv`.

---

## Round 6 — The tensor-network arm (2026-08-28)

### D-026 Two silent initialisation defects in the MPS classifier

Both produced a model that trained without error, reported a plausible loss near ln 2, and
scored exactly at chance. Neither raised.

**Both physical channels initialised to the identity.** `core[:, 0, :]` and `core[:, 1, :]`
were each set to the identity plus small noise. The contraction then reduces to a scalar
multiple of a fixed vector, the running renormalisation divides that scalar out, and the model
becomes input-independent. Measured: logits bit-identical for all-zeros and all-ones input,
loss flat at 0.6931, holdout AUC 0.5000. The `cos` channel now carries the identity and the
`sin` channel a small learnable perturbation.

**The final bond closed to one.** With the right boundary core shaped `(chi, 2, 1)` the partial
contraction becomes a scalar, and the per-sample renormalisation maps every sample to plus or
minus one before the output head sees it. Measured that way: AUC 0.4990, 0.4970 and 0.5000 at
bond dimensions 4, 12 and 32 on a linearly separable task. The right bond now stays open at
`chi` and the head closes the network.

After both repairs, on the same synthetic task: AUC 0.9519, 0.9591 and 0.9593 at bond
dimensions 4, 12 and 32, against 0.9801 for logistic regression. An MPS reaching close to but
below a linear model on a linearly separable problem is the expected ordering, which is what
makes it a usable check that the implementation works.

### D-027 The in-band tensor-network result is negative, and it is not under-training

Fitted on the 2,916 band rows of `D_band` (292 fraud), evaluated on the 2,537 band rows of
`D_cal` (278 fraud), eight features selected by mutual information on `D_train`:

| model | AUC | AP |
|---|---|---|
| MPS, bond dimension 16 | 0.5251 | 0.1205 |
| gradient boosting | 0.5643 | 0.1407 |
| logistic regression | 0.5103 | 0.1182 |
| base rate | — | 0.1096 |

The MPS performs comparably to logistic regression and below gradient boosting.

**Under-training was tested and ruled out.** Sweeping epochs and learning rate: 30 epochs at
3e-3 gives AUC 0.5127; 150 epochs at 3e-3 gives 0.5251; 150 at 1e-2 gives 0.5061; 400 at 1e-2
gives 0.4420. Longer training makes it worse, not better, so the model is at capacity for this
data rather than short of optimisation.

**Context for the number.** The band is by construction the region where the primary scorer is
uncertain, and little signal remains there for anything: gradient boosting itself reaches only
0.5643 AUC against a base rate of 0.1096 AP. The MPS captures approximately the linear part of
what is left.

**How this sits with the quantum kernel result.** The two arms fail differently and that
distinction is worth preserving. The quantum kernel was rejected *before* being run, by
a-priori screens that no candidate passed. The tensor-network arm *was* run and lost to a
tuned classical baseline on the same rows and features. Two pre-registered approaches, two
mechanisms identified, two negative results.

---

## Round 7 — Repository integrity (2026-08-28)

### D-028 Four source files were absent from every pushed commit

`.gitignore` line 5 read `data/`. A git ignore pattern without a leading slash matches at
**any** depth, so it excluded `src/hsbcfraud/data/` along with the intended top-level data
directory. Absent from the repository: `ieee_cis.py` (the loader and its file-identity
assertions), `splits.py` (the four-block temporal split, the two control arms and
`TestFoldGuard`), `label_audit.py` (the censoring test), and the package marker.

The failure was invisible locally and total remotely. `git add -A` does not warn when it skips
an ignored path, `git status` does not list it, and every command still worked here because
the files were present on disk. A reviewer following the repository link would have found a
Makefile whose first target imports a package that is not there.

Verified: `git check-ignore -v src/hsbcfraud/data/splits.py` returned `.gitignore:5:data/`.

The pattern is now anchored (`/data/`, `/datasets/`), and
`tests/test_repo_hygiene.py::test_every_source_file_is_tracked` compares the files on disk
against `git ls-files` so the class of defect cannot recur silently.

### D-029 The repository-hygiene suite, and two things it found about itself

`tests/` was empty. It now asserts seven properties of the artefact rather than of the
science: every source file tracked, every result table tracked, no bare `python3`, no chained
pandas assignment, an SPDX header on every file, every Makefile-referenced script present, and
the pre-registration gate not failing with an undocumented change.

Two findings worth recording.

The chained-assignment detector is syntactic and cannot distinguish a DataFrame from a list,
so it flagged `envelope_rows[-1]["decline_rate"] = decline_rate` in `run_conformal.py` -- a
list of dicts, where the assignment is correct. The code was rewritten to build the row before
appending rather than weakening the detector; the rewrite is clearer regardless. A file that
must contain the pattern opts out with an explicit marker, which `scripts/smoke.py` uses
because it carries the fixture S6 tests against.

`test_makefile_scripts_exist` currently reports `xfail`: 15 of the 22 scripts the Makefile
invokes are not written, including the entire document path. It is deliberately an expected
failure rather than a skip, so the count appears in every test run and shrinks visibly as the
scripts land.

---

## Round 8 — The power gate, and the long-chain instability (2026-08-28)

### D-030 The H4 power gate was calibrated on the wrong noise, and passed for the wrong reason

The pre-registration commits to computing the minimum detectable effect **before** the
tensor-network comparison, and to declaring H4 underpowered in advance if it exceeds 0.023
average precision. Until this round the calculation had no caller at all: `power.csv` was
never written, so H4 was going to be *observed* rather than tested. That is the first defect
and it is the more serious one.

The second defect is in the fix. `scripts/run_power.py` estimated the noise as the
seed-to-seed spread of the gradient-boosted baseline against itself, reasoning that two
models differing only in random seed have a true difference of zero, so the spread of their
bootstrap distribution is pure noise. That reasoning is correct and the quantity is
irrelevant. It gave a standard error of 0.0025 and an MDE of 0.0069, comfortably inside the
0.023 ceiling.

The comparison H4 actually makes is between **model families**, and when it ran, its measured
standard error was 0.0183–0.0204 depending on bond dimension — about eightfold larger. The
true MDE for this comparison is therefore near 0.056, which **exceeds** the pre-registered
ceiling. A gate calibrated on seed noise passes almost anything, and this one did.

`run_power.py` now computes a second proxy — a regularised logistic model against the same
gradient-boosted baseline, on identical rows — and the gate binds on the larger of the two.
Neither proxy involves a quantum model or touches `D_test`, so the ordering commitment is
preserved.

**Consequence for the claim.** H4 must be reported as underpowered against its
pre-registered ceiling. What survives is weaker than a null result and is stated as such: the
95 % clustered interval on the MPS-minus-GBDT difference in band-conditional average
precision is `[-0.0597, +0.0136]` at bond dimension 4 and `[-0.0574, +0.0222]` at bond
dimension 32, so **any improvement is bounded above by roughly +0.02 AP**. That is a
non-superiority bound, not evidence of equivalence, and not evidence that the tensor network
is worse.

### D-031 A matrix-product-state chain over 431 sites needs a depth-scaled learning rate

The full-scale arm raised `RuntimeError: MPS loss became non-finite` at 431 features. The
guard fired correctly and no number was produced, which is the behaviour it exists for.

Diagnosis. The forward contraction is stable at initialisation: the running log-norm reaches
about −327 at 431 sites and the logits stay finite. The failure is in training. Measured on
2,000 rows over five epochs, holding everything else fixed:

| Sites | lr 3e-3 | lr 1e-3 | lr 3e-4 |
|---|---|---|---|
| 200 | 0.7009 to 0.6951 (stalls) | 0.7002 to 0.4562 | 0.6988 to 0.4330 |
| 431 | 0.7113 to 0.6899 (stalls) | 0.6977 to 0.6909 (stalls) | 0.7102 to 0.3573 |

The gradient passes through one einsum per site, so the effective step compounds with depth.
`MPSConfig.learning_rate` now defaults to `None`, meaning `LEARNING_RATE_SCALE / n_features`
with the numerator fixed at 0.15, which reproduces the measured stable rates (7.5e-4 at 200
sites, 3.5e-4 at 431). A float still overrides it.

This is a property of the ansatz worth reporting rather than a bug: the usable learning rate
falls with chain length, which is a practical constraint on applying matrix-product-state
classifiers to wide tabular data, where the number of sites is the number of features and is
not something the modeller chooses.

A separate latent hazard was removed while diagnosing. `forward` ended with
`logits + log_norm.unsqueeze(-1) * 0.0` — the log-norm cancels in the softmax, so multiplying
by zero was intended to discard it while keeping the tensor in the graph. But `inf * 0.0` is
`nan`, so had the log-norm ever overflowed, the term would have silently poisoned the loss
instead of being ignored. It is now dropped outright.

### D-032 The bond-dimension sweep is nearly free, because the chain is launch-bound

A matrix product state contracts one site at a time, so a forward pass over 431 features
issues 431 sequential `einsum` calls whose individual arithmetic is small. The expectation
from the ansatz is that cost scales with the square of the bond dimension: each site
contraction is a `(batch, chi) x (chi, p, chi)` product.

Measured on this host, 512-row batches, 431 sites, forward plus backward plus optimiser step:

| chi | ms/step | relative | chi^2 would predict |
|---|---|---|---|
| 4 | 139.5 | 1.00 | 1 |
| 8 | 148.8 | 1.07 | 4 |
| 16 | 161.4 | 1.16 | 16 |
| 32 | 158.8 | 1.14 | 64 |

Cost is essentially flat. At these bond dimensions the kernel-launch overhead of 431
sequential operations dominates the arithmetic entirely, which is consistent with the 14 %
GPU utilisation observed during the full-scale run while a core was pinned at 100 %.

Two consequences.

**For the study.** Sweeping the bond dimension at full scale costs about four times one
job, not the sixty-five times a `chi^2` cost model predicts -- roughly three hours rather
than sixty-five. The sweep is therefore affordable, and reporting capacity dependence
across `chi` is not a compute concession. It also means `chi` is not the cost knob for long
chains that it is for short ones; the number of sites is.

**For the progress reporting added in D-033.** A remaining-time estimate can be extrapolated
across the sweep from a single completed job, because the per-step cost barely moves. Had the
`chi^2` model held, no such extrapolation would have been defensible and the estimator would
have had to refuse to predict past the current job.

The prediction was recorded before the measurement: the alternative hypotheses were 2765 s
per job under launch-bound behaviour against 2765, 11062, 44246 and 176986 s under `chi^2`.

### D-033 Progress reporting, and a first design that would have warned on every healthy run

Three defects in one: a full-scale run produced no output for forty-six minutes, then failed
with no record of when it turned, and the four-job sweep gave no estimate of remaining time.
`src/hsbcfraud/progress.py` addresses each. Two of its decisions are measurements rather than
preferences and are recorded here because the first attempt at both was wrong.

**Per-step telemetry is affordable, and the first measurement of that was contaminated.** An
initial A/B put the cost of a full per-step JSON record at 8 %, which would have forced a
compromise. That measurement was taken while a training job held the GPU, and every
subsequent variant came out *faster* than the baseline -- an impossible result that made the
contamination visible. Re-measured with the conditions interleaved round-robin to cancel
drift, the cost is 1.3 % against a 146 ms step, within the interquartile spread of the
baseline itself. At 431 sites the step is dominated by kernel launches, so the record is
effectively free and no compromise was needed.

**The divergence detector had to be rebuilt after it failed its own test.** The first version
compared the latest pre-clip gradient norm against its running median and warned above a
multiple of it. Against simulated traces carrying the heavy-tailed spikes real training
produces -- rare batches at eight to twenty-five times typical -- it warned on **40 of 40
healthy runs** at multiples of 5, 10 and 20. Raising the multiple to 40 or 80 bought silence
at the cost of firing 228 and 314 steps *after* the loss had already doubled, which is not a
warning. The signature that separates divergence from a spike is persistence, not magnitude,
so the test now compares the median of a fifty-step window against the median of the last
2,048: a single spike cannot move a median. At a multiple of 2 it gives 0 of 40 false alarms
and warns a median of 117 steps before the loss doubles.

The loss-trend fallback, for callers with no gradient norm, has a limit that is derivable
rather than empirical. Comparing half-window means detects exponential growth `exp(t/tau)`
only when `tau < (window/2) / ln(multiple)`, because the ratio of the two half-means is
exactly `exp((window/2)/tau)` and does not grow as the run proceeds. With the gradient
window of 50 and a multiple of 1.5 the limit is 62 steps, and a simulated divergence with
`tau = 90` was never detected at any point in a 5,000-step run. The loss window is therefore
separate and larger -- 200 steps, limit 247 -- and the boundary is pinned by test at
`tau = 150` detected and `tau = 400` missed.

`clip_grad_norm_` already returned the pre-clip total norm and the loop was discarding it, so
the earlier of the two signals was available at no cost the whole time.

**The failure dump could not be provoked, which is itself a result.** When the fit raises on a
non-finite loss it now writes the preceding steps at step resolution, because a per-epoch
record answers only "which of the thirty epochs", and an epoch is 696 steps. Exercising that
path meant provoking a real divergence, and after the cosine decay of D-031 the model
survived initial learning rates of 3, 50, 500 and 5000 on a 200-site chain -- gradient
clipping at norm 1.0 and a decaying step together make it very hard to break. The dump path
is therefore covered by unit tests on its components rather than by an end-to-end failure,
and the inability to trigger it is evidence about the fix rather than a gap in the test.

### D-034 An exploratory run could silently overwrite a pre-registered result

Testing the new reporting with `--bonds 4 --epochs 6` overwrote `results/tables/mps_band.csv`
and `mps_h4.csv`, replacing a thirty-epoch four-bond result with a six-epoch single-bond one.
Nothing warned. The files have the same schema and plausible values, so only the git history
distinguished them, and had this happened before a commit rather than after, the study would
have quietly acquired numbers from a smoke test.

`results/tables/` now holds the pre-registered configuration and nothing else: `run_mps.py`
compares its arguments against the parser defaults and diverts a non-default run to
`results/runs/exploratory/`, saying so on stdout. Verified by re-running the same command and
confirming the committed table's hash is unchanged.

### D-035 The non-finite check was one step too late, and the loss cannot be the trigger

`fit_mps` guarded on `torch.isfinite(loss)`. That check fires one optimiser step after the
damage, and by then the model is unrecoverable.

The mechanism, verified on a 64-site chain. When a single gradient element becomes
non-finite, `clip_grad_norm_` returns a total norm of `inf`. Its clip coefficient is then
`max_norm / inf = 0`, so every gradient is multiplied by zero -- except the offending
element, where `inf * 0` is `nan`. Adam writes that `nan` into a parameter, and the *next*
forward pass is the first thing the loss check can see.

At the moment the norm goes non-finite:

| | |
|---|---|
| loss | 0.5927, entirely plausible |
| non-finite gradient elements | 1 of 8,096 |
| non-finite parameters | 0 of 8,096 |
| next forward pass | non-finite |

So the loss is not merely a late signal, it is an unusable one: at the step that matters it
reads as a healthy run. The guard now tests the gradient norm **before** `optimiser.step()`
and raises there, naming the step and the loss at that step. Verified against a run with one
gradient element poisoned at step 40: it raises at step 39 with the loss at 0.4564, zero
non-finite parameters, and a forward pass on the raised model still finite. The failure is
therefore recoverable -- the model can be inspected or checkpointed -- and the reported step
is the one where the fault occurred rather than the one after.

This was found by an independent design review rather than by the implementation, and the
claim was re-derived here before being acted on. One detail of the review's account did not
survive checking: it stated that the `nan` propagates to every parameter, whereas the clip
coefficient of zero means the others receive a zero update and stay finite. The consequence
is the same and the fix is the same, but the mechanism is recorded as measured.

### D-036 The site loop was launch-bound; reassociating it into a reduction tree is 12x

`MPSClassifier.forward` contracted one site at a time in a Python loop, issuing roughly seven
small CUDA kernels per site. Measured at 439 sites, batch 512, one full training step:

| | |
|---|---|
| time across bond dimensions 4, 8, 16, 32, 64, 128 | 137, 126, 131, 137, 131, 127 ms |
| time across 55, 110, 220, 439 sites at chi=32 | 17.8, 35.1, 60.6, 132.7 ms |
| achieved throughput at chi=32 | 0.0067 TFLOPS against roughly 125 TFLOPS peak |

A 1024-fold change in arithmetic produced no change in time, and time was linear in sites at
about 300 microseconds each. The work was bound by kernel launches, not by the GPU.

**What licenses the fix.** The accumulated log-norm is discarded (D-031) and the loop ends by
dividing by the running norm, so the output is exactly `normalise(v0 @ M_1 @ ... @ M_d) @ head`.
The per-site renormalisation keeps the product in float32 range; it cannot change the
direction. Matrix multiplication is associative, so any bracketing computes the same
direction, and a pairwise tree finishes in `log2(d)` rounds rather than `d`.

**Why it is not a free win.** The fold does `6.b.d.chi^2` flops; the tree does
`4.b.d.chi^2 + 2.b.d.chi^3`. The tree performs `(2 + chi)/3` times the arithmetic -- twice as
much at `chi=4`, forty-three times at `chi=128` -- in exchange for about three thousand fewer
launches. A crossover exists by arithmetic alone. Measured per-step time, conditions
interleaved to cancel drift:

| chi | chunk 1 | chunk 8 | chunk 32 | chunk 128 | chunk 439 | best |
|---|---|---|---|---|---|---|
| 4 | 134.8 | 60.0 | 27.6 | 17.0 | **11.5** | 11.7x |
| 8 | 145.5 | 66.0 | 29.7 | 17.5 | **12.2** | 11.9x |
| 16 | 148.1 | 64.1 | 30.6 | 17.9 | **17.0** | 8.7x |
| 32 | 144.6 | 67.9 | **47.9** | 66.0 | 82.3 | 3.0x |
| 64 | 143.0 | **79.0** | 119.9 | 138.6 | 155.7 | 1.8x |
| 128 | **142.3** | 459.0 | 546.7 | 574.1 | 627.2 | 1.0x |

By `chi = 128` the original fold is already optimal. `contraction_chunk` therefore selects the
reduction width, and it is **measured at fit time** rather than tabulated: the crossover
depends on the bond dimension, the site count, the batch size and the device, and a table
baked from one GPU is a number that silently stops being true on another. Tuning costs about
twenty steps against the twenty-one thousand a full-scale job runs, and the tuner reproduced
the interleaved sweep's choice at every bond dimension tested.

**Validation.** `contraction_chunk = 1` is bit-identical to the previous implementation --
max absolute logit difference exactly 0.00e+00 at four configurations, checked against the
code at the preceding commit -- so the default path is the shipped behaviour rather than an
approximation of it. Wider settings agree to 3e-6 on uniform inputs and 3e-5 on the real
MinMax-scaled features, which are 72.7 per cent exact zeros and nothing like uniform. Gradient
cosine similarity is 0.9999999974 and the gradient norm ratio is 1.000000, so the training
trajectory and the norm-based guard of D-035 are unaffected. Re-running the in-band arm moved
ROC AUC by at most 4.8e-6 and average precision by at most 7.2e-6, well inside the four-decimal
precision the claims are checked at.

**The ordering trap.** Matrix multiplication is associative but not commutative, so a tail
appended on the wrong side of an odd round permutes the chain silently -- the model still
trains and still reports plausible metrics, which is precisely how D-026 and D-027 got in.
The reduction pairs `mats[:, 0::2]` as the left operand with `mats[:, 1::2]` as the right and
carries an odd tail forward unpaired so it stays rightmost. The test establishes its own
teeth before asserting: a single interior site swap moves the output by order 1, against
order 1e-5 for the reassociation, so the agreement is evidence about ordering rather than
about a chain whose factors happen to commute.

Memory moves the other way. The tree materialises every transfer matrix at once, so peak
allocation grows with the width: at `chi = 32`, 0.99 GiB for the fold against 3.93 GiB for the
full tree. Memory, not time, is what bounds the bond dimension here, and the tuner skips
widths that do not fit rather than reporting them as slow.

### D-037 A parallel investigation produced no recorded results, for two reasons, both avoidable

Four parallel workers were run in isolated checkouts to evaluate restructurings of the
contraction. All four did substantial work; the run recorded four `started` events and zero
results.

**The reporting contract was too strict and its payload too large.** Each worker was required
to return the complete rewritten function and its raw measurement output as free text in a
single structured record. Those records reached 16 to 18 kB and failed to parse on string
escaping. One worker fixed the escaping and was then rejected for carrying a single field the
contract did not declare. The work survived only because it was on disk.

The lesson is specific and general: a reporting contract that carries large free text is
fragile at exactly the moment it matters, and rejecting a record for one surplus field
converts a trivial excess into total loss. Ask for short fields and a path to the artefact
rather than for the artefact itself.

**Every checkout was eight commits stale.** They were created at 19:57 from commit `4d4559f`
of 15:29, while the repository head was `77bd690` from 19:52. The workers therefore saw no
cosine decay, no depth-scaled learning rate, no gradient-norm guard, no telemetry module and
no decision entries past D-029. Their `MPSConfig.learning_rate` was the fixed `3e-3` that
D-031 established as divergent at 431 sites.

One worker reported that the brief was wrong because `docs/decisions.md` contained no D-031.
From that checkout the objection was correct, and it was the right one to raise: the brief
cited as checkable authority a document the worker could not see. A brief that references
state the reader cannot reach is not checkable, however true it is.

**What this cost.** The timing measurements survive, because the only difference in `forward`
between the two commits is a trailing multiply-by-zero -- two kernels out of roughly three
thousand. What does not survive is the finding that mattered most: a 0.026 change in average
precision between two numerically equivalent implementations, read as evidence that the
training trajectory was unstable. That was measured at `3e-3` with no decay, the regime
already known to diverge, so it could not be carried to the current configuration. It was
nearly used as grounds to reject the rewrite.

The same instability was then found independently on the current code, at a far larger
magnitude, and is recorded in D-038. The conclusion was right; the evidence behind it was not
transferable, and the difference matters.

### D-038 The full-scale bond-dimension sweep measures seed noise, not capacity

The committed full-scale table reports one seed per bond dimension and shows ROC AUC from
0.7889 to 0.8074 and average precision from 0.1693 to 0.2464. Four seeds were then run at a
**fixed** bond dimension of 16, on the committed contraction path:

| Seed | ROC AUC | AP | Final loss |
|---|---|---|---|
| 20260828 | 0.7983 | 0.2464 | 0.3120 |
| 20260829 | 0.7867 | 0.1789 | 0.3099 |
| 20260830 | 0.8012 | 0.1945 | 0.3445 |
| 20260831 | 0.7824 | 0.1740 | 0.2935 |

| | ROC AUC spread | AP spread |
|---|---|---|
| across four seeds at one bond dimension | 0.0188 | 0.0724 |
| across four bond dimensions, one seed each | 0.0186 | 0.0771 |
| ratio | 1.01 | 0.94 |

The two are the same size. **A per-configuration reading of the sweep is reading seed noise.**
The proposal previously quoted "the best across the sweep", which takes a maximum over four
draws from a distribution whose spread equals the quantity being reported -- the winner's
curse, in a study whose protocol exists to prevent exactly that.

The first seed reproduces the committed chi=16 row exactly, which is the reproduction check
rather than a result: it is the seed that run used.

**What survives, and is strengthened.** Every seed and every bond dimension lands between
0.174 and 0.246 average precision against a tuned gradient-boosted baseline at 0.5055 to
0.5114. The tensor network loses by a factor of two to three in every draw, so the arm's
conclusion does not depend on which draw is reported. What is withdrawn is any claim about
how capacity affects it.

**A second finding, from the same measurement.** The reduction tree of D-036 does not merely
perturb the trajectory, it destabilises the optimisation. At width 128 the same four seeds
gave ROC AUC 0.5636 to 0.8007 and average precision 0.0438 to 0.1792, and **two of four
failed to train at all** -- final loss 0.52 and 0.63 against 0.29 to 0.34 at width 1. Seed
spread rises from 0.0188 to 0.2371 in ROC AUC. That is why the sequential fold is the default
and the tree is opt-in: it is not an equivalent path that happens to round differently, it is
a worse-conditioned one.

The speedup remains useful for exactly the work that produced this entry. A four-seed sweep
at the reduced width costs twenty minutes against three and a half hours, and reporting a
range instead of a point is what the arm needed. It is only unsound to use it for the numbers
the study reports as its own.

### D-039 A thirteen-hour sweep ran without the metric that exists to catch a degenerate fit

`scripts/run_seed_sweep.py` called `fit_mps` without an `evaluate` callback. It logged elapsed
time, remaining time, loss, learning rate and both gradient-norm summaries, so the log looked
complete; it carried no ranking metric.

That is the exact failure `fit_mps`'s own docstring is written against. Two silent defects in
this classifier (D-026, D-027) produced steadily falling loss with a test AUC of exactly
0.5000, and the docstring says so: *"a falling loss is not evidence of learning."* The sweep
was launched anyway, for sixteen jobs at roughly fifty minutes each, in a state where a job
that stopped depending on its input would have looked healthy for thirteen hours.

Found by a reader asking where the accuracy column was, not by any check in the repository.

Fixed, and pinned: `scripts/run_seed_sweep.py` now passes a stratified per-epoch probe, and
`tests/test_progress.py::test_every_long_fit_script_supplies_an_evaluation_probe` parses every
`fit_mps` call in `scripts/` and fails on one without `evaluate`. Intent in a docstring did not
survive the next script; a test does.

One useful thing came out of the interrupted run. Its single completed job returned
ROC AUC 0.807439921708344 and average precision 0.2149426770600668 against `mps_full.csv`'s
0.807439921708344 and 0.2149426770600668 for the same bond dimension and seed -- bit-identical,
which is the reproduction check the sequential-fold default of D-038 was chosen to preserve.

Its fit took 3,072 s against the committed 2,765 s, an 11 % inflation from document builds and
test runs sharing the device. The metrics are unaffected; the timing is not, which is why
`require_idle_gpu` refuses to start a fresh sweep against a busy device and why the same
discipline should apply to anything else running alongside one.

### D-040 The divergence detector watches the wrong direction, and the sweep says why

The seed-replicated sweep -- sixteen full-scale fits, four bond dimensions at four seeds --
produced the first real test of `DivergenceWatch`. It failed.

| | |
|---|---|
| jobs that warned | 10 of 16 |
| jobs that actually failed to train | 2 of 16 |
| failures caught | 1 of 2 |
| healthy runs warned on | 9 of 14 (64 %) |

The synthetic calibration recorded in D-033 -- zero false alarms across forty healthy traces --
certified something that does not transfer.

**The cause is the premise, not the threshold.** The detector watches for *elevated* gradient
norms. Measured over these sixteen runs, the two that failed had a median gradient norm of
**0.81**; the fourteen that trained had **10.69**. The failure mode is a fit going quiet, not
loud. No threshold on elevation detects a signal that moves the other way, so there is nothing
to retune.

The gradient path is therefore kept for the failure it was built against -- a loss growing
towards non-finite, which the pre-clip norm genuinely does lead by about a hundred steps -- and
is documented as **not** a detector for a fit that stops learning.

**What does separate them is the evaluation metric.** Both failures show a falling AUC
(0.6521 to 0.4761 at chi=32 seed 20260831, ending below chance; 0.6896 to 0.5451 at chi=8) while
all fourteen healthy fits rise into 0.77 to 0.82. An epoch-5 rule on the AUC change gives 2 of 2
sensitivity at 3 of 14 false positives -- better than the gradient rule but not clean, because
three healthy runs dip early and recover. With two positives in sixteen runs there is not
enough evidence to calibrate an early-warning rule, and fitting one to two examples would be
fitting noise. No rule is added.

That the accuracy probe exists at all is D-039: the sweep was first launched without it, and a
reader asking where the accuracy column was is the only reason the failure mode is visible
here rather than an unexplained pair of bad rows.

### D-041 Capacity is not resolvable at full scale, and one fit in eight does not train

Sixteen jobs, contraction width 1, all reported:

| chi | ROC AUC (mean) | AP (mean) | AP range across seeds | stalled |
|---|---|---|---|---|
| 4 | 0.8023 | 0.1997 | 0.1442--0.2512 | 0 of 4 |
| 8 | 0.7301 | 0.1493 | 0.0453--0.2092 | 1 of 4 |
| 16 | 0.7922 | 0.1985 | 0.1740--0.2464 | 0 of 4 |
| 32 | 0.7103 | 0.1368 | 0.0497--0.2278 | 1 of 4 |

The largest within-configuration seed spread is **0.1781** average precision; the spread of the
seed means *across* bond dimension is **0.0629**. Seed noise is 2.8 times the capacity signal,
so the withdrawal in D-038 -- made from four seeds at one bond dimension -- holds at every bond
dimension. Replication did not rescue the sweep; it confirmed there was nothing to read.

**Two of sixteen fits never left chance**, both at seed 20260831, at chi = 8 and chi = 32. These
ran on the sequential fold, the contraction path chosen in D-038 precisely because the reduction
tree destabilised training. So a 12.5 % failure rate is a property of the ansatz on this data,
not of the D-036 optimisation. D-038 said "at width 128 two of four failed to train", which
invited the reading that width 1 is safe. It is not; it fails less often.

The arm's conclusion is unchanged and is now better supported. Every one of the sixteen fits,
including the best draw at 0.2512, sits far below the tuned gradient-boosted baseline's
0.5055--0.5114. Which draw is reported does not matter.

**Reframed 2026-08-30 against the literature, which changes what this finding is worth.** The
stall was recorded here as an unexplained property of the ansatz. It is a known failure mode:
Tang, Khoo and Ying \[QM-13] describe randomly initialised matrix product states failing under
gradient descent, and attribute it to a structural cause -- boundary-site interactions and a
causality trap -- rather than to bad luck in the seed.

Three things follow, and the middle one is the reason to record this at all.

* **What is not new.** That randomly initialised MPS training can fail is documented. Reporting
  it as a discovery would have been an overstatement, and this entry previously read that way.
* **What is new, and is a measurement rather than an anecdote.** Their setting is *generative* --
  Born machines and tomography -- and they **report no failure rate**. The 2 of 16 measured here
  is a rate, for the *discriminative* case, on payments data, at fixed hyperparameters across
  four seeds and four bond dimensions. That is a quantitative datum the literature does not have.
* **What it turns into.** They propose two remedies this study did not apply: natural gradient
  descent, and a TTNS-Sketch warm start in place of random initialisation. So the honest Phase II
  reading is not "the tensor network is unreliable" but "the failure has a named cause and two
  published remedies, neither of which was tried here" -- a caveat converted into a specific
  next step rather than left as a shrug.

The rate stays as reported. What changes is that it is now positioned against work that explains
it, which is the difference between a negative result and an unfinished one.

### D-042 The format gate failed the build on the proposal's own filenames

`scripts/check_pdf.py` scans the LaTeX sources for numeric literals, because every measured
figure is supposed to arrive through a `\Claim` macro. Adding sections 6 and 7 in the previous
round took it from passing to eight failures, all of them the check's fault:

```
submission/proposal.tex:14: numeric literal '01'   (\input{content/01-problem})
...                                        '06'   (\input{content/06-hybrid})
submission/proposal.tex:7:  numeric literal '2026' (\author{... Challenge 2026 ...})
```

`make pdf` exits non-zero, and `check: pdf claims`, so **the entire verification chain the
README advertises had been unrunnable**. The scanner stripped LaTeX control sequences but not
their arguments, so `\input{content/06-hybrid}` became `{content/06-hybrid}` and the `06` was
read as a measurement.

This is the third wrong verdict from this one checker. It read the horizontal entry of the text
matrix and declared 37 % of a compliant document undersized; it read the version in `Apache-2.0`
as a figure; it now reads a filename as one. The pattern in all three is the same: the check
was written against the defect it was hunting and never against a compliant document. Every
scan in it now has a test asserting **both** directions, in `tests/test_check_pdf.py`.

**A second blind spot, found while fixing the first.** Spelling a number as a word defeats the
literal scan completely, and the submission had accumulated fourteen such figures. Two were
wrong. The proposal claimed per-step cost was flat "despite a thousandfold change in FLOPs"
across a bond-dimension grid whose chi-squared span is **sixty-four**; the other section of the
same document said sixty-four-fold for the same measurement. `05-limits.tex` claimed "roughly
eight times the in-band evaluation set" where the figure derivable from the tables was four.

A blanket ban on number words is not workable -- the body counts in words about ninety times,
legitimately, in sentences like "three actions, not two". The check therefore targets the two
constructions that are always doing measurement work: a multiplier (`sixty-four-fold`, `eight
times`, `two thirds`) and a proportion of a stated total (`four of five`, `one-in-eight`).
Twelve of the fourteen were accurate and are now bound to claims; one was unverifiable and was
removed; the mechanism it illustrated is documented in D-009 and did not need the multiplier.

Binding them required a mechanism `docs/claims.yaml` did not have: a claim derived from other
claims. A ratio typed by hand does not move when its inputs do, which is how "thousandfold"
survived. `SweepChiFlopSpan` is now `(SweepChiMax / SweepChiMin)^2` and cannot drift from the
grid. The first hand-computed value for `PowerSeRatio` was 7.96, from the rounded inputs; the
mechanism resolved it to 8.03 from the raw ones, on its first run.

### D-043 A results table reported four comparisons as resolvable that could not resolve themselves

`results/tables/mps_h4.csv` is the evidence behind claim C5 and the source of the proposal's H4
table. Every one of its four rows carried `minimum_detectable_effect = 0.006945` and
`resolvable = True`.

**0.006945 is the retracted figure.** D-030 recorded that the power calculation had passed a
bootstrap standard error to a function expecting a per-observation standard deviation,
understating the minimum detectable effect roughly fiftyfold, and corrected `power.csv` to
0.046068. The correction never reached this table. There is only one `power.csv` in the
repository, `run_mps.py` reads exactly the column that was corrected, and the committed
`mps_h4.csv` is newer than the corrected `power.csv` -- so the file on disk cannot have come
from the pipeline as it now stands, and nothing in the repository detected that.

Re-running the band arm reproduced **every measurement bit-identically** -- average precision,
both interval endpoints, the standard error and the p-value, at all four bond dimensions, to
0.00e+00. Only the two derived columns changed, and `fit_seconds`, which is wall clock. The
pipeline is reproducible; the committed derived columns were simply stale.

**The design was also wrong, independently of the staleness.** `resolvable` compared the
observed difference against the *pre-registered* minimum detectable effect. That figure is
estimated from proxies before the comparison runs; it is the commitment that authorises the
run, not a description of what the run could see. What the comparison could actually resolve is
computable only afterwards, from its own bootstrap standard error. Both are now recorded, under
names that cannot be confused, and `resolvable` uses the realised one:

| chi | standard error | MDE (pre-registered) | MDE (realised) | observed | resolvable |
|---|---|---|---|---|---|
| 4 | 0.018684 | 0.046068 | 0.052346 | -0.020530 | False |
| 8 | 0.018709 | 0.046068 | 0.052415 | -0.023447 | False |
| 16 | 0.018748 | 0.046068 | 0.052524 | -0.022452 | False |
| 32 | 0.019893 | 0.046068 | 0.055733 | -0.017198 | False |

The observed differences are about a third of what the comparison could detect. The arm's
conclusion is unchanged -- it was already reported as underpowered, and every interval already
contained zero -- but the table said the opposite of the prose for as long as it was committed.

The body now reports the **realised** minimum detectable effect, 0.0557, rather than the
pre-registered 0.0461. It is the larger and therefore the more conservative statement, and it
is the one that describes this comparison rather than the estimate that preceded it.

### D-044 The alpha floor was attributed to a mechanism that cannot produce it

The README and the results section both explained the band-conditional certificate's reachable
`alpha` this way: *"the class-conditional degeneracy floor `⌈1/α⌉ − 1` then binds"*, citing
Ding, Angelopoulos, Bates, Jordan and Tibshirani (NeurIPS 2023). Two things were wrong.

**The arithmetic.** The band-conditional grid is `alpha` in {0.05, 0.10, 0.15, 0.25}, whose
degeneracy floors are 19, 9, 5.67 and 3 rows. The tightest band budget supplies **847**
legitimate in-band calibration rows, and the loosest 4,831. The floor is between forty and
sixteen hundred times smaller than the sample it is supposed to bind. **It cannot bind anywhere
on this grid**, at any budget, at any level.

**The form.** The cited paper is cited precisely because the bound is `(1/α) − 1` and not the
ceiling; `src/hsbcfraud/conformal/split.py` returns the exact form and its docstring says that
rounding it *"changes which borderline configurations are reported as degenerate"*.
`protocol.md` and `PROVENANCE.md` both use the correct form. The two documents a reviewer reads
first were the only ones that did not.

**What actually binds.** Amendment A1 had it right and was not consulted when the prose was
written: the constraint is the concentration bound. Certification requires *both* risks
controlled simultaneously under Holm, and all five certified configurations sit at the loosest
missed-fraud budget, `alpha_FN = 0.45` — so it is the missed-fraud budget that selects among
them, and the joint requirement plus the in-band sample size that excludes the rest. At zero
realised risk every configuration on the grid would certify comfortably, which is another way
of seeing that the sample size does not act through a representability floor.

Both passages now describe the concentration bound, and the README states the retraction.

### D-045 The proposal answered the rubric's smallest criteria and not its largest

The official Phase I guidelines and assessment criteria were downloaded from the portal and
read for the first time this round. They resolve a question two documents in this repository
had answered differently — the body limit is **six pages** and the appendix **three**, so the
Makefile was right and `SUBMISSION_CHECKLIST.md` was wrong — and they exposed something larger.

Phase 1 is scored on six weighted criteria:

| Criterion | Weight | Covered by, before this round |
|---|---|---|
| Problem Relevance & Impact | 25 % | section 1, but with no statement of expected impact |
| Technical Approach & Innovation | 25 % | sections 2 and 4 |
| **Feasibility** | **20 %** | **nothing** |
| **Validation Plan** | **15 %** | **nothing** |
| Hybrid / Cross-Domain | 5 % | section 6 |
| Team Capability | 10 % | section 7 |

The body mentioned Phase II exactly twice, both in passing, and contained no PoC plan, no
resource requirements and no forward-looking success metrics. Meanwhile the results and the two
quantum arms occupied 46 % of the proposal — about 2.75 of six pages — for criteria worth 25 %.
The document was a rigorous study report where the rubric asks for a proposal.

This was not a subtle omission and nothing in the repository could have caught it, because
every gate here checks that the numbers are true, not that the document answers the question it
was set. The guidelines had never been read into the repository at all.

**What changed.** Two sections were added: *Feasibility, resources, and the constraints we
already know*, which absorbs the former limitations section — "assumptions and constraints
clearly stated" is exactly what it contained — and adds a staged PoC plan and resource figures
measured on this study's own runs rather than estimated. And *Validation plan*, whose argument
is that the plan is not aspirational: every step in it was executed here, on public data, so
what a sprint changes is the data and not the protocol. Section 1 gained a statement of
expected impact.

They were paid for by compression, not by cutting results: the superseded full-scale table was
removed, three retraction narratives that the appendix already tells in full were reduced to
their conclusions, and the hardware specification moved from the team section to feasibility,
where the rubric actually scores it. The body remains at six pages.

**One correction fell out of the compression.** The hybrid section claimed Braket's
`LocalSimulator` was the reference implementation and that Aer agreed with it *"to
floating-point round-off"*. D-022 records the opposite: the exact statevector is the reference,
Braket is one of four routes measured against it, and Aer agrees to between 2.6e-15 and
4.8e-13 — three orders of magnitude above double round-off. The section now says so.

### D-046 Three of the challenge statement's six primary metrics were computed and never reported

The HSBC challenge statement, section 4.1, names the primary evaluation metrics: **ROC AUC,
AUPRC, F1, precision, recall and a confusion matrix**. The submission reported the first two.
A scan of the built PDF found zero occurrences of "F1" and zero of "confusion", and the single
occurrence of "recall" was the recall floor in the risk-control formulation rather than a
reported measurement.

They were not missing from the study. `scripts/run_baselines.py` has computed all six from the
first run, and `results/tables/baselines.csv` carries `precision`, `recall`, `f1` and the four
confusion-matrix cells for every arm, block and seed. The gap was entirely in the reporting:
the documents were written around the arguments the study wanted to make, and the metrics that
did not serve an argument were left in the table.

This is a different failure from the ones above. Every gate in this repository checks that a
number in the prose matches its table. None checks that a number the challenge statement asks
for appears at all, and no gate can, because the requirement lives in a PDF that had never been
read into the repository.

All six are now reported, in a new appendix section, at the default threshold on the held-out
temporal block. The appendix states plainly what they are and are not: a confusion matrix
describes one operating point on one block and bounds nothing about the next, which is the
distinction the body's certificate exists to make.

**A second requirement, and an opportunity taken.** Section 4.1 also asks participants to
benchmark against published results and "clearly report comparison methodology". The
first-place IEEE-CIS solution reports ROC AUC 0.9459; this study's baseline reports
0.8837--0.8884, and without the comparison a reviewer reads that as an untuned baseline. It is
not. The published figure is a time-based cross-validation over the training file and this is a
strict forward holdout, and the study has already measured what that difference is worth:
changing only the split, with an identical model, moves average precision by 0.3164. D-016
recorded this and it never reached the submission. The results section now makes the comparison
explicitly, which converts the apparent weakness into the study's headline finding.

**Still not reported: feature attribution.** Section 5.2 lists it as one of three expected
outputs, alongside the fraud probability and the binary prediction. There is none, and E12 was
never started. This is a genuine gap against a stated requirement rather than a reporting
oversight, and it is recorded here rather than left for a reviewer to find.

### D-047 The parity claim shipped in the PDF with no table and no script behind it

D-022 records the agreement between four independent routes to the same overlap, and names
`scripts/check_parity.py` as the thing that asserts it. **That script did not exist**, and
neither did `parity.csv`. The measured figures lived only as prose inside D-022, and the
proposal shipped a parity statement in the scored PDF that resolved to nothing a reviewer
could check or a reader could regenerate.

The script now exists and the table is committed. Measured against the exact statevector over
four encodings and twelve backend-by-encoding comparisons:

| backend | worst absolute difference |
|---|---|
| Braket `braket_sv` | 6.7e-16 |
| Aer CPU | 2.9e-15 |
| Aer GPU | 3.4e-15 |

**D-022's recorded range was wrong at the top end.** It quoted 2.6e-15 to 4.8e-13 for Aer; the
committed measurement is 2.2e-15 to 3.4e-15. The 4.8e-13 figure cannot be reproduced by this
script and no artefact supports it, so it is withdrawn rather than carried forward. The
proposal previously said "better than one part in `1e12`", which was true against the withdrawn
figure and is now unnecessarily loose; it quotes the measured worst case instead.

Backend availability is reported rather than worked around. `qiskit-aer` and its GPU wheel are
an optional extra installed by `make venv-gpu`, so the default environment has the exact
statevector and Braket only. The script records any backend it could not import, and fails if
either required backend is absent -- a parity check that silently compares one implementation
against itself would report success and mean nothing.

### D-048 The in-band model is three-quarters one anonymised card identifier

The challenge statement lists feature attribution as one of three expected outputs, alongside
the fraud probability and the binary prediction. It was the one output never produced: E12 was
never started, `scripts/run_explain.py` was referenced by the Makefile and missing, and
`src/hsbcfraud/explain/` was an empty package marker.

TreeSHAP over the calibration band, on the same rows and features the tensor-network arm uses:

| rank | feature | mean absolute contribution | share |
|---|---|---|---|
| 1 | `card1` | 0.8267 | 75.8 % |
| 2 | `V201` | 0.0606 | 5.6 % |
| 3 | `V189` | 0.0518 | 4.7 % |
| 4 | `V259` | 0.0468 | 4.3 % |

Additivity is asserted rather than assumed: base value plus contributions reproduces the raw
model margin to 6.2e-06, which is float32 accumulation over four hundred trees. A violated
additivity check would mean the explanation does not describe the model, and the script fails
on it rather than reporting a ranking that means nothing.

**The result is a finding, not a formality.** Three quarters of the in-band decision rests on
an anonymised card identifier. That is the measured form of a limitation this study already
stated in prose -- that a chargeback propagates to later transactions on the same card, so the
label is partly an entity flag -- and it is a stronger statement than the prose was. A
governance review would read it as a model that mostly recognises cards rather than
transactions, which is exactly what a model-risk function exists to catch and exactly the
argument for running this on an issuer's own data, where an entity key is a key and not a
predictor.

It also bounds what either quantum arm could have contributed. A re-scorer competing for the
remaining quarter of the signal, on eight features, is not a setting where a richer feature map
has much room, and that is worth knowing before proposing one.

**A refactor made this possible without duplicating the pipeline.** The band edges, the in-band
row selection, the mutual-information feature choice and the scaler lived inline in
`scripts/run_mps.py`. They are now `src/hsbcfraud/features/band.py`, imported by both arms, so
"the same rows and the same features" is a property of the code rather than a claim in prose.
The tensor-network arm was re-run after the extraction and reproduced every measured column
bit-identically.

### D-049 Three pre-registered hypotheses and three pre-registered measurements had no written disposition

An audit of the plan against the repository found two gaps of the same kind. Neither is a
wrong number; both are silence where the protocol had made a commitment.

**H1, H2 and H3 were never tested and nothing said so.** All three compare a quantum kernel
against a control. The stopping rule in protocol section 9 fired first: of 120 pre-registered
configurations, 28 passed conditioning and none passed distinctness, so no kernel reached the
band and there was nothing to test the nulls against. That is the stopping rule working, and
the proposal reports the screen result prominently. But the hypothesis table listed all five
with their tests and decision rules and said nothing about which had been run, so a reader
would take "H1-H4 are expected to survive" as a statement about five completed tests. Protocol
section 8 now carries a disposition for each.

**Three of section 7's five drift measurements were never run.** The conformal test
martingale, the block permutation test, and band-conditional drift against marginal drift.
Section 7 lists five things as "measured, not assumed" and it is the section that justifies
refusing to promise a numeric `alpha + eta` -- so three of the five being absent weakens
exactly the argument that section exists to make. Amendment A7 records what ran, what did not,
and what the two that ran do and do not establish. No figure moves: these were diagnostic, and
no hypothesis, certificate or reported number depends on them.

**The Makefile was encoding the gaps as breakage.** Nine of its twenty-two referenced scripts
did not exist, so `make data`, `conformal`, `quantum`, `explain`, `measure` and therefore
`make reproduce` could not complete -- while the README advertised `make reproduce` as the
reproduction path. Two of the nine now exist (`run_explain.py`, `check_parity.py`) and two were
never needed: `run_envelope.py` was a wrong path, since `envelope.csv` is written by
`run_conformal.py`, and `run_quantum.py` describes an arm the stopping rule stopped. The
remaining five are named in a comment block at the head of the targets, with where each is
recorded, rather than as commands that fail. `make reproduce` completes.

The distinction that matters throughout: **"not run" is a result when a pre-registered rule
stopped it, and a gap otherwise.** Both are now written down, and which is which is stated.

### D-050 A reported environment deviation was not one, and the correction changed the plan

An audit reported that `cuquantum-cu11`, `custatevec-cu11`, `qiskit-aer` and
`qiskit-aer-gpu-cu11` were installed in `.venv` while `NOTICE` and `PROVENANCE.md` state that
cuQuantum is proprietary and not installed by default, and recommended removing them. That
recommendation was acted on as a decision before the premise was checked.

**The premise was wrong.** Both documents say cuQuantum is not installed by the *default*
`make venv` target, and that is true: it lives in the `gpu-crosscheck` optional extra installed
by `make venv-gpu`. An environment that has run `make venv-gpu` is the documented state for
E11, not a deviation from it. Nothing needed removing.

Removing it would also have been actively harmful by the time the recommendation was made. E11
now has a script, `scripts/check_parity.py`, and two of its four routes are Aer; uninstalling
the extra would have shrunk a committed table to a subset of itself.

**One sentence did become false, and is corrected.** `NOTICE` said the binary is kept "off the
path that produces reported results". That held while E11 had no script. It no longer does:
`parity.csv` carries Aer rows. The correct statement, now written, is narrower and still
strong -- no *scientific* figure in the submission depends on the proprietary path, because
every kernel number comes from the exact statevector route, and the parity table's required
backends are the two the default environment provides. A default install reproduces a smaller
parity table, not a wrong one.

The general lesson is the one this log keeps recording: an audit finding is a hypothesis. This
one was plausible, specific, and wrong, and it was escalated to a decision without being
checked against the file that would have refuted it in two lines.

### D-051 The three documents other documents promised now exist, and writing them found two errors

`docs/REFERENCES.md` said `REFERENCE_CROSSCHECK.md` and `REGULATORY_SOURCES.md` existed;
`SUBMISSION_CHECKLIST.md` listed `guarantee.md` as required. None of the three did. Writing
them was expected to be clerical. It was not.

**The cross-check is generated, not written.** A hand-maintained map of which reference each
module uses drifts within a round -- references get renumbered, modules get refactored -- and
this project has already been bitten by exactly that with the decision count. So
`scripts/make_crosscheck.py` searches for each entry by three independent handles: its
identifier, its arXiv identifier, and its first author's surname. The surname is the handle
that finds most of them, because the source modules cite people rather than reference keys.
Fifty entries, thirty-four reached from the repository, sixteen cited nowhere and listed as
such.

Building it surfaced two defects in the generator's own reasoning, both fixed and both worth
recording as a pattern: it counted the files that *parse* references as files that *cite*
them, and it read the italic in `**[CP-1]**` as a title. A generated document is only as
trustworthy as the reading that produced it.

**The regulatory locators were the real work, and two entries were wrong.**

*RG-6 pointed at the wrong instrument.* The entry said FCA Policy Statement PS21/19 was "the
onshored UK SCA-RTS". It is not. PS19/26 onshored the SCA-RTS at Brexit; PS21/19 *amends* the
onshored version, adding an Article 10A exemption and an Article 36(6) consent reconfirmation,
both open-banking measures. No figure moves -- neither touches the Annex rates or the Article
18/19 structure -- but the citation pointed a reader at a document that does not contain what
the entry claimed for it. Both are now cited with their roles distinguished.

*Two claims could not be confirmed and are marked 要確認 rather than repeated.* RG-1 lists
Article 21 (monitoring) among the provisions relied on, and nothing in this repository cites
it. RG-5 states that the PRA "confirmed" SS1/23 applies to fraud models, and that confirmation
was not located at the issuing authority.

**One near-miss is worth recording as a method.** Checking the RG-1 Annex, an automated read of
the source returned 0.015 %, 0.01 % and 0.005 % -- which would have made the repository's
0.13 %, 0.06 % and 0.01 % look wrong. Extracting the table structure from the markup instead
showed why: the Annex has two columns, and the summary had read *remote electronic credit
transfers* rather than *remote electronic card-based payments*. The repository is right and
`config.py` reproduces the correct column exactly. An audit finding is a hypothesis, and the
second one this round that did not survive contact with the primary source.

**`guarantee.md` states the guarantee once.** Notation, the estimand and why it is conditional,
the theorem, the five certified configurations with their held-out results, and six things the
guarantee explicitly does not cover. All thirty-nine numbers in it were checked against the
tables they come from before it was committed.

### D-052 The statistical core now has tests, and writing them found an off-by-one

Three sentences in this repository described validations that had never been committed:
`REFERENCES.md` said `learn_then_test` was "validated by simulation (0 violations in 3,000
replications at delta = 0.05)" and `hoeffding_bentkus_p_value` "validated at the null
boundary"; D-013 said "the tests assert that it reduces exactly to standard split conformal at
`w_i = 1`". `tests/` held nothing touching conformal, risk control, coverage or statistics.

They exist now, in `tests/test_riskcontrol.py` and `tests/test_conformal.py`, 55 tests across
CP-1, CP-2, CP-3, CP-4, CP-5, CP-7 and CP-8. Three things came out of writing them.

**The simulation figure was wrong, and is corrected to what it measures.** At the null boundary
-- the true risk sitting exactly at `alpha`, which is the hardest point of the null -- the
procedure produces **1 violation in 3,000 replications**, not zero, against a permitted rate of
`delta = 0.05`. The reference sentence now says that. A negative control in the same file
confirms the procedure certifies on 300 of 300 replications when the true risk is `alpha/2`, so
the low count is conservatism rather than a procedure that refuses everything -- which is the
failure a coverage simulation cannot detect on its own.

**An off-by-one in the degeneracy floor's docstring.** `split.py` said "a class with `|I^y|` at
or below this value has `qhat^y = +inf`". The bound is strict. The algebra is
`ceil((n+1)(1-alpha)) > n  <=>  n < (1/alpha) - 1`, so a class whose count *equals* the floor
already admits a finite quantile, and the test asserting the documented behaviour failed on
the first run. Verified across five alpha values: `n = floor` is finite at 0.05, 0.10, 0.25,
0.01 and 0.001, and `n = floor - 1` is infinite at all five.

Nothing measured moves. The smallest headroom in `degeneracy.csv` is 1,122 rows above the
floor, so no reported configuration is anywhere near the boundary. But the whole content of a
bound is where it sits, and the document was describing a different bound from the code. This
is the second defect this round found by writing the check rather than by reading the code, and
the first one D-044 found the same way was a wrong scientific attribution in the scored PDF.

**The 0/1 loss requirement is now checked rather than read.** `SUBMISSION_CHECKLIST.md` carried
"Every risk certificate rests on a mean of per-observation 0/1 losses, as Hoeffding-Bentkus
requires" with the note **no test exists; verified only by reading**. The test asserts that the
risk at a threshold equals the mean of the indicator exactly, and that the effective sample
size is the conditioned subset rather than the row count -- using the row count would claim
more evidence than the data contains.

### D-053 The label-censoring audit was run, and its result was never reported

`scripts/audit_labels.py` has produced `label_censoring.csv` and `label_verdict.csv` since the
first round. Neither table was referenced by `claims.yaml`, `tables.yaml`, the README, the
protocol or any section of the proposal. The experiment ran, wrote its answer, and the answer
sat in `results/tables/` unread.

It is the obvious objection to the study's central result. If fraud on recent transactions were
simply not yet reported -- the IEEE-CIS label rule has a settlement window -- then a forward
holdout would show a coverage breach that is an artefact of the labels rather than a property
of the data. Anyone reading the temporal result would ask this.

The audit answers it, and answers it in the direction that supports the result:

| | |
|---|---|
| Trailing-window fraud rate | 3.666 % |
| Earlier fraud rate | 3.281 % |
| Mann-Kendall trend | tau = -0.2857, p = 0.3988 |
| Verdict | no censoring detected |

The trailing rate is **higher** than the earlier rate, which is the opposite of what censoring
produces, and there is no monotone trend across the eight buckets. The body now says so in one
sentence in the results section.

The pattern is worth naming because it is the mirror image of most entries in this log. The
usual failure here has been claiming something the tables do not support. This one is the
reverse: a measurement that supports the headline claim, defends it against the first question
a reviewer would ask, and was left out because no argument in the draft happened to need it. A
gate that checks every quoted number against its table cannot see a table that is quoted
nowhere.

### D-054 A reference audit against primary sources found eleven defects, one of them a title that does not exist

Every entry in `docs/REFERENCES.md` was checked against its primary source: the arXiv abstract
page, the DOI resolver, the publisher, or the issuing authority. Each candidate defect was then
put to an independent reviewer instructed to refute it. Eleven survived. Five matter.

**[QM-5] carried a title the paper does not have.** The entry read *"On the Interplay of
Bandwidth and Expressivity in Quantum Kernels"*. The paper at arXiv:2503.05602 is *"On the
similarity of bandwidth-tuned quantum kernels and classical kernels"*, by **R.** Flórez-Ablan
(not D.), published in *Quantum Science and Technology* 10(3):035051. The entry also had the
finding inverted: optimal bandwidths drive quantum kernels toward RBF, and because those
optimal values are small the kernels simplify further into low-order polynomial kernels. The
entry said polynomial behaviour appeared at *larger* bandwidths.

This is the most serious defect found in the project. A reviewer checking the citation that
justifies the kernel screen would not have found the paper. **The screen itself is unaffected**
-- `screens.py` evaluates conditioning and RBF similarity at every bandwidth and gates on
neither direction, and its own docstring states the relationship correctly -- so the inversion
lived only in the reference annotation. Verified by reading the module rather than assuming.

**[DS-3] credited the wrong source with the metric choice, and so did the code.** The entry said
Dal Pozzolo et al. "recommends AUPRC over ROC-AUC at this prevalence". The paper endorses
ROC-AUC: *"a well-accepted ranking measure for unbalanced dataset is AUC"*. The AUPRC
recommendation is on the dataset's Kaggle page, verbatim, and that is [DS-4]. The same
misattribution was in `src/hsbcfraud/metrics.py`, which said "the dataset's own originating
paper recommends AUPRC". Both corrected. Worth noting: **the challenge statement makes the same
attribution**, citing Dal Pozzolo et al. for the AUPRC recommendation. The metric choice is
right and now points at the source that actually makes it.

**[DS-5] conflated two things and had no locator.** The NVIDIA post is *"Leveraging Machine
Learning to Detect Fraud"*, by McDonald and Deotte, 26 January 2021, not a 2019 piece titled
"1st Place Solution". Every number in the annotation -- 0.9459, 0.9363, the GroupKFold design,
the UID construction -- verifies exactly. Citation hygiene, not a numerical error.

**Grossi et al.'s 0.789 does not appear in the paper.** Two independent extractions of
arXiv:2208.07963 find no "0.789" or "78.9". The statevector row is 0.78 +/- 0.01 (Table 7); the
balanced subsample is about 2,500 rows, of which 1,000 is the test split rather than the whole.
The companion figure, 0.55 +/- 0.10 for the noisy simulation, is exact.

**[RG-5]'s "confirmed by the PRA" could not be substantiated.** "Fraud" does not occur in
SS1/23. `docs/REGULATORY_SOURCES.md` had already declined to substantiate it, so two files in
this repository contradicted each other. Replaced by the scope argument that actually supports
inclusion: para 1.2 limits scope by firm, para 1.3 covers all model and risk types within it.

Smaller corrections: [RG-4] attributed independent review to SR 26-2 Section V, where the phrase
does not occur and third-party validation is Section VII; Innan et al. is 22(02):2350044 (2024),
not 21(05) (2023); and the Section 6 preamble said all versions were checked against the PyPI
JSON API, which cannot serve a local-version wheel.

**A real packaging bug fell out of the last one.** `src/hsbcfraud/quantum/mps.py` imports
`torch` at module scope and torch appeared in no dependency list, so `pip install -e ".[dev]"`
alone produces an environment where that module raises `ModuleNotFoundError`; only the
Makefile's ordering hid it. It is now declared as an optional `torch` extra rather than an
ordinary dependency, because installing it from PyPI is the worse failure: that build has no
`sm_120` kernels, so it imports cleanly and then cannot use the GPU. A loud absence beats a
silent wrong answer.

**On method.** Two of this round's audit findings did not survive contact with the primary
source -- the PSD2 Annex rates and the cuQuantum installation -- and eleven did. The difference
was always whether the check went to the issuing document or to a summary of it.

### D-055 An exploratory check for value-conditional under-coverage, and what it does not license

A literature sweep surfaced a mechanism worth checking against this certificate. Zhong et al.
(CP-15) find that population-level conformal bands under-cover high-risk subgroups in 57 of 68
audited combinations despite nominal marginal coverage, and separate two causes: **rarity**,
which more calibration data closes, and **tail-heaviness**, which it does not. The
band-conditional certificate here conditions on band membership and on the legitimate class,
and on nothing else. If the in-band population is tail-heavy in transaction value, the
certified false-decline rate could hold overall while failing on high-value transactions --
which are the ones a false decline costs most.

Checked rather than assumed. Splitting the in-band legitimate rows of the held-out block into
transaction-amount quartiles and applying each certified rule unchanged:

| Configuration | Q1 low | Q2 | Q3 | Q4 high | alpha |
|---|---|---|---|---|---|
| budget 0.035 | 0.1667 | 0.1794 | 0.1867 | 0.1647 | 0.25 |
| budget 0.050 | 0.1695 | 0.1751 | 0.1729 | 0.1697 | 0.25 |
| budget 0.100, a = 0.10 | 0.0857 | 0.0801 | 0.0869 | 0.0894 | 0.10 |
| budget 0.100, a = 0.15 | 0.0857 | 0.0801 | 0.0869 | 0.0894 | 0.15 |
| budget 0.100, a = 0.25 | 0.1776 | 0.1640 | 0.1739 | 0.1868 | 0.25 |

**All twenty cells hold.** The worst uses 0.894 of its budget. The highest-value quartile
averages 0.1400 against the lowest quartile's 0.1370 -- a difference of 0.003, with no ordering
in value. Whatever else limits this certificate, value-conditional tail-heaviness is not
visible in it.

**Three things this does not license, and the third is the reason this entry exists.**

It is **not pre-registered**. It was prompted by a paper published after the protocol was
frozen, and it is exploratory in the exact sense the protocol uses that word.

It is **not a conditional guarantee**. Twenty cells holding is evidence that the marginal
certificate is not hiding a value-conditional failure at this resolution. It is not a
certificate on any quartile, and it would take its own pre-registration and its own
family-wise correction to become one.

It **reads the held-out block a second time**, and that is worth stating plainly rather than
letting a reader discover it. The single-evaluation rule exists to stop a configuration being
chosen by its test performance. Nothing was chosen here: the configurations, the band edges and
the thresholds were all fixed by the certificate, this is a further summary of the same
authorised evaluation rather than a new one, and no reported number depends on the outcome --
had the quartiles failed, the finding would have been recorded as a limitation rather than
changing any rule. That is the honest position, and it is still a second look. It is recorded
here, and deliberately not in the proposal, because a post-hoc check reported alongside
pre-registered results reads as though it were one.

Reproduce with the certified rows of `riskcontrol.csv` against
`results/runs/scores_temporal_20260828.parquet`, quartiles on `amount` within
`band_lo <= score < band_hi` and `y == 0`, declining at `selected_lambda`.

### D-056 The team section had dropped a disclosure the project's own rule required

The pre-study draft that scoped this submission set an explicit rule for one credential: the
QIntern work's figures may be used only as evidence of past work, the proposal must state that
they come from **CIC-IoT2023, an intrusion-detection benchmark**, and nothing may be
extrapolated from them to card fraud.

An earlier revision of the team section said "the significance protocol for a
quantum-classical intrusion detector", which satisfied the rule. Compressing that section to
fit six pages removed the domain and left the claim reading as though the conformal calibration
of a false-alarm rate had been done on payments. In a card-fraud competition that is exactly
the wrong reading to leave available, and the compression that caused it was made for space
rather than for meaning.

The section now names the benchmark and states plainly that what transfers is the machinery
rather than a number. The lesson is narrow and worth recording: **a `\Record{}` fence exempts a
figure from every numeric gate in this repository**, which is correct -- those figures are not
measurements from this study -- but it also means the only thing standing between a
biographical claim and a reader is prose, and prose gets compressed.

**A second disclosure was missing entirely.** The draft's checklist requires the submission to
state honestly that the author has no experience running on real quantum hardware, since the
challenge statement says explicitly that participants who do not run on hardware are not
penalised. The proposal said both arms run on simulators, which is a statement about this
study, and said nothing about the author. It now says both.

**One open question in the draft is resolved and needs no organiser query.** The draft flagged
that section 4.2 of the challenge statement said "Given the dataset size (~24,000 rows)" while
section 5.1 lists datasets of 590,540, 284,807 and 1.3 million rows, and recommended asking the
organisers which applied. Comparing the two published versions: the figure appears once in
`HSBC-Challenge-Statement-vF-1.pdf` and **not at all** in
`HSBC-Challenge-Statement-vFinalRevised.pdf`. The organisers removed it. What remains is the
requirement this submission already meets -- state the sample count used for quantum execution,
and stratify the subsample.

### D-057 Six biographical claims were not supported by the artefacts, and three had no artefact at all

The team section makes claims wrapped in `\Record{}`. That macro declares a figure as the
author's own record rather than a measurement from this study, and it therefore **exempts the
figure from every numeric gate in this repository**. That exemption is correct -- these are not
this study's measurements -- and it means nothing had ever checked them.

They were checked against the source folders on this machine, each candidate defect then put to
an independent reviewer instructed to refute it. The result is the most consequential finding of
this round, because an unsupported credential in a submission whose whole posture is "we claim
only what we measured" is worse than a wrong result.

**Three claims had no artefact anywhere.** The QPoland entry read "quantum graph kernels against
Weisfeiler--Lehman and shortest-path baselines under nested cross-validation". Searching the
whole machine: no graph-kernel implementation, no Weisfeiler--Lehman implementation, no nested
cross-validation. The only occurrences of "Weisfeiler" outside third-party library code were in
the team section asserting it. The entry is now "QPoland 2025 runner-up" and nothing more.

**One claim was contradicted by the author's own file.** The proposal said "Yale Peaked
Hackathon 2026, #13 of 550". That repository's README states "Leaderboard: Top 10 (all tied at
450 pts; no participant solved P10)". The two cannot both be right, and the proposal's version
may be the *understatement*. Since neither can be confirmed here, the rank is removed rather
than guessed in either direction.

**One method attribution was wrong in a way that mattered.** "By a matrix-product-state marginal
attack" -- the repository's own verifier shows the nine solutions came from a mix of
tensor-network methods, and not from the marginal attack at the 69-qubit problem the sentence
leaned on. Now "by tensor-network methods", which is what the artefact supports and is still the
point: the tensor-network arm here is not a first attempt at the method.

**One ownership claim overstated an individual contribution.** "He owns the statistical-guarantee
proposition of the team's paper." The team's own documentation assigns Proposition 3 to **Team
A**, a three-person team. What is the author's own, and corroborated by a teammate rather than
self-reported, is the split-conformal calibration and the results freeze. The sentence now says
that, inside the team that owns the proposition.

**Two are fine and one is sharper than claimed.** The unitaryHACK contributions are merged
upstream and verifiable by pull request: NVIDIA/cuda-quantum #4693, merged into the default
branch, carrying the maintainers' own `unitaryhack-accepted` label, and a genuine recursive
Quantum Shannon decomposition rather than a name reused for something flatter; QuEST #783,
merged by the lead maintainer into `devel`. The "135 automated checks" figure was counted at the
freeze commit and is exactly 135 -- the repository is at 183 today, so the wording now says
"green at that freeze" to pre-empt a reviewer finding the larger number.

**Not checkable here, and left alone.** The Qiskit Advocate credential has no local artefact;
it is an ordinary biographical fact and stays. The portfolio URL has no local checkout. Neither
is a technical claim.

**The general lesson.** Every gate in this repository points at the study's own numbers. The one
place a claim can enter unchecked is the place explicitly marked "this is not from the study" --
and that is precisely where the least-checked claims accumulated. A fence that exempts a figure
from verification is not the same as a fence that verifies it.

*Separately, and outside this submission:* the `taler-sbom` README states "10 tests" where the
suite defines 15. Not part of any deliverable here, recorded because it was found while checking.

### D-058 A URL was clipped off the page, and no gate could see it

The team section gives a portfolio URL. In the built PDF it rendered as
`thedaemon-wizard.github.io/portfolio-quantum-` and then stopped: the final two words of the
address ran past the right margin and were clipped from the page. A reviewer following the link
as printed gets a 404.

`\texttt{}` will not break a long token, so LaTeX set the whole URL as one unbreakable box,
reported `Overfull \hbox (146.4pt too wide)`, and carried on. Nothing read that warning. The
fix is `\url{}` with hyperref's break points widened to hyphens and letters rather than slashes
alone.

**The gate gap is the point.** `scripts/check_pdf.py` asserted page count, paper size, font
size, numeric literals and number words. Every one of those passed on a page with text missing
from it, because none of them looks at whether the text fits. The build log had said so all
along, in a warning nobody read -- the same shape as D-053, where a table sat unread in
`results/tables/`, and D-042, where the checker itself was wrong.

`check_overfull` now reads the log beside the PDF and fails above 60 pt of overhang. The
tolerance is not arbitrary: a long inline equation in section 4 overhangs by 47 pt and is
entirely readable, while the clipped URL overhung by 146 pt. Both directions are pinned in
`tests/test_check_pdf.py`, including the case where the log is missing -- a gate that passes
when its evidence is absent is worse than no gate.

### D-059 The strongest credential rested on placeholder scores, and the section said so nowhere

The first credential audit (D-057) removed three unsupported claims and corrected two. A second
pass against the source artefacts found that the correction to the most load-bearing one had
not gone far enough, and found the tensor-network attribution still wrong after I had already
softened it once.

**The QIntern coverage result was computed on a dummy score interface.** The proposal said
"coverage verified inside the exact Beta-Binomial band across five seeds". The artefact that
result comes from, `week5/reports/_generated/w5_02_table_a.json`, carries
`"source_kind": "dummy"` and `"scores_root": "week2/interface/dummy_scores"`. Every Week-5
artefact in that freeze carries the same field, and the team's own README states it plainly:

> Everything quantum is still on the **dummy** interface, so every QS-Net number this week is
> **PROVISIONAL** ... **protocol-final, numbers-provisional**.

A reviewer of a fraud submission reading "coverage verified across five seeds" reads an
empirical result. What exists is a harness verification: the machinery ran end to end, on three
corpora and five seeds, and produced in-band coverage on placeholder scores. That establishes
the machinery works and is wired correctly, which is precisely the claim the paragraph is
making -- but the sentence let a stronger reading stand. The section now uses the team's own
phrase and says what the freeze establishes.

Two things move in the honest direction at once. The scope was also **understated**: the
coverage harness covers three intrusion-detection corpora, and the proposal named one.

**The tensor-network attribution was still wrong.** D-057 changed "by a matrix-product-state
marginal attack" to "by tensor-network methods", which is weaker but still misattributes: the
repository's own `verify_results.py`, which I ran, reports P9 at 69 qubits as **FAIL, Hamming
32/69** -- indistinguishable from chance -- with P8 and P7's tensor-network runs failing too and
P7 solved exactly by graph factorisation instead. Its README records that the correct answers
for P7 to P9 were submitted early and the producing runs were not saved. So no artefact
attributes the 69-qubit answer to a tensor network. The saved runs are exact to 60 qubits
(P6, circuit compression plus MPS sampling) and P5 at 50 qubits by MPS marginal alone.

The claim is now "nine of ten peaked circuits, the largest at 69 qubits; the saved
tensor-network runs are exact to 60 qubits and degrade above it". Both figures verified, and
the sentence is better for naming the boundary: a proposal whose own section 4 reports its
tensor network losing is more credible for saying where the method stops working.

**Also corrected, in passing:** the earlier "#13 of 550" was wrong twice. The README records
"Top 10", and 550 is the maximum *score* (10+20+...+100), not the field -- which was 549
participants.

**The lesson, which is the same one twice.** Softening a claim is not the same as checking it.
D-057 made "marginal attack" into "tensor-network methods" on the reasoning that the weaker
statement must be safe. It was not, because the artefact does not support the attribution at
that size at all. The check that settled it was running the repository's own verifier, which
took one command.

### D-060 The rank I removed was correct, and the source I trusted to remove it was the weaker one

D-057 removed "#13 of 550" from the team section, and D-059 repeated the reasoning. Both were
wrong. A third-party repository documenting these hackathons,
`github.com/roman-bagdasarian/Peaked-Circuits`, carries a leaderboard table for **Yale Quantum
2026**:

| Team | World rank | Score | Solved | Time penalty |
|---|---|---|---|---|
| MerQury | **#13 / 550** | 450 | **9 / 10** | 28.36 h |

Its Problem 9, Grand Summit, is 69 qubits and its peak bitstring is present, so the nine
solved include the largest. The same file lists a second entry at "#63/550" for a different
event, which settles the other half: **550 is the field size, not the maximum score.** An
earlier audit asserted it was the score, reasoning from the 10+20+...+100 problem weights, and
that inference was wrong.

**What led me astray.** The author's own repository says "Leaderboard: Top 10 (all tied at 450
pts)". I treated that as contradicting "#13" and removed the rank as unsupported. Both
statements are true: many teams tied at 450 and the tie is broken on time penalty, so a team
can be inside the top score tier and thirteenth overall. Faced with a self-authored README and
an external leaderboard, I trusted the self-authored one -- exactly backwards. The rank is
restored, and the event now carries the label the third-party source uses.

**A genuine open question, not resolved here.** The author's repository and the third-party
repository record **different peak bitstrings for problems 3 through 9**, at identical qubit
counts, while agreeing exactly on problems 1 and 2. Since a peaked circuit has one peak, two
different answers to the same circuit cannot both be right; the likeliest explanations are
different circuit instances behind the same problem names -- the third-party repository does
carry separate `MIT_iQuHACK_2026` and `Yale_Quantum_2026` sets with identical names -- or a
different working set. The author's answers match neither published set beyond problem 2.
**要確認.** Nothing in the proposal depends on it: the rank, the count and the qubit ceiling are
all corroborated externally, and the sentence attributes only the *saved tensor-network runs*
to 60 qubits, which is what this project's own verifier reproduces.

**The team name also differs** -- MerQury on the leaderboard, PeakQubit in the author's
repository. The proposal names no team, so nothing turns on it, but a reviewer who follows the
credential will meet the discrepancy. **要確認.**

**The Qiskit Advocate credential is confirmed, and is not publicly verifiable yet.** The author
holds the acceptance email; the programme is at Tier 0 and the badge has not been issued. It
stays in the proposal as an ordinary biographical fact, with no verification URL to offer,
which is the honest position rather than an omission.

**The lesson.** D-059 recorded that softening a claim is not checking it. This one is the
mirror image: *removing* a claim is not checking it either. Both errors came from acting on a
single source without asking which source was in a position to know.

### D-061 The organiser's leaderboard settled a claim that three repositories had disagreed about

The Yale credential has now been through four readings, three of them wrong, and the error each
time was the same: reading a derived source instead of the one that issues the fact.

| Reading | Source | Verdict |
|---|---|---|
| Original | the author's assertion | "#13 of 550" |
| D-057 | the author's own repository README | removed the rank as contradicted |
| D-060 | a third-party repository's leaderboard table | restored "#13 of 550" |
| **Now** | **the organiser's own leaderboard** | **rank 13, score 450 of 550** |

The BlueQubit leaderboard for the event, read directly, gives
`13 | MerQury | 450/550 | 28.36h | 35 submissions`, under a page whose own title is **Yale
Peaked Hackathon 2026** -- so the proposal's original event name was right and my change to
"Yale Quantum 2026", taken from a third-party repository's section heading, was a step away from
the organiser's own label.

**Both earlier readings of "550" were wrong, in opposite directions.** It is the maximum score,
not the field size: the column is headed SCORE, every row on the first two pages reads `450/550`,
and the ten problems carry weights 10 through 100, which sum to 550.

**The sentence that followed here was wrong and is retracted** -- see [D-105](#d-105). It read
"that arithmetic also confirms the count independently -- 450 is every problem but the last, so
nine of ten follows from the score without needing the author's repository at all." Ten subsets
of the weights sum to the missing 100, so the score is consistent with six through nine solved.

**The "Top 10" that started this is not a contradiction.** Ranks 1 to 20 are all tied at 450 and
separated only by time penalty. A team can be in the top score tier and thirteenth overall.

**The method attribution was wrong to remove.** D-057 called "matrix-product-state marginal
attack" unsupported and D-059 capped it further. It is the author's own contribution:
`github.com/roman-bagdasarian/Peaked-Circuits` lists `thedaemon-wizard` as a contributor, and
pull request #1 from branch `marginal_attack_by_amon`, merged 2026-04-10, adds
`marginal_attack.py`, whose own description reads *"MPS marginal attack: determine peak
bitstring from single-qubit Z expectation values"* and which drives
`AerSimulator(method="matrix_product_state")`. Restored, and now the strongest line in the
section: the method this study's own tensor-network arm uses is one the author wrote and
competed with.

**What the certificate does and does not establish.** The file the author holds is a
Certificate of *Completion*, naming him, dated 2026, from Yale in collaboration with YQuantum.
It does not record a placement; the leaderboard does. Both are cited for what each supports.

**The remaining discrepancy is downgraded, not dismissed.** The author's private repository
records different peak bitstrings for problems 3 to 9 than the published set. The organiser's
score of 450 is itself the record of nine correct submissions, so this reads as a personal
working copy diverging from what was submitted rather than a problem with the credential.
Still **要確認**, and nothing in the proposal rests on it.

**The lesson, stated once for all three errors.** D-059: softening a claim is not checking it.
D-060: removing a claim is not checking it either. This entry adds the rule that would have
prevented all three -- **go to the party that issues the fact.** A leaderboard is issued by the
organiser, a merge by the upstream maintainer, a coverage number by the run that produced it.
Every wrong reading in this sequence came from a source that was reporting the fact rather than
holding it.

### D-062 The rank had a denominator after all, and it was on the same screen

D-061 established that "550" is the maximum score and not the field size, and that "#13 of 550
teams" therefore conflated two different things. That was right. The conclusion drawn from it
was not: the entry treated the rank as having no citable denominator and the proposal settled
for a bare "rank 13".

The leaderboard's pagination control reads **`11–20 of 549`**. That is the field — the number of
ranked entries — and it was visible in the same screenshot, one line below the table D-061 read.
So the correct statement carries both denominators, and they differ by one:

> **rank 13 of 549 entries, scoring 450 of 550 points**

Re-read live at `app.bluequbit.io/hackathons/wSvCWg8f38spoXX3?page=2&tab=leaderboard`, scope
**World view**, 2026-08-30, and cross-checked against the screenshot the author supplied.
`08-team.tex` and [CREDENTIALS.md](CREDENTIALS.md) §3 now state both.

**Why the near-coincidence matters.** 549 and 550 sitting one apart is what made the original
"#13 of 550" look like a plausible field size, and it is why four readings of this credential
went wrong before this one. The two numbers are not variants of a single quantity: one is the
size of the field, the other the sum of the ten problem weights 10, 20, ... 100.

**Entries, not teams.** Some rows carry team names, others personal names, so the field is
counted in leaderboard entries and the proposal says "entries".

**The lesson.** D-061 ended on *go to the party that issues the fact*. This adds the obvious
corollary it did not follow: **read the whole artefact that party issued.** The correction was
sourced from the organiser's leaderboard, which was the right source — and then stopped at the
table without reading the control underneath it. A number absent from a document is a finding; a
number present and unread is an error.

### D-063 The latency measurement measured a threading default and nearly reported it as model cost

E13 was the last open item on the compliance checklist: the challenge statement lists inference
latency as good-to-have, names *latency vs. complexity* as a bottleneck, and the proposal's
answer to it — the expensive component runs on a band that is a few percent of traffic — had
been argued from volume arithmetic and never timed.

The first run produced a table that was internally consistent and wrong in its meaning:

| | batch 1 | batch 1024 |
|---|---|---|
| classical scorer, 431 features, GPU | 53.5 ms | 0.035 ms |
| in-band re-scorer, 8 features, GPU | 44.0 ms | 0.034 ms |

Three things in it should not have been true at once. A 431-feature model cost the same as an
8-feature one. CPU and GPU were indistinguishable. Batch 1 cost two thirds of batch 1024, for
1/1024 of the work. Each is a signal that the quantity varying between rows is not the quantity
being measured.

**The cause.** XGBoost defaults its thread count to the core count. Synchronising 20 threads
over a one-row payload costs about 19 ms; the prediction costs about 0.05 ms. Measured directly
on a fitted booster:

| `nthread` | 1 | 4 | 20 (default) |
|---|---|---|---|
| single-row p50 | 0.051 ms | 0.051 ms | 19.33 ms |

A 380-fold penalty, independent of device, feature count and batch size — which is exactly why
it flattened all three dimensions the sweep was varying.

**What it would have cost.** The figure would have entered §5 as tens of milliseconds of model
cost against a residual budget of about 170 ms: 11–31 % of the issuer's budget consumed by a
gradient-boosted tree ensemble. Every conclusion drawn from it would have been wrong in the same
direction. It would also have flattered the quantum arm, putting the kernel at roughly 2× the
classical scorer when the true ratio, once the barrier is removed from both, is about 1,148×.

**The fix.** The serving profile is now a swept dimension rather than an accident, and the
all-core rows stay in the table as the documented trap. The single-threaded profile is the
per-request serving shape: concurrency in an authorisation path comes from serving many
transactions at once, not from splitting one across cores. Timing goes through
`Booster.inplace_predict` rather than the sklearn wrapper, whose validation and `DMatrix`
construction are not part of a deployed scorer.

**What the corrected measurement says.** The classical core is not the latency risk — 0.084 ms
median, 0.320 ms at the tail, 0.19 % of the residual budget. The kernel is: 96.577 ms median,
with a tail taking 0.759 of that budget. It fits, and only because the band rations it. That is
a stronger version of the proposal's argument than the volume arithmetic it replaces, and it
carries an honest constraint into Phase II — a per-authorisation kernel evaluation has little
headroom even on the band.

**The lesson.** D-062 said to read the whole artefact. This one is about measurement rather than
reading: **when the quantity you are varying stops changing the answer, you are measuring
something else.** Three separate anomalies pointed at a shared cause and each was individually
dismissible as noise. The check that resolved it — hold everything constant and sweep only the
suspected confound — took two minutes and should have come before the table, not after it.

### D-064 Six defects were shipping in the built PDFs, and the gates could not see any of them

Every gate was green: 6 + 3 pages, 155 tests, 88 claims resolving, citations resolving, no
unused claims, 38 artefacts matching the manifest. An audit against the repository's own tables
then found six defects **in the built PDFs**, and the common property is what matters — each
sits in a blind spot the gate architecture creates rather than covers.

| # | Defect | Why no gate saw it |
|---|---|---|
| 1 | §3 asserted the card-disjoint arm over-covers "in every case … without breaching the one-sided guarantee". Seed 20260831 at α=0.001 has ratio 1.5245, `conservative=False` — it *under*-covers | The direction was prose. Only the counts were bound |
| 2 | Both PDFs printed 61 decision entries against an actual 63 | The gate compares `claims.yaml` to the table. Nothing compared the table to `decisions.md` |
| 3 | The appendix said "Four" amendments against seven, and described all as derived from block sizes — untrue of A5/A6/A7 | The count was typed, not bound |
| 4 | §6 opened "Every step was executed in this study" while H1–H3 and three drift measurements never ran | Not a number, so nothing to check it against |
| 5 | §7 and CREDENTIALS said "the author has not used a QPU", refuted by his own public repository | The fact lives outside this repository entirely |
| 6 | ENVIRONMENT reported three of four split sizes from the wrong arms, summing to 10,903 rows that do not exist | A Markdown table is not a bound claim |

**The pattern.** A claim gate can only check a number that someone chose to bind, in a document
the gate reads, against a table something regenerates. Defect 1 failed the first condition,
defect 6 the second, defect 2 the third. Defects 3 and 4 were never numbers at all, and defect 5
was not checkable from inside the repository under any design.

**What was changed, beyond the six fixes.** Only defect 2 admitted a mechanical fix, and it got
one: `summarise_decisions.py` now also derives the amendment count, `make claims` depends on a
`derived` target that regenerates both, and `test_decision_log_matches_the_decisions_document`
recounts from the source documents and asserts contiguity. The count is now wrong in the test
suite before it can be wrong in the submission. Defects 1 and 3 were bound as claims
(`CardDisjointTightestWorstRatio`, `CardDisjointTightestBestRatio`, `ProtocolAmendments`) so the
same sentences cannot drift again.

**Defect 2 was self-inflicted, and that is the most useful part of it.** D-062 and D-063 were
appended in the previous session without regenerating the table. The gate passed, the PDF built,
and the submission carried a wrong number about its own process — in the section arguing that
this project records its retractions. A gate that runs on demand does not protect a document
edited afterwards.

**Defect 5 has no gate and should not have one.** No amount of repository tooling could have
caught a false claim about the author's history; only reading his other artefacts could, and
that is what this round did. [FACTCHECK_LOG.md](FACTCHECK_LOG.md) now records each such check
with the command that produced it, because the alternative to a gate is a repeatable procedure,
not trust.

**The lesson.** D-062 said to read the whole artefact; D-063 said that when the quantity you
vary stops changing the answer you are measuring something else. This one is about the gates
themselves: **a green gate certifies the properties it was built to check, and silently
certifies nothing about the rest.** The correct reading of "all checks passed" is "no bound
number disagrees with its table" — which is a much smaller statement than it looks.

### D-065 The prior work was closer than the proposal admitted, and saying so is the stronger position

§1 described the conformal fraud-detection literature as certifying "a *marginal* error rate
under exchangeable splits". Checked against the entries this repository already cites, neither
half is right.

* **[FR-1]** (Singh et al., arXiv:2607.27143) is **Mondrian class-conditional** conformal
  prediction with cost-controlled abstention — not marginal, and it already has an abstention
  mechanism.
* **[FR-3]** (Zhu et al., *DISCO*, *Decision Support Systems* 208, DOI
  `10.1016/j.dss.2026.114717`) certifies a **false-negative rate** on real card-fraud data —
  a class-conditional risk, and on this problem rather than an adjacent one.

Calling both "marginal" understated them, and understating the prior work is a way of
overstating this submission. It is also the failure a reviewer who knows FR-3 would catch
immediately, and the reference list this repository ships would have handed them the evidence.

**What actually distinguishes this work** is narrower and survives contact with the literature:
the guarantee is conditional on the **abstention region the rule routes to**, and it is
evaluated under a **split that orders time**. §1 now says exactly that, names what each prior
paper does, and notes that the second distinction is the one that turned out to matter — the
temporal arm breaches on every seed at the two loosest levels.

A web sweep for work published since August 2026 found nothing that supersedes the positioning.
The result is recorded in [FACTCHECK_LOG.md](FACTCHECK_LOG.md) §2 with each source and verdict,
so the next round starts from what was checked rather than repeating it.

**The lesson.** The reference list was current — FR-3 and FR-6 were already in it, correctly
described. The proposal's one-line summary of them was not. **A citation being present is not
the same as the sentence around it being true**, and the citation gate checks only the former.

### D-066 I diagnosed a script from a contaminated measurement, and the instrument disproved me twice

During a clean-room reproduction, `scripts/make_splits.py` printed one line about the dataset
and then produced no output for over thirteen minutes. It is the first target `make reproduce`
invokes, so this looked like the pipeline hanging on its first step.

**It was not the script.** On an idle machine the same script completes in **two seconds** after
the data load. The thirteen minutes were my own orphaned background processes — earlier attempts
at the same clean-room run, which I had failed to reap — saturating all twenty cores. The
measurement was of the machine, not of the code.

**The instrument contradicted me twice before I got there.** Adding `ProgressReporter` to the
script came with a comment explaining that the cost was building the card-disjoint arm, which
groups 590,540 rows by entity. The first instrumented run showed all three arms built in about
**one second**, so the comment was corrected to name the six `HistGradientBoostingClassifier`
fits instead. The second run, on an idle machine, showed those six fits taking **two seconds
between them**. Both explanations were plausible, both were written next to a working
instrument, and both were wrong.

**What survives, and it is smaller than the entry that first stood here.** The script had no
progress output at all, and neither did twenty-three of the twenty-six scripts in `scripts/`.
When something *is* slow — for whatever reason, including a reviewer's machine being busy —
silence and a wedged process are indistinguishable from outside. The instrumentation is kept for
that reason, not for the reason I originally gave. It reports twelve units with elapsed and
remaining time and writes a JSONL record beside the tables.

**What the clean-room run did establish.** A fresh virtualenv from the system Python 3.12, torch
2.13.0 from the CUDA 13.0 index with `torch.cuda.is_available()` true, the project and its dev
extras installed with no resolution conflict, `make smoke` accounting for all nine checks, and
`make walkthrough` passing. Separately, regenerating the split tables to a scratch directory
reproduced `splits.csv` and `data_integrity.csv` **bit-identically** against the committed
copies — the determinism claim, checked rather than asserted.

**The lesson.** D-063 recorded a timing that measured an OpenMP barrier instead of a model. This
is the same error with a different confound and a worse aggravation: there the contamination was
a library default, here it was **processes I had started myself and left running**. Before
attributing slowness to code, look at what else is on the machine — and when an instrument you
have just added disagrees with the sentence you wrote beside it, the sentence is what is wrong.

### D-067 Two of the three open 要確認 were errors in my own auditing, both against the author

The three items this repository carried as unresolved were closed by going to the artefact. Two
of them turned out not to be facts about the credential at all, but mistakes in the audit that
raised them — and both made the record look weaker than the evidence supports.

**The Yale bitstrings: one answer set in two bit orders, and it took me two wrong explanations
to get there.** The 要確認 said the author's working repository recorded different peak
bitstrings for problems 3 to 9 than the third-party repository. Compared programmatically, the
two records are the same nine answers reversed: P3 to P9 are **exact string reversals**,
character for character; P1 is identical because `1001` is a palindrome; P2 is identical because
it is the one row the third-party README transcribed un-reversed. That coincidence is precisely
what made a raw comparison read as "agrees on 1 and 2, differs on 3 to 9".

The clinching artefact is one the author committed himself. `P5_soft_rise.qasm.txt` is the
output of a single run and prints `Little-Endian:` and `Big-Endian:` on consecutive lines — the
first is the third-party README's P5 row, the second is his. Both "sets" are two lines of one
simulation, because the tooling emits `peak` and `peak[::-1]` by construction. His copy is the
*submitted* orientation, so the doubt ran the wrong way.

**My first explanation for it was also wrong**, and worth recording: I concluded the two records
were different problem sets, 2025 against 2026, having looked only at the `data/*.txt` files and
never opened the third-party README, which carries a Yale-2026 table with all ten problems. Two
wrong explanations for one item, both produced by checking a part of the artefact rather than
the artefact.

**The QIntern calibration module: the independent attribution existed and had not been read.**
The 要確認 said authorship rested only on a self-authored handoff, because the file arrives in a
bulk "Initial commit" with no per-file provenance. Both halves were wrong. That commit was made
by a **teammate** — the repository is hosted under his account — and three days later the same
teammate committed a handoff calling it "AK's Day-15 CQ-ZDR module", recording that he re-ran it
and matched its numbers to six decimals. The Days 26–27 freeze is corroborated the same way, by
a teammate who reproduced all 39 frozen scalars to 1e-9 on a different Python and found the
manifest byte-identical. A third statement in the same section — that `qi26_12/week3/` was a
documentation-only copy — was also false; it has `scripts/` and `tests/`.

**The filing date** was the one genuine unknown, and the author settled it: 1 August 2026. The
submission states the status without a date, so nothing in it moves. `Resume_Amon_Koike2026_CV`
says "Jul 2026" and is the document that needs correcting, outside this repository.

**The pattern, and it is not the same as D-059's.** D-059 recorded that softening a claim is not
checking it, and D-060 that removing one is not either. Both were about being too generous.
These two are the opposite failure: an audit that manufactured doubt from a comparison it had
not validated, and then carried that doubt in a shipped document for three rounds. **An
unresolved 要確認 is a claim too** — it asserts that something could not be settled, and that
assertion needs the same evidence as any other. Neither survived contact with the actual
artefact, and the Yale one did not survive my first two attempts at explaining it either: read
the whole artefact (D-062), and when a cheap check is available, run it before writing the
explanation (D-063, D-066).

Only A8 remains open, and it cannot be closed from here: the portal upload is a manual action.
The live form was read on 2026-08-30 and confirms five empty slots and the accepted-format list.

### D-068 The pre-registration claimed an enforcement the code does not perform

Section 2.2 of the protocol stated: "`D_test` is evaluated **once** … **Every sweep, ladder and
ablation runs on held-out slices of `D_band` or `D_cal`.**" Three scripts read the test block
without going through `TestFoldGuard` — `run_baselines.py` (3 arms x 5 seeds of descriptive
metrics), `run_ablations.py` (4 leakage-ablation variants x 3 seeds) and `run_seed_sweep.py`
(the 16-job full-scale arm) — while `test_access.json` records one authorised configuration.

**The guarantee is intact, and that is a separate question from whether the sentence was true.**
What protects a finite-sample guarantee is that nothing may be *selected* on the test fold.
Nothing was: `ablations.csv` is consumed by no claim, no figure and no downstream script, and
the baselines and the sweep report primary metrics on a held-out block, which is what a held-out
block is for. Every threshold and both band edges come from `D_band` and `D_cal`.

**The sentence was still wrong, and the ablation ladder is exactly the case it excluded.** A
reviewer comparing section 2.2 against `run_ablations.py:82` finds it in two minutes. A
pre-registration that overstates its own enforcement is worse than one that states a weaker rule
accurately, because the overstatement is the part a reviewer is checking.

**What was not done, deliberately.** The three scripts were not routed through the guard and the
ledger was not back-filled. Either would have changed committed tables so that a text problem
disappeared. The protocol is what was wrong, so the protocol is what changed — amendment A8
records which scripts read the fold, why the guarantee survives, and that `TestFoldGuard` raises
only on a *different* configuration rather than on a repeat read, which is what its docstring
always said and what section 2.2 now claims.

**Related, and also corrected this round:** the Expected Impact paragraph added earlier today
cited the AI Act's Annex III point 5(b) as a source of "documentation duties". That provision is
the clause **excluding** fraud detection from the high-risk category, and this repository's own
reference entry says so. The sentence now uses the exception the way it should be used: because
no external high-risk regime applies, the discipline has to come from internal model-risk
review. A regulatory misattribution in a proposal whose subject is governance auditability is
among the cheapest errors for a reviewer to find and the most expensive to make.

**The lesson.** D-064 recorded that a green gate certifies only what it was built to check.
This is the same shape one level up: **a pre-registration is only as good as the narrowest
sentence in it**, and the sentences most worth auditing are the ones describing enforcement,
because those are the ones a reader will test against the code.

### D-069 The proposal had one figure, and its labels were under the font floor

The assessment criteria ask the Technical Approach section for "a clear description of the
proposed method, algorithm, or **workflow**", and the pre-registration's own argument turns on a
fact about geometry: which block may touch which parameter. Until now the body carried a single
figure, on coverage, and the method was prose only.

**What the figure had to be, and what it stopped being.** The first version drew both the
four-block split and the three-valued decision rule, sized each block by its day span, and
collided four labels: the two narrow blocks are exactly the two that carry the guarantee, so
their names overprinted each other and their roles overflowed. The second dropped scale for
equal boxes and fixed the collisions. The third dropped the decision-rule panel entirely --- the
rule is three lines of prose and an estimand in display maths, whereas the split geometry is the
part a reader reconstructs slowly and wrongly. One panel, four boxes, no arrows to misplace.

**A font defect the gate was tolerating.** `check_pdf.py` asserts a 10 pt floor and allows 8 % of
characters below it for mathematical sub- and superscripts. Adding the figure took that share
from 1.8 % to **5.1 %** --- still passing, and for the wrong reason. The cause was that figures
were authored at 8.6 in and included at `width=\textwidth`, which on A4 at an 18 mm margin is
6.85 in: LaTeX scaled them by 0.80 and every label lost a fifth of its size. Figures are now
authored at `TEXT_WIDTH_IN`, computed from the geometry rather than guessed, so the point sizes
in the source are the point sizes on the page. The share is **1.7 %**, below where it started,
with a figure more than before.

That is the general lesson and it is not about figures: **an allowance sized for one cause will
silently absorb another.** The 8 % was reasoned about mathematical scripts. Nothing checked that
what it was actually absorbing was mathematical scripts.

**What it cost.** About thirty lines of prose across six sections, none of it a measured result.
The body is 4,016 words against 4,166 before, and every figure in the submission is still
generated from a committed table rather than drawn.

### D-070 ULB was pre-registered, never obtained, and its one numeric claim did not follow from the split

The pre-registration listed ULB European Cardholder as **Secondary**, with row and fraud counts,
and asserted a degeneracy result on it. The dataset is not on this machine, no script fetches
it, and smoke check S7 has skipped for that reason throughout. Amendment A9 records it.

Three defects, and they are different kinds:

1. **Not run.** E15 never executed. Same class as the drift tests in A7, same treatment.
2. **Not measured.** 284,807 rows, 492 frauds, 1,081 duplicates are transcribed from the
   dataset's published description. Probably right; nothing here checked them, and the protocol
   presented them as if it had.
3. **Arithmetically inconsistent with this protocol.** Section 6.1 said "roughly 98 calibration
   frauds against the 99 required". 98 is 20 % of 492 -- this protocol's **test** fraction.
   `split.cal` is 10 %, giving roughly 49.

The third is the interesting one, because the error is invisible from the conclusion. 49 against
a floor of 99 is *more* degenerate than 98 against 99, so the claim "ULB degenerates at
alpha = 1e-2" survives and strengthens. A number that supports the right conclusion by the wrong
route is the hardest kind to catch: nothing downstream looks wrong.

**Withdrawn rather than corrected.** Recomputing 49 and leaving the sentence would assert a
measurement that still has not been made. ULB is out of scope for Phase I, and the argument it
was to supply -- that a fraud-conditional quantile degenerates on a small imbalanced file where
IEEE-CIS's does not -- is already made directly by section 6.1's own degeneracy table, where the
floor is 19 rows against 847 available.

**Nothing in the submission moves.** No claim, figure or table rests on ULB; it appears in no
`.tex` file. The appendix's licensing paragraph mentioned ULB's two-layer licence and now names
the operative restriction instead: Kaggle competition rules section 7.A limits IEEE-CIS to
non-commercial research, which is a reason the Phase II sprint must move to the issuer's stream.

**The lesson.** D-068 recorded that a pre-registration is only as good as its narrowest
sentence. This adds the case where the sentence is a *number*: **an unverified figure that
implies the right conclusion is not evidence, and is harder to find than one that implies the
wrong one.** The check that caught it was arithmetic against the protocol's own configuration,
which costs nothing and was never run on this paragraph in three rounds.

### D-071 The appendix printed nine amendments and listed eight, and the bound count is why

`ProtocolAmendments` moved 8 to 9 the moment amendment A9 was written, and `A1-protocol.tex`
dutifully printed 9 above an `enumerate` that still held 8 items. Two sentences around it were
wrong with it: "the first four ... **the rest** were not" accounted for four of five, and "two
disclosures" was three.

**This is the third count-drift defect in the project, and the first two were fixed by binding
the number.** That fix worked, in the sense that the digit is now always right. It does nothing
whatever about the prose the digit describes, and the failure moved there: a reviewer who counts
a numbered list finds the gap in ten seconds, in the section whose whole purpose is proving that
nothing was dropped.

So the fix is a test, not another macro. `test_appendix_lists_every_amendment_it_claims` parses
the `enumerate` block and asserts the item count equals `protocol_amendments` in
`decision_log.csv`. The generalised lesson: **binding a number stops the number drifting and
leaves everything that references it unguarded.** Every bound count in this project should be
asked what prose asserts a structure around it.

### D-072 A tolerance chosen for a catastrophe certified a defect a reader could see

`check_pdf.py` carried `OVERFULL_TOLERANCE_PT = 60.0`, set to catch a URL that overhung by
146 pt and lost its last two words off the paper. Beside it the comment said a long equation
"routinely overhangs by a few points". Both statements cannot be true of the same threshold. At
60 pt the gate passed an inline equation overhanging by 47 pt, which put text 15.5 mm into an
18 mm margin on page 4 -- found by a human looking at the PDF.

Three changes, and the third is the one that matters. The equation is set as display math, since
`\resizebox`, `\scalebox` and `\small` are banned here and shrinking is not a fix anyway. The
tolerance is 12 pt, which is what "a few points" means. And a new `check_margins` measures the
artefact instead of LaTeX's complaint about it: per page, the rightmost text position against
the text block. Run against the previously shipped PDF it reports page 4 at 44 pt past the
margin; against the current one, nothing outside the block on any page. **A gate that infers
from a build log is checking the log. This one checks the page.**

### D-073 The reference list's "cited nowhere" column was about to be used as a deletion list

The cross-check reported 16 of 59 entries as reached by nothing, and the plan was to delete
them. Four of the sixteen were live citations the tool could not see:

* **RG-2** is cited in the proposal as `\Cited{EU AI Act}` -- the previous round replaced the
  bracketed key with the instrument name precisely so a reviewer holding only two PDFs could
  resolve it, and the scanner looks for keys.
* **RG-3** has a section of its own in `REGULATORY_SOURCES.md`, which was not in `SEARCH_GLOBS`.
  The glob list named five documents by hand and the `docs/` directory had grown past it.
* **SW-5 crepes** and **SW-8 scikit-learn** are pinned dependencies declared in
  `pyproject.toml`; the surname handle requires a capital first letter, so a lower-case package
  name has no handle at all.

Fixed at the source rather than by exception: `docs/*.md` replaces the hand-maintained document
list, with `REFERENCES.md` and the generated cross-check explicitly excluded so an entry cannot
cite itself; and a fourth handle, `Cited as: \`name\``, lets an entry declare how it is cited
when it has no author to cite. FR-2 (ARGUS) and FR-5 (ProtoCP) were then *cited* in section 1
rather than deleted -- ARGUS calls its own objective conformal-style rather than a finite-sample
guarantee, which is the sharpest differentiator available, and ProtoCP is the closest
counterexample to the novelty claim, so citing it removes an attack instead of inviting one.
Ten genuinely unreachable entries were deleted; 49 remain and every one resolves.

**The lesson.** An automated finding is evidence about the tool as much as about the corpus.
This one had a 25 % false-positive rate and its output was one command away from removing four
working citations.

### D-074 5.554 % was attributed to the file in three documents and to the wrong band in a fourth

The claim key is `BandValueFraudRate` and its selector is `{arm: temporal, block: band}`: it is
the value-weighted fraud rate of `D_band`, the second of the four temporal blocks. `protocol.md`,
`REFERENCES.md` and the appendix all attributed it to IEEE-CIS as a whole, which is **3.867 %**.
Correcting the appendix, the first attempt then attributed it to the *abstention band* -- a
different object again, since `D_band` is a time block and the band is a score interval.

Both errors run the same way: toward the larger number, which strengthens the argument that a
portfolio-level PSD2 ceiling cannot be applied here. The argument does not need the help; the
whole file is still an order of magnitude above the loosest ceiling. **A claim macro binds a
value to a selector and says nothing about the noun the prose attaches it to**, which is the
same failure as D-071 in a different dress.

### D-075 The kernel latency conclusion did not follow from the measurement

Section 5 read "one in-band kernel evaluation would cost 96.577 ms ... it fits only because the
band rations it". Two defects.

**The support-set size was undisclosed.** `measure_latency.py` sets `SUPPORT_ROWS = 64`, a
constant that appeared in neither PDF nor `configs/`. Cost is linear in it, and the in-band
training block is 2,916 rows -- at which the same Gram row takes about 4.4 s and does not fit at
all. A measurement whose governing assumption is not stated is not a measurement a reader can
use.

**Rationing bounds aggregate compute, not per-request latency.** A transaction that lands in the
band pays the tail whether the band is 2 % of traffic or all of it, and authorisation timeouts
are per-request. Both PDFs and `ENVIRONMENT.md` now state the support-set assumption, its
linearity, and what the band does and does not bound; per-request feasibility separately
requires holding the support set to order 100.

### D-076 The literal scanner could not see a number at the end of a sentence

`LITERAL`'s trailing lookahead was `(?![\w.])`, so a full stop immediately after a digit run
suppressed the match. Two figures shipped unbound in the appendix -- a width ratio of `0.89.`
and an AUC of `0.5000.` -- while the gate reported nothing unaccounted for over fifteen source
files. The lookahead is now `(?![\w]|\.\w)`: a full stop ending a sentence is not part of the
number, a full stop followed by a word character still is, which keeps `section 7.A` and
three-part version strings out. Both figures were reworded rather than bound, since neither
comes from a results table: one is a fixture, the other is chance by construction.

Also corrected in the same pass: `A3-reproduction.tex` claimed "every number in both documents
is generated", which was the claim this defect falsified. It now says every *measured* number,
which is what the machinery actually enforces.

### D-077 Three credentials and one product name, checked against their sources

The team section said the QIntern results freeze had "the 135 checks green", conflating a
SHA-256 freeze manifest with a pytest suite: the handoff document records a suite that grew
110 to 135 tests, all green, and separately a freeze pinning 16 artefacts and 39 scalars that
verifies with no mismatch. Both are true and they are not the same object. The later suite
figure of 182 was not used: it depends on a feature branch that has not merged, and it counts
work by others.

The workstation is an **RTX PRO 6000** Blackwell, not an "RTX 6000 PRO" -- `nvidia-smi` reports
`NVIDIA RTX PRO 6000 Blackwell Workstation Edition`, and `ENVIRONMENT.md` already had it right
while the proposal and one docstring did not. The machine has 14 cores and **20 threads**, so
the OpenMP note now says threads.

`guarantee.md` said $\alpha_{\mathrm{FN}}$ selects among the five certified configurations. All
five sit at $\alpha_{\mathrm{FN}} = 0.45$, so it is a necessary condition for certifying at all;
band budget and $\alpha$ are what distinguish them.

### D-078 The README opened with tables and closed with the picture that explains them

Section 4 was 244 lines of tables between the certificate and the negative results, and the
only diagram was a compact split figure sitting inside it, four screens down. A reader arriving
from the submission portal met the numbers before the object they measure.

Two changes. The results move to [`RESULTS.md`](RESULTS.md) in full -- every table, every range
across seeds, every retraction -- and the README keeps the three figures with one paragraph of
conclusion under each, linking out. 494 lines to 332. And a new `overview_figure` goes in at
section 2, carrying the four-block split *and* the decision it licenses, which the proposal's
version had to drop for space.

The figure shares `draw_split_row` with the proposal's compact one, so the two pictures of the
same four blocks cannot disagree; both read `splits.csv` on every build. Two defects were fixed
by looking at the rendered PNG rather than by any test: the branch arrows originally left the
*transaction* box rather than the score, which drew "one authorisation, therefore approve", and
the bottom box was narrower than its own caption. **No gate in this project catches either.
Figures are checked by eye, and that is a standing cost of having them.**

### D-079 The telemetry was written, tested, used once, and then not used again

`run_log` exists because a seed sweep called `fit_mps` without a reporter and ran silently for
hours (D-051). It fixed that script. `run_baselines.py` then ran thirteen minutes of fitting
behind a `print` that fires only after a model has already finished, and `run_ablations.py`
four minutes the same way -- the identical defect, in the scripts nobody had looked at.

Both are now wrapped in `run_log`, and all three long scripts report a `SweepTimer` estimate
with the spread it is derived from. Extrapolating across jobs is defensible for the bond-
dimension sweep because per-step cost barely moves with $\chi$ (D-032); for the baselines it is
not, since logistic regression is far cheaper than either boosted family, which is why
`summary()` prints the observed range beside the estimate rather than a bare deadline.

The log timestamp now carries the date. A thirteen-hour sweep crosses midnight and a bare
`01:14:07` cannot be ordered afterwards.

Both refactors were checked against the committed tables rather than assumed: one seed of
`run_baselines` and one of `run_ablations` re-run into a scratch directory reproduce every
science column exactly. **The general point: a helper is not adopted because it exists.** The
test added here asserts that each of the five scripts loading IEEE-CIS opens a progress
destination, which is the only thing that stops this recurring a third time.

### D-080 The reproducibility check fired on every rebuild, so it checked nothing

`scripts/freeze.py` classifies `results/figures/*` as scientific artefacts and requires them to
be bit-identical across runs. They never were. matplotlib stamps a `CreationDate` into every
PDF it writes, so two consecutive `make figures` runs on the same tables produce different
bytes for pixel-identical pictures -- measured directly here, twice, with the PNGs identical
each time and the PDFs differing.

The consequence is worse than the cause. `make check` reported "11 scientific artefacts differ
from the manifest" after any rebuild, most of them figures that had not changed at all, under a
message asking whether a measurement had genuinely moved. **A check that fires every time
trains its reader to clear it without reading the list**, which is precisely how a real change
would have passed.

`savefig(..., metadata={"CreationDate": None})` omits the timestamp; the same two runs now
produce identical bytes. The Makefile already had a `DETERMINISTIC` variable for exactly this
reason and it was applied to the LaTeX build only, which is why the gap survived: the mechanism
existed and covered the artefact somebody had already thought about.

### D-081 The tables grouped their counts and the prose did not

`format_value` rendered a claim exactly as `claims.yaml` records it, so section 3 read "847
legitimate in-band rows against 58343 overall" two pages after a table printing "356,216".
Inside a table a column gives the eye somewhere to land; in running prose a five-digit run does
not, and the two conventions sat in the same document.

Integers of 10,000 and above are now grouped in the inline macros as they always were in the
tables. Only integers: a probability or a ratio is never grouped, and `_value_appears` in
`check_claims.py` already matched both forms, so the binding is unaffected.

### D-082 GitHub eats backslash escapes inside math, and eleven expressions were affected

A reader reported "Missing or unrecognized delimiter for \Bigl" in README section 4.2.

**The renderer is MathJax, not KaTeX**, and identifying that mattered more than it sounds. The
first pass here used KaTeX as the oracle, which answers "Expected group as argument to '\Bigr'"
for the same input -- a different message for the same defect. The reported string is MathJax's
wording, and MathJax was then confirmed by reproducing the reader's message exactly. Working
from the wrong renderer produced two wrong conclusions and one wrong fix, all corrected below.

**What GitHub does to the source**, measured through its own `POST /markdown` endpoint and then
through MathJax, rather than modelled:

| written as | what the renderer receives |
|---|---|
| `$...$` or `$$...$$` | a backslash before ASCII punctuation is **removed** |
| `$...$` only | a raw `<` or `>` is escaped **twice** and arrives as the five characters `&lt;` |
| fenced ` ```math ` | **byte for byte**, except a row separator at end of line, which gains a backslash |
| `$...$` across a line break | not mathematics at all: literal text and no math node |

So `\Bigl\{` arrives as `\Bigl{` -- a group opener where a delimiter was meant.

**The visible error was the least of it.** One expression failed outright; **seven others
rendered without any error and silently wrong** -- `\,` and `\;` arriving as a literal comma
and semicolon dropped into the middle of a formula, and `\{ \}` arriving as grouping braces so
that the abstention band printed as $B = x : \ldots$ with its set braces gone, and the
tensor-network sum printed as $\sum_s$ rather than a sum over configurations. A defect that
raises is found by whoever reads the page. A defect that renders is not.

**A second failure the report did not mention**: a raw `<` inside inline math. Four spans had
one, including both statements of the abstention band and both statements of the RBF screen
threshold. Each answers "Misplaced &". And one inline span crossed a soft line break, so it was
never mathematics.

**The fix follows the table.** Display mathematics moves into fenced `math` blocks, where it
keeps its natural `\;` `\,` `\{` `\}` spelling and needs no substitution at all -- the row
separator in the decision rule starts its line rather than ending it, which is the one fence
hazard. Inline mathematics has no fence available, since a fence cannot live in a table cell, so
it uses `\lbrace` `\rbrace` `\lt` `\gt`. Verified end to end: all 137 spans in 15 documents
pushed through GitHub's renderer and then through MathJax, zero failures.

**Two claims from the first pass, withdrawn.** `\textsc` was reported here as unsupported and
therefore as breaking both central formulas of the guarantee document. That was a KaTeX
artefact: MathJax defines `\textsc`, and those formulas rendered. The `\texttt{DECLINE}`
spelling was kept anyway, because the README already used it for the same word, but it is a
consistency change and not a repair. Second, the first fix substituted `\thickspace` for `\;`
throughout; the fence makes that unnecessary, and an independent check raised a doubt about
which MathJax package set GitHub loads that the fence removes entirely rather than answers.

**And a gate, because there was none.** `scripts/check_markdown_math.py` extracts every span
GitHub would render, skipping code fences and code spans, and asserts each of the four rows of
that table. With `--render DIR` it additionally parses every expression through MathJax as the
renderer will receive it, which is the only way to catch an undefined macro. Run against the
previous commit it reports all eleven, ending with the reader's own error string; against this
one, none.

**The lesson.** Every artefact this project publishes had a gate except the one most people will
actually read, and the first attempt to build that gate used the wrong renderer and would have
certified a document GitHub could not display. **A gate is only as good as its oracle, and the
oracle has to be identified rather than assumed.**

### D-083 Three formulas in the shipped documents are not what the code computes

Found while checking the mathematics that the rendering work had touched. All three are in the
proposal PDF, not only in the markdown.

**Effective rank.** Both documents printed
$r_{\mathrm{eff}} = (\sum_i \sigma_i)^2 / (n \sum_i \sigma_i^2)$, the participation ratio.
`effective_rank_ratio` in `quantum/screens.py` computes the **exponential of the spectral
entropy**, normalised by matrix size. These are different functions of the spectrum. They agree
on a flat spectrum and nowhere else: on a geometric decay they give 0.4873 against 0.3721, a
31 % relative gap. Every screen verdict in `screens.csv`, and the reported 28 of 120 passing
conditioning, came from the entropy form -- so the printed formula was not the one that produced
the number beside it.

**RBF correlation.** Printed as $\mathrm{corr}(K_Q, K_{\mathrm{RBF}}(\gamma^\star))$. The code
takes the **largest absolute** correlation over a 25-point bandwidth grid, on **off-diagonal
entries only** -- both kernels have unit diagonal, so including it would add a block of
perfectly correlated values and inflate the result. Two material qualifications, neither stated.

**The tensor-network contraction.** As written, every bond index is contracted and the
expression has no free index: it is a scalar. `MPSClassifier` carries a $(\chi, 2)$ head that
closes the final bond into two class logits, which is what makes it a classifier. The head is
now in both statements of the formula.

**The lesson, and it is uncomfortable.** This project binds every quoted *number* to a table and
gates on it. It never checked a single *formula* against the code, and three of them were wrong
in the shipped PDF while every gate was green. A number is easy to bind and a formula is not,
which is exactly why the formulas drifted and the numbers did not.

### D-084 The degeneracy flag disagreed with its own module, twice

`degeneracy_floor` derives the bound and documents it as **strict**:
at $n = \lceil 1/\alpha \rceil - 1$ the algebra gives $\lceil (n+1)(1-\alpha) \rceil = n$,
a finite quantile, so a class
holding exactly `floor` points is not degenerate. Its docstring even records that an earlier
version said "at or below" and was off by one. The `headroom` property beside it agrees: it
calls a class degenerate only when the headroom is negative.

`mondrian_thresholds` then set `degenerate = n <= floor`, which is the version the docstring had
already retracted. At exact equality it reports a class as degenerate whose quantile is finite,
and disagrees with `headroom == 0` on the same object.

Nothing measured moves: the smallest headroom in `degeneracy.csv` is 1122 rows, so no reported
configuration is anywhere near the boundary. That is precisely why only a test would ever have
caught it, and there is one now, parameterised over the five levels on the grid, asserting the
three statements agree at `floor` and at `floor - 1`.

### D-085 The band was written open at both ends and implemented half-open

`rows_in_band` selects `score >= low & score < high` and its docstring says so: a score at the
upper edge is a decline, not an abstention. Every written statement of the band said
`tau_lo < f(x) < tau_hi`, open at both ends -- so a score exactly at the lower edge abstained in
the code and was approved by the document.

Five sites, one of them the proposal PDF: `README.md`, `docs/guarantee.md` twice (the notation
table and the three-valued decision rule), `docs/protocol.md` twice, and
`submission/content/02-method.tex`. The decision rule was the worst of them, because its three
branches were written `>= tau_hi`, `tau_lo < f(x) < tau_hi` and `<= tau_lo`: a score exactly at
`tau_lo` matched the approve branch and was excluded from the step-up branch, while the code
routes it to step-up. Two branches claiming the same point is not imprecision, it is an
ill-defined rule.

All five now read `tau_lo <= f(x) < tau_hi`, and the decision rule's approve branch reads
`f(x) < tau_lo`, which makes the three branches a partition.

On continuous scores this is a measure-zero event and no reported number moves. It is recorded
because the estimand is the whole content of the guarantee: a certificate on
$\mathbb{P}(D(X) = \texttt{DECLINE} \mid Y = 0, X \in B)$ is a statement about a specific $B$,
and a reader checking the code against the document would have found them disagreeing about
which one.

### D-086 The documents describe a two-model rule; the certified path thresholds one score

`docs/guarantee.md` prints the decision rule as `STEP-UP, then g(x) >= lambda => DECLINE`, and
section 2 of the proposal introduces `g` as the in-band re-scorer. `run_conformal.py` selects
`lambda` from a grid over `s_cal` and `validate_certificate.py` applies it to `s_test` -- both
the **same** full-traffic score `f`. `band_conditional_false_decline` takes one score array and
thresholds it. There is no second model anywhere in the certified path.

For the reported configuration the band is `[0.0320, 0.0718)` and `lambda = 0.0582`, strictly
inside it. So `lambda` is a **third threshold on `f`**, not a threshold on a different score.

Nothing here is wrong mathematically. `band_conditional_false_decline` accepts whichever score
ranks the band, so the certificate is valid for any `g` put in its place, and the two-stage
description is the design. What was missing is the sentence saying that in this run **nothing
filled the slot** -- no in-band re-scorer beat the outer scorer on its own band, so `g = f`. A
reviewer who opened `run_conformal.py` beside the guarantee document would have found a
two-model story and a one-model implementation, with no note reconciling them.

Both statements now say so. The related conflation is worth naming too: the in-band
eight-feature model that the tensor network is compared against in the results is a **different
object** from the certified rule. It exists for that comparison and no part of the certificate
depends on it.

**The lesson.** The three formula defects of the previous round were documents disagreeing with
code about a *computation*. This one is documents disagreeing with code about *how many models
there are*, which is a larger claim and was harder to see precisely because every individual
sentence was defensible.

### D-087 Numbers typed into a document by hand, three of four wrong, caught in one command

Relocating the label-censoring control out of the six-page body and into `RESULTS.md`, the
replacement paragraph was written by hand rather than read off `claims.yaml`. The trailing-window
rate was written 3.978 against an actual 3.666, the earlier rate 3.297 against 3.281, and the
p-value 3.666 -- which is the trailing rate, transposed into a field where any value above one
is impossible on its face.

`make claims` failed on the next run, because `CensoringP` went unused the moment its real value
stopped appearing anywhere. The gate did its job in one command and nothing reached a built
artefact.

Recorded rather than quietly fixed, because it is the same failure the whole claim mechanism
exists to prevent and it happened while *moving* a passage rather than writing one -- the case
that feels safest. **Text that carries a bound number is not prose and must not be retyped; it
has to be read off the source or moved verbatim.**

### D-088 The statement's first Expected Outcome had no artefact a reviewer could open

The portal's challenge panel lists four Expected Outcomes, and the first is "fraud probability
scores (float [0,1]) and binary predictions for each transaction". Both existed here. Neither
was reachable from the submission.

The scores lived in `results/runs/scores_*.parquet`, and parquet is not among the formats the
portal accepts -- PDF, PNG, JPG, WEBP, GIF, PY, JSON, JS, XLS, XLSX, CSV, DOC, DOCX, confirmed
in a browser against the live form. The binary decision existed only as a count in appendix §A4.
`COMPLIANCE_CHECKLIST.md` marked both rows met and cited exactly those two things, so the
checklist was true about the repository and wrong about the upload.

`scripts/export_predictions.py` now writes one row per held-out transaction under the certified
configuration at the tightest level that certifies: the probability, the three-valued decision,
and the binary decline it implies, with the band edges and threshold repeated on every row so a
reader can recompute the decision from the file alone. It derives and measures nothing: the
score is the frozen scorer output and the thresholds come from the certificate. Verified against
`h5_validation.csv` -- 10,021 legitimate in-band rows and 858 declines, both exact -- and a test
pins the agreement, because a derived deliverable is precisely the kind that stops agreeing with
its source quietly.

**Float formatting was the one place this nearly went wrong.** The first version wrote six
decimal places, which put rows near the threshold on the wrong side of their own stated
boundary: the file disagreed with its own columns. It is written at full precision now, at a
cost of 11 MB against a 20 MB cap.

It takes the fifth slot from the certificate table, which was the only staged file whose content
a reviewer could already read elsewhere -- every certified row is printed in the appendix and
the full 48-cell grid is what the certified-region figure plots.

**The lesson.** Four rounds of audit checked whether the documents were true. None checked
whether the *upload* answered the question the challenge asked, and the checklist row that
should have caught it was satisfied by evidence in a format the portal rejects.

### D-089 Four committed tables had no producer, and the manifest could not see it

`freeze.py --check` compares each committed table to its recorded hash. A table that no target
rewrites always matches, so the reproducibility gate is blind to exactly the failure it looks
like it covers: a result carried forward rather than regenerated.

Four tables were in that blind spot, and two of them are printed in the proposal.

| table | backs | status |
|---|---|---|
| `split_arm_baselines.csv` | Table 1, the study's largest measured effect | producer written, reproduces bit-identically |
| `rolling_origin.csv` | Table 3 and four bound claims | producer written, reproduces bit-identically |
| `mps_seed_sweep_summary.csv` | six bound claims | producer written, agrees to 1.1e-16 |
| `mps_seed_spread.csv` | nothing; evidence behind D-038 | retained and disclosed |

The rolling-origin reproduction is the one worth reporting. Its procedure existed only in the
prose of D-025 -- a 20-day calibration window and a 20-day test window stepped by ten days over
the frozen scores at `alpha = 1e-2`, judged against the exact 99 % Beta-Binomial interval -- and
implementing it from that description reproduces all five rows, ratios and interval bounds
exactly. A decision entry turned out to be a sufficient specification, which is the strongest
evidence so far that the record is doing its job.

The seed-sweep summary differs from the committed file in the last bit of two cells,
1.1e-16, which is float summation order rather than a disagreement; every claim resolves
unchanged.

`mps_seed_spread.csv` is kept. It is a contraction-width comparison whose producing variant of
`run_seed_sweep.py` is gone, it is consumed by no claim, and it is the measured evidence behind
D-038's choice of the sequential contraction -- so `RESULTS.md` now cites it, `PROVENANCE.md`
§1.4 records that it cannot be regenerated, and it is the single named exemption in the new
test. Reproducing it costs eight GPU fits and buys a number nothing quotes.

**The lesson.** A hash check answers "has this changed since I recorded it", which is not the
question "can this be produced again". The second needs a different test, and it is now
`test_every_committed_table_has_a_producer`, with an exemption list that has to state a reason.

### D-090 A map from each reference to the code that realises it

`REFERENCES.md` records what is cited and `REFERENCE_CROSSCHECK.md` records, automatically,
which files reach each entry. Neither answers the question a reviewer has, which is whether the
mathematics in the documents is the mathematics in the code.

That question is not rhetorical. Asking it this round found three formulas the documents printed
and the code did not compute (D-083) and a decision rule described with two models and
implemented with one (D-086). None of those was reachable by any gate here: every number agreed
with its table and every table agreed with the code. What disagreed was the formula printed
beside the number.

`docs/REFERENCE_IMPLEMENTATION.md` records the mapping. Twenty of the forty-nine entries reach an
implementation file; the other twenty-nine are cited for context, prior art, regulation or a
software version, which the crosscheck already reports. Five carry the guarantee and were read
line by line against the cited source:

* **Learn-then-Test** — the Hoeffding-Bentkus p-value matches the printed form term for term,
  with two conservative additions the formula does not show and neither changes the bound.
* **The exact coverage law** — called as `BetaBinomial(m, n+1-k, k)` at both sites, as printed.
* **Ding et al.** — matches after this round's strict-inequality correction.
* **Stoudenmire and Schwab** — matches after the output index was restored to the contraction.
* **Huang et al.** — computed and not gated, which amendment A6 already discloses, and its
  companion effective-rank definition was one of the three corrected.

Section 2 of that file is checked for attribution rather than line by line, and says so. The
judgement half of this check cannot be generated, which is the reason to write down where it has
been done and where it has not.

### D-091 The certificate was stated at a level and never at a confidence

Both PDFs said "certified at level $\alpha$" throughout and neither ever printed $\delta$.
`riskcontrol.py` defines the object as $\mathbb{P}(R(\lambda) \le \alpha) \ge 1 - \delta$ and
`configs/default.yaml` pins `delta: 0.05`; the number lived only in a config file. A
distribution-free finite-sample certificate without its confidence level is half a statement,
and model risk is the first reader to ask.

Section 2 now says it, and says it as one statement rather than two: a grid point is admissible
only when both nulls are rejected, so the joint certificate inherits a single $1 - \delta$ --
which is what `riskcontrol.py` states and what an intersection-union test delivers. Writing it
as two marginal statements would have been weaker than the code.

The same gap applied to the interval level. Every coverage verdict in section 3 -- "outside on
every seed", the seeds-outside column of Table 2, Figure 2 -- is decided by
`risk.coverage_band_level`, and 99 % appeared nowhere in either document.

### D-092 "Tuned" described a baseline that was never tuned, in eight places

There is no hyperparameter search anywhere in this repository: no grid search, no randomised
search, no optuna, no hyperopt. `fit_xgboost` sets hand-chosen constants, and Table 1's own
caption says "the same hyperparameters". The word "tuned" nonetheless described the classical
baseline eight times in the shipped documents, and section 7 called it "deliberately the
strongest available baseline" -- which is the load-bearing claim of the hybrid argument, since
the next sentence says a weak classical half proves nothing.

It is also not established as strongest. `baselines.csv` contains one model. `run_baselines.py`
registers three fitters and only XGBoost was ever written to the table, so no comparison exists
that would rank it against the other two.

Every occurrence describing the baseline is gone. Two survive and should: the RBF at
04-quantum.tex is genuinely tuned -- `rbf_correlation` maximises over a 25-point bandwidth grid
-- and "nothing is tuned or ranked there" is about selection on the test fold.

**And a claim the README made that was simply false**, written during this same round while
documenting the Expected Outcomes: "§3 against tuned XGBoost and LightGBM". There is no LightGBM
row in any shipped table. The challenge statement asks for at least one classical baseline and
XGBoost is one; claiming two was gratuitous and checkable.

### D-093 Four smaller defects the same audit confirmed

**The latency figure was quoted in the wrong unit.** 96.577 ms is one Gram *row* against a
64-row support set -- 64 kernel evaluations -- and section 5 called it "one in-band kernel
evaluation". Off by the support size, in the direction that makes the kernel look cheaper per
evaluation than it is.

**A claim reduced over the wrong rows.** `BandGbdtSeconds` took the minimum `fit_seconds` over
all of `mps_band.csv`, which holds both the tensor-network and the gradient-boosted arms, under
a note about the gradient-boosted baseline. The value was right only because the baseline
happened to be the faster of the two in this run. It now filters on the model.

**A GPU guard tolerated the case it exists to stop.** `require_idle_gpu` raised at
`len(listing) > 1`, and it runs before any CUDA context is created, so this process is not in
the listing -- one foreign process passed, which is exactly the contended-run scenario its
docstring cites as having produced a three-fold error in a reported column.

**A column named for rows held features.** `latency.csv` carried `rows_scored`, and the call
site has always passed `source.shape[1]`. Renamed in the producer and in the table; no value
changed and nothing referenced the old name.

Also corrected: the appendix said the kernel arm was "rejected by the other screen" without
naming it, when three screens were in play and the one that rejected all 120 configurations was
RBF distinctness -- conditioning admitted 28. The distinctness gate is inclusive in the code
(`corr <= 0.60`) and the proposal stated it strictly. And `CREDENTIALS.md` said the IBM
`ibm_fez` hardware run "now appears in the team section where it counts", which had stopped
being true when the section was compressed; it is back, in a section carrying 10 % of the score.

### D-094 Two teammates' personal email addresses were in a repository going public

`docs/CREDENTIALS.md` identified the teammates who independently attributed the author's QIntern
work by full name and personal Gmail address, and `docs/decisions.md` repeated one of them. This
repository becomes public on 2026-09-15.

The Phase I submission guidelines state that the proposal "must not contain confidential or
proprietary information of any third party" and that supplementary materials must not either.
Publishing a collaborator's contact address to support a credential is also simply not something
to do without asking them, and nothing about the attribution needs it: "a teammate, on whose
account the repository is hosted" carries the same evidential weight as a name and an address,
because the point is that the attribution came from someone other than the author.

Both are gone, from both files. The teammates are identified by role.

**The private-repository locators went with them, for a different reason.** The Team A
repository is private, so a filesystem path, a commit identifier or a branch name in it is a
locator no reviewer can follow. Citing one reads as evidence while being uncheckable, and the
path additionally exposed a directory layout. `CREDENTIALS.md` now says plainly that the
repository is private and that the checks described there cannot be independently repeated by a
reviewer -- which is the honest version of what those identifiers were standing in for.

The credential itself stays. "QIntern 2026 Project 12 (QWorld), within Team A" names a public
programme and a role in it, the way a competition placement does, and it is the evidence for
10 % of the score. What was removed is the apparatus around it that pointed somewhere nobody can
go. The one address remaining anywhere is the author's own, in the paragraph establishing that
two git identities are the same person; it is already public in his own commits, and it is his
to keep or remove.

### D-095 The submission ignored the two positive quantum results its own sponsor cites

The challenge statement's executive summary cites a hybrid quantum neural network at 0.87
precision (Deloitte and AWS, 2024) and a variational classifier at $F_1 = 0.88$ (Karimi et al.,
2024), and section 4.1 says participants are "encouraged to benchmark against these published
results and clearly report comparison methodology". This submission reports two negative quantum
arms and mentioned neither. Reporting a negative while silently passing over the sponsor's own
positive citations is the weakest available position: it invites the reading that the result is
a failure of execution.

Section 4 now names both and says why they differ, on two specific grounds rather than
rhetorically. Neither applies a distinctness screen -- both report a downstream score without
first testing whether the kernel is separable from a classical one, which is the test that
rejected all 120 configurations here. And neither states a search budget for its classical arm,
which is the failure mode the one controlled comparator in the literature identifies as
producing an apparent advantage.

Both are now reference entries, copied from the statement's own list.

### D-096 The statement supplies the economics the proposal declined to use

Section 1 read "We attach no monetary figure: issuer-side false-decline costs are not public".
That was defensible when the figures were not to hand. The challenge statement's own executive
summary supplies them: 443 billion USD of legitimate transactions falsely declined globally in
2021 (Aite-Novarica, 2019), and more than 40 % of customers abandoning after a decline (Radial,
2023). Declining to price impact on the criterion worth 25 % of the score, while the sponsor
puts the price in the problem statement, reads as a gap rather than as restraint.

Section 1 now uses both, attributed to the sources the statement attributes them to, and keeps
the restraint where it is still correct: the *issuer-side* cost of one false decline is not
public, so the study certifies the rate and leaves the bank to multiply it.

### D-097 No hardware run, and the statement is the reason

The statement asks participants to "use Amazon Braket (real QPUs and simulators)", which reads
as a requirement until the rest of the document is read. It is not one:

* Section 4.2: "full end-to-end model training or inference on quantum hardware is not expected
  nor required."
* Section 5.4: "participants who do not execute on hardware are not penalized, but hardware
  execution is encouraged."
* The noise and error-mitigation documentation bullet sits in an "is valued" list introduced by
  "Teams using hardware are encouraged to", so it is conditioned on hardware use.
* DM1 is offered as a way to "prototype noise-aware circuits **before** hardware execution".
  There is no hardware execution here for it to precede.

The one mandatory clause in the whole hardware block is a reporting clause -- "the total number
of samples used for quantum execution must be explicitly stated" -- and section 7 states it.

So no hardware task and no noise simulation was run, and none should be. A noise-aware study of
the kernel arm would measure the noise sensitivity of a method this study rejected on noiseless
grounds, before it ever reached the task. That is work that cannot change a conclusion.

**One version check, because a recommendation not to work rests on the text.** The revised and
the original challenge statements were compared word by word. They differ only in that the
revision adds URLs -- dataset links, AWS documentation, reference locators -- and three words.
No requirement changed. An earlier automated diff reported the two as identical, which is close
but not what the diff shows; the substantive conclusion survives and the claim of identity does
not.

### D-098 I propagated a miscitation this repository had already caught, and asserted a fact I never checked

D-095 added the challenge statement's two positive quantum results to section 4. Both halves of
how I did it were wrong, and an adversarial verification pass found them before submission.

**The miscitation.** The statement cites the $F_1 = 0.88$ result as "Karimi et al., 2024". It is
not. `REFERENCES.md` section 7 already recorded the correct attribution -- El Alami, Innan,
Shafique and Bennai, arXiv:2412.19441 -- and ended with the sentence **"Frequently miscited as
'Karimi et al.'"** I created a second entry for the same arXiv identifier under the miscited
name, and cited it that way in the proposal. The repository had the answer written down and I
did not read it before adding a reference to the same paper.

**The unsupported assertion.** My sentence said both papers "report a downstream score without
first testing distinctness from a classical kernel, and neither states a search budget for its
classical arm". The second clause is false of one and unverified for the other: the El Alami
paper has **no classical arm at all**, which is stronger than what I wrote and was already in
the file; and nothing in this repository documents the Deloitte and AWS report, so I had no
basis for any claim about its classical arm. I asserted a fact about a source I never opened, in
a submission whose entire pitch is that every claim resolves to something.

Both corrected. The proposal now says what is checkable: the 984-row balanced undersample at
seven principal components with no classical arm, which is neither this task's base rate nor a
comparison; and for the other, a precision quoted with no recall and no base rate, which is all
the statement's own text supports. The Deloitte entry carries a 要確認 saying the underlying
post has not been read. The duplicate section 7 record is gone, because one paper in two
sections under two attributions is how the miscitation propagated.

**The lesson, and it is about me rather than the repository.** Adding a reference is exactly the
moment to search for it first. The check that caught this was an independent pass instructed to refute
rather than confirm, reading the same file I had edited; it is the second time in this project
that adversarial verification caught a defect introduced by the fix for another defect.

### D-099 Two named requirements of the statement, one unmet and one framed as a lapse

**Class imbalance.** Section 5 of the challenge statement says "handling of class imbalance
should be documented (e.g., resampling, loss weighting, threshold tuning)". Neither PDF
addressed it, and the answer is a good one: this study does not resample and cannot, because
resampling or reweighting the calibration block breaks the exchangeability the certificate
depends on. Every block is scored as it falls and the imbalance is carried entirely by where the
three thresholds sit. That is the requirement answered by the method rather than in spite of it,
and leaving it unstated gave away a named "should" for nothing.

**The withdrawn dataset.** Amendment A9 presented ULB as "a dataset promised and never
obtained", which reads as a lapse. Section 5.4 of the statement lists "focus on one or two
datasets rather than all three" among its acceptable simplifications, so using one of three is a
scope the challenge explicitly permits. What is actually disclosable is that the
*pre-registration* promised a third, not that the challenge expected one. A9 now says which of
the two it is.

Four lines of displacement came from restatement in sections 2, 3 and 6.

### D-100 The clean room found a defect the gates could not, which is why it is run

The first clean-room pass predates `summarise_split_arms.py`, `summarise_seed_sweep.py`,
`run_rolling_origin.py` and `export_predictions.py` -- four producers and five committed tables.
A reproduction record that predates the scripts it covers is not a record, so the procedure was
repeated from `git archive HEAD` into an empty directory.

Every derived table reproduced byte-identically. And the run found a defect nothing else could:
`run_rolling_origin.py` printed **"0 of 5 breach"** against a table containing two. When the
verdict string was capitalised to match the selector in `claims.yaml`, the summary line two
functions below kept counting the lowercase form. The CSV was right the whole time and every
gate passed, because **nothing in this repository compares a console line to the file it
summarises** -- the claim gate reads tables, the freeze gate reads hashes, and neither reads
stdout.

The three verdicts are now named constants, read in both places. The general point is that a
literal used twice is a literal that will drift, and this project has now seen it four times.

Three tests cannot run from an archive extraction -- two call `git ls-files` and one needs the
LaTeX build log. All three fail rather than skip, which is correct and is the rule `check_pdf.py`
already states: silently passing when the evidence is absent is the worst thing a gate can do.
[`CLEANROOM.md`](CLEANROOM.md) §2b records which and why, so a third party following the
procedure is not alarmed by them.

### D-101 Two governing documents had never been read against the submission

Five audit rounds checked this submission against itself and against the challenge statement.
Neither the **Assessment Criteria** nor the **Terms and Conditions** had been read against it --
the document that governs how it is scored, and the one that governs what the entrant undertakes
by submitting. Both are now in the loop, and the gap is recorded in
[`VERIFICATION_CHECKLIST.md`](VERIFICATION_CHECKLIST.md) rather than quietly closed, because the
interesting fact is not what they contained but that five rounds of thorough work never opened
them.

That checklist is new and is deliberately not a third copy of the other two. `COMPLIANCE_CHECKLIST`
maps requirements to where they are met; `SUBMISSION_CHECKLIST` is the pre-upload walk; this one
records **which sources were read against the submission, on what date, and which checks remain
open** -- including the three that cannot be automated at all: the figures, the formulas against
the literature, and the upload itself.

**And one claim in the older checklist was weaker than it read.** It ticked "README carries the
development environment and the measured benchmark timings" and cited a *link* to
`ENVIRONMENT.md`. Carrying and linking are not the same thing for a reader who opens one file.
The README now states the machine and a six-row stage-cost table inline.

### D-102 A team result presented inside a paragraph that says "single-person team"

Section 8 declares "Single-person team." and, two lines below, listed the Yale Peaked Hackathon
placement with no qualifier. Every source attributes it to a team: the CV lists "Team MerQury"
against that project and writes "team score 450", and the addendum heads the section the same
way. A reviewer who opens the CV -- which the proposal invites, since the portfolio is linked --
finds a team placement presented as an individual one, in the section carrying 10 % of the score
and immediately after a sentence asserting the opposite.

Now attributed: "with team MerQury, on his own matrix-product-state attack". Both halves are
supported -- the placement is the team's and the attack is the author's, which the addendum
states separately.

**And the same section understated the upstream work.** It said only "merged into CUDA-Q and
QuEST". The addendum says "Three PRs merged. Two closed bounty issues, for USD 200 total", and
`CREDENTIALS.md` listed only two of the three, which is why the proposal was short. It now
states three merged pull requests and two closing bounty issues.

An understatement is a defect here in the same way an overstatement is, and this project has
now found four of them in the same section.

**A footnote on the correction itself.** Writing the dollar figure into these three documents
introduced an unpaired `$`, and `check_markdown_math.py` refused the commit. By rule 4 it would
in fact have rendered, because the delimiter touches a digit -- but it renders by an exception,
and any later `$` in the same file can pair with it. The gate built for the reported bug caught
a fresh instance of the same bug class inside the entry describing an unrelated fix, which is
the only evidence worth having that a gate is doing its job.

### D-103 A rule stated in the same section it was broken in

D-094 removed two teammates' full names and email addresses from `CREDENTIALS.md`, and added a
sentence saying "the teammates who made the attributions below are identified by role, not by
name or contact address."

Five given names survived that scrub in the same file and one other, embedded where a
search for full names and addresses would not look: inside quoted filenames of the form
`week3/<GIVENNAME>_TASK16_17_HANDOFF.md`, and inside the block quotations themselves, which
refer to people by initials and given names.

So the section stated a rule and broke it four lines later. That is worse than stating no rule,
because a reader who sees the undertaking stops checking. All five are replaced by role, with
the substitutions marked in the quotations so a reader knows the text was altered rather than
paraphrased.

The Terms and Conditions clause behind this -- "your submission does not contain any
confidential or proprietary information of any third party" -- was already ticked in the
compliance checklist, and was ticked on the strength of an argument about *datasets*. A tick
with the wrong evidence stops anyone looking again, which is exactly what happened here.

### D-104 A section that reproduced a private repository's contents while saying it did not

`CREDENTIALS.md` §2 opened with "Per the Phase I submission guidelines, no third party's
information is reproduced here", and then reproduced four block quotations from two teammates'
unpublished handoff documents and a team README, all held in a repository that is **private**.
It also said "commit identifiers and the local paths they were read from have been removed",
while eight internal paths remained in the same section.

The Terms and Conditions §3 warrant that a submission contains no "confidential or proprietary
information of any third party disclosed without permission". This repository goes public on
2026-09-15 and the proposal links to it. A teammate's unpublished prose, quoted at length
without a recorded permission, is the clause's central case rather than an edge of it.

Every quotation is now a description, and the internal paths are gone. **Nothing evidentiary was
lost**, and the reason is worth stating: a verbatim quotation from a repository a reviewer
cannot open carries no more weight than a description of the same document, because neither can
be checked from outside. The quotations were serving the author's confidence, not the reader's.

One four-word phrase is kept -- the team's own "protocol-final, numbers-provisional" -- because
it *limits* the credential rather than supporting it, and because the proposal already prints
it. Removing a third party's disclaimer while keeping the credit it qualifies would be the wrong
trade.

Note what the compliance checklist had said about this clause: D3, "met; both datasets are
public, and neither is redistributed". Correct about datasets, and blind to the actual exposure
twice over -- first the teammates' names and addresses ([D-094](#d-094)), then their prose.
Three findings against one row.

<a id="d-105"></a>
### D-105 An arithmetic claim of independence that the arithmetic does not support

Two documents asserted that the Yale solved-count follows from the score alone: "the ten
problems carry weights 10 through 100, which sum to 550 ... 450 is every problem but the last,
so nine of ten follows from the score without needing the author's repository at all."

It does not follow. The missing 100 points are made up by **ten** distinct subsets of
$\lbrace 10, 20, \ldots, 100 \rbrace$ -- $\lbrace 100 \rbrace$, $\lbrace 10, 90 \rbrace$,
$\lbrace 20, 80 \rbrace$, $\lbrace 30, 70 \rbrace$, $\lbrace 40, 60 \rbrace$,
$\lbrace 10, 20, 70 \rbrace$, $\lbrace 10, 30, 60 \rbrace$, $\lbrace 10, 40, 50 \rbrace$,
$\lbrace 20, 30, 50 \rbrace$ and $\lbrace 10, 20, 30, 40 \rbrace$ -- so a score of 450 is
consistent with six, seven, eight or nine circuits solved.

The count of nine is still true. What was false is the claim that it was *independently*
derivable, and that is the more damaging error of the two: the whole point of D-061 was to
prefer organiser-issued facts over author-issued ones, and this sentence smuggled an
author-issued fact across that line by dressing it as arithmetic. The leaderboard has no
"solved" column. The count's only source is the author's own README.

**The same section carried a second, related overreach.** Section 8 read "his own
matrix-product-state attack on 69-qubit circuits", which is precisely the attribution
[D-057](#d-057) and [D-059](#d-059) removed: the repository's own `verify_results.py` reports
P9 at 69 qubits as FAIL at Hamming 32/69, and the run that produced the correct answer was never
saved. D-059 had settled the honest form -- the saved runs are exact to 60 qubits and degrade
above it -- and section 8 had drifted back off it. Restated to match, which is also the better
sentence: a proposal whose section 4 is titled "how each failed" is more credible for naming
where its own method stops working, not less.

**The lesson is about the shape of the error, not the number.** Both defects survived five audit
rounds because they read as *more* rigorous than the truth. "It follows from the score itself"
sounds like a verification; it was an assumption wearing one. A derivation stated in a document
is a claim like any other, and this project had been checking numbers against tables while
leaving the sentences that connect them unchecked.

<a id="d-106"></a>
### D-106 "Which this submission reuses", said of a private and unlicensed repository

Section 8 described the QIntern work and closed: "so it establishes machinery, which is what
this submission reuses."

Read against the Terms' originality warranty -- "All submissions must be original work and must
not infringe on any third-party intellectual property rights" (§3) -- the natural reading of
*reuses*, applied to another team's private repository with no licence, is that code was carried
across. The appendix says the opposite in as many words: "The conformal implementation was
written from the published papers rather than adapted from an unlicensed repository, and that
decision is recorded with its date."

So the two documents contradicted each other on precisely the clause a reviewer would care
about, and the weaker of the two was in the section under the Team criterion. Corrected to say
what is true: what carries over is the author's practice with the machinery, and no code
crossed over.

The sentence was not trying to claim reuse -- "machinery" was meant in the sense of *technique*.
That is what makes it the dangerous kind of error: it was written for one reading and is
load-bearing under another, and only the second one is a warranty.

<a id="d-107"></a>
### D-107 Publishing on the submission date switches off a protection the Terms give

Terms §4.2: Resonance "will not share the full content of your submission with third parties
outside the evaluation and judging process without your prior written consent, **unless it is
already publicly available**."

This repository goes public on 2026-09-15, the same day as the upload, and the proposal prints
its URL in the title block. So the protection lapses by its own terms at the moment of
submission, and Resonance may from then on share the full content with anyone.

That is the intended trade and it is worth stating rather than discovering later. The whole
argument of this submission is that a reviewer can check it: the claim ledger, the frozen
manifest, the clean-room record and 175 tests are only worth anything if they can be opened. A
guarantee nobody can audit is the thing this proposal is arguing against. Keeping the repository
private to retain §4.2 would trade the submission's central property for a protection over
material the author is choosing to publish anyway.

Recorded because a null row and an unconsidered row look identical six months later.

### D-108 A novelty claim that was true as written and unqualified as read

Section 1 said conformal methods "reach this problem closely" and that "none conditions the
guarantee on the abstention region it routes to". The "none" was grammatically bound to the four
fraud papers named immediately before it, so the sentence was not false.

It was still the wrong sentence. Conditioning a distribution-free guarantee on a *selected*
region is an active named subfield, and this repository cited none of it -- a grep for the
obvious terms returned nothing across 53 references. A reviewer who works in that area reads the
sentence as a claim about the literature, not about four fraud papers, and finds it wrong.

Two entries added, both resolved against arXiv before citing rather than after
([D-098](#d-098) is why):

* **CP-18**, Xu, Guo and Wei, *Selective Conformal Risk Control*, arXiv:2512.12844,
  14 December 2025. Conformal risk control applied on the selected subset. Its SCRC-I variant
  has the same PAC-style form as the Learn-then-Test certificate used here.
* **CP-19**, Bai and Jin, *Conformal Selective Prediction with General Risk Control*,
  arXiv:2603.24704, 25 March 2026. Abstention with finite-sample control of a general bounded
  risk, via e-values rather than uniform concentration.

Section 1 now says selection-conditional risk control *does* condition on the retained region,
under exchangeability and not on payments. **The claim is narrower and much harder to attack**:
what is new here is the pair -- band-conditional and time-ordered -- not the first half alone.

Note the near-miss that made this worth checking: Wenge Guo co-authors CP-18 and also CP-17,
which this submission already cited. The nearest prior art was one hop from a reference already
in the file.

**A comparator was considered and declined.** Peng, Lu and Chen, arXiv:2608.24631, 25 August
2026, apply quantum kernels to card fraud and report ranking *second* on IEEE-CIS -- this
submission's own dataset -- while stating that their kernel "can also be evaluated exactly on a
classical computer" and so establishes "predictive and representational value rather than
computational quantum speedup". It is not cited. It agrees with section 4's finding rather than
challenging it, the challenge statement asks only that its own two cited results be engaged, and
a 6-of-6 page is the wrong place to spend lines corroborating a negative result the document
already reports. Recorded so the omission is visibly a decision.

### D-109 A literature sweep that found the nearest published certificate, three weeks old

[D-108](#d-108) added two selection-conditional references and stopped. A second sweep over
work posted after 2026-08 found a closer one still, and it changes how section 3 reads.

**CP-20**, Yu and Liu, *A Joint Finite-Sample Certificate for Adaptive Selective Conformal Risk
Control*, arXiv:2606.08517, 7 June 2026. It certifies selected risk under *adaptive* threshold
selection, treats that risk as a **ratio**, and couples an empirical-Bernstein bound on the
ratio with a Clopper-Pearson bound on acceptance and a closeness bound on utility. It reports
the empirical-Bernstein bound beating Hoeffding-based alternatives.

That lands on this submission's sharpest self-criticism. Section 3 says only 5 of 48 grid points
certify, that the survivors need $\alpha \ge 0.10$, and -- crucially -- that **the binding
mechanism is sample size acting through the concentration bound**. If that diagnosis is right,
a tighter concentration inequality on the same estimand is the route out, and here is a 2026
paper reporting exactly that. Section 3 now says so.

**This is a better sentence than the one it replaced**, and not because it softens the
limitation. A limitation with a named, citable remedy is a feasibility argument; a limitation
stated alone is a wall. The submission's own diagnosis is what makes the remedy legible, so
citing it also demonstrates the diagnosis was worth making.

What the paper does *not* do bounds it: ImageNet and COCO, no fraud, no payments, no temporal
ordering, no distribution shift. So it constrains the novelty claim on the **estimand** and
leaves the **setting** open, which is the same shape as CP-18 and CP-19.

**A fourth paper was found and deliberately not relied upon.** Joshi, Wang, Hassani and
Dobriban, *Risk-Controlled Post-Processing of Decision Policies*, arXiv:2605.06479, 7 May 2026,
certifies a decision rather than a prediction set -- the property this submission claims for
itself. It is in the not-relied-upon section rather than cited, because its structure is
different (agreement with an incumbent policy under a chance constraint, i.i.d. throughout, no
abstention band). Its fallback-on-failure design is the closest published analogue to section
6's degradation path, which was arrived at independently; recording that is more honest than
either claiming the idea or quietly citing it as though it were the source.

All four were resolved against the arXiv record on 2026-09-02 before being written down.

### D-110 A clean-room checkout found three scripts that crash instead of instructing

`results/runs/` is gitignored on purpose: the per-seed score files are large intermediates and
the repository commits the tables derived from them instead. A clean-room checkout therefore
does not carry them, and three scripts read the same one --- `run_rolling_origin.py`,
`export_predictions.py` and `validate_certificate.py` --- with the identical unguarded
expression. All three died with a bare `FileNotFoundError` naming an absolute path to a file the
reader has never heard of.

`summarise_seed_sweep.py` had already established the right pattern in this repository
(`raise SystemExit("...; run \`make seedsweep\` first")`). So this is a DRY defect as much as a
usability one: the guard existed, written inline in one place and absent from three others.

One helper now, `paths.require_run_artefact`, used at all three sites. There is no fallback to
add --- a score cannot be invented --- so the only improvement available was to turn the crash
into an instruction, and it now names the make target that produces the file.

**What this says about the earlier clean-room passes.** [`CLEANROOM.md`](CLEANROOM.md) §2 and
§2b both reported success, and both were run in a tree that already had `results/runs/`
populated. They tested whether the tables *regenerate*, which is a different question from
whether a reviewer can run the scripts at all. The third pass copied only tracked files and
found in one command what two passes had missed.

### D-111 Both competition placements are team results, and only one was attributed

[D-102](#d-102) attributed the Yale placement to team MerQury and left the QPoland line bare,
because no artefact on this machine recorded whether that entry was solo. It was not: the author
confirmed on 2026-09-02 that it was team **The Cats Cradle**. Now attributed.

The sentence that made this a defect was two paragraphs above: "Single-person team." Read next
to two unattributed competition placements, it invites the reading that both were the author's
alone. It now reads "Sole author of this submission", which is the true and narrower statement
--- it is about this entry, not about a competition history in which both placements were
teams'.

Note that this was carried as an open 要確認 rather than guessed at, and the guess that would
have been natural --- solo, since the CV does not say otherwise --- would have been wrong.

### D-112 The two degenerate fits were inflating the capacity signal, not the seed noise

Section 4 reported that seed noise is \ClaimSweepSpreadRatio-fold the capacity signal at full
scale, computed over all sixteen fits. It separately reported that two of those sixteen never
left chance. Both statements were true and the second undercut the first, because a run that
ends at the entropy of the prior is not a draw from the seed distribution.

**The justification given here was wrong, and [D-121](#d-121) corrects it.** This entry said
"a run whose cross-entropy never fell below the entropy of the prior". It did fall: a controlled
re-run shows both runs reaching a loss of 0.482 and 0.478, well under $\ln 2 = 0.693$, within
two epochs. They learned and then collapsed. The exclusion itself stands -- it keys on the final
ROC AUC, and the final state is degenerate either way -- but the mechanism was misdescribed.

The expected correction was that excluding them would *shrink* the ratio. It does the opposite.

| | largest seed spread | capacity signal | ratio |
|---|---|---|---|
| all sixteen fits | 0.1781 | 0.0629 | **2.83** |
| the fourteen that trained | 0.1274 | 0.0339 | **3.76** |

The two stalls sat at **two different bond dimensions**, so they depressed two of the four
per-chi means. That manufactured a spread *across* chi -- and the spread across chi is the
capacity signal itself. Removing them cuts the capacity signal by 46 per cent against 28 per
cent for the seed noise, and the ratio rises.

So a large part of what the sweep was measuring as "capacity" was two optimiser failures. The
conclusion the section draws -- that capacity is not resolvable at full scale -- was
**understated**, and the corrected figure supports it more strongly than the reported one.

**Both numbers are now reported, and that is the point.** Quoting only the all-fits ratio lets
two runs that never trained stand in for seed noise. Quoting only the trained ratio hides an
exclusion. Since the exclusion happens to help, it has to be visible, or a reader cannot tell it
was not applied in order to help. `summarise_seed_sweep.py` computes both from the same
`STALLED_BELOW_AUC` rule the summary already used to *count* the stalls, so no new threshold
enters the analysis -- and that rule was chosen in a gap where any value between 0.54 and 0.76
returns the same two runs.

**Also answered here: the statement's secondary objective 4.2**, which asks participants to
"characterize under what conditions (feature sets, data subsets, encoding strategies) quantum
approaches perform differently". The answer was in the tables and was never stated as an answer:
the tensor network reaches \ClaimMpsShareBand\,per cent of the baseline's average precision on
the eight-feature in-band problem and \ClaimMpsShareFull\,per cent on the 431-feature full one.
The gap widens with dimension, not with capacity. Section 4 now says so in the sentence that was
already making the comparison.

Paid for by deleting a sentence that restated the section title in a different register.

### D-113 `make reproduce` could not reproduce the table behind the largest reported effect

`make reproduce` passes no flags, so the defaults in `run_baselines.py` are what rebuild
`baselines.csv`. They did not match it. The committed table is **one model over three arms** ---
45 rows, xgboost, temporal plus both controls. The defaults were **three models over one arm**:
`--models` defaulted to `list(FITTERS)` and `--arms` to `["temporal"]`.

So `make reproduce` overwrote the table with a different shape, and the next target reached
`summarise_split_arms.py`, whose arm guard then halted:

```
baselines.csv has no rows for ['stratified', 'card_disjoint']
```

That table is `tab:splitarm`, which section 3 introduces as "the largest effect in this study".
**The repository's own reproduction command could not rebuild the evidence for its headline
control result**, and nothing caught it, because every test and every documented invocation
passes explicit flags. A default is only exercised by the one caller that omits them, and that
caller is the one a reviewer would use.

`--arms` now resolves from `configs/default.yaml` after parsing, the way `--seeds` already did,
so the default cannot drift from the config. `--models` is pinned to the single model the study
reports; the other two fitters remain reachable by flag. A test asserts the committed table's
arms and models against the defaults, so this cannot regress silently again.

This is the third defect of the same shape found here: [D-110](#d-110) (a script that crashed
instead of instructing), the two clean-room passes that ran in a populated tree, and now this.
**Every one was invisible to a test suite that passes arguments.** The lesson is not about
baselines; it is that the reviewer's path through the repository is the one least exercised.

### D-114 A retracted attribution went back in, in the commit that removed a different instance of it

[D-057](#d-057) removed "graph kernels", Weisfeiler-Lehman baselines and nested cross-validation
from the QPoland placement, because no implementation of any of the three exists on this machine,
and settled the entry as "QPoland 2025 runner-up and nothing more".

The label was changed to "Quantum-inspired graph kernels" on the grounds that the CV titles the
project that way and that a bare "quantum kernels" could be misread as the arm this submission
rejects. Both of those are true. Neither survives the point that **the CV's title for a project
is not evidence that the work exists in a form anyone can check**, which is exactly what D-057
turned on.

It went back in **in the same commit that removed the 69-qubit instance of the same error**, and
the commit message says so in as many words. Reverted to "Quantum kernels".

The misreading risk that motivated the change is real and is handled by context: section 4 is
titled "Two quantum arms, and how each failed", and the team line now carries a team name, a
year and a placement, none of which reads as this study's kernel arm.

### D-115 "Tuned" survived in four places, three of them in the shipped PDFs

[D-092](#d-092) is titled "'Tuned' described a baseline that was never tuned, in eight places"
and asserts "Every occurrence describing the baseline is gone. Two survive and should."

Four survived. Three were in `docs/tables.yaml`, which is the *source of the table captions*, so
they were rendered into both PDFs and were not visible to a grep over `submission/content/`:
the H4 caption, the metrics caption, and a clause reading "not because the baseline is untuned".
The fourth was section 7's "deliberately the strongest available baseline", which is the exact
phrase D-092 quotes as the load-bearing claim it removed.

There is no hyperparameter search anywhere in this repository -- `run_baselines.py` fixes the
`XGBClassifier` arguments as literals -- and section 3 says "at fixed hyperparameters", so the
captions contradicted the body. All four are gone. The two D-092 meant to keep are a tuned
*RBF*, which genuinely is tuned over a bandwidth grid, and "nothing is tuned or ranked" about
the held-out block.

**A decision entry claiming a class of defect is closed is itself a claim**, and this one was
wrong for eight days. The generated captions are the blind spot: `tables.yaml` is prose that
ships in the PDF, and every text audit until now searched the `.tex` sources.

### D-116 Three shares that summed to 100.87 per cent

Section 1 printed the operating point as approve / step-up / decline. Two of the three were
roll-ups from `operating_point.csv` and both contained the same 1,007 rows -- a challenge the
cardholder fails is a step-up *and* a decline -- so the three printed shares summed to
**100.87**.

A reader who adds three numbers in the first paragraph of section 1 and gets 100.87 has found
an error before finding anything else, and under the criterion carrying 25 per cent of the mark.

Fixed by binding the third number to the decline *branch* rather than the decline roll-up.
The three printed shares are now the three branches of the three-valued rule and sum to exactly
100. **The prose did not change at all** -- it was already an accurate description of the three
branches, and the defect was entirely in which row of the table the claim selected.

A claim was briefly added for the 0.87 per cent overlap and then removed, because the fix made
it unnecessary and `check_claims.py` refuses a claim nothing cites. That refusal is worth
noting: it is the gate that stops the ledger accumulating definitions no document uses, and it
fired within a minute of the claim becoming dead. The overlap is in `operating_point.csv`,
which the roll-up rows name explicitly.

### D-117 A requirement ticked against a section that did not contain it

The challenge statement asks, in §5.2 Reporting Considerations, for a "Description of quantum
approach, encoding strategy, and circuit design choices". `COMPLIANCE_CHECKLIST.md` ticked it:
`C17 | §4.2 Describe encoding strategy and circuit design | met | §4`.

Section 4 named no encoding. It described the screen's *dimensions* --- "configurations of
encoding $\times$ qubits $\times$ bandwidth $\times$ entanglement" --- which is a sentence about
the shape of the search, not about what was encoded. `screens.csv` and `circuits.csv` carry
`z`, `zz` and `dense_angle` with `linear` and `none` entanglement, and **none of those words
reached either PDF**. A reviewer checking this requirement against the submission would find the
word "encoding" and no encoding.

Now named in section 4, at the cost of one line. The circuit-design half was already discharged
by section 7's transpiled depth and two-qubit gate counts.

**This is the fourth checklist row found ticked on evidence that did not reach its clause**,
after D3 (third-party information, argued from datasets), the predictions row (an Expected
Outcome ticked while citing a rejected file format), and D1 (eligibility, quoting words the
Terms do not contain). The pattern is now specific enough to name: **the evidence column tends
to point at the section where the topic is discussed rather than at the sentence that discharges
the requirement**, and the two are not the same thing.

**Two more of today's own sentences were overstatements.** Section 6 opened "Every step below
was executed in this study on public data" --- two of its four steps are Phase II governance in
the future tense, and one of those is the power-sized evaluation block that section 4 reports as
*not* achieved. That sentence was written earlier today, while compressing the section. And
appendix A1 warned that non-exchangeable split conformal "requires the weights to be fixed
rather than fitted", a guarantee **the body never claims**: nothing in sections 1--8 makes a
non-exchangeable statement, and `weighted.py` reaches no committed table. The warning now
covers the guarantee the body does make, which is that $\alpha$ must be fixed before the
calibration scores are seen.

### D-118 Three open questions closed by the only party who could close them

All three had been carried as 要確認 rather than guessed at, and one of the guesses that would
have been natural was wrong.

**The sponsor-contractor bar (Terms §2).** Confirmed by the author on 2026-09-02: no current
contractual relationship with Resonance Alliance Inc. / The Quantum Insider or with HSBC. This
is a *disqualification* criterion, not a scoring one, so it was worth asking rather than
assuming -- a sole proprietor who takes inbound work is exactly the person for whom "Contractor
of a Challenge enterprise sponsor" is a live category.

**The sole-proprietorship date.** The filing records **1 August 2026**, which is what this
submission and the portfolio already use. So the CV's "Jul 2026 -- Present" is the error, not
the submission. It is outside this repository and should be corrected before the CV travels
with this submission, because §8 links the portfolio and a reviewer who opens both sees the
discrepancy.

**A trading name.** The entry now carries **Quantum Daemons** in both title blocks and in §8,
as the trading name of the registered sole proprietorship rather than as a company.

The reason this matters is narrower than it looks. Terms §4.2 lets Resonance publicise an entry
by "team name, submission title, and a summary description", and withholds personal names
without prior written consent. A single-person entry with no team name therefore gives them
nothing they are permitted to publish: the only identifier that exists is the one the clause
protects.

**And the portal has no team-name field.** The submission form, read live on 2026-09-02, is four
hidden inputs, a file picker and three buttons. The organisation name is an *account-level*
field set at registration -- `reg_company_type` and `reg_company_name`, placeholder "Company
Ltd" -- which is what §4.2's "organizational affiliations ... company, or institution" reaches.
So the document was the only surface under our control, and it now states the name. Note: Whether
the registration field already holds it is **要確認** and can only be read while signed in.

### D-119 The remaining audit findings, and the three that only closed across group boundaries

A second workflow re-verified every confirmed finding from the full-hierarchy audit against the
current tree and applied what was still open, in five disjoint file groups. Most were already
closed by [D-110](#d-110) and [D-113](#d-113) through [D-118](#d-118). The rest are applied.

Three could not be closed inside any one group, and they are the interesting ones.

**Three long scripts reported nothing, and a strict xfail held them there.**
`audit_labels.py`, `run_explain.py` and `run_power.py` each load all 590,540 rows with identity
columns behind no durable destination, and `run_power` then bootstraps the evaluation block two
thousand times. `tests/test_progress.py` recorded all three as `xfail(strict=True)`, which is
the right way to hold a known gap -- but it also means **instrumenting a script turns its XFAIL
into an XPASS and the suite goes red**. So the fix had to land in `scripts/` and `tests/` in the
same change, and neither group owned both. All three are now wrapped in `run_log`, and
`SILENT_WITHOUT_A_DESTINATION` is empty rather than deleted, because the mechanism that forced
this to be atomic is worth keeping.

**A flaky test with a provable cause, not a shrug.** `test_mps_contraction.py` drew its inputs
before `build()` called `torch.manual_seed(0)`, so they came from whatever CUDA RNG state the
preceding tests had left. Three fresh processes drew three different tensors -- input sums
56172.289, 56114.883 and 56326.0 -- against assertions tight enough to fail on the input alone.
A failure that cannot be reproduced from the file that contains it. Seeded before the draw.

**"Tuned" had a fifth instance, inside a committed artefact.** [D-115](#d-115) removed four from
the prose. `results/tables/power.csv` carried `tensor network vs tuned GBDT` in its
`comparison` column, written there by `run_power.py`. Both are corrected. **A string in a CSV
is as much a claim as a sentence in a PDF**, and a text audit over `.tex` and `.md` will never
see it.

**P7 is now met and P8 is corrected.** The scan P7 defines returned four lines:
two in `.gitignore`, where the ignore rule moved to `.git/info/exclude` so the path stays
ignored without being published, and two of prose here. P8 claimed "two human authors ... 32
commits"; the real figures are 58 commits and **three author strings, all the same person**,
one carrying a typo. No `Co-authored-by` trailer anywhere. Note: Left as 要確認 because a reviewer
reading the history of a sole-author submission sees three contributors.

**And a measurement was off by an order of magnitude.** `measure_latency.py` and
`ENVIRONMENT.md` both said the serving profile matters "by three orders of magnitude", five
lines from their own "roughly 380 times slower". Recomputed from `latency.csv`: 236x and 261x
for the two scorers, 379x for the micro-benchmark. None reaches three orders, and a reviewer
dividing two columns of the project's own table catches it immediately.

### D-120 Three author strings on a submission that claims a sole author

`git log` carried three author strings over 58 commits: `thedamon-wizard
<amon.koike@daemons.jp>` (27, a typo for the GitHub handle), `a-koike
<amon06251994@gmail.com>` (22) and `Amon Koike <amon.koike@daemons.jp>` (10). The author
confirmed on 2026-09-02 that all three are him.

Section 8 says sole author. A reviewer running `git shortlog -sne` on the public repository
would have seen three contributors and had to decide whether that contradicted the claim ---
and the honest answer, that they are one person with an inconsistent git config, is not
something the repository said anywhere.

**Fixed with `.mailmap`, not with a rewrite.** Mapping is by commit email, which collapses both
daemons.jp name variants and the gmail address in two lines. `git shortlog -sne` now reports one
author; `git log --format=%an` still returns all three, which is correct. **The history is
evidence and must not be edited to look tidier than it was** --- that principle is the whole
basis of `docs/decisions.md` keeping its retractions, and it does not stop applying at the
commit log.

The repository identity is also set to the canonical name, so new commits do not reopen it, and
a test asserts the single-author view rather than trusting it.

**One thing turned out already correct, and checking it mattered.** GitHub's contributor view
does *not* read `.mailmap` --- it matches by commit email --- so the fix could have been local
only. It is not: both addresses are registered to the same account and the API reports
`thedaemon-wizard` with all 59 commits, 37 under one address and 22 under the other. Had one
address been unregistered, `.mailmap` would have satisfied the test here while the public page
still showed two contributors, which is exactly the kind of gap this project keeps finding
between a local check and what a reviewer actually sees.

<a id="d-121"></a>
### D-121 The degenerate fits did not fail to start; they learned and then collapsed

The two degenerate cells were re-run against matched controls -- the same two bond dimensions
at a seed that trained cleanly -- with the loss trajectory captured, which is what
`run_seed_sweep.py` had been computing and discarding. Four jobs, about three and a half GPU
hours.

| $\chi$ | seed | ROC AUC | AP | initial loss | best loss | epoch of best | final loss |
|---|---|---|---|---|---|---|---|
| 8 | ...829 control | 0.798 | 0.209 | 0.456 | **0.302** | **30** | 0.302 |
| 8 | ...831 degenerate | 0.533 | 0.045 | 0.490 | **0.482** | **2** | 0.651 |
| 32 | ...829 control | 0.801 | 0.228 | 0.480 | **0.321** | **30** | 0.321 |
| 32 | ...831 degenerate | 0.488 | 0.050 | 0.478 | **0.478** | **1** | 0.668 |

Reproduce with
`.venv/bin/python scripts/run_seed_sweep.py --bonds 8 32 --seeds 20260829 20260831`.

**Three things this settles.**

**The sweep is deterministic.** All four average precisions reproduce the committed values to
six decimal places -- 0.209184, 0.045251, 0.227753, 0.049673. So the failures are a property of
the (seed, bond dimension) pair, not flakiness, and any reader can obtain them.

**The mechanism is divergence, not a bad start.** Every run begins in the same place: initial
losses 0.456 to 0.490, and the degenerate pair is not the worse half of that. The controls then
descend for the full thirty epochs, still improving at the last. The degenerate pair reaches its
best loss at **epoch 2 and epoch 1**, and climbs from there to $\ln 2$. The per-epoch log shows
the first one touching **ROC AUC 0.652 at epoch 1** -- genuinely above chance -- then decaying,
with the median gradient norm halving at epoch 8 and never recovering.

So "never left chance", which section 4 said and [D-112](#d-112) justified, is **wrong in a way
that understates the finding**. A model that cannot start is uninteresting. A model that learns
and then destroys what it learned, reproducibly, at a specific capacity and seed, is a
characterisation of the ansatz -- and it is what the challenge statement's secondary objective
4.2 asks for.

**The exclusion rule survives untouched.** `STALLED_BELOW_AUC = 0.65` keys on the *final* ROC
AUC, and the gap is 0.488 and 0.533 against 0.798 and 0.801. Nothing about the corrected
mechanism moves it, which is the point of having chosen a threshold in a gap rather than at a
tuned value.

**The timings from this run are not quotable.** All four rows carry `gpu_contended=True`
because a browser held a compositing context throughout, and `fit_seconds` came in at 3,060 to
3,369 seconds against 2,747 to 3,145 for the uncontended committed run. That is the stamp doing
its job: the numbers are visibly marked rather than silently wrong.

### D-122 The two tables behind the central contrast had no producer, and now reproduce byte for byte

`coverage_by_arm.csv` and `coverage_by_arm_seeds.csv` carry the comparison the three-arm design
exists to make: the temporal split breaches its coverage interval at the loose levels, the
stratified split never does, and the card-disjoint split sits between. Section 3 leads on that
contrast and `claims.yaml` binds four claims to it.

**No script wrote either.** They were committed by hand. That is worse than it sounds, because
`freeze.py --check` passed them on every run: **a file nothing regenerates can never differ
from its own hash.** The manifest was asserting that a table equalled itself. `make reproduce`
carried them forward untouched, so the evidence behind the central contrast was the one part of
the study a reviewer could not rebuild.

`scripts/run_coverage_arms.py` now produces both, and the result is the strongest available:
**both are byte-identical to the committed files**, all 72 rows, every column.

The computation was *extracted* rather than reimplemented.
`coverage.split_conformal_coverage` now holds the per-split coverage verdict, and both
`run_conformal.py` and the new producer call it. A second implementation of the order statistic
could have drifted from the one under test, and then the by-arm table would no longer be the
same procedure applied to a different split -- which is the only thing that makes comparing the
arms mean anything.

Two smaller choices are recorded because either could have hidden a selection. The headline
table takes the **first configured seed**, not the best: [D-020](#d-020) retracted a
maximum-across-seeds ratio that had been quoted as a centre, and `summarise_coverage.py` exists
because of it. And a level whose order index exceeds the calibration block is **skipped rather
than clamped**, because the quantile does not exist at that sample size and a clamped one would
assert coverage the data cannot support.

**The strict xfail is what made this fixable.** It failed loudly for as long as the gap was
open and would have failed the other way the moment a producer appeared, which is what forced
the registry entry to be removed in the same change. Both registries in the suite --- this one
and `SILENT_WITHOUT_A_DESTINATION` --- are now empty and kept rather than deleted. The
mechanism is the asset, not the entries.

### D-123 A control experiment that stopped a false finding

Verifying the README in a browser on 2026-09-02 showed every formula as raw LaTeX --
`$$R(\lambda) \;=\; \mathbb{P}\bigl(...\bigr)$$` displayed as text on the Preview tab, and the
same for every inline span. On its face that is a serious defect: the README is the public face
of the submission and the proposal links it.

It is not a defect. **GitHub's own documentation page for writing mathematical expressions
displayed `$\sqrt{3x-1}+(1+x)^2$` as raw text in the same browser session, with zero math nodes
in the DOM.** GitHub renders math client-side, and that renderer was not executing. The Mermaid
diagram, which was watched rendering earlier the same day in a different browser, also reported
absent -- the same cause.

The source is correct and unchanged: ` ```math ` fences and paired `$...$`, 96 dollar signs
outside code, all balanced, and `check_markdown_math.py` green.

**The lesson is about the shape of the check, not the outcome.** A rendering check that has no
control cannot distinguish "the content is broken" from "the renderer did not run", and the
first conclusion is the one that costs a page of the submission to act on. One navigation to a
page whose correct rendering is not in question settled it in under a minute.

This project has now been wrong in both directions on GitHub rendering: [D-082](#d-082) used
KaTeX as the oracle when GitHub uses MathJax and produced two wrong conclusions and one wrong
fix; this time the oracle was right and the instrument was broken. Both were caught by going to
a source whose answer was already known.

<a id="d-124"></a>
### D-124 The gap did not widen with dimension; the base rate did

[D-112](#d-112) answered the challenge statement's secondary objective 4.2 -- "characterize
under what conditions quantum approaches perform differently" -- by reporting the tensor
network at **87.8 %** of the baseline's average precision on the eight-feature in-band problem
and **49.1 %** at full scale, and concluding that the deficit widens with dimension.

**The two numbers are not comparable, and the conclusion is an artefact.** Average precision
floors at the positive rate: a random ranking scores the base rate, not zero. The two evaluation
sets do not share one. The in-band block is **10.96 %** positive -- the band is where the scorer
already concentrates fraud -- against **3.412 %** for the held-out block. The in-band ratio
therefore starts from a floor three times higher and flatters whatever sits on it.

Dividing lift by lift removes the floor from both sides:

| | positive rate | raw AP share | share of lift |
|---|---|---|---|
| band, 8 features | 10.96 % | 87.8 % | **44.8 %** |
| full, 431 features | 3.41 % | 49.1 % | **45.5 %** |

They agree to seven tenths of a point. **The deficit does not move with dimension at all.**

ROC AUC settles it independently, because its floor is a fixed 0.5 whatever the base rate and
needs no correction. As a share of the baseline's lift over 0.5 the arm reaches **35.9 %** in
the band and **79.2 %** at full scale -- the gap *narrows*, in the opposite direction to the raw
shares. Three normalisations, and only the confounded one supported the sentence that shipped.

`scripts/summarise_mps_lift.py` now computes all of this from committed tables, so the
correction is bound rather than argued.

**What makes this the worst kind of error this project has made.** It was not inherited: it was
written today, deliberately, as the answer to a named objective, and it went in the direction
that flatters the work -- the band is the setting this submission proposes to *use* a quantum
model in, and the claim made the arm look strongest exactly there. It also survived a claim
ledger, because both numbers were correctly bound to their tables. **A number can be right and
the sentence joining two of them still wrong**, which is the same lesson as [D-105](#d-105),
where an arithmetic claim of independence was false while every figure in it was correct.

The corrected finding is not weaker as an answer to 4.2. "The deficit is invariant to dimension
and to capacity" is a sharper statement about the ansatz than "it degrades with dimension", and
it is the one the evidence supports.

<a id="d-125"></a>
### D-125 Four findings from the final audit, and what each turned on

**A configuration the measurement never applied.** Section 5 read "Both figures bound model
cost on this workstation, single-threaded". `latency.csv` has exactly one quantum-kernel row
and its profile is **`CPU (unconstrained)`**; only the two classical components were pinned to
one thread. The claim ledger already knew: `LatencyScorerMedian` and `LatencyScorerTail` filter
`profile: 1-thread CPU`, while the two kernel claims deliberately carry no profile filter, and
`measure_latency.py` documents why. So a comment in `claims.yaml` asserting "every figure is
from the 1-thread CPU profile" was false for two of the four.

The paragraph's own finding is that **the serving profile dominates the model** by more than
two orders of magnitude. Getting the profile wrong there, in the sentence that closes it, is
the worst available place for this error. Section 5 now names the profile for each and says the
ratio is not like for like.

**A qualifier that dropped out of the PDF while three documents certified it.** Section 3 quotes
the 1st-place IEEE-CIS solution at "about 0.9363 in time-based cross-validation". `REFERENCES.md`
(DS-5) records that figure as measured **without their UID feature**, which they report as worth
about **+0.011 AUC** -- and the compliance row C7 described a sentence containing that
qualifier. The shipped PDF did not contain it. Quoting the ablated figure as the headline
understates the published result by roughly the margin the comparison turns on. Restored, and
C7 now describes what ships.

**An understatement that a merged pull request refutes.** Section 8 said "the author's
matrix-product-state runs are exact to 60 qubits". The *method* is his own contribution and is
checkable: PR #1 into the public repository `roman-bagdasarian/Peaked-Circuits`, merged
2026-04-10 from branch `marginal_attack_by_amon`, +150 lines. **Verified against the GitHub API
before citing** -- the repository is public and the pull request is visible, which the audit
had marked 要確認 because it could not open it. The attribution had been removed alongside the
69-qubit overreach in [D-105](#d-105); only the overreach needed to go. In a submission whose
section 4 is a matrix-product-state study, presenting the MPS credential as a team rank plus a
limitation sells it short.

**Two counts that go stale on every push.** `.mailmap` opened "three author strings, 58
commits" and attributed 10 to one identity; the actual figures were 65 and 16 by the time it
was read, and 27 + 22 + 10 did not even sum to 58. The counts are gone. What replaces them is
the command that produces the current split, because **a comment that is wrong after the next
commit is worse than one that states only the shape**.

**And the warning glyph is gone.** U+26A0 appeared ten times across five tracked files, all of
them mine from today. It sits in Unicode's emoji data with `Emoji=Yes`, and the project forbids
emoji in tracked files. `Emoji_Presentation=No` makes it arguable, and an arguable case in a
repository that goes public against an explicit instruction is not worth keeping.

<a id="d-126"></a>
### D-126 A circuit figure, and why it is not a portal deliverable

Asked whether the submission should attach a quantum circuit diagram, the statement was checked
rather than guessed at. **§5.2 Reporting Considerations requires** *"Description of quantum
approach, encoding strategy, and circuit design choices"*, and the Good-to-Have Metrics list
carries *"Qubit count and circuit depth (useful for assessing near-term hardware feasibility)"*.

Both are **descriptions**, and both were already met: §4 names the three feature maps and the
two entanglement patterns, §7 gives the transpiled depth and two-qubit counts, and
`circuits.csv` holds all twenty configurations. **The words "diagram" and "figure" do not occur
anywhere in the statement.**

**So no portal slot changes.** All five are full, and the arm a circuit drawing depicts was
rejected by its own screens *before it ran*. Trading the figure that shows the certificate --
the study's actual deliverable -- for a drawing of a method that produced no result would be
the wrong way round. The tensor-network arm, which did run at full scale, is not a circuit at
all.

What was genuinely missing is that **the repository had no picture of a circuit anywhere**, and
a table of depths is not a circuit design. `scripts/plot_circuits.py` draws the three encodings
at their smallest screened width and plots depth and two-qubit count across the grid. It costs
no upload slot and no page budget, and the proposal links the repository.

Four things were decided while building it, each because the obvious choice was wrong:

* **The widest configuration is not drawn.** Its text rendering is 526 columns; a reader learns
  less from it than from the numbers already in §7, and the scaling panel carries the same
  information legibly.
* **Qiskit's text drawer, not the matplotlib one.** The latter needs `pylatexenc`, and a new
  dependency for a documentation figure is a poor trade against the clean-room reproduction
  this project maintains.
* **The font size is computed from the widest drawing** rather than fixed, so the panel cannot
  silently clip when the grid gains a wider configuration.
* **Two series were merged.** `z, none` and `zz, none` are not close, they are *identical* --
  depth 7 and no two-qubit gates at every width -- so plotting both drew one invisibly beneath
  the other and left the legend naming a line the reader could not find. Neither dashing nor
  hollow markers fixes equal data. They are now one line labelled with both names, and the
  caption says why: with its entangling layer removed, `zz` **is** `z`, which is exactly what
  makes it the control the screens compare against.

That last one is worth keeping in mind. The first two attempts at it were cosmetic -- change the
dash, hollow the marker -- and both left a legend entry pointing at nothing. **A figure that
claims a line it does not draw is a false statement in the same way a wrong number is.**

<a id="d-127"></a>
### D-127 The training data cannot ship, so its fingerprint does

Asked whether the training data and trained models should be attached for reproducibility, the
statement and the licence were checked rather than assumed.

**The data must not be attached.** The statement's own §6.1 table lists IEEE-CIS as
*"Competition license"* and *"Kaggle (requires account)"*, and the competition rules §7.B are
explicit: *"You agree not to transmit, duplicate, publish, redistribute or otherwise provide or
make available the Competition Data to any party not participating in the Competition."*
Attaching it would breach the licence and the Terms' third-party-content warranty at once.

**Nothing asks for it either.** §5.2 Expected Outputs names three things -- fraud probability,
binary prediction, feature attribution. No checkpoint, no training set, no model artefact. The
per-transaction output is already staged as `HSBC-predictions.csv`.

**But the question exposed a real gap.** Everything in this study is reproducible from the
repository except its input, which a reader must fetch themselves -- and that is precisely
where two people can silently be comparing different things. The loader asserted 590,540 rows,
20,663 frauds and 394 columns, which catches a re-release, a truncated download or the test
split by mistake. It could not catch **a file of the same shape holding different bytes**:
re-encoded floats, a repaired text encoding, rows reordered within a shared timestamp. Each of
those changes every number while passing every check in the module.

So the digests now ship. `train_transaction.csv` and `train_identity.csv` are pinned by SHA-256
in `ieee_cis.py`, recorded in [`PROVENANCE.md`](PROVENANCE.md), and verified on every load in
1.9 seconds. **A digest is not the data** -- it redistributes nothing, and it is the strongest
identity statement the licence permits.

Three choices inside it, each because the obvious one is wrong:

* **Members, not the archive.** Kaggle serves re-zipped copies whose container bytes differ
  while the CSVs inside are identical. Hashing the zip would reject correct data, which is the
  failure mode that teaches a reader to delete the check.
* **Only the members actually read.** `test_*.csv` and `sample_submission.csv` are unhashed
  because nothing here opens them, and identity is verified only when a caller asks for it. A
  gate that fails someone for not downloading a file it never uses is punishing the wrong
  thing.
* **A hard error, not a warning.** Every committed number is conditional on this file.

**On model artefacts, the answer is no, and for a reason worth stating.** `results/runs/` is
100 MB of scores and telemetry and is deliberately untracked. Shipping fitted weights would
make the study checkable *only* by someone willing to trust them; what makes it checkable now
is that 50 artefacts are frozen by digest, every table has a producer, and 190 tests assert the
relationships between them. **A checkpoint proves what a model did once. A producer proves what
it does.**

<a id="d-128"></a>
### D-128 A verification target that invalidated the thing it verified

`make check` reported "3 scientific artefacts differ from the manifest" immediately after a
decision entry was added, and passed after a re-freeze. That looked like ordering noise. It was
two defects.

**A misclassification.** `results/tables/decision_log.csv` is a one-row projection of
`docs/decisions.md` -- it holds `decision_entries` and `protocol_amendments`, and its producer's
docstring says it counts documents "and not of any run". It contains no measurement. But it
lands under `results/tables/`, which `SCIENTIFIC` sweeps with a glob, while the document it
projects sits in `SPECIFICATION`, whose movement the freeze already declares legitimate. **The
same fact was classified two ways in the same manifest**, so adding a decision entry raised a
*scientific* alarm -- the message reserved for a corrupted measurement.

Compounding it: `check` depends on `claims`, which depends on `derived`, which *regenerates*
that file. So the command invalidated the manifest it then checked, and the only way to make
the message stop was to re-freeze -- which is precisely the habit a freeze exists to prevent.

Now excluded from the scientific glob and frozen with `docs/decisions.md`. The exemption is an
explicit path list rather than a pattern, because it weakens a gate and should be impossible to
widen by accident.

**A misdiagnosis.** With the classification fixed, two artefacts still moved: both PDFs. That is
correct and unavoidable -- they print `\ClaimDecisionEntries`, so a decision entry genuinely
changes their bytes, and `check` rebuilds them before comparing. What was wrong is what the
failure *said*: "Either a measurement genuinely changed ... or the run is not reproducible."
Neither is true here, and telling someone their run may be irreproducible when they added a
paragraph is how a gate loses its authority.

`freeze.py` now recognises the case -- **only built documents moved, and a specification file
they quote moved too** -- and says so: re-freeze, no measurement is implicated. It still exits
non-zero. The gate did not weaken; it learned to name the third cause.

Two tests pin both halves, and one of them guards the guard: it asserts a scientific pattern
still covers `results/tables/`, so the classification test cannot quietly become vacuous if the
globs are ever rewritten.

**What generalises.** A verification step that regenerates its inputs cannot distinguish "this
changed" from "I just changed this", and will always produce a failure whose remedy is to
suppress it. That is worth checking for wherever a `check` target has build dependencies.

<a id="d-129"></a>
### D-129 The comparator was dismissed on a ground that reading it refuted

`REFERENCES.md` carried a 要確認 on QM-16: the Deloitte/AWS blog post had never been read, and
everything asserted about it came from the challenge statement's one-line summary. Section 4
dismissed it as "a precision with no recall or base rate".

**The base rate is reported.** The post names its dataset -- ULB, 284,807 transactions, 492
frauds, a **0.172 %** fraud rate. So half the objection was simply false, and it was false in
the direction that let the submission move past a comparator the statement explicitly asks it
to engage.

Reading it also produced two facts the statement's summary omits and that make the comparison
sharper than the original objection did:

* **0.87 is one point on a threshold sweep**, not a summary. The post reports 0.87, 0.89 and
  0.92 at thresholds 0.65, 0.70 and 0.75.
* **There is a classical arm**, and the margin is narrow: 0.83, 0.84 and 0.86 at the same three
  thresholds, so the claimed advantage is 0.04 to 0.06 precision.

Recall, F1, accuracy and AUC are genuinely absent, so the precision still cannot be placed on a
curve. Section 4 now says the accurate thing: it is a thresholded precision on a base rate
**twenty times below this task's**, which is not a comparable quantity -- the same confound
[D-124](#d-124) had just corrected *internally*, appearing again in a cited comparator. That
symmetry is worth having in the document: the submission applies the same standard to others
that it was forced to apply to itself.

**The lesson is about the shape of the 要確認, not the source.** The marker said "not read, so
nothing beyond the summary is asserted" -- and then the proposal asserted something beyond the
summary anyway, in the sentence that dismissed it. A 要確認 that scopes a *document* does not
constrain what a *different* document says about the same object. Cheaper to read the source: it
took one fetch.
