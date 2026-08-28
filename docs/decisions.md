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
cuStateVec=True; device memory 629 → 5565 MiB (+4936, 4096 expected)."*

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

`hsbcfraud.conformal.weighted` implements Barber et al. eq. (11), and the tests assert that
it reduces exactly to standard split conformal at `w_i = 1` (verified to machine precision
at three (n, alpha) combinations). The point mass at `+inf` is included: omitting it is the
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
| temporal (days 141-181 held out) | 0.8805 | 0.5083 |
| stratified random | 0.9619 | 0.8031 |
| card-disjoint | 0.8474 | 0.5365 |

A random split inflates AUC by **+0.081** and average precision by **+0.295** over the
temporal split, on the same data with the same model.

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
| 200 | 0.7009 → 0.6951 (stalls) | 0.7002 → 0.4562 | 0.6988 → 0.4330 |
| 431 | 0.7113 → 0.6899 (stalls) | 0.6977 → 0.6909 (stalls) | 0.7102 → 0.3573 |

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
