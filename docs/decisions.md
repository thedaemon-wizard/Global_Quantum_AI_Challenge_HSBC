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
signal that a temporal split leaves available, since 84.8 % of test-block entities also
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

### D-019 The central experimental result: the guarantee breaks, and only under time

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

### D-020 Correction to D-019: the headline ratio is 1.44, not 1.49

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
