# Certified abstention for card-fraud decisions, with two quantum arms reported as measured

Phase I submission, Global Quantum + AI Challenge 2026, HSBC track — *Quantum-Enhanced Credit
Card Fraud Detection for Digital Payment Ecosystems*.

The contribution is a **distribution-free certificate on a decision an issuer actually
makes**, and an honest measurement of where two quantum approaches sit relative to a tuned
classical baseline. Neither quantum arm produced a usable result, and they failed differently:
the kernel was rejected by its own a-priori screens, while the tensor network ran and yielded a
non-superiority bound from an underpowered comparison. Both are reported in the body, with the
evidence that produced them.

---

## 1. Deliverables

The five files uploaded to the portal, produced by `make submission` and staged in
[`submission/portal/`](submission/portal). The portal accepts PDF, PNG, JPG, WEBP, GIF, PY,
JSON, JS, XLS, XLSX, CSV, DOC and DOCX, in five slots, with a 20 MB cap; `assemble_submission.py`
refuses to stage anything outside that list.

| # | Uploaded as | Format | What it is |
|---|---|---|---|
| 1 | `HSBC-proposal.pdf` | PDF, 6 pp. | The concept proposal. Built from [`submission/content/`](submission/content) with every figure bound to a table |
| 2 | `HSBC-appendix.pdf` | PDF, 3 pp. | Pre-registration and amendments, what we got wrong, the reproduction record, and the challenge statement's primary metrics in full (ROC AUC, AUPRC, $F_1$, precision, recall, confusion matrix) |
| 3 | `HSBC-certificate.csv` | CSV | The certificate itself: all 48 pre-registered configurations, which certify, the selected threshold and the band edges — [`riskcontrol.csv`](results/tables/riskcontrol.csv) |
| 4 | `HSBC-riskcontrol.py` | PY | The Learn-then-Test implementation the certificate rests on — [`riskcontrol.py`](src/hsbcfraud/conformal/riskcontrol.py) |
| 5 | `HSBC-certified-region.png` | PNG | Which (band budget, $\alpha$, $\alpha_{\mathrm{FN}}$) cells certify and which do not — [`certified_region.png`](results/figures/certified_region.png) |

The proposal maps to the six assessment criteria as: problem framing and expected impact (§1),
technical approach (§2, §4), feasibility and resources (§5), validation plan (§6), hybrid
design (§7), team (§8). The challenge statement's own requirements are covered as: primary
metrics (appendix §A4), comparison against a classical baseline (§3), published-leaderboard
comparison with methodology (§3), quantum encoding and circuit design (§4), quantum execution
sample count (§7), behaviour under temporal distribution shift (§3), feature attribution (§5,
from [`attribution.csv`](results/tables/attribution.csv) — see [D-048](docs/decisions.md)), and
inference latency (§5, from [`latency.csv`](results/tables/latency.csv)).

**Supporting documents**, not uploaded but referenced from the proposal and public in this
repository:

| Document | What it carries |
|---|---|
| [`docs/protocol.md`](docs/protocol.md) | The frozen pre-registration and its dated amendments |
| [`docs/guarantee.md`](docs/guarantee.md) | The guarantee stated once: notation, theorem, golden values, and six things it does **not** cover |
| [`docs/decisions.md`](docs/decisions.md) | Every decision and every retraction, in order |
| [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) | The machine, the pinned versions, and the measured wall-clock cost of each stage |
| [`docs/COMPLIANCE_CHECKLIST.md`](docs/COMPLIANCE_CHECKLIST.md) | Every requirement in the four official documents, with where it is met |
| [`docs/REFERENCES.md`](docs/REFERENCES.md) | The reference list |
| [`docs/REFERENCE_CROSSCHECK.md`](docs/REFERENCE_CROSSCHECK.md) | Which module or document reaches each reference — generated, not written |
| [`docs/REGULATORY_SOURCES.md`](docs/REGULATORY_SOURCES.md) | Official locators for every instrument, and what was verified at the issuing authority |
| [`docs/PROVENANCE.md`](docs/PROVENANCE.md) | Where every algorithm and dataset came from |
| [`docs/CREDENTIALS.md`](docs/CREDENTIALS.md) | Every biographical claim in the team section, what backs it, and what is unresolved |
| [`docs/FACTCHECK_LOG.md`](docs/FACTCHECK_LOG.md) | What was checked, against which source, on what date, and which claims the check overturned |
| [`docs/CLEANROOM.md`](docs/CLEANROOM.md) | Reproducing from an empty directory: the procedure, what it costs, and which steps were actually exercised |
| [`docs/SUBMISSION_CHECKLIST.md`](docs/SUBMISSION_CHECKLIST.md) | The pre-submission verification list |

---

## 2. What is claimed, and what is not

| | Claim | Status | Evidence |
|---|---|---|---|
| C1 | A three-valued decision rule (approve / step-up / decline) admits a finite-sample, distribution-free risk certificate on the **band-conditional** false-decline rate and the false-negative rate simultaneously | Supported at loose levels only: 5 of 48 configurations, none below $\alpha = 0.10$ | [`riskcontrol.csv`](results/tables/riskcontrol.csv), [protocol §6](docs/protocol.md) |
| C2 | Split-conformal coverage **fails under a temporal split** and holds under a stratified one; the card-disjoint control *over*-covers on 2 of 5 seeds at $\alpha = 0.01$ and *under*-covers on 1 of 5 at $\alpha = 0.001$ | Supported, with the direction of each deviation stated separately by level | [`coverage_seed_summary.csv`](results/tables/coverage_seed_summary.csv), [`coverage_by_arm_seeds.csv`](results/tables/coverage_by_arm_seeds.csv) |
| C3 | The size of the temporal breach is **origin-dependent**, not a fixed property of the data | Supported; this narrows C2 | [`rolling_origin.csv`](results/tables/rolling_origin.csv) |
| C4 | A fidelity quantum kernel is **rejected before being run** by two a-priori screens | Supported | [`screens.csv`](results/tables/screens.csv), [protocol §9 and amendment A6](docs/protocol.md) |
| C5 | A matrix-product-state classifier does not beat a tuned GBDT in the band | Weaker than a null result: **underpowered** against the pre-registered ceiling: the effect this comparison could resolve is 0.0557 AP against a ceiling of 0.023. What survives is a non-superiority bound of about +0.02 AP | [`mps_h4.csv`](results/tables/mps_h4.csv), [`power.csv`](results/tables/power.csv), [D-030](docs/decisions.md) |
| C6 | At **full scale** the same classifier loses by roughly a factor of two in average precision, across 16 fits at four seeds | Supported | [`mps_seed_sweep_summary.csv`](results/tables/mps_seed_sweep_summary.csv) |
| C7 | The bond-dimension sweep is **launch-bound**, not governed by $\chi$ | Supported; this is what makes C6 affordable | [`mps_seed_sweep.csv`](results/tables/mps_seed_sweep.csv), [D-032](docs/decisions.md) |

**Not claimed.** No quantum advantage of any kind. No portfolio-level PSD2 compliance
figure (see [amendment A2](docs/protocol.md)). No claim that the certificate binds the
unconditional false-decline rate — it does not, and §1 of the protocol says why.

---

## 3. The certificate

### 3.1 Estimand

The certified quantity is conditional on the band, not marginal:

$$R(\lambda) \;=\; \mathbb{P}\bigl(\,D(X) = \texttt{DECLINE} \;\bigm|\; Y = 0,\; X \in B\,\bigr)$$

where $B = \{x : \tau_{\mathrm{lo}} < f(x) < \tau_{\mathrm{hi}}\}$ is the abstention band and
$D$ the composite rule. The marginal rate $\mathbb{P}(D(X)=\texttt{DECLINE} \mid Y=0)$ is
reported alongside and is deliberately **not** the headline: it is dominated by
$\tau_{\mathrm{hi}}$ and would barely move if the in-band scorer were replaced by a coin
flip. A certificate that does not bind the component it licenses is not evidence about that
component.

### 3.2 Risk control

Learn-then-Test (Angelopoulos, Bates, Candès, Jordan & Lei, *AoAS* 19(2):1641–1662, 2025)
controls the family-wise error over a **pre-specified finite grid** $\Lambda$. For each
$\lambda \in \Lambda$ the Hoeffding–Bentkus p-value is

$$p^{\mathrm{HB}}_\lambda \;=\; \min\Bigl\{\, \exp\bigl(-n\,h_1(\hat{R}(\lambda) \wedge \alpha,\ \alpha)\bigr),\;\; e\,\mathbb{P}\bigl(\mathrm{Bin}(n,\alpha) \le \lceil n\hat{R}(\lambda)\rceil\bigr) \,\Bigr\}$$

with $h_1(a,b) = a\log\frac{a}{b} + (1-a)\log\frac{1-a}{1-b}$. Two risks are controlled
jointly — the band-conditional false-decline rate at $\alpha$ and the false-negative rate at
$\alpha_{\mathrm{FN}}$ — with Holm across the grid.

**The bound applies only to a mean of per-observation losses.** An earlier
`recall_shortfall`, defined as $\max(0, \text{floor} - \text{recall})/\text{floor}$, is a
nonlinear transform of a mean and therefore produced quantities that were not valid
p-values. It was replaced by a genuine 0/1 loss averaged over frauds; controlling the FNR at
$\alpha_{\mathrm{FN}}$ is equivalent to a recall floor at $1 - \alpha_{\mathrm{FN}}$, so
nothing is lost. See [D-024](docs/decisions.md).

### 3.3 Coverage validation, and why the naive check is wrong

For a sound split-conformal system the number of test errors is not merely bounded in
expectation — it follows an exact predictive law:

$$E \;\sim\; \mathrm{BetaBinomial}\bigl(m,\; n + 1 - k,\; k\bigr), \qquad k = \lceil (n+1)(1-\alpha) \rceil$$

Checking $\hat{r} \le \alpha$ instead is a **one-sided test against the wrong null**: on a
correctly calibrated system it passes only 51.15 % of the time. Every coverage row in
[`coverage_by_arm.csv`](results/tables/coverage_by_arm.csv) is judged against the
Beta-Binomial interval, not against $\alpha$.

### 3.4 The reachability limit, stated up front

At the tightest band budget (2 % of traffic) $D_{\mathrm{cal}}$ contributes only 847
legitimate in-band rows; the four budgets on the grid give 847, 1572, 2259 and 4831. What
binds is the concentration bound: Hoeffding-Bentkus under Holm, over the decision grid and
both risks, needs the observed risk to sit far enough below $\alpha$ for the bound to reject,
and how far depends on that sample size. **The band-conditional certificate is sample-starved
by construction**, and the reachable $\alpha$ is capped by the band budget rather than by the
method. This is [amendment A1](docs/protocol.md), recorded before the arm ran.

An earlier version of this paragraph attributed the cap to the class-conditional degeneracy
floor of Ding, Angelopoulos, Bates, Jordan & Tibshirani (NeurIPS 2023), and printed it as
$\lceil 1/\alpha \rceil - 1$ rather than the exact $(1/\alpha) - 1$ that paper establishes.
Both were wrong. On this grid the floor is 19 rows at $\alpha = 0.05$ and 3 at $\alpha = 0.25$,
against 847 available, so it cannot bind anywhere here ([D-044](docs/decisions.md)).

---

## 4. Results

### 4.1 What a random split buys you

Before any conformal machinery. One gradient-boosted model, fixed hyperparameters, five
seeds, three splits of the same file:

| Split | ROC AUC | AP | AP vs forward holdout |
|---|---|---|---|
| stratified random | 0.9658 | 0.8254 | **+0.3164** |
| temporal (forward holdout) | 0.8851 | 0.5090 | — |
| card-disjoint | 0.8509 | 0.5394 | +0.0304 |

Ranges do not overlap across seeds (stratified AP 0.8171–0.8356 against temporal
0.5055–0.5114). **A random split reports 62 % more average precision than a forward holdout
on identical data and an identical model.**

That is the largest effect in this study. It is larger than anything either quantum arm could
have contributed, and observing it required no quantum method — only the discipline of
running the control. It is also why every number below comes from a temporal split, though a
random one would flatter all of them.

### 4.1a The split, and which block may touch which parameter

![The four-block temporal split](results/figures/architecture.png)

Shaded blocks carry the guarantee. Band edges and every threshold come from
$D_{\mathrm{band}}$; $\lambda$ is certified on $D_{\mathrm{cal}}$; nothing is selected on
$D_{\mathrm{test}}$. This geometry is what removes the *selective* break in exchangeability —
conditioning on band membership would be conditioning on a data-dependent event if the edges
had been estimated on the data used to certify. Only the temporal break survives, and §4.2
measures it.

### 4.2 Coverage by split arm

Nominal versus empirical false-decline rate on $D_{\mathrm{test}}$, calibrated on
$D_{\mathrm{cal}}$. `finite_sample_ok` is the Beta-Binomial verdict.

![Empirical over nominal coverage by split arm, against the exact Beta-Binomial interval](results/figures/coverage_by_arm.png)

Grey bands are the exact interval; red points fall outside it. The interval widens sharply as
$\alpha$ tightens, which is why the temporal arm's apparent compliance at the tightest level is
weak evidence rather than a pass.

| Arm | $\alpha$ | Ratio (mean) | sd | Range | Seeds outside the interval |
|---|---|---|---|---|---|
| temporal | 0.010 | 1.436 | 0.055 | [1.349, 1.494] | **5 of 5** |
| temporal | 0.005 | 1.456 | 0.036 | [1.421, 1.493] | **5 of 5** |
| temporal | 0.002 | 1.314 | 0.067 | [1.205, 1.385] | **4 of 5** |
| temporal | 0.001 | 0.943 | 0.020 | [0.923, 0.977] | **0 of 5** |
| stratified | 0.010 | 1.008 | 0.038 | [0.952, 1.056] | **0 of 5** |
| stratified | 0.005 | 1.039 | 0.037 | [0.990, 1.084] | **0 of 5** |
| stratified | 0.002 | 1.019 | 0.071 | [0.943, 1.127] | **0 of 5** |
| stratified | 0.001 | 0.962 | 0.140 | [0.798, 1.158] | **0 of 5** |
| card-disjoint | 0.010 | 0.916 | 0.133 | [0.785, 1.076] | **2 of 5** |
| card-disjoint | 0.005 | 0.917 | 0.055 | [0.851, 0.993] | **0 of 5** |
| card-disjoint | 0.002 | 0.904 | 0.157 | [0.783, 1.166] | **0 of 5** |
| card-disjoint | 0.001 | 0.882 | 0.362 | [0.650, 1.525] | **1 of 5** |

Five seeds per arm and level, each judged against the exact Beta-Binomial interval rather than
against $\alpha$.

**The temporal arm breaches on every seed** at the two loosest levels and on four of five at
the third. **The stratified arm breaches on none, at any level.** That contrast is the result
the three-arm design was built to produce.

The card-disjoint arm is not clean, and the earlier version of this section was wrong to say
it was: it is outside on two of five seeds at $\alpha = 0.01$ and one of five at
$\alpha = 0.001$, where its spread (sd 0.362) is an order of magnitude wider than the
temporal arm's. Its 0.001 row is a single seed at ratio 1.525 against four between 0.65 and
0.76 — noise, not a systematic breach, but not a pass either.

Two further cautions. The temporal arm does **not** breach at the tightest level: at
$\alpha = 0.001$ its mean ratio is 0.943 with every seed inside, and the Beta-Binomial band
there is wide enough to have little power. And "the breach is temporal" is stronger than three
non-randomised arms can carry — the arms differ in more than time, and by a two-sample
classifier the card-disjoint arm is the *most* distinguishable of the three (0.6554 against
the temporal arm's 0.5525). What the data supports is that the breach appears only in the arm
ordered by time, on every seed, and that neither control reproduces it.

### 4.3 The certificate holds on held-out data

All five certified configurations, applied unchanged to $D_{\mathrm{test}}$:

| Band budget | $\alpha$ | $n$ legit in band | Declined | Realised | Fraction of budget |
|---|---|---|---|---|---|
| 0.035 | 0.25 | 3,314 | 580 | 0.1750 | 0.70 |
| 0.050 | 0.25 | 4,835 | 830 | 0.1717 | 0.69 |
| 0.100 | 0.10 | 10,021 | 858 | 0.0856 | **0.86** |
| 0.100 | 0.15 | 10,021 | 858 | 0.0856 | 0.57 |
| 0.100 | 0.25 | 10,021 | 1,762 | 0.1758 | 0.70 |

**Five of five hold.** This is H5, and it had never been run — the study's headline deliverable
had no held-out evidence behind it until now.

Read it for what it is. That the machinery transfers to unseen data is a real check. That it
transfers while holding a ceiling of 0.10 to 0.25 is a weak one, because those ceilings are
loose, which [amendment A1](docs/protocol.md) predicted from the band's sample size before the
arm ran. The pre-registered test was also specified for the wrong mechanism — Beta-Binomial
suits split conformal, not Learn-then-Test — so both it and the correct binomial tail are
reported ([amendment A5](docs/protocol.md)).

### 4.4 What the certificate actually reaches

![Which band budget, alpha and missed-fraud budget cells certify](results/figures/certified_region.png)

Filled cells certify; open cells have no admissible threshold. The sparsity is the point —
a reader told "the certificate holds" would otherwise assume it holds everywhere.

Five of 48 configurations certify. All five sit **strictly inside** the band — the smallest
margin below $\tau_{\mathrm{hi}}$ is 0.0136 — which is the property an earlier version failed:
every one of its 32 "certified" configurations sat exactly at the boundary, where the flagged
set is empty by construction and the risk is structurally zero ([D-024](docs/decisions.md)).

| Band budget | $\alpha$ | $\alpha_{\mathrm{FN}}$ | $n_{\text{legit,band}}$ | Admissible $\lambda$ | Selected |
|---|---|---|---|---|---|
| 0.035 | 0.25 | 0.45 | 1572 | 1 | 0.0582 |
| 0.050 | 0.25 | 0.45 | 2259 | 2 | 0.0541 |
| 0.100 | 0.10 | 0.45 | 4831 | 1 | 0.0537 |
| 0.100 | 0.15 | 0.45 | 4831 | 1 | 0.0537 |
| 0.100 | 0.25 | 0.45 | 4831 | 2 | 0.0426 |

These are **loose levels**, and saying so is the point. Nothing certifies below $\alpha =
0.10$, nothing certifies at the 2 % band budget at all, and every surviving configuration
needs $\alpha_{\mathrm{FN}} = 0.45$. The mechanism is sample size, not method: certification
appears only once the band budget is widened enough to put a few thousand legitimate rows in
$D_{\mathrm{cal}}$. That is [amendment A1](docs/protocol.md) showing up in the measurement
exactly where it was predicted to.

### 4.5 The breach is origin-dependent

Re-running the same procedure at five rolling calibration origins:

| `cal_start` (day) | Ratio | Verdict |
|---|---|---|
| 101 | 0.684 | conservative |
| 111 | 0.893 | inside |
| 121 | 1.411 | **breached** |
| 131 | 1.198 | **breached** |
| 141 | 1.126 | inside |

Two of five breach and the earliest **reverses** the sign. So the honest statement is not
"temporal splits inflate the rate by 1.4×" — it is that the deviation depends on where the
calibration window is placed, and a single origin cannot establish its magnitude. This
narrowing is recorded in [D-025](docs/decisions.md); an earlier draft quoted 1.49× from a
single seed's maximum, which was an overclaim.

### 4.6 Quantum kernel: rejected by the screens, before it ran

Two a-priori gates over 120 configurations (encoding × qubits × bandwidth × entanglement):

$$\text{effective rank } r_{\mathrm{eff}} = \frac{\left(\sum_i \sigma_i\right)^2}{n \sum_i \sigma_i^2}, \qquad \rho_{\mathrm{RBF}} = \mathrm{corr}\bigl(K_Q,\, K_{\mathrm{RBF}}(\gamma^\star)\bigr)$$

A kernel is usable only if it is **not** exponentially concentrated — effective rank
$r_{\mathrm{eff}}$ inside a usable band — **and not** reproducible by a tuned RBF
($\rho_{\mathrm{RBF}} < 0.60$). Result: **28 of 120 pass conditioning, 0 pass distinctness, 0 pass both.** The
closest any configuration came was $\rho_{\mathrm{RBF}} = 0.6291$ against a 0.60 threshold.

The kernel arm was therefore not run on the decision task. Reporting a screen that rejects
its own headline method is the point of pre-registering it.

### 4.7 Tensor network: it ran, and it lost

The matrix-product-state classifier (Stoudenmire & Schwab, NeurIPS 29:4799, 2016) contracts

$$f(\mathbf{x}) \;=\; \sum_{\{s\}} A^{s_1}_{\alpha_1} A^{s_2}_{\alpha_1\alpha_2} \cdots A^{s_N}_{\alpha_{N-1}} \prod_{j=1}^{N} \phi^{s_j}(x_j), \qquad \phi(x) = \bigl[\cos\tfrac{\pi x}{2},\ \sin\tfrac{\pi x}{2}\bigr]$$

with bond dimension $\chi$ swept rather than tuned.

![Average-precision difference, MPS minus GBDT, with intervals, by bond dimension](results/figures/mps_h4.png)

Every interval contains zero. In-band, on identical rows and features,
against a tuned GBDT — see [`mps_band.csv`](results/tables/mps_band.csv) and
[`mps_h4.csv`](results/tables/mps_h4.csv).

| $\chi$ | AP (MPS) | AP (GBDT) | $\Delta$AP | 95 % clustered CI | $p$ | Holm |
|---|---|---|---|---|---|---|
| 4 | 0.1202 | 0.1407 | -0.0205 | [-0.0593, +0.0149] | 0.864 | not rejected |
| 8 | 0.1173 | 0.1407 | -0.0234 | [-0.0630, +0.0122] | 0.895 | not rejected |
| 16 | 0.1183 | 0.1407 | -0.0225 | [-0.0613, +0.0136] | 0.884 | not rejected |
| 32 | 0.1235 | 0.1407 | -0.0172 | [-0.0582, +0.0200] | 0.806 | not rejected |

Every interval contains zero, so the correct reading is **not** "the MPS is worse" — it is
that the comparison cannot separate them. Note also that AP does **not** increase with $\chi$:
the ordering here is 0.1202, 0.1173, 0.1183, 0.1235, and §4.8 shows why no ordering should be
read from a single seed per configuration.

**And the power gate that was supposed to tell us this in advance was itself wrong.**
`power.csv` initially recorded an MDE of 0.0069 against the 0.023 ceiling, estimated from the
seed-to-seed spread of the baseline against itself. But the comparison H4 makes is between
*model families*, whose measured standard error turned out to be 0.018–0.020 — about
eightfold larger. The effect this comparison could actually resolve is 0.0557 AP, which
**exceeds** the pre-registered ceiling of 0.023 — as does the 0.0461 the corrected
pre-registration estimated. The body reports the measured figure rather than the estimate
([D-043](docs/decisions.md)).
H4 is therefore reported as **underpowered**, exactly as the protocol commits to doing when
this happens. The gate now computes a cross-family proxy (logistic against GBDT, still no
quantum model) and binds on the larger of the two. See [D-030](docs/decisions.md).

What survives is narrower than a null result and is stated as such: **any improvement from
the tensor network is bounded above by roughly +0.02 AP** at 95 % confidence. That is a
non-superiority bound. It is not evidence of equivalence, and it is not evidence that the
tensor network is worse.

### 4.8 At full scale: sixteen fits, no readable capacity signal, two failures

Four bond dimensions at four seeds each, all 431 features, scored once on $D_{\mathrm{test}}$:

| $\chi$ | ROC AUC (mean) | AP (mean) | AP range across seeds | Never left chance |
|---|---|---|---|---|
| 4 | 0.8023 | 0.1997 | 0.1442–0.2512 | 0 of 4 |
| 8 | 0.7301 | 0.1493 | 0.0453–0.2092 | 1 of 4 |
| 16 | 0.7922 | 0.1985 | 0.1740–0.2464 | 0 of 4 |
| 32 | 0.7103 | 0.1368 | 0.0497–0.2278 | 1 of 4 |
| **tuned GBDT** | **0.8851** | **0.5090** | 0.5055–0.5114 | 0 of 5 |

Three things to read.

**The margin is not close.** The best of sixteen draws reaches AP 0.2512 against the
baseline's 0.5055–0.5114. Every draw loses by a factor of two or more, so which one is
reported does not matter.

**Capacity is not resolvable.** The largest spread across seeds at one bond dimension is
0.1781 AP; the spread of the seed means across bond dimension is 0.0629. Seed noise is 2.8
times the capacity signal. This settles [D-038](docs/decisions.md) at every $\chi$ rather than
at one, and replication is what established it — an earlier draft read the single-seed
ordering as though it meant something.

**Two of sixteen fits never trained.** Both at seed 20260831, at $\chi = 8$ and $\chi = 32$,
on the sequential contraction that [D-038](docs/decisions.md) chose *because* the reduction
tree destabilised training. A one-in-eight failure rate is a property of the ansatz on this
data, not of the optimisation. The per-epoch AUC shows the shape: 0.6521 at epoch 1, falling
to 0.4761 by epoch 30 — below chance. A loss-only log would have shown a flat curve and left
open whether it was slow learning or none ([D-039](docs/decisions.md), [D-040](docs/decisions.md)).

**And the fit time is flat across $\chi$**: the sixteen jobs took 2747 to 3145 seconds,
against the sixty-four-fold spread a $\chi^2$ cost model predicts. A 431-site chain is bound
by the launch overhead of its sequential contractions, not by their arithmetic; the competing
predictions were written down before the measurement ([D-032](docs/decisions.md)). Capacity
is nearly free here and depth is the cost, which is the opposite of the intuition carried
over from short chains.

---

## 5. Negative results, and why they are in the body

Findings in this repository that contradict what the work set out to show, and one that
contradicts a claim an earlier draft had already written down:

* The **quantum kernel** was eliminated by a screen the protocol fixed in advance.
* The **tensor network** ran and did not beat gradient boosting — though the comparison
  turned out to be underpowered, so the result is a non-superiority bound rather than a
  clean null ([D-030](docs/decisions.md)).
* The **power gate meant to catch that in advance was itself mis-calibrated**, and passed a
  comparison it should have flagged. It was estimating seed noise where family noise was
  needed, and understated the standard error eightfold.
* The **temporal coverage breach** turned out to be origin-dependent, which removed a clean
  headline number ([D-019](docs/decisions.md), [D-020](docs/decisions.md),
  [D-025](docs/decisions.md) — the central result was narrowed twice).
* A **certificate that appeared to succeed was vacuous**: all 32 certified configurations
  sat at $\lambda = \tau_{\mathrm{hi}}$, where the flagged set is empty by construction, and
  it rested on an invalid p-value besides ([D-024](docs/decisions.md)).
* The **first latency measurement measured the library's threading default**, not the model —
  an OpenMP barrier over one row cost 19 ms against 0.05 ms of prediction, which flattened
  device, feature count and batch size alike. Reported as-is it would have put a tree ensemble
  at a tenth of the issuer's latency budget ([D-063](docs/decisions.md)).

[`docs/decisions.md`](docs/decisions.md) carries every entry including the retractions —
a memory-bandwidth witness retracted twice, a clustered-bootstrap width claim asserted from
theory and withdrawn when measured (0.89×, the opposite direction), and a `.gitignore`
pattern that silently excluded four source files from three pushed commits.

---

## 6. Reproduction

```
make venv         # python3.12, torch cu130 first (the PyPI build has no sm_120)
make smoke        # S0-S8 environment assertions; any failure stops the build
make reproduce    # every measurement target in order, then the SHA-256 manifest
make check        # rebuild PDFs, then the claim, citation and manifest gates
make walkthrough  # trace the certificate against the committed tables (seconds, no GPU)
```

**Two levels of check, and they answer different questions.** `make walkthrough` runs
[`notebooks/walkthrough.py`](notebooks/walkthrough.py), which recomputes the certificate chain
— blocks, certified configurations, held-out validation, the coverage arms, both quantum arms —
from `results/tables/` and asserts each step. It refits nothing, so it costs about a second and
answers *is what is reported internally consistent?* `make reproduce` refits everything and
answers *do the tables regenerate?* The full-scale tensor-network sweep is deliberately excluded
from `reproduce` at 13 GPU-hours; run it with `make seedsweep`.

The procedure for reproducing from an empty directory, and what it was measured to cost, is in
[`docs/CLEANROOM.md`](docs/CLEANROOM.md).

The pre-registration is enforced in code. [`scripts/check_protocol.py`](scripts/check_protocol.py)
hashes the **guarantee-bearing parameters** — every $\alpha$ grid, $\delta$, $\rho$, the
decision-grid size, the FWER method, the band budget and recall-floor grids, the split
fractions and the decline budget — and exits 1 on an undocumented change to any of them; exit
2 means changed *with* an amendment. It does not hash the protocol's prose, so a corrected
figure does not require an amendment. [`configs/default.yaml`](configs/default.yaml) is a
readable record generated from the committed defaults, and `--verify-dump` fails if it drifts.

The test-fold loader keeps a persisted ledger in
[`results/tables/test_access.json`](results/tables/test_access.json) — the authorised
configuration hash and how many times it has been requested — and raises on a second distinct
configuration. The single-evaluation rule is a mechanism, not a promise.

`scripts/run_mps.py` refuses to run the H4 comparison until `power.csv` exists, because a
power gate that can be computed afterwards is not a gate.

---

## 7. Repository map

### 7.1 Documents

| Path | Contents |
|---|---|
| [`docs/protocol.md`](docs/protocol.md) | Pre-registration, frozen before any model was fitted. Estimand, split, decision rule, band freezing, null hypotheses H1–H5, screens, out-of-scope claims, and its dated amendments |
| [`docs/decisions.md`](docs/decisions.md) | Every entry, including the retractions and the reason for each |
| [`docs/REFERENCES.md`](docs/REFERENCES.md) | 59 entries: conformal theory (CP-1…CP-17), fraud prior art (FR-1…FR-6), quantum ML evidence (QM-1…QM-15), datasets (DS-1…DS-5), regulation (RG-1…RG-6), software (SW-1…SW-10), and sources deliberately **not** relied upon. [`REFERENCE_CROSSCHECK.md`](docs/REFERENCE_CROSSCHECK.md) reports which are reached from the repository and which are not |
| [`NOTICE`](NOTICE) | Third-party licences, including why `cuquantum-cu11` is not installed by default |

### 7.2 Implementation

| Path | Role |
|---|---|
| [`src/hsbcfraud/conformal/riskcontrol.py`](src/hsbcfraud/conformal/riskcontrol.py) | Learn-then-Test, Hoeffding–Bentkus p-values, Holm, the half-open in-band grid |
| [`src/hsbcfraud/data/splits.py`](src/hsbcfraud/data/splits.py) | Four-block day-snapped split, three arms, `TestFoldGuard` |
| [`src/hsbcfraud/stats.py`](src/hsbcfraud/stats.py) | Card-clustered bootstrap, exact McNemar, TOST, the two MDE forms |
| [`src/hsbcfraud/quantum/screens.py`](src/hsbcfraud/quantum/screens.py) | Effective rank, RBF correlation, Huang geometric difference |
| [`src/hsbcfraud/quantum/kernel.py`](src/hsbcfraud/quantum/kernel.py) | Fidelity kernel across four backends, measured against an exact statevector reference |
| [`src/hsbcfraud/quantum/mps.py`](src/hsbcfraud/quantum/mps.py) | MPS classifier, depth-scaled learning rate, two documented silent-bug fixes |

### 7.3 Result tables

Every number quoted in prose resolves to a file in
[`results/tables/`](results/tables/), hashed in the manifest and checked by
`scripts/check_claims.py`. `tests/test_repo_hygiene.py` asserts that each one is tracked by
git, because a table present locally but untracked makes its claim unverifiable by a
reviewer.

---

## 8. Licence and data

Code is Apache-2.0 ([`LICENSE`](LICENSE)). Raw data is never committed: IEEE-CIS is
distributed under Kaggle competition rules rather than an open licence, and the ULB dataset
is two-layer — the **database** under the Open Database License (ODbL) and its **contents**
under the Database Contents License (DbCL) v1.0. Describing it as ODbL alone, as the challenge
statement does, is incomplete; [`NOTICE`](NOTICE) records both layers. Neither dataset is
redistributed here, and both are downloaded from Kaggle by the reader.
