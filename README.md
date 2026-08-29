# Certified abstention for card-fraud decisions, with two quantum arms reported as measured

Phase I submission, Global Quantum + AI Challenge 2026, HSBC track — *Quantum-Enhanced Credit
Card Fraud Detection for Digital Payment Ecosystems*.

The contribution is a **distribution-free certificate on a decision an issuer actually
makes**, and an honest measurement of where two quantum approaches sit relative to a tuned
classical baseline. Both quantum arms returned negative results. They are reported as such,
in the body, with the evidence that produced them.

---

## 1. What is claimed, and what is not

| | Claim | Status | Evidence |
|---|---|---|---|
| C1 | A three-valued decision rule (approve / step-up / decline) admits a finite-sample, distribution-free risk certificate on the **band-conditional** false-decline rate and the false-negative rate simultaneously | Supported at loose levels only: 5 of 48 configurations, none below $\alpha = 0.10$ | [`riskcontrol.csv`](results/tables/riskcontrol.csv), [protocol §6](docs/protocol.md) |
| C2 | Split-conformal coverage holds under a stratified and a card-disjoint split, and **fails** under a temporal split | Supported | [`coverage_by_arm.csv`](results/tables/coverage_by_arm.csv) |
| C3 | The size of the temporal breach is **origin-dependent**, not a fixed property of the data | Supported; this narrows C2 | [`rolling_origin.csv`](results/tables/rolling_origin.csv) |
| C4 | A fidelity quantum kernel is **rejected before being run** by two a-priori screens | Supported | [`screens.csv`](results/tables/screens.csv), [D-019](docs/decisions.md) |
| C5 | A matrix-product-state classifier does not beat a tuned GBDT in the band | Weaker than a null result: **underpowered** against the pre-registered ceiling (MDE 0.0461 against 0.023). What survives is a non-superiority bound of about +0.02 AP | [`mps_h4.csv`](results/tables/mps_h4.csv), [`power.csv`](results/tables/power.csv), [D-030](docs/decisions.md) |
| C6 | At **full scale** the same classifier loses by roughly a factor of two in average precision | Supported | [`mps_full.csv`](results/tables/mps_full.csv) |
| C7 | The bond-dimension sweep is **launch-bound**, not governed by $\chi$ | Supported; this is what makes C6 affordable | [`mps_full.csv`](results/tables/mps_full.csv), [D-032](docs/decisions.md) |

**Not claimed.** No quantum advantage of any kind. No portfolio-level PSD2 compliance
figure (see [amendment A2](docs/protocol.md)). No claim that the certificate binds the
unconditional false-decline rate — it does not, and §1 of the protocol says why.

---

## 2. The certificate

### 2.1 Estimand

The certified quantity is conditional on the band, not marginal:

$$R(\lambda) \;=\; \mathbb{P}\bigl(\,D(X) = \texttt{DECLINE} \;\bigm|\; Y = 0,\; X \in B\,\bigr)$$

where $B = \{x : \tau_{\mathrm{lo}} < f(x) < \tau_{\mathrm{hi}}\}$ is the abstention band and
$D$ the composite rule. The marginal rate $\mathbb{P}(D(X)=\texttt{DECLINE} \mid Y=0)$ is
reported alongside and is deliberately **not** the headline: it is dominated by
$\tau_{\mathrm{hi}}$ and would barely move if the in-band scorer were replaced by a coin
flip. A certificate that does not bind the component it licenses is not evidence about that
component.

### 2.2 Risk control

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

### 2.3 Coverage validation, and why the naive check is wrong

For a sound split-conformal system the number of test errors is not merely bounded in
expectation — it follows an exact predictive law:

$$E \;\sim\; \mathrm{BetaBinomial}\bigl(m,\; n + 1 - k,\; k\bigr), \qquad k = \lceil (n+1)(1-\alpha) \rceil$$

Checking $\hat{r} \le \alpha$ instead is a **one-sided test against the wrong null**: on a
correctly calibrated system it passes only 51.15 % of the time. Every coverage row in
[`coverage_by_arm.csv`](results/tables/coverage_by_arm.csv) is judged against the
Beta-Binomial interval, not against $\alpha$.

### 2.4 The reachability limit, stated up front

At the tightest band budget (2 % of traffic) $D_{\mathrm{cal}}$ contributes only 847
legitimate in-band rows; the four budgets on the grid give 847, 1572, 2259 and 4831. The
smallest risk level any Mondrian class-conditional quantile can represent is bounded below
by the degeneracy floor $\lceil 1/\alpha \rceil - 1$ (Ding, Angelopoulos, Bates, Jordan &
Tibshirani, NeurIPS 2023). **The band-conditional certificate is sample-starved by
construction**, and the reachable $\alpha$ is capped by the band budget rather than by the
method. This is [amendment A1](docs/protocol.md), recorded before the arm ran.

---

## 3. Results

### 3.1 What a random split buys you

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

### 3.2 Coverage by split arm

Nominal versus empirical false-decline rate on $D_{\mathrm{test}}$, calibrated on
$D_{\mathrm{cal}}$. `finite_sample_ok` is the Beta-Binomial verdict.

| Arm | $\\alpha$ | Ratio (mean) | sd | Range | Seeds outside the interval |
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
against $\\alpha$.

**The temporal arm breaches on every seed** at the two loosest levels and on four of five at
the third. **The stratified arm breaches on none, at any level.** That contrast is the result
the three-arm design was built to produce.

The card-disjoint arm is not clean, and the earlier version of this section was wrong to say
it was: it is outside on two of five seeds at $\\alpha = 0.01$ and one of five at
$\\alpha = 0.001$, where its spread (sd 0.362) is an order of magnitude wider than the
temporal arm's. Its 0.001 row is a single seed at ratio 1.525 against four between 0.65 and
0.76 — noise, not a systematic breach, but not a pass either.

Two further cautions. The temporal arm does **not** breach at the tightest level: at
$\\alpha = 0.001$ its mean ratio is 0.943 with every seed inside, and the Beta-Binomial band
there is wide enough to have little power. And "the breach is temporal" is stronger than three
non-randomised arms can carry — the arms differ in more than time, and by a two-sample
classifier the card-disjoint arm is the *most* distinguishable of the three (0.6554 against
the temporal arm's 0.5525). What the data supports is that the breach appears only in the arm
ordered by time, on every seed, and that neither control reproduces it.

### 3.3 What the certificate actually reaches

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

### 3.4 The breach is origin-dependent

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

### 3.5 Quantum kernel: rejected by the screens, before it ran

Two a-priori gates over 120 configurations (encoding × qubits × bandwidth × entanglement):

$$\text{effective rank } r_{\mathrm{eff}} = \frac{\left(\sum_i \sigma_i\right)^2}{n \sum_i \sigma_i^2}, \qquad \rho_{\mathrm{RBF}} = \mathrm{corr}\bigl(K_Q,\, K_{\mathrm{RBF}}(\gamma^\star)\bigr)$$

A kernel is usable only if it is neither exponentially concentrated ($r_{\mathrm{eff}}$
inside a usable band) **and** not reproducible by a tuned RBF ($\rho_{\mathrm{RBF}} <
0.60$). Result: **28 of 120 pass conditioning, 0 pass distinctness, 0 pass both.** The
closest any configuration came was $\rho_{\mathrm{RBF}} = 0.6291$ against a 0.60 threshold.

The kernel arm was therefore not run on the decision task. Reporting a screen that rejects
its own headline method is the point of pre-registering it.

### 3.6 Tensor network: it ran, and it lost

The matrix-product-state classifier (Stoudenmire & Schwab, NeurIPS 29:4799, 2016) contracts

$$f(\mathbf{x}) \;=\; \sum_{\{s\}} A^{s_1}_{\alpha_1} A^{s_2}_{\alpha_1\alpha_2} \cdots A^{s_N}_{\alpha_{N-1}} \prod_{j=1}^{N} \phi^{s_j}(x_j), \qquad \phi(x) = \bigl[\cos\tfrac{\pi x}{2},\ \sin\tfrac{\pi x}{2}\bigr]$$

with bond dimension $\chi$ swept rather than tuned. In-band, on identical rows and features,
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
the ordering here is 0.1202, 0.1173, 0.1183, 0.1235, and §3.7 shows why no ordering should be
read from a single seed per configuration.

Every interval contains zero, so the correct reading is **not** "the MPS is worse" — it is
that the comparison cannot separate them.

**And the power gate that was supposed to tell us this in advance was itself wrong.**
`power.csv` initially recorded an MDE of 0.0069 against the 0.023 ceiling, estimated from the
seed-to-seed spread of the baseline against itself. But the comparison H4 makes is between
*model families*, whose measured standard error turned out to be 0.018–0.020 — about
eightfold larger. The true MDE is near 0.056, which **exceeds** the pre-registered ceiling.
H4 is therefore reported as **underpowered**, exactly as the protocol commits to doing when
this happens. The gate now computes a cross-family proxy (logistic against GBDT, still no
quantum model) and binds on the larger of the two. See [D-030](docs/decisions.md).

What survives is narrower than a null result and is stated as such: **any improvement from
the tensor network is bounded above by roughly +0.02 AP** at 95 % confidence. That is a
non-superiority bound. It is not evidence of equivalence, and it is not evidence that the
tensor network is worse.

### 3.7 At full scale, the margin is not close

All 431 features, trained on $D_{\mathrm{train}}$ and scored once on $D_{\mathrm{test}}$:

| $\chi$ | ROC AUC | AP | Final loss | Fit (s) |
|---|---|---|---|---|
| 4 | 0.8074 | 0.2149 | 0.3456 | 2765 |
| 8 | 0.7989 | 0.1789 | 0.3075 | 3071 |
| 16 | 0.7983 | 0.2464 | 0.3120 | 2840 |
| 32 | 0.7889 | 0.1693 | 0.3652 | 2898 |
| **tuned GBDT** | **0.8837–0.8884** | **0.5055–0.5114** | — | 16 |

Two things to read here. The tensor network is not competitive — a factor of two in average
precision, at a fraction of the baseline's speed. And **the fit time is flat across $\chi$**,
against the sixty-four-fold spread a $\chi^2$ cost model predicts. A 431-site chain is bound
by the launch overhead of its sequential contractions, not by their arithmetic; the competing
predictions were written down before the measurement ([D-032](docs/decisions.md)). Capacity
is nearly free here and depth is the cost, which is the opposite of the intuition carried
over from short chains.

---

## 4. Negative results, and why they are in the body

Three findings in this repository contradict what the work set out to show, and one
contradicts a claim an earlier draft had already written down.

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

[`docs/decisions.md`](docs/decisions.md) carries all 38 entries including the retractions —
a memory-bandwidth witness retracted twice, a clustered-bootstrap width claim asserted from
theory and withdrawn when measured (0.89×, the opposite direction), and a `.gitignore`
pattern that silently excluded four source files from three pushed commits.

---

## 5. Reproduction

```
make venv        # python3.12, torch cu130 first (the PyPI build has no sm_120)
make smoke       # S0-S8 environment assertions; any failure stops the build
make reproduce   # every measurement target in order, then the SHA-256 manifest
make check       # rebuild PDFs, then the claim, citation and manifest gates
```

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

## 6. Deliverables

### 6.1 Documents

| Path | Contents |
|---|---|
| [`docs/protocol.md`](docs/protocol.md) | Pre-registration, frozen before any model was fitted. Estimand, split, decision rule, band freezing, null hypotheses H1–H5, screens, out-of-scope claims, and four dated amendments (A1–A4) |
| [`docs/decisions.md`](docs/decisions.md) | Every entry, including the retractions and the reason for each |
| [`docs/REFERENCES.md`](docs/REFERENCES.md) | Conformal theory (CP-1…CP-13), fraud prior art (FR-1…FR-6), quantum ML evidence (QM-1…QM-10), datasets (DS-1…DS-5), regulation (RG-1…RG-6), software (SW-1…SW-9), and sources deliberately **not** relied upon |
| [`NOTICE`](NOTICE) | Third-party licences, including why `cuquantum-cu11` is not installed by default |

### 6.2 Implementation

| Path | Role |
|---|---|
| [`src/hsbcfraud/conformal/riskcontrol.py`](src/hsbcfraud/conformal/riskcontrol.py) | Learn-then-Test, Hoeffding–Bentkus p-values, Holm, the half-open in-band grid |
| [`src/hsbcfraud/data/splits.py`](src/hsbcfraud/data/splits.py) | Four-block day-snapped split, three arms, `TestFoldGuard` |
| [`src/hsbcfraud/stats.py`](src/hsbcfraud/stats.py) | Card-clustered bootstrap, exact McNemar, TOST, the two MDE forms |
| [`src/hsbcfraud/quantum/screens.py`](src/hsbcfraud/quantum/screens.py) | Effective rank, RBF correlation, Huang geometric difference |
| [`src/hsbcfraud/quantum/kernel.py`](src/hsbcfraud/quantum/kernel.py) | Fidelity kernel across four backends; Braket parity to 3.9e-16 |
| [`src/hsbcfraud/quantum/mps.py`](src/hsbcfraud/quantum/mps.py) | MPS classifier, depth-scaled learning rate, two documented silent-bug fixes |

### 6.3 Result tables

Every number quoted in prose resolves to a file in
[`results/tables/`](results/tables/), hashed in the manifest and checked by
`scripts/check_claims.py`. `tests/test_repo_hygiene.py` asserts that each one is tracked by
git, because a table present locally but untracked makes its claim unverifiable by a
reviewer.

---

## 7. Licence and data

Code is Apache-2.0 ([`LICENSE`](LICENSE)). Raw data is never committed: IEEE-CIS is
distributed under Kaggle competition rules rather than an open licence, and the ULB database
carries ODbL share-alike terms on the database itself. `scripts/fetch_data.py` reconstructs
both from a token.
