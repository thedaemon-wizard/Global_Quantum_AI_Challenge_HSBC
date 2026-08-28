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
| C5 | A matrix-product-state classifier does not beat a tuned GBDT in the band | Weaker than a null result: **underpowered** against the pre-registered ceiling. What survives is a non-superiority bound of about +0.02 AP | [`mps_h4.csv`](results/tables/mps_h4.csv), [`power.csv`](results/tables/power.csv), [D-030](docs/decisions.md) |

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

### 3.1 Coverage by split arm

Nominal versus empirical false-decline rate on $D_{\mathrm{test}}$, calibrated on
$D_{\mathrm{cal}}$. `finite_sample_ok` is the Beta-Binomial verdict.

| Arm | $\alpha$ | Empirical | Ratio | Errors | Beta-Binomial band | Verdict |
|---|---|---|---|---|---|---|
| temporal | 0.010 | 0.01494 | 1.494 | 1667 | [973, 1266] | **breached** |
| temporal | 0.005 | 0.00746 | 1.493 | 833 | [458, 665] | **breached** |
| temporal | 0.002 | 0.00264 | 1.322 | 295 | [161, 292] | **breached** |
| temporal | 0.001 | 0.00098 | 0.977 | 109 | [69, 162] | inside |
| stratified | 0.010 | 0.00952 | 0.952 | 1085 | [993, 1292] | inside |
| stratified | 0.001 | 0.00116 | 1.158 | 132 | [70, 164] | inside |
| card-disjoint | 0.010 | 0.00896 | 0.896 | 990 | [960, 1256] | inside |
| card-disjoint | 0.001 | 0.00076 | 0.760 | 84 | [67, 161] | inside |

The stratified and card-disjoint arms sit inside the band at **all four** $\alpha$; the two
rows each shown here are the endpoints of that grid. The temporal arm breaches at three of
four — and **not** at the tightest level, $\alpha = 0.001$, where it lands at 0.977. That
exception is kept in the table rather than dropped: an arm that fails at loose levels and
passes at the tightest is not behaving like a simple upward bias, and at $\alpha = 0.001$
the Beta-Binomial band is wide enough (69–162 errors) that it has little power to detect one.

The breach is therefore **temporal** rather than a sampling artefact — which is the
decomposition the three-arm design exists to produce — but its size is not constant across
$\alpha$, and §3.3 shows it is not constant across calibration origins either.

### 3.2 What the certificate actually reaches

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

### 3.3 The breach is origin-dependent

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

### 3.4 Quantum kernel: rejected by the screens, before it ran

Two a-priori gates over 120 configurations (encoding × qubits × bandwidth × entanglement):

$$\text{effective rank } r_{\mathrm{eff}} = \frac{\left(\sum_i \sigma_i\right)^2}{n \sum_i \sigma_i^2}, \qquad \rho_{\mathrm{RBF}} = \mathrm{corr}\bigl(K_Q,\, K_{\mathrm{RBF}}(\gamma^\star)\bigr)$$

A kernel is usable only if it is neither exponentially concentrated ($r_{\mathrm{eff}}$
inside a usable band) **and** not reproducible by a tuned RBF ($\rho_{\mathrm{RBF}} <
0.60$). Result: **28 of 120 pass conditioning, 0 pass distinctness, 0 pass both.** The
closest any configuration came was $\rho_{\mathrm{RBF}} = 0.6291$ against a 0.60 threshold.

The kernel arm was therefore not run on the decision task. Reporting a screen that rejects
its own headline method is the point of pre-registering it.

### 3.5 Tensor network: it ran, and it lost

The matrix-product-state classifier (Stoudenmire & Schwab, NeurIPS 29:4799, 2016) contracts

$$f(\mathbf{x}) \;=\; \sum_{\{s\}} A^{s_1}_{\alpha_1} A^{s_2}_{\alpha_1\alpha_2} \cdots A^{s_N}_{\alpha_{N-1}} \prod_{j=1}^{N} \phi^{s_j}(x_j), \qquad \phi(x) = \bigl[\cos\tfrac{\pi x}{2},\ \sin\tfrac{\pi x}{2}\bigr]$$

with bond dimension $\chi$ swept rather than tuned. In-band, on identical rows and features,
against a tuned GBDT — see [`mps_band.csv`](results/tables/mps_band.csv) and
[`mps_h4.csv`](results/tables/mps_h4.csv).

| $\chi$ | AP (MPS) | AP (GBDT) | $\Delta$AP | 95 % clustered CI | $p$ | Holm |
|---|---|---|---|---|---|---|
| 4 | 0.1185 | 0.1407 | −0.0222 | [−0.0597, +0.0136] | 0.888 | not rejected |
| 8 | 0.1200 | 0.1407 | −0.0207 | [−0.0599, +0.0147] | 0.860 | not rejected |
| 16 | 0.1215 | 0.1407 | −0.0192 | [−0.0614, +0.0159] | 0.835 | not rejected |
| 32 | 0.1228 | 0.1407 | −0.0179 | [−0.0574, +0.0222] | 0.810 | not rejected |

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

[`docs/decisions.md`](docs/decisions.md) carries all 29 entries including the retractions —
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

The test-fold loader persists an access counter to
`results/tables/test_access.json` and raises on a second distinct configuration — the
single-evaluation rule is a mechanism, not a promise.

`scripts/run_mps.py` refuses to run the H4 comparison until `power.csv` exists, because a
power gate that can be computed afterwards is not a gate.

---

## 6. Deliverables

### 6.1 Documents

| Path | Contents |
|---|---|
| [`docs/protocol.md`](docs/protocol.md) | Pre-registration, frozen before any model was fitted. Estimand, split, decision rule, band freezing, null hypotheses H1–H5, screens, out-of-scope claims, and four dated amendments (A1–A4) |
| [`docs/decisions.md`](docs/decisions.md) | D-001…D-029, including every retraction and the reason for it |
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
