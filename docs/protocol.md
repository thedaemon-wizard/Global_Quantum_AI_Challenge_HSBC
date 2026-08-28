# Pre-registration protocol

**Frozen 2026-08-28, before any model was fitted and before any block other than the file
header was read.** `scripts/check_protocol.py` hashes this file together with
`configs/default.yaml` and refuses to run an experiment if either has changed since the
freeze without a dated entry in `docs/decisions.md` recording the change and its reason.

This document exists because two of the guarantees claimed in this work are only valid if
their parameters were chosen without reference to the data.

* **Learn-then-Test** (Angelopoulos, Bates, Candès, Jordan & Lei, *Annals of Applied
  Statistics* 19(2):1641–1662, 2025) controls the family-wise error rate over a **finite,
  pre-specified** grid of decision parameters. Enlarging the grid after seeing which value
  looks good inflates the error rate the procedure is supposed to bound.
* **Non-exchangeable split conformal** (Barber, Candès, Ramdas & Tibshirani, *Annals of
  Statistics* 51(2):816–845, 2023, eq. 11) requires the weights `w_i` to be **fixed and
  not data-dependent**. Fitting the decay `rho` on the calibration set voids the theorem.

Everything below is therefore a commitment, not a description.

---

## 1. Estimand

The quantity certified is the **false-decline rate on the legitimate class, conditional on
the transaction being routed into the abstention band**:

> R(λ) = P( D(X) = DECLINE | Y = 0, X ∈ B )

where `D` is the composite decision rule of §3, `B` is the frozen band of §4, and `Y = 0`
denotes a transaction not linked to a chargeback reported within the observation window.

The unconditional rate P(D(X) = DECLINE | Y = 0) is reported alongside it and is **not**
the headline. The reason is stated plainly because it is the objection a careful reviewer
raises first: the unconditional rate is dominated by the upper threshold `τ_hi` and would
be almost unchanged if the in-band scorer were replaced by a coin flip. A certificate that
does not bind the component it is said to license is not evidence about that component.

### 1.1 What the estimand is conditional on, and what that excludes

Three conditions are carried explicitly rather than left for a reader to discover.

1. **Incumbent filtering.** IEEE-CIS and ULB contain only transactions an existing
   authorisation stack already approved. Declined transactions are structurally absent and
   can never generate a chargeback. The estimand is therefore defined on the
   approved-and-settled population, conditional on the incumbent's decline policy. This is
   exactly the population on which an issuer's champion/challenger comparison runs, so the
   restriction is the right one — but it is a restriction.
2. **Label construction.** The IEEE-CIS `isFraud` flag is set on a reported chargeback and
   then **propagated to subsequent transactions sharing a user account, email address or
   billing address**; a transaction is labelled 0 only if nothing is reported within 120
   days (competition host, Kaggle discussion 101203). Many `Y = 1` rows are therefore not
   fraudulent transactions but transactions linked to a card that later had one. "Fraud
   detection" on this file is partly entity-contamination detection, and no claim here
   asserts otherwise.
3. **Row-level exchangeability does not hold.** The propagation rule clusters rows by
   entity. Measured on the file: 84.8 % of the `card1` values present in the test block
   also appear in the training block, with a median of 4 and a 95th percentile of 107
   transactions per value. All confidence intervals are therefore card-level block
   bootstrap, and the guarantee is stated at block level.

---

## 2. Data and split

**Primary:** IEEE-CIS `train_transaction.csv`, 590,540 rows, 20,663 frauds (3.4990 %), 394
columns, spanning exactly 182.00 days, `TransactionDT` monotonically increasing.

**Secondary:** ULB European Cardholder, 284,807 rows, 492 frauds, after removing the 1,081
exact duplicate rows **before** splitting.

Sparkov is not used in Phase I.

### 2.1 Blocks

Ordered by `TransactionDT`, cut at day boundaries (86,400-second buckets counted from the
file minimum), contiguous, no overlap:

| Block | Fraction | Role |
|---|---|---|
| `D_train` | 0.60 | Fits the classical scorer. Never used for calibration. |
| `D_band` | 0.10 | The **only** data that may determine the band edges, the in-band scorer's hyperparameters, the kernel bandwidth, or the screen thresholds. Frozen and hashed at the end of this stage. |
| `D_cal` | 0.10 | Certifies the **composite** rule under Learn-then-Test. |
| `D_test` | 0.20 | Reported once. Full imbalanced fold. Never subsampled, never resampled. |

Two control arms run on the same proportions: a **stratified random** split, and a
**card-disjoint** split. Their purpose is to decompose any degradation into a temporal
component and a sampling component, rather than attributing all of it to time.

### 2.2 The single-evaluation rule

`D_test` is evaluated **once**, for the single pre-registered configuration selected on
`D_band` and certified on `D_cal`. Every sweep, ladder and ablation runs on held-out slices
of `D_band` or `D_cal`.

This is enforced in code, not by discipline: the test-fold loader increments a counter
persisted to `results/tables/test_access.json` and raises on the second distinct
configuration. A reviewer can read the counter.

---

## 3. Decision rule

Three-valued, matching the actions an issuer actually has:

```
f(x) >= tau_hi          ->  DECLINE
f(x) <= tau_lo          ->  APPROVE without step-up
tau_lo < f(x) < tau_hi  ->  STEP-UP (SCA challenge); the in-band scorer g re-ranks,
                            and g(x) >= lambda decides
```

The abstain tier is **step-up authentication**, not manual review. Routing even one percent
of an issuer's authorisation volume to human analysts is not operationally possible; a 3-D
Secure challenge scales. This choice is what makes the band's traffic budget a realistic
control rather than an accounting fiction, and §6 reports the budget in challenges per day
as well as as a fraction.

---

## 4. The band, and why freezing it is what makes the guarantee valid

`B = { x : tau_lo < f(x) < tau_hi }`.

The edges are chosen on `D_band` **only**, by the rule fixed here: take the traffic budget
`beta` from `configs/default.yaml` and place the interval symmetrically in score quantiles
around the operating threshold of §5. The band is defined by **fixed quantiles**, not fixed
score cutoffs, so the routed volume — and therefore the step-up cost — stays constant under
drift while the risk content is allowed to move and is measured.

There are two distinct breaks of exchangeability in this architecture, and conflating them
is the error this section exists to prevent.

* **Break 1, temporal.** The calibration block precedes the test block in time.
* **Break 2, selective.** If the band edges were estimated from the same data used to
  certify, conditioning on band membership would be conditioning on a data-dependent event.

Break 2 is removed **by construction**: once frozen on `D_band`, membership in `B` is a
fixed measurable predicate, and filtering a sequence by a fixed predicate preserves
exchangeability of the retained subsequence. Break 1 remains and is treated in §7.

A consequence that must be stated rather than hidden: the guarantee now lives on the
band-conditional law, and drift inside a narrow score band straddling the decision boundary
is systematically worse than marginal drift. §7 measures both.

---

## 5. Operating point

The threshold is **not** chosen by Youden's J. It is chosen by a Neyman–Pearson rule:

> maximise recall subject to the **value-weighted**, rolling-90-day fraud rate on the
> approved-without-step-up branch remaining at or below the PSD2 SCA-RTS Annex reference
> rate for the target exemption-threshold tier.

Reference rates for remote card-based payments (Commission Delegated Regulation (EU)
2018/389, Annex): **0.13 % at ETV EUR 100, 0.06 % at EUR 250, 0.01 % at EUR 500.**

Three properties of that regime are load-bearing and are stated in the proposal because
getting them wrong is a common and visible error:

1. Article 18 governs eligibility to **exempt a transaction from strong customer
   authentication**. It does not regulate the authorisation decline threshold.
2. Article 19 defines the rate as **total value of fraudulent remote transactions divided
   by total value of all remote transactions**, i.e. value-weighted, not count-weighted,
   computed on a **rolling 90-day** basis.
3. The rate is computed over the payment service provider's remote card-based portfolio,
   combining authenticated and exempted transactions.

All fraud rates in this work are therefore reported **both** count-weighted and
value-weighted, with the value-weighted figure carrying the regulatory comparison.

---

## 6. Certified risks and the parameter grid

Both sides of the envelope are certified. Certifying only the false-decline side would
leave uncertified the side PSD2 caps and the side the loss reserve is held against.

| Item | Commitment |
|---|---|
| Risks | `false_positive_rate` (= band-conditional false-decline rate) **and** `recall` |
| Procedure | Learn-then-Test with Hoeffding–Bentkus p-values, FWER controlled by Holm–Bonferroni across the joint grid |
| Guarantee form | PAC: `P( R(lambda_hat) <= alpha ) >= 1 - delta` |
| `alpha` grid | {1e-2, 5e-3, 2e-3, 1e-3} |
| `delta` | 0.05 |
| Recall floor | 0.60 |
| Band traffic budget grid | {0.005, 0.01, 0.02, 0.05}, one value carried to test |
| Coverage band level | 0.99, exact Beta-Binomial |

The `alpha` grid is an order of magnitude tighter than a textbook conformal sweep. At
`alpha = 0.05` the certificate would say that up to five percent of legitimate customers
may be declined, which is not an operating point any issuer runs.

**Conformal Risk Control** (Angelopoulos et al., ICLR 2024) bounds the *expected* risk and
is reported alongside as the tighter-but-weaker alternative. The two are never used
interchangeably, and the proposal states which is which.

### 6.1 Class-conditional degeneracy

Mondrian class-conditional calibration degenerates when a class has fewer than
`(1/alpha) - 1` calibration points: the quantile becomes `+inf` and the class enters every
prediction set regardless of score (Ding, Angelopoulos, Bates, Jordan & Tibshirani, NeurIPS
2023). Measured counts in `D_cal` at the 60/10/10/20 split:

| Class | Count in `D_cal` | Smallest admissible `alpha` |
|---|---|---|
| legitimate | 56,993 | 1.8e-5 |
| fraud | 2,061 | 4.9e-4 |

The whole `alpha` grid clears both. The guarantee lives on the legitimate class, where
degeneracy never binds — a design property, not luck. On ULB the fraud-conditional quantile
is degenerate at `alpha = 1e-2` (roughly 98 calibration frauds against the 99 required),
which is why IEEE-CIS is primary and ULB is the stress case.

### 6.2 Minimum detectable effect — computed before the arm runs

The power calculation is a **gate**, not a report. With band size `n_band` and the
per-observation variance of average precision estimated on `D_band`, the minimum detectable
difference at 80 % power and a two-sided 0.05 level is computed and written to
`results/tables/power.csv` **before** the quantum comparison executes.

Pre-committed decision rule: **if the minimum detectable effect exceeds 0.023 average
precision — the point estimate reported by Chaves et al. (arXiv:2603.06473) for a
comparable architecture, whose own standard deviation is 0.09 — the comparison is declared
underpowered in advance and reported as such.** An equivalence test whose margin is wider
than the effect it is testing for is not evidence of equivalence.

---

## 7. Non-exchangeability: what is claimed, and what is measured

**No numeric `alpha + eta` is promised.** This is a deliberate refusal, and the reason is
that such a number cannot be produced honestly:

* the additive penalty `eta` of Oliveira, Orenstein, Ramos & Romano (*JMLR* 25:1–38, 2024)
  depends on the beta-mixing coefficients of the transaction stream, which are not
  estimable from the data;
* the drift parameter `epsilon` in Barber et al.'s closed forms is unobservable, and
  estimating it from the calibration set would make the weights data-dependent, which the
  theorem forbids.

A model risk committee cannot accept a control whose stated level is "alpha plus an
unquantified correction". So the structure of the claim is:

1. **certify** at `alpha` under exchangeability, stating the assumption;
2. **measure** the realised risk gap across successive time windows, and report that
   measurement as a headline result rather than a caveat;
3. **cite** the theory for why the degradation is bounded in form — Oliveira et al. for
   plain split conformal under beta-mixing, including their Appendix D extension to
   risk-controlling prediction sets, and Barber et al. eq. (11) as the weighted robustness
   variant — without presenting either bound as an evaluated number.

`rho = 0.999` is pinned here for the weighted variant. Note for honesty in the write-up:
Barber et al.'s two closed forms both route through their Lemma 1, which assumes the
observations are independent. A serially correlated transaction stream falls under their
separate covariate-time-series treatment, so the `rho^k` changepoint bound is **not** led
with.

Measured, not assumed:

* conformal test martingale over the `TransactionDT`-ordered stream;
* block permutation test;
* realised risk gap per time window;
* **band-conditional drift against marginal drift**, since §4 makes the former the relevant
  quantity;
* temporal arm against the stratified-random control arm.

---

## 8. Null hypotheses

Stated in advance, in the direction that makes them refutable.

| ID | Null hypothesis | Test | Decision |
|---|---|---|---|
| H1 | The quantum kernel does not improve band-conditional average precision over the bandwidth-matched RBF control | paired card-level block bootstrap on AP, two-sided | reject at p < 0.05 after Holm correction across the kernel family |
| H2 | The quantum kernel's decisions do not differ from the RBF control's at the certified operating point | exact McNemar on paired decisions | reject at p < 0.05 after Holm |
| H3 | Removing entanglement from the feature map does not change band-conditional AP | paired block bootstrap | reject at p < 0.05 |
| H4 | The tensor-network classifier does not improve band-conditional AP over the tuned GBDT | paired block bootstrap | reject at p < 0.05 |
| H5 | Realised risk on `D_test` does not exceed the certified `alpha` | exact Beta-Binomial tail probability | reject at p < 0.01 |

**H1–H4 are expected to survive.** The 2026 literature is consistent on this point, and the
result of this work does not depend on refuting them. What would make the work wrong is
claiming a rejection that the statistics do not support.

Test statistics are matched to estimands rather than applied uniformly: average precision is
a threshold-free ranking quantity and is compared by paired bootstrap; decisions at a fixed
threshold are compared by exact McNemar. Comparing two kernels each at its own operating
point produces discordance tables that are not comparable, so all McNemar comparisons are
made at a common threshold.

---

## 9. Screens, and the stopping rule

The quantum arm runs only if its feature map passes **both** a-priori screens on `D_band`:

1. **Geometric difference** `g(K_classical || K_quantum)` (Huang et al., *Nature
   Communications* 12:2631, 2021) above threshold;
2. **Eigenspectrum effective rank** within the band a classical RBF kernel occupies on the
   same data — outside it the Gram matrix is either near-rank-one or near-uniform;

and a third diagnostic recorded jointly with the second:

3. **Effective-rank ratio against best-fit-RBF correlation**, a two-dimensional criterion. A
   feature map has to be simultaneously well-conditioned and distinct from an RBF kernel to
   be worth evaluating. In a preliminary measurement on isotropic uniform data, no
   bandwidth satisfied both at once: at natural bandwidth the eight-qubit ZZ map has
   effective-rank ratio 0.98 and correlation 0.03 with any RBF, while tightening the
   bandwidth to reach the RBF-like region drove the correlation monotonically to 0.71. That
   measurement establishes that the instrument fires; whether real fraud features behave
   the same way is the question the screen answers on `D_band`.

**If the screens reject every candidate, the quantum arm is not run and the rejection is the
reported result.** The challenge statement asks participants to characterise the conditions
under which quantum approaches perform differently; a screened negative is a direct answer
to that question, not a failure to answer it.

---

## 10. Claims fixed in advance as out of scope

Recorded here so that no later result can be presented as though it had been sought.

* No claim of quantum accuracy advantage on AUC-ROC or AUPRC.
* No claim of computational speed-up. A fidelity kernel needs O(n²) circuit evaluations;
  the measured cost on this machine is 19.5 µs per pair at 8 qubits.
* No unqualified use of the phrase "distribution-free finite-sample guarantee" without the
  exchangeability condition and the measured gap attached.
* No claim that class-conditional conformal prediction with abstention is novel in fraud
  detection. It is not; §1.2 of the proposal names the prior work.
* No claim of EU AI Act high-risk compliance. Annex III point 5(b) explicitly excludes AI
  systems used for detecting financial fraud.
* No merchant-side false-decline economics presented as issuer benefit.
* No number from the author's prior intrusion-detection work presented as a forecast for
  card fraud.
