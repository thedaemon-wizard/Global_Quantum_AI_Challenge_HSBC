# Pre-registration protocol

**Frozen 2026-08-28, before any model was fitted and before any block other than the file
header was read.** `scripts/check_protocol.py` hashes the **guarantee-bearing fields of
`configs/default.yaml`** — every alpha grid, `delta`, `rho`, the decision-grid size, the FWER
method, the band traffic-budget and recall-floor grids, the split fractions and the decline
budget — and refuses to proceed if any of them changed since the freeze without a matching
amendment below and a corresponding entry in `docs/decisions.md`.

It does **not** hash this file's prose. An earlier version of this paragraph said it hashed
"this file together with `configs/default.yaml`", which was written before the script existed
and was never true of it. Prose here can be corrected — a mis-transcribed count, a stale
percentage — without tripping a gate whose purpose is to catch movement in the quantities a
theorem depends on. Fields that enter no guarantee (output paths, model-fitting seeds,
plotting choices) are deliberately outside the hash: a gate annoying enough to be bypassed
protects nothing.

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
   entity. Measured on the file: 85.0 % of the `card1` values present in the test block
   also appear in the training block, with a median of 4 and a 95th percentile of 107
   transactions per value. All confidence intervals are therefore card-level block
   bootstrap, and the guarantee is stated at block level.

---

## 2. Data and split

**Primary:** IEEE-CIS `train_transaction.csv`, 590,540 rows, 20,663 frauds (3.4990 %), 394
columns, spanning exactly 182.00 days, `TransactionDT` monotonically increasing.

**Secondary:** ULB European Cardholder, 284,807 rows, 492 frauds, after removing the 1,081
exact duplicate rows **before** splitting. *Withdrawn by amendment A9: the dataset was never
obtained and nothing here was run on it. These counts are transcribed from its published
description, not measured.*

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

`D_test` is evaluated **once for selection**: exactly one pre-registered configuration is
chosen on `D_band`, certified on `D_cal`, and carried to `D_test` unchanged. Nothing is tuned,
ranked or chosen using the test block.

*Amended 2026-08-30 (A8).* This paragraph previously read "Every sweep, ladder and ablation
runs on held-out slices of `D_band` or `D_cal`", which the code does not do. See A8 for exactly
which scripts read the test block and why the guarantee is unaffected.

This is enforced in code, not by discipline: the test-fold loader keeps a persisted ledger in
`results/tables/test_access.json` recording the authorised configuration hash and how many
times it has been requested, and raises on a second **distinct** configuration. A reviewer can
read the ledger.

An earlier version of this paragraph, and of the appendix, said the loader "increments a
counter" that "a reviewer can read". It did not: it stored the hash alone, so the file said
nothing about how often the fold had been touched. The count was added rather than the claim
withdrawn, because the count is what a reader actually wants.

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
2023). Measured counts in `D_cal` at the 60/10/10/20 split, temporal arm, which is the
primary arm and the one `results/tables/degeneracy.csv` reports:

| Class | Count in `D_cal` | Smallest admissible `alpha` |
|---|---|---|
| legitimate | 58,343 | 1.7e-5 |
| fraud | 2,121 | 4.7e-4 |

An earlier revision of this table read 56,993 and 2,061. Those figures came from the
stratified arm of a superseded split and were never the temporal ones; the counts above are
regenerated from `results/tables/splits.csv` and agree with `degeneracy.csv` row for row.

The whole `alpha` grid clears both. The guarantee lives on the legitimate class, where
degeneracy never binds — a design property, not luck.

*Amended by A9.* This paragraph continued: "On ULB the fraud-conditional quantile is degenerate
at `alpha = 1e-2` (roughly 98 calibration frauds against the 99 required), which is why
IEEE-CIS is primary and ULB is the stress case." That was never measured, and its arithmetic
does not match this protocol's own split: 98 is 20 % of ULB's 492 frauds, which is the **test**
fraction, while `split.cal` is 10 % and yields roughly 49. The direction of the conclusion is
unaffected — 49 against a floor of 99 is more degenerate than 98, not less — but the number was
wrong and unverified, so it is withdrawn rather than corrected in place.

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

**Disposition, added 2026-08-30.** Three of the five were never tested, and this table read as
though they had been.

* **H1, H2 and H3 were not tested, and cannot be.** All three compare a quantum kernel against
  a control, and section 9's stopping rule fired first: of 120 pre-registered configurations,
  28 passed conditioning and **none** passed distinctness, so no kernel was carried to the
  band. There is no kernel to test the nulls against. This is the stopping rule working as
  designed, not an omission -- but "not tested because the arm stopped" and "tested and not
  rejected" are different statements, and only the record distinguishes them.
* **H4 was tested and is reported as underpowered rather than as a null**, which is what the
  protocol commits to when the minimum detectable effect exceeds its ceiling.
* **H5 was tested and holds** on the held-out block, for all five certified configurations.

No hypothesis was tested and then left unreported, and no hypothesis is claimed as a null that
was not tested.

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
  detection. It is not; section 1 of the proposal names the prior work (FR-1 and FR-3) and
  states what is different here.
* No claim of EU AI Act high-risk compliance. Annex III point 5(b) explicitly excludes AI
  systems used for detecting financial fraud.
* No merchant-side false-decline economics presented as issuer benefit.
* No number from the author's prior intrusion-detection work presented as a forecast for
  card fraud.

---

## Amendment A1 — 2026-08-28, before any calibration risk was observed

**What changed:** the band traffic-budget grid becomes {0.02, 0.035, 0.05, 0.10}, the
decision grid is fixed at 11 points, and the certified `alpha` is reported per achievable
(budget, alpha) pair rather than as a single level across the whole grid.

**Why this does not compromise the guarantee.** The change was derived from *sample sizes
and the concentration bound alone*. No risk value from `D_cal` was computed, inspected or
used. The quantity that drove it — how many legitimate transactions fall in a band of a
given width — is a property of the split, which was fixed in E1, and of the Hoeffding-Bentkus
bound, which is a formula. Learn-then-Test requires the grid to be fixed before the risks are
evaluated; it does not require the grid to be chosen without arithmetic.

**What was found.** The band-conditional risk of section 1 is conditioned on two events at
once — legitimate, and inside the band — and the band is deliberately a few percent of
traffic. At a 10 % calibration block (58,343 legitimate rows measured) the reachable `alpha`
is capped by the band budget:

| target `alpha` | minimum band budget that certifies |
|---|---|
| 1e-2 | 0.035 |
| 5e-3 | 0.070 |
| 2e-3 | 0.175 |
| 1e-3 | 0.345 |

computed at an assumed true risk of `alpha/3`, with Holm correction over an 11-point grid
and two risks. The original grid {0.005, 0.01, 0.02, 0.05} therefore could not have
certified anything below `alpha = 1e-2`, and only at its widest point.

The **unconditional** false-decline rate, which uses the whole calibration block, reaches
`alpha = 5e-4` comfortably (p = 1.7e-4 against a Holm level of 2.3e-3).

**Consequences, all reported rather than hidden.**

1. Two certificates are reported side by side, with their different characters stated: the
   unconditional one is tight but does not constrain the in-band scorer; the band-conditional
   one constrains it but is capped by sample size. Neither alone is the whole story.
2. The reachable (budget, alpha) frontier is a **result**, not a configuration choice. It is
   the guarantee-versus-abstention trade-off that section 6 promised, derived from the
   concentration bound rather than asserted.
3. It settles a design question that was previously argued on operational grounds alone.
   A certificate at `alpha = 1e-2` requires routing at least 3.5 % of traffic into the band.
   Sending 3.5 % of an issuer's authorisation volume to **manual review** is not possible;
   sending it to **3-D Secure step-up** is ordinary. The statistics independently force the
   architecture the operational argument already preferred.

**Grid size is now a stated cost.** Holm correction runs over grid points times risks, so a
41-point grid costs roughly a 30 % larger minimum band budget than an 11-point grid at the
same `alpha`. Eleven points is chosen as the smallest grid that still resolves the frontier.

---

## Amendment A2 — 2026-08-28, operating point rule corrected

**What changed:** the operating point is selected by an issuer decline-rate budget, not by
the PSD2 reference fraud rate. The PSD2 comparison is retained and reported, but as a
distance measurement rather than as a constraint.

**Why.** Section 5 committed to "maximise recall subject to the value-weighted fraud rate on
the approved branch remaining at or below the PSD2 Annex reference rate". Applied to
IEEE-CIS this is a category error, and the measurement makes it plain.

Measured on `D_band` (58,326 rows), value-weighted, as Article 19 defines it:

| decline rate | approved-branch value fraud rate |
|---|---|
| 0 % | 5.554 % |
| 5 % | 2.556 % |
| 20 % | 1.055 % |
| 50 % | 0.347 % |
| 60 % | 0.247 % |

Reaching **any** of the three tiers requires declining **94.74 %** of traffic. The figure is
the same for all of them, and that is not a transcription slip: the threshold that satisfies
the loosest ceiling (ETV EUR 100, 0.13 %) already drives the achieved value-weighted rate to
9.74e-05, below even the strictest (EUR 500, 0.01 %), so one threshold clears the whole ladder
and `results/tables/envelope.csv` reports one decline rate three times.

An earlier version of this paragraph gave 94.7 % for the loosest tier and **98.2 %** for the
strictest. The 98.2 % appears in no table and never did; the measurement gives 94.74 % for
every tier. Corrected here, and `ClaimPsdDeclineRate` is bound to the `etv_eur = 500` row so
the quoted figure resolves to the strictest ceiling it is used to discuss.

The reason is not that the model is weak. The PSD2 reference rates govern a payment service
provider's **entire remote card portfolio**, which is overwhelmingly ordinary traffic.
IEEE-CIS is a fraud-detection benchmark assembled by sampling for fraud density: at a
5.554 % value-weighted fraud rate it sits roughly forty times above the loosest ceiling
before any model is applied. A portfolio-level threshold cannot be applied to an enriched
sample and read as an operating constraint.

**What replaces it.** The operating point is the highest threshold whose decline rate stays
within an issuer-plausible budget. Published figures put all-in card-not-present decline
rates in the range of roughly 0.5-3 % of volume, so the pre-registered budget is **2 %**,
with 1 % and 3 % reported alongside as sensitivity.

**What survives from the PSD2 framing, and it is the part that mattered.** The structural
argument is unaffected: the regulated quantity is value-weighted, is computed on a rolling
90-day basis, and imposes a portfolio-level ceiling on the fraud side while nothing
regulates the false-decline side. A one-sided objective is therefore the wrong target and a
two-sided envelope is the right one. What is withdrawn is only the use of those specific
numbers as a threshold selector on this specific dataset.

**The distance measurement is kept as a result.** The decline rate required to bring an
enriched benchmark down to portfolio-level fraud rates is the honest way to say how far this
data sits from deployment, and it is reported for that reason.

**Guarantee integrity.** As with A1, this was derived from `D_band` alone. No `D_cal` risk
value and no `D_test` quantity was computed, inspected or used.

---

## Amendment A3 — 2026-08-28, decline budget and the reporting of the recall floor

**What changed:** the pre-registered decline budget moves from 2 % to 5 %, and the recall
floor is reported as a **frontier across floors** rather than certified at a single
pre-chosen value.

**Why the budget moved.** Measured on `D_band` and `D_cal`, with the band corrected to sit
below the decline threshold:

| decline budget | outer-threshold recall | any configuration certifiable? |
|---|---|---|
| 2 % | 0.354 | no, at any band width, alpha or recall floor tested |
| 5 % | 0.561 | yes, all 24 combinations tested |

At 2 % the outer threshold catches 35.4 % of fraud and even declining the entire band
reaches only 62.7 %; the concentration bound cannot establish a recall floor from a band
that small. At 5 % the outer threshold reaches 56.1 % and every tested combination
certifies. The boundary is between the two, and it is a property of the data and the bound,
not a preference.

**Why the recall floor is now a frontier.** The floor of 0.60 in section 6 was chosen before
any data were seen, which was the right procedure but produced a number with no particular
justification. Certifying at one blind value and reporting only that would present an
arbitrary choice as a finding. All of 0.45, 0.50, 0.55 and 0.60 certify at a 5 % decline
budget, so the frontier is reported and the operating choice is left where it belongs — with
the institution that knows its own loss tolerance.

**The operational consequence is the same one A1 reached by a different route.** A certified
envelope needs roughly 5 % declines plus a 5-20 % step-up band: about a tenth of
authorisation traffic receives some intervention. That is routine for 3-D Secure step-up and
impossible for manual review, so the statistics again select the architecture.

**A bug was fixed in the process, and it mattered.** The band was initially centred *on* the
decline threshold and extended above it, so the in-band scorer was being asked to re-rank
transactions already routed to decline; with the threshold at the 98th percentile the upper
edge also clipped to the maximum score for any budget above 4 %. The band now occupies the
traffic immediately beneath the decline threshold, which is what the three-region decision
rule in section 3 actually describes.

**Guarantee integrity.** Derived from `D_band` and `D_cal` only. `D_test` was not consulted.

---

## Amendment A4 — 2026-08-28, estimand scale, the recall loss, and the decision grid

**What changed:** three things, all forced by defects an adversarial audit found in the
certification, and all documented here because each moves a pre-registered quantity.

**1. The band-conditional risk gets its own alpha scale.** Section 6 pre-registered
`alpha in {1e-2, 5e-3, 2e-3, 1e-3}` for a quantity described in section 1 as band-conditional.
Those are two different scales. The unconditional false-decline rate is what a bank's control
document states and belongs at that order; the band-conditional rate is measured over the
region where the model is uncertain and is intrinsically percent-scale. Measured on `D_cal` at
a 5 % band: declining the top decile of the band gives a band-conditional rate of 0.036,
declining half the band gives 0.241. Requiring that quantity below 1 % admits only the rule
that declines nobody.

The configuration now carries three grids: `alpha_grid` for the unconditional certificate
(unchanged), `alpha_band_grid` = {0.05, 0.10, 0.15, 0.25}, and `alpha_fn_grid`.

**2. The recall constraint becomes a false-negative rate.** The previous
`max(0, floor - recall) / floor` is a nonlinear transform of a mean. Hoeffding-Bentkus bounds
the mean of independent [0, 1] losses; it says nothing about a function of such a mean, so
those values were not valid p-values. `missed_fraud_rate` assigns loss 1 to a fraudulent
transaction approved by both stages and 0 otherwise, averaged over fraudulent transactions.
Controlling it at `alpha_fn` is equivalent to a recall floor at `1 - alpha_fn`, so the
commitment is unchanged in substance and now satisfies the theorem's hypotheses.

**3. The decision grid excludes its upper endpoint and follows the score distribution.**
Band membership is `lo <= score < hi` and the in-band rule is `score >= lambda`, so a grid
closing at `hi` contains a point where the flagged set is empty **by construction**. Its risk
is structurally zero whatever the data say, which makes it the only point clearing the
family-wise level. That single parameterisation artefact is why every certificate issued
before this amendment selected the rule that declines nobody. `in_band_grid` spaces the grid
by quantiles of the in-band calibration scores and drops the top quantile.

**Effect, measured.** Before: 32 of 32 certified configurations sat at the band top, declining
nobody. After: 0 of 5. The certified rules decline between 362 and 1,081 in-band transactions
and lift recall from 0.561 to between 0.595 and 0.637.

**What is withdrawn.** Amendments A1 and A3 described a guarantee-versus-abstention frontier
built from the vacuous certificates. A1's arithmetic about sample size and reachable alpha
remains valid as arithmetic; its framing as a frontier does not. A3's "all 24 combinations
certify" is withdrawn. `results/tables/riskcontrol.csv` and `tradeoff.csv` are regenerated.

**Guarantee integrity.** The grid change is a reparameterisation over calibration scores,
which are observed before any risk is evaluated, so the grid remains finite and fixed before
testing. The alpha and loss changes are stated here before the regenerated certificates were
read, and `D_test` was not consulted in reaching any of the three.

## Amendment A5 — the pre-registered test for H5 was specified for the wrong mechanism

**Dated 2026-08-29, after the certificate was produced and before it was validated.**

§8 pre-registers H5 as an "exact Beta-Binomial tail probability". That law is correct for
**split conformal**: the threshold there is the `k`-th order statistic of the calibration
scores, coverage is `Beta(n+1-k, k)` distributed, and the test-error count is therefore
Beta-Binomial.

Learn-then-Test does not select `lambda` that way. It tests a null hypothesis per grid point
under family-wise error control and returns the admissible set; the guarantee is
`P(R(lambda_hat) > alpha) <= delta`. Conditional on `lambda_hat` — which depends only on
`D_cal` — the declines among in-band legitimate test rows are independent Bernoulli draws, so
the count is **Binomial**, and under the certificate its rate is at most `alpha`. The valid
and conservative test is `P(Bin(m, alpha) >= observed)`.

`scripts/validate_certificate.py` reports **both**. The binomial tail is the operative
statistic; the Beta-Binomial is carried alongside because it was pre-registered, and dropping
it silently would conceal that the pre-registration named a test for a mechanism the estimand
does not have. On this data the two agree on the verdict at every certified configuration, so
nothing turns on the correction — but that is luck, not licence.

The amendment does not touch a guarantee-bearing parameter, so the protocol hash is unchanged.

## Amendment A6 — the geometric-difference screen was pre-registered and never gated

**Dated 2026-08-29, recorded on discovery rather than on decision.**

§9 pre-registers two a-priori screens, the first being the Huang geometric difference
`g(K_classical || K_quantum)` above a threshold, with the effective-rank-versus-RBF-correlation
diagnostic listed third. What `scripts/screen_kernels.py` actually gates on is the second
screen and the third diagnostic. `results/tables/screens.csv` computes
`geometric_difference` — it ranges from 1.15 to 364.05 with a mean of 57.6 — and **no
`passes_*` column consults it**.

This was not decided; it was discovered during an audit of the submission. The screen was
dropped in implementation and the protocol was never amended, which is the failure the
pre-registration exists to prevent.

The consequence must be stated plainly. Huang's criterion treats a *large* geometric
difference as the favourable direction, so this is the screen most configurations would have
cleared. Gating on it would have made the kernel arm's rejection *less* decisive, not more.
Reporting the rejection while silently omitting the screen that would have weakened it is the
direction of error that flatters the finding.

The rejection stands on the two screens that were applied — 28 of 120 configurations pass
conditioning, 0 pass distinctness, and the median RBF correlation is 0.974 with a maximum of
exactly 1.000 — but the body now says which screens were gated and which was not.

---

## Amendment A7 — 2026-08-30, three pre-registered drift tests were not run

**What changed:** section 7 lists five things as "measured, not assumed". Three of them were
never executed: the **conformal test martingale** over the `TransactionDT`-ordered stream, the
**block permutation test**, and **band-conditional drift against marginal drift**. No script
was written for any of them, and `exchangeability.csv` and `drift.csv` do not exist.

**Why this is recorded rather than quietly dropped.** Section 7 is the part of the protocol
that says what this study does *instead* of promising a numeric `alpha + eta`. Its whole
argument is that the degradation is measured rather than bounded. Three of the five
measurements not existing weakens that argument, and a reader who takes the list at face value
would over-read what the study established.

**What was measured, and what it does and does not cover.** Two of the five did run, and they
carry the temporal claim:

* the **realised risk gap per time window** — five rolling calibration origins,
  `rolling_origin.csv`, reported in the body. This is what establishes that the breach is
  origin-dependent, and it is the finding that narrowed the central claim;
* the **temporal arm against the stratified-random control** — three arms, five seeds each,
  `coverage_seed_summary.csv`, also reported in the body.

What is therefore *not* established: that the deviation is a martingale-detectable drift
rather than a level shift, and whether band-conditional drift differs from marginal drift. The
second matters, because section 4 makes the band-conditional quantity the relevant one, and
this study has not shown that the two behave differently.

**Why no numbers move.** These tests were diagnostic, not gating. No hypothesis, no
certificate and no reported figure depends on them, so nothing in the results changes. What
changes is the strength of the claim section 7 can carry, and that is stated here rather than
left to be inferred from an absent table.

**Where it belongs in Phase II.** All three are cheap on an issuer's own stream and none of
them touches the guarantee, so they belong in the first stage of the sprint, alongside the
split-integrity work.



## Amendment A8 — 2026-08-30, the single-evaluation rule described more than the code enforces

**What changed:** section 2.2 stated that "every sweep, ladder and ablation runs on held-out
slices of `D_band` or `D_cal`". Four scripts read the test block, and three of them do not go
through `TestFoldGuard`:

| Script | What it reads from `D_test` | Guard |
|---|---|---|
| `run_conformal.py` | the certified configuration | `authorise()` |
| `validate_certificate.py` | H5, the held-out validation | `authorise()` |
| `run_baselines.py` | descriptive metrics, 3 arms x 5 seeds | none |
| `run_ablations.py` | 4 leakage-ablation variants x 5 seeds | none |
| `run_seed_sweep.py` | the 16-job full-scale arm | none |
| `run_mps.py` | the in-band comparison | `authorise()` |

**Why the guarantee is unaffected, and why this is still a defect.** The rule that protects a
finite-sample guarantee is that nothing may be *selected* on the test fold. Nothing was:
`ablations.csv` is consumed by no claim, no figure and no downstream script — it is a
standalone diagnostic — and `baselines.csv` and the seed sweep report the primary metrics on
the test block, which is what a held-out block is for. The certified threshold and the band
edges come from `D_band` and `D_cal` only, and `test_access.json` records the one configuration
that was authorised.

What is genuinely wrong is the sentence. "Every sweep, ladder and ablation runs on held-out
slices" is stronger than the code, and the ablation ladder is exactly the case it excludes. A
reviewer comparing section 2.2 with `run_ablations.py:82` finds the divergence in two minutes,
and a pre-registration that overstates its own enforcement is worse than one that states a
weaker rule accurately.

**What was not done.** The scripts were not changed to route through the guard, and the ledger
was not back-filled. Both would alter committed tables to make a text problem disappear, which
is the wrong direction: the protocol is the thing that was wrong, so the protocol is what
changed. `TestFoldGuard` also raises only on a *different* configuration hash rather than on a
repeat read; that is the behaviour its own docstring describes, and it is what section 2.2 now
claims.

**Guarantee integrity.** No `D_cal` risk value and no threshold was derived from anything in
the table above. The change is to the description of the rule, not to the rule.

## Amendment A9 — 2026-08-30, ULB was pre-registered as a secondary dataset and never used

**What changed:** section 5 listed ULB European Cardholder as **Secondary**, with row and fraud
counts, and section 6.1 asserted a degeneracy result on it. Neither was ever executed. The
dataset is not on this machine, no script fetches it, `ulb.csv` does not exist, and smoke check
S7 has skipped for that reason throughout.

**Three separate things were wrong, and they are worth separating.**

1. **Not run.** E15, the ULB stress case, never executed. This is the same class of omission as
   the three drift measurements in A7, and it gets the same treatment: recorded, not quietly
   dropped.
2. **Not measured.** The counts in section 5 — 284,807 rows, 492 frauds, 1,081 exact duplicates
   — are transcribed from the dataset's published description. They may well be right; nothing
   here checked them, and the protocol presented them as though they had been.
3. **Arithmetically inconsistent with this protocol.** Section 6.1 said "roughly 98 calibration
   frauds against the 99 required". 98 is 20 % of 492, which is this protocol's **test**
   fraction. `split.cal` is 10 %, giving roughly 49. The conclusion survives and strengthens —
   49 against a floor of 99 is more degenerate than 98 — but a number that does not follow from
   the split it is quoted beside is not evidence, whichever way it points.

**Why ULB is out of scope rather than deferred.** The comparison it was meant to supply — that a
fraud-conditional quantile degenerates on a small, heavily imbalanced file where IEEE-CIS's does
not — is an argument about sample size that the IEEE-CIS measurements already make directly:
section 6.1's own degeneracy table shows the floor at 19 rows against 847 available, two orders
of magnitude clear. ULB would have been a second illustration of a point already carried.

**What this changes in the submission.** Nothing. No claim in either PDF, no figure and no table
rests on ULB; `grep` finds it in no `.tex` file. `docs/PROVENANCE.md` §1.2 already records that
it is not reconstructible here. This amendment exists so that the pre-registration says the same
thing the repository does.

**Guarantee integrity.** No `D_cal` risk value and no threshold was ever derived from ULB,
because ULB was never loaded.
