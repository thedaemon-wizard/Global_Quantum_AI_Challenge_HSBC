# The guarantee: notation, statement, golden values, and what it does not cover

`docs/REFERENCES.md`, `README.md` and the proposal all refer to a guarantee. This file states
it once, in full, so that the other documents can point here instead of restating it and
drifting from each other.

Every number below is reproduced from `results/tables/` and is checked by
`scripts/check_claims.py`. If a figure here disagrees with a table, the table is right.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $f$ | the classical scorer, fitted on $D_{\mathrm{train}}$ |
| $\tau_{\mathrm{lo}}, \tau_{\mathrm{hi}}$ | band edges, fixed on $D_{\mathrm{band}}$ and never re-estimated |
| $B$ | the abstention band, $\lbrace x : \tau_{\mathrm{lo}} \le f(x) \lt \tau_{\mathrm{hi}} \rbrace$ |
| $g$ | the in-band re-scorer, where a quantum model would enter |
| $\lambda$ | the in-band decision threshold, selected on $D_{\mathrm{cal}}$ |
| $\alpha$ | the false-decline budget |
| $\alpha_{\mathrm{FN}}$ | the missed-fraud budget |
| $\delta$ | the family-wise error level of the selection |
| $Y$ | the label, $1$ for fraud |

The decision rule is three-valued:

```math
D(x) = \begin{cases}
\texttt{DECLINE} & f(x) \ge \tau_{\mathrm{hi}}
\\ \texttt{STEP-UP},\ \text{then } g(x) \ge \lambda \Rightarrow \texttt{DECLINE} & \tau_{\mathrm{lo}} \le f(x) < \tau_{\mathrm{hi}}
\\ \texttt{APPROVE} & f(x) < \tau_{\mathrm{lo}}
\end{cases}
```

## 2. The estimand, and why it is conditional

The certified quantity is the **band-conditional** false-decline rate:

```math
R(\lambda) \;=\; \mathbb{P}\bigl(D(X) = \texttt{DECLINE} \,\bigm|\, Y = 0,\; X \in B\bigr)
```

Not the marginal false-decline rate. The marginal rate is dominated by $\tau_{\mathrm{hi}}$ and
would be almost unchanged if $g$ were replaced by a coin flip, so a certificate on it would not
bind the component it is said to license. The conditional quantity is harder and has a smaller
sample, and the cost of that choice is reported rather than hidden.

The second risk controlled jointly is the missed-fraud rate, a $0/1$ loss averaged over frauds,
equivalent to a recall floor at $1 - \alpha_{\mathrm{FN}}$.

## 3. The statement

Learn-then-Test selects $\hat\lambda$ from a grid fixed in advance, testing a null at each grid
point with Hoeffding-Bentkus $p$-values under Holm correction across the family. The guarantee
is over the draw of $D_{\mathrm{cal}}$:

```math
\mathbb{P}\bigl(R(\hat\lambda) > \alpha\bigr) \;\le\; \delta
```

Three properties of this statement matter more than the inequality itself.

**It is distribution-free.** No assumption about the score distribution, the model class, or
the fraud process. It requires exchangeability between $D_{\mathrm{cal}}$ and the deployment
population, and nothing else.

**It is finite-sample.** Not asymptotic. The bound holds at the calibration sizes actually
available, which for the tightest band budget is 847 legitimate in-band rows.

**It is conditional on $\hat\lambda$ being selected the way the procedure says.** Selecting
$\lambda$ by looking at $D_{\mathrm{test}}$, or re-choosing the band edges after seeing
calibration risks, voids it. Both are enforced in code rather than by discipline:
`TestFoldGuard` keeps a persisted ledger of the authorised configuration and its evaluation
count, and `scripts/check_protocol.py` hashes every guarantee-bearing parameter.

## 4. Golden values

The five configurations that certify, out of 48 pre-registered. All five sit at the loosest
missed-fraud budget, $\alpha_{\mathrm{FN}} = 0.45$ --- which is therefore a necessary condition
for certifying rather than what distinguishes the five from each other; band budget and $\alpha$
do that. Source: [`riskcontrol.csv`](../results/tables/riskcontrol.csv).

| Band budget | $\alpha$ | $\alpha_{\mathrm{FN}}$ | Band $[\tau_{\mathrm{lo}}, \tau_{\mathrm{hi}})$ | In-band legit. cal. rows | $\hat\lambda$ |
|---|---|---|---|---|---|
| 0.035 | 0.25 | 0.45 | [0.031981, 0.071798) | 1572 | 0.058202 |
| 0.050 | 0.25 | 0.45 | [0.025988, 0.071798) | 2259 | 0.054104 |
| 0.100 | 0.10 | 0.45 | [0.015407, 0.071798) | 4831 | 0.053650 |
| 0.100 | 0.15 | 0.45 | [0.015407, 0.071798) | 4831 | 0.053650 |
| 0.100 | 0.25 | 0.45 | [0.015407, 0.071798) | 4831 | 0.042632 |

Applied unchanged to the held-out block — same band edges, same $\lambda$, no recalibration —
all five hold. Source: [`h5_validation.csv`](../results/tables/h5_validation.csv).

| Band budget | $\alpha$ | In-band legit. test rows | Declined | Realised risk | Holds |
|---|---|---|---|---|---|
| 0.035 | 0.25 | 3314 | 580 | 0.1750 | yes |
| 0.050 | 0.25 | 4835 | 830 | 0.1717 | yes |
| 0.100 | 0.10 | 10021 | 858 | 0.0856 | yes |
| 0.100 | 0.15 | 10021 | 858 | 0.0856 | yes |
| 0.100 | 0.25 | 10021 | 1762 | 0.1758 | yes |

The worst realised risk uses 0.856 of its budget. These are loose ceilings — 0.10 to 0.25 — and
holding them is a real check that the machinery transfers, not a demonstration that the rule is
tight.

## 5. What the guarantee does not contain

Stated here rather than distributed through the other documents, because every item is
something a reader could reasonably assume and should not.

**It does not bind the unconditional false-decline rate.** Section 2 explains why that is the
point rather than a caveat. A certificate on the marginal rate would be easier to obtain and
would say nothing about the abstention band.

**It does not survive a temporal split.** This is the study's central measured result, not a
theoretical caveat. Split-conformal coverage on a forward holdout falls outside the exact
Beta-Binomial interval on every seed at the two loosest levels; a stratified random split
falls outside on none. The guarantee assumes exchangeability, and time breaks it.

**It does not come with a numeric $\alpha + \eta$.** The additive penalty for beta-mixing
depends on mixing coefficients that are not estimable from the data, and the drift parameter in
the weighted variant is unobservable. Protocol section 7 refuses to promise a corrected level
and measures the realised gap instead. Three of the five measurements it lists were not run;
amendment A7 records which.

**It does not certify below $\alpha = 0.10$.** Nothing certifies at the tightest band budget at
any level. The mechanism is sample size acting through the concentration bound, not the
class-conditional degeneracy floor — that floor is 19 rows at $\alpha = 0.05$ against 847
available and cannot bind anywhere on this grid ([D-044](decisions.md)).

**It is not conditional on anything but the band and the class.** It says nothing about
coverage within a subgroup -- by transaction value, by merchant category, by device. An
exploratory check found no value-conditional failure: all twenty amount-quartile cells of the
five certified configurations hold on the held-out block, worst at 0.894 of budget
([D-055](decisions.md)). That is evidence against one specific failure mode at one resolution.
It is not a conditional certificate, and it was not pre-registered.

**It is not a statement about a quantum model.** Both quantum arms returned negative results.
The certificate wraps whatever scores the band, and it is reported precisely because it
survives a negative quantum result.

**It does not extrapolate past the approved-and-settled population.** IEEE-CIS contains only
transactions an existing stack approved, and a chargeback propagates to later transactions on
the same card, so the label is partly an entity flag. Shapley attribution puts 75.8 % of the
in-band model's contribution on `card1` alone ([D-048](decisions.md)).

## 6. Where each piece lives

| Component | Implementation |
|---|---|
| Learn-then-Test, Hoeffding-Bentkus, Holm | [`conformal/riskcontrol.py`](../src/hsbcfraud/conformal/riskcontrol.py) |
| Split-conformal threshold, degeneracy floor | [`conformal/split.py`](../src/hsbcfraud/conformal/split.py) |
| Exact Beta-Binomial coverage law | [`conformal/coverage.py`](../src/hsbcfraud/conformal/coverage.py) |
| Four-block split, `TestFoldGuard` | [`data/splits.py`](../src/hsbcfraud/data/splits.py) |
| Band edges, in-band rows, feature choice | [`features/band.py`](../src/hsbcfraud/features/band.py) |
| Certification run | [`run_conformal.py`](../scripts/run_conformal.py) |
| Held-out validation (H5) | [`validate_certificate.py`](../scripts/validate_certificate.py) |
| Pre-registration and its amendments | [`protocol.md`](protocol.md) |
