# References

Every source this project relies on, with a locator that resolves. `scripts/check_claims.py`
runs a citation gate over the documents listed in `docs/claims.yaml`: any citation-shaped
string in those documents must resolve to an entry here, so a reference cannot be cited in
prose without appearing in this file.

Entries are grouped by the role they play in the work. `docs/REFERENCE_CROSSCHECK.md` records
which module or table uses each one and what it was checked against.

---

## 1. Conformal prediction and distribution-free risk control

**[CP-1]** Vovk, V., Gammerman, A. and Shafer, G. *Algorithmic Learning in a Random World*.
Springer, 2005 (2nd edition 2022). DOI 10.1007/b106715.
The split-conformal threshold as the order statistic `k = ceil((1-alpha)(n+1))`, `qhat = s_(k)`.
Implemented in `src/hsbcfraud/conformal/split.py::conformal_threshold`.

**[CP-2]** Vovk, V. "Conditional Validity of Inductive Conformal Predictors". *Proceedings of
Machine Learning Research* 25:475-490, 2012. arXiv:1209.2673.
Label-conditional validity, and the Beta law of conditional coverage that compounds to the
Beta-Binomial predictive distribution used to validate thresholds. Implemented in
`src/hsbcfraud/conformal/coverage.py`.

**[CP-3]** Vovk, V., Lindsay, D., Nouretdinov, I. and Gammerman, A. "Mondrian Confidence
Machine". Royal Holloway, University of London, technical report, 2003.
The class-conditional construction. Implemented in
`src/hsbcfraud/conformal/split.py::mondrian_thresholds`.

**[CP-4]** Ding, T., Angelopoulos, A. N., Bates, S., Jordan, M. I. and Tibshirani, R. J.
"Class-Conditional Conformal Prediction with Many Classes". *Advances in Neural Information
Processing Systems* 36:64555-64576, 2023. arXiv:2306.09335.
The degeneracy floor: a class with fewer than `(1/alpha) - 1` calibration points has an
infinite classwise quantile. Note the exact form, which is not `ceil(1/alpha) - 1`.
Implemented in `src/hsbcfraud/conformal/split.py::degeneracy_floor`; the arithmetic for this
study's calibration block is in `results/tables/degeneracy.csv`.

**[CP-5]** Barber, R. F., Candes, E. J., Ramdas, A. and Tibshirani, R. J. "Conformal
prediction beyond exchangeability". *Annals of Statistics* 51(2):816-845, April 2023.
DOI 10.1214/23-AOS2276. arXiv:2202.13415.
Equation (11), the weighted quantile with a point mass at infinity, and equation (3), the
coverage-gap bound. Implemented in `src/hsbcfraud/conformal/weighted.py`.
**Used with a stated limitation.** The paper's two closed-form bounds, `2*eps/(1-rho)` under
bounded drift and `rho^k` after a changepoint, both route through its Lemma 1, which assumes
independent observations. A transaction stream is serially correlated and falls under the
paper's separate covariate-time-series treatment, so neither closed form is quoted here as an
evaluated number. See `docs/decisions.md` D-013.

**[CP-6]** Oliveira, R. I., Orenstein, P., Ramos, T. and Romano, J. V. "Split Conformal
Prediction and Non-Exchangeable Data". *Journal of Machine Learning Research* 25, 2024.
Plain split conformal retains finite-sample validity for stationary beta-mixing processes with
an additive penalty, and Appendix D extends the argument to risk-controlling prediction sets.
Cited as the reason the simple temporal design is defensible; the penalty itself is not
evaluated, because it depends on mixing coefficients that are not estimable from the data.

**[CP-7]** Angelopoulos, A. N., Bates, S., Candes, E. J., Jordan, M. I. and Lei, L. "Learn
then Test: Calibrating Predictive Algorithms to Achieve Risk Control". *Annals of Applied
Statistics* 19(2):1641-1662, June 2025. DOI 10.1214/24-AOAS1998. arXiv:2110.01052.
The PAC risk-control construction this study certifies with. Implemented in
`src/hsbcfraud/conformal/riskcontrol.py::learn_then_test`, validated by simulation
(0 violations in 3,000 replications at delta = 0.05).

**[CP-8]** Bates, S., Angelopoulos, A. N., Lei, L., Malik, J. and Jordan, M. I.
"Distribution-Free, Risk-Controlling Prediction Sets". *Journal of the ACM* 68(6):43, 2021.
arXiv:2101.02703.
The Hoeffding-Bentkus p-value. Implemented in
`src/hsbcfraud/conformal/riskcontrol.py::hoeffding_bentkus_p_value`, validated at the null
boundary.

**[CP-9]** Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L. and Schuster, T. "Conformal
Risk Control". *International Conference on Learning Representations*, 2024.
arXiv:2208.02814.
Bounds the expected risk. Reported alongside the PAC form as the tighter-but-weaker
alternative; the two are never used interchangeably.

**[CP-10]** Tibshirani, R. J., Barber, R. F., Candes, E. J. and Ramdas, A. "Conformal
Prediction Under Covariate Shift". *Advances in Neural Information Processing Systems* 32,
2019. arXiv:1904.06019.
Weighted conformal under pure covariate shift, requiring a known or estimated likelihood
ratio. **Distinguished from [CP-5]**, which gives approximate coverage under arbitrary shift
with fixed weights. Fraud tactics change `P(Y|X)`, not only `P(X)`, so the covariate-shift
assumption is not adopted here.

**[CP-11]** Podkopaev, A. and Ramdas, A. "Distribution-free uncertainty quantification for
classification under label shift". *Proceedings of Machine Learning Research* 161:844-853
(UAI), 2021. arXiv:2103.03323.
Fraud prevalence drift is label shift, which covariate-shift weighting does not cover.

**[CP-12]** Gibbs, I. and Candes, E. J. "Adaptive Conformal Inference Under Distribution
Shift". *Advances in Neural Information Processing Systems* 34, 2021. arXiv:2106.00170.

**[CP-13]** Gibbs, I. and Candes, E. J. "Conformal Inference for Online Prediction with
Arbitrary Distribution Shifts". *Journal of Machine Learning Research* 25(162):1-36, 2024.
arXiv:2208.08401.
Positioned as the Phase II monitoring layer, not as the Phase I guarantee.

---

## 2. Conformal prediction applied to fraud detection -- the prior art this work is positioned against

This section exists because the claim "conformal prediction for fraud is an empty field" is
false and would be refuted by one search. The novelty claimed here is narrower; see
`README.md` and the proposal's related-work paragraph.

**[FR-1]** Singh, M., Srikantha, A. and Lakhanpal, S. "Cost-Sensitive Conformal Prediction and
Human-in-the-Loop Abstention for Imbalanced High-Stakes Decision Support: A Multi-Domain
Benchmark". arXiv:2607.27143, 29 July 2026.
Mondrian class-conditional conformal prediction with cost-controlled abstention across 15
imbalanced tabular datasets, including the ULB credit-card set. Uses seed-governed random
splits.

**[FR-2]** Nayak and Bushara. "Uncertainty-Aware Fraud Detection Using Hybrid Transformer With
Gated Token Mixing and Conformal Risk Control" (ARGUS). *IEEE Access* 14:77557-77573, 2026.
DOI 10.1109/ACCESS.2026.3694614.
Reports AUPRC 0.880, coverage 0.88, selective risk 0.015 on the ULB dataset. The paper
describes its own objective as conformal-style rather than a finite-sample guarantee.

**[FR-3]** Zhu et al. "DISCO: Decoupling representation learning and risk control for reliable
credit card fraud detection". *Decision Support Systems* 208, 2026.
DOI 10.1016/j.dss.2026.114717.
Conformal risk control with a formal guarantee on the false-negative rate.

**[FR-4]** Mapaila and Senekane. *Technologies* 14(4):212, 3 April 2026.
DOI 10.3390/technologies14040212.
Split conformal prediction with abstention-based routing on PaySim.

**[FR-5]** Chen, Gong, Cheng and Jin. "Temporal Graph Prototype-conditioned Conformal
Prediction for Fraud Detection" (ProtoCP). *KDD* 2026. DOI 10.1145/3770855.3818061.
arXiv:2608.15768.

**[FR-6]** Wang, T., Kang, J., Yan, Y., Kulkarni, A. and Zhou, D. "Non-exchangeable Conformal
Prediction for Temporal Graph Neural Networks" (NCPNET). *KDD* 2025.
DOI 10.1145/3711896.3737064. arXiv:2507.02151.

---

## 3. Quantum machine learning -- the state of the evidence

**[QM-1]** Huang, H.-Y., Broughton, M., Mohseni, M., Babbush, R., Boixo, S., Neven, H. and
McClean, J. R. "Power of data in quantum machine learning". *Nature Communications* 12:2631,
2021. DOI 10.1038/s41467-021-22539-9.
The geometric difference `g(K_classical || K_quantum)` used here as an a-priori screen.

**[QM-2]** Thanasilp, S., Wang, S., Cerezo, M. and Holmes, Z. "Exponential concentration in
quantum kernel methods". *Nature Communications* 15:5200, 18 June 2024.
DOI 10.1038/s41467-024-49287-w. arXiv:2208.11060.
Four sources of concentration: embedding expressivity, global measurements, entanglement and
noise. The paper states that kernel-alignment training is also susceptible, which closes the
usual escape route.

**[QM-3]** Kubler, J. M., Buchholz, S. and Scholkopf, B. "The Inductive Bias of Quantum
Kernels". *Advances in Neural Information Processing Systems* 34, 2021. arXiv:2106.03747.
An exponentially large feature space makes generalisation harder, not easier.

**[QM-4]** Slattery, L., Shaydulin, R., Chakrabarti, S., Pistoia, M., Khairy, S. and Wild,
S. M. "Numerical evidence against advantage with quantum fidelity kernels on classical data".
*Physical Review A* 107:062417, 2023. arXiv:2211.16551.

**[QM-5]** Florez-Ablan, D., Roth, M. and Schnabel, J. "On the Interplay of Bandwidth and
Expressivity in Quantum Kernels". arXiv:2503.05602, 29 July 2025.
Optimal bandwidth tuning drives fidelity and projected quantum kernels toward radial basis
function kernels, and at larger bandwidths toward polynomial kernels. This is the trap the
two-dimensional screen in `src/hsbcfraud/quantum/screens.py` is designed to detect.

**[QM-6]** Kakavand, Strohmeyer and Schlotter. arXiv:2604.18837, April 2026.
Nine tabular datasets, 8,400 SVM fits, hardware-validated. No pairwise quantum-classical
comparison significant; an 18.1 percentage-point balanced-accuracy deficit. Recommends
spectral pre-screening, which this study implements.

**[QM-7]** Bowles, J., Ahmed, S. and Schuld, M. "Better than classical? The subtle art of
benchmarking quantum machine learning models". arXiv:2403.07059, 2024.
Out-of-the-box classical models outperform quantum classifiers; removing entanglement often
does not hurt. The source of this study's commitments to a tuned classical baseline, an
entanglement ablation, multiple seeds and released code.

**[QM-8]** Chaves, Kumar, Chagas, Linerud, Sorem, Mancilla and Bell (Oxford Quantum Circuits
and Mastercard). "A Mixture-of-Experts Framework for Practical Hybrid-Quantum Models in Credit
Card Fraud Detection". arXiv:2603.06473, 1 May 2026.
The nearest architectural neighbour: a classical primary expert, a router, and a quantum
secondary. Reports average precision 0.793 +/- 0.085 against 0.770 +/- 0.096 for XGBoost, a
difference the authors do not claim as significant, under random stratified cross-validation
with the time column removed.

**[QM-9]** Park, S. and Simeone, O. "Quantum Conformal Prediction for Reliable Uncertainty
Quantification in Quantum Machine Learning". *IEEE Transactions on Quantum Engineering* 5,
art. 3103224, 2024. DOI 10.1109/TQE.2023.3333224. arXiv:2304.03398.
Shot noise makes the nonconformity score a random variable, so a naive conformal wrapper
around a shot-based quantum kernel is not automatically valid. This study's Phase I response
is an exact statevector simulator with a fixed configuration.

**[QM-10]** Spencer, Nicholls and Caprio. "Adaptive Conformal Prediction for Quantum Machine
Learning". *Transactions on Machine Learning Research*, May 2026. arXiv:2511.18225.
Time-varying hardware noise undermines conformal guarantees even when the data are
exchangeable. Named as the Phase II hardware mitigation path.

---

## 4. Datasets and evaluation protocol

**[DS-1]** IEEE-CIS Fraud Detection. Kaggle, 2019. Provided by Vesta Corporation.
https://www.kaggle.com/competitions/ieee-fraud-detection
Distributed under competition rules, not an open licence. Measured properties of the training
file, asserted on load in `src/hsbcfraud/data/ieee_cis.py`: 590,540 rows, 20,663 frauds
(3.4990 %), 394 columns, span exactly 182.00 days, `TransactionDT` monotonically increasing.

**[DS-2]** IEEE-CIS label definition. Competition host, Kaggle discussion 101203.
A reported chargeback sets `isFraud = 1` and the flag propagates to later transactions sharing
a user account, email address or billing address; a transaction is labelled 0 only if nothing
is reported within 120 days. This is why the estimand in `docs/protocol.md` section 1.1 is
stated as it is, and why `scripts/audit_labels.py` exists.

**[DS-3]** Dal Pozzolo, A., Caelen, O., Johnson, R. A. and Bontempi, G. "Calibrating
Probability with Undersampling for Unbalanced Classification". *IEEE Symposium Series on
Computational Intelligence*, 2015. DOI 10.1109/SSCI.2015.33.
Originating paper for the ULB European Cardholder dataset; recommends AUPRC over ROC-AUC at
this prevalence.

**[DS-4]** European Cardholder dataset. Machine Learning Group, Universite Libre de Bruxelles.
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Licence, read from the Kaggle API on 2026-08-28: *"Database: Open Database, Contents: Database
Contents"* -- that is, the database under ODbL and its contents under DbCL v1.0. The challenge
statement describes it as ODbL alone, which is incomplete.

**[DS-5]** Deotte, C. "IEEE-CIS Fraud Detection: 1st Place Solution", via NVIDIA Developer
Blog, 2019.
AUC-ROC 0.9459 on the private leaderboard; about 0.9363 in local validation without the UID
feature, under time-based GroupKFold cross-validation. The UID construction
`card1_addr1 + floor(day - D1)` and its reported contribution of about +0.011 AUC. This study
measures that feature at +0.0015 average precision under a forward holdout; see
`docs/decisions.md` D-021.

---

## 5. Regulation and model governance

Official locators for each instrument are in `docs/REGULATORY_SOURCES.md`. Summarised here
only to the extent the project relies on them.

**[RG-1]** Commission Delegated Regulation (EU) 2018/389 supplementing Directive (EU) 2015/2366
as regards regulatory technical standards for strong customer authentication.
Article 18 (transaction risk analysis exemption), Article 19 (fraud rate calculation:
**value-weighted**, rolling 90 days), Article 21 (monitoring), and the Annex reference fraud
rates for remote card-based payments: 0.13 % at EUR 100, 0.06 % at EUR 250, 0.01 % at EUR 500.
**Used as a structural argument, not as a threshold selector.** See `docs/protocol.md`
amendment A2: these are portfolio-level rates and IEEE-CIS is a fraud-enriched benchmark at a
5.554 % value-weighted rate.

**[RG-2]** Regulation (EU) 2024/1689 (the Artificial Intelligence Act), Annex III point 5(b).
Credit scoring is high-risk **with the exception of AI systems used for the purpose of
detecting financial fraud**. Recital 58 reinforces the exception. This project therefore makes
no AI Act high-risk compliance claim.

**[RG-3]** Regulation (EU) 2026/1744, published in the Official Journal 24 July 2026.
Defers stand-alone Annex III high-risk obligations to 2 December 2027.

**[RG-4]** Board of Governors of the Federal Reserve System, OCC and FDIC. SR 26-2, *Revised
Guidance on Model Risk Management*, 17 April 2026. Supersedes SR 11-7 and SR 21-8.
Section V: validation, ongoing monitoring, outcome analysis, benchmarking to other models,
independent review, and third-party model validation.

**[RG-5]** Prudential Regulation Authority. Supervisory Statement SS1/23, *Model risk
management principles for banks*, 17 May 2023, effective 17 May 2024.
Confirmed by the PRA to apply to fraud models.

**[RG-6]** Financial Conduct Authority. Policy Statement PS21/19, 29 November 2021.
The onshored UK SCA-RTS, preserving the reference fraud rates and article structure of [RG-1].

---

## 6. Software

Versions are pinned in `pyproject.toml` and were checked against the PyPI JSON API on
2026-08-28; `NOTICE` carries the full licence list.

**[SW-1]** Qiskit 2.5.2. Apache-2.0. https://github.com/Qiskit/qiskit
**[SW-2]** Qiskit Machine Learning 0.9.1. Apache-2.0.
https://github.com/qiskit-community/qiskit-machine-learning
Supplies `FidelityStatevectorKernel`, the production kernel path here.
**[SW-3]** Amazon Braket SDK 1.126.0 and amazon-braket-default-simulator 1.40.1. Apache-2.0.
https://github.com/amazon-braket/amazon-braket-sdk-python
**[SW-4]** MAPIE 1.5.0. BSD-3-Clause. https://github.com/scikit-learn-contrib/MAPIE
**[SW-5]** crepes 0.9.1. BSD-3-Clause. https://github.com/henrikbostrom/crepes
**[SW-6]** XGBoost 3.4.1. Apache-2.0. https://github.com/dmlc/xgboost
**[SW-7]** SHAP 0.52.0. MIT. https://github.com/shap/shap
**[SW-8]** scikit-learn 1.9.0. BSD-3-Clause.
**[SW-9]** PyTorch 2.13.0+cu130. BSD-3-Clause.

---

## 7. Sources consulted and deliberately not relied upon

Recorded because excluding a source is a decision.

**Innan, N. et al.** "Financial Fraud Detection: A Comparative Study of Quantum Machine
Learning Models". *International Journal of Quantum Information* 21(05), 2023.
DOI 10.1142/S0219749923500442. The reported QSVC F1 of 0.98 was obtained on 200 rows of
BankSim, balanced 100/100, with four features and no classical baseline. Not comparable to a
full imbalanced test fold, and not a ULB result.

**El Alami, Innan, Shafique and Bennai.** "Comparative Performance Analysis of Quantum Machine
Learning Architectures for Credit Card Fraud Detection". arXiv:2412.19441; *Applied
Intelligence*, 2026, DOI 10.1007/s10489-026-07110-7. The VQC F1 of 0.88 was obtained on a
984-row balanced undersample of ULB reduced to seven principal components, with no classical
baseline. Frequently miscited as "Karimi et al."

**Grossi, M. et al.** "Mixed Quantum-Classical Method for Fraud Detection with Quantum Feature
Selection". *IEEE Transactions on Quantum Engineering*, 2022. arXiv:2208.07963. The QSVM
accuracy of 0.789 is an ideal-simulator figure on a 1,000-row balanced subset of proprietary
data; the paper's own noisy-simulation row is 0.55 +/- 0.10.

**PeerJ Computer Science**, 2 September 2025, PMC12453863. The challenge statement attributes
a ULB stacking AUC-ROC of 0.9887 to this paper. The paper reports 0.898 for its stacking model,
and the string 0.9887 does not appear in it. This project re-derives its own ULB baseline.

**Shanaa and Abdallah.** *F1000Research* 14:664, 2025. DOI 10.12688/f1000research.166350.2.
Peer-review status is two approved with reservations and one not approved; the reported recall
of 0.925 at precision 0.957 is far outside the range other ULB studies achieve at comparable
precision. Not used as a target.
