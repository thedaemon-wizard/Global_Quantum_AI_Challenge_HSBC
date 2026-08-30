# Provenance

Where every input came from, what licence governs it, and what was verified about it before
it was used. Nothing here is taken on trust from a filename.

---

## 1. Data

### 1.1 IEEE-CIS Fraud Detection (primary)

| | |
|---|---|
| Source | Kaggle competition `ieee-fraud-detection`, `train_transaction.csv` and `train_identity.csv` |
| Obtained | Downloaded by the project owner from the competition data page; staged locally as `datasets/ieee-fraud-detection.zip` |
| Licence | **Kaggle competition rules**, not an open licence. Redistribution is not permitted. |
| In this repository | **Never committed.** `.gitignore` carries an anchored `/datasets/` pattern. |
| Verified | 590,540 rows, 20,663 frauds (3.4990 %), 394 columns, `TransactionDT` strictly increasing, spanning exactly 182.00 days. Asserted at load time by `src/hsbcfraud/data/ieee_cis.py`, so a substituted or truncated file fails immediately rather than producing plausible numbers. |

**The label is not what its name suggests.** `isFraud` is set on a reported chargeback and
then propagated to subsequent transactions sharing a user account, email address or billing
address; a transaction is labelled 0 only if nothing is reported within 120 days (competition
host, Kaggle discussion 101203). Many positive rows are therefore not fraudulent transactions
but transactions on a card that later had one. This is recorded in `docs/protocol.md` §1.1 and
is why the study describes the task as partly entity-contamination detection.

**Consequence for inference.** The propagation rule clusters rows by entity. Measured on this
file, 85.0 % of `card1` values in the test block also appear in training, with a median of 4
and a 95th percentile of 107 transactions per value. Row-level exchangeability does not hold,
so every interval in this study is a card-level block bootstrap and the guarantee is stated at
block level.

**The population is conditional on an incumbent.** The file contains only transactions an
existing authorisation stack already approved; declined attempts cannot generate a chargeback.
No result here extrapolates to the declined population.

### 1.2 ULB European Cardholder (secondary, stress case)

| | |
|---|---|
| Source | Machine Learning Group, Université Libre de Bruxelles |
| Licence | Two-layer: the **database** under ODbL, its **contents** under DbCL v1.0. Share-alike applies to a derived database; not redistributed here. |
| In this repository | **Not present, and not reconstructible from this repository.** There is no fetch script; the file was never downloaded to this machine, which is why E15 was not run. Obtain it from the source above if you want to reproduce the stress case. |
| Role | Stress case only. Its fraud-conditional quantile is degenerate at `alpha = 1e-2` (roughly 98 calibration frauds against the 99 required by the floor `(1/alpha) - 1`), which is precisely why IEEE-CIS is primary. |

### 1.3 Sparkov

Not used in Phase I. Listed here so its absence is a decision on the record rather than an
oversight.

---

## 2. Code

### 2.1 Written from the literature, not ported

The conformal and risk-control implementations in `src/hsbcfraud/conformal/` were written from
the published papers. A repository implementing similar procedures was found during the survey
but carries no licence file, which makes it non-redistributable and unsafe to adapt regardless
of intent. It was not read while implementing. Recorded as D-010 in `docs/decisions.md`, with
the date.

Every algorithm traces to a numbered entry in [REFERENCES.md](REFERENCES.md):

| Component | Source |
|---|---|
| Learn-then-Test, Hoeffding-Bentkus, Holm | CP-7, CP-8 |
| Mondrian class-conditional calibration | CP-3 |
| Class-conditional degeneracy floor `(1/alpha) - 1` | CP-4 (Ding et al., NeurIPS 2023) |
| Non-exchangeable weighted quantile with the `+inf` atom | CP-5 (Barber et al., AoS 51(2), eq. 11) |
| Geometric difference `g(K_C \|\| K_Q)` screen | QM-1 (Huang et al., 2021) |
| Matrix-product-state classifier | QM-12 (Stoudenmire and Schwab, NeurIPS 29:4799, 2016) |
| Corroborating negative result on quantum kernels for card fraud | QM-10 (Faryad, arXiv:2608.15718, August 2026) |

### 2.2 Third-party software

Pinned versions and their licences are in [NOTICE](../NOTICE). Two entries need attention
rather than a line in a table.

**`cuquantum-cu11`** arrives under NVIDIA's proprietary SLA, not an open licence. It is
therefore **not** installed by the default `make venv` target; it has its own `make venv-gpu`
and its own NOTICE section. Installing it is a deliberate act with a licence consequence, so
it is not something a first-time `make` should do silently.

E11 is the one experiment that needs it. `scripts/check_parity.py` compares four routes to the
same overlap and two of them are Aer, so reproducing every row of `parity.csv` requires
`make venv-gpu`. The script requires only the two routes the default environment provides --
the exact statevector and Braket -- and reports the others as unavailable rather than
silently comparing an implementation against itself. No scientific figure in the submission
depends on the proprietary path.

**`qiskit-aer` and `qiskit-aer-gpu-cu11`** are separate distributions that install the *same*
package. Whichever pip unpacks last wins. Resolved together, the CPU build can land second and
Aer then reports devices `('CPU',)` with no error anywhere — a GPU cross-check silently becomes
a second CPU run. The GPU wheel is reinstalled last and smoke check S1 asserts the device by
reading `nvidia-smi` rather than by asking the library.

---

## 3. Results

Every table in `results/tables/` is produced by a script in `scripts/`, committed, and hashed
in `MANIFEST.sha256.json`. Every number quoted in prose is bound to a table row in
[claims.yaml](claims.yaml) and recomputed by `scripts/check_claims.py`.

Intermediate artefacts under `results/runs/` are **not** committed and **not** hashed: they are
large, derived, and reconstructible. The manifest covers what a reviewer reads, not what the
pipeline happens to write.

**What the manifest cannot tell you.** It records that a file has not changed since the freeze.
It cannot tell you the file was produced by the code currently in the tree. That link is
`make reproduce` followed by `make check`, and it is the only claim of reproducibility this
project makes.

---

## 4. Hardware and environment

Results were produced on a single Linux host with an NVIDIA GPU of compute capability 12.0.
This matters for one non-obvious reason: the default PyPI build of torch does not carry
`sm_120`, so `make venv` installs torch from the CUDA 13.0 index **first**, before the project
and its dependencies. A resolver that pulls the default wheel later would produce an
environment where CUDA is available, tensors move to the device, and every kernel fails at
launch.

`make smoke` asserts the interpreter version, the torch build, the reachable architectures,
the Aer device, the pandas copy-on-write behaviour and the dataset identity before any
experiment runs. Three of its nine checks are optional and are reported as skipped rather than
silently passing.
