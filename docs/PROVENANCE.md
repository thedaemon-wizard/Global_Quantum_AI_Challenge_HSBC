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
| Licence | **Kaggle competition rules**, not an open licence. Section 7.A restricts use to **non-commercial purposes only**; section 7.B forbids redistribution. See the note below. |
| In this repository | **Never committed.** `.gitignore` carries an anchored `/datasets/` pattern. |
| Verified | 590,540 rows, 20,663 frauds (3.4990 %), 394 columns, `TransactionDT` monotonically non-decreasing (17,191 ties, so it is ordered but not strictly increasing — the loader asserts non-decreasing, which is what the temporal split needs), spanning exactly 182.00 days. Asserted at load time by `src/hsbcfraud/data/ieee_cis.py`, so a substituted or truncated file fails immediately rather than producing plausible numbers. |

**The licence bounds what this dataset can ever be used for, and that shapes Phase II.**
Competition rules section 7.A, read at
<https://www.kaggle.com/competitions/ieee-fraud-detection/rules> on 2026-08-30:

> "You may access and use the Competition Data for **non-commercial purposes only**, including
> for participating in the Competition and on Kaggle.com forums, and for **academic research and
> education**. The Competition Sponsor reserves the right to disqualify any participant who uses
> the Competition Data other than as permitted by the Competition Website and these Rules."

Section 7.B adds: "You agree not to transmit, duplicate, publish, redistribute or otherwise
provide or make available the Competition Data to any party not participating in the
Competition."

Phase I is research, which 7.A permits. **A commercial proof of concept is not**, and no amount
of care with this file would make it so. That is not a limitation the proposal works around --
it is a second, independent reason the Phase II sprint must run on the issuer's own
authorisation stream, alongside the scientific reasons (a settled chargeback window, a real
entity key, and a population not conditioned on someone else's incumbent). The protocol ports;
the data does not.

Practical consequence for a reviewer: every number in this submission is reproducible by anyone
who accepts the same competition rules, and none of it may be carried into a deployed system.

**The data cannot ship, so its fingerprint does.** Section 7.B forbids redistribution, which
makes the input the one part of this study a reader must fetch themselves -- and therefore the
one place where two people can silently be comparing different things. A digest is not the
data: it redistributes nothing, and it lets anyone with legitimate Kaggle access prove they
hold the same bytes *before* comparing a single result.

| Member | SHA-256 |
|---|---|
| `train_transaction.csv` | `3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642` |
| `train_identity.csv` | `b63c725d8377be90a995268d97f347c17d456b95db45807adcf9f59cd603c37c` |

Computed 2026-09-02 from the archive every committed number was produced on, and asserted on
every load by `src/hsbcfraud/data/ieee_cis.py`.

Three choices in that check are deliberate. **Members, not the archive**: Kaggle serves re-zipped
copies whose container bytes differ while the CSVs inside are identical, so hashing the zip
would reject correct data. **Only the members actually read**: `test_*.csv` and
`sample_submission.csv` are unhashed, because nothing here opens them and a reader who
downloaded only what this study needs should not be failed for it; identity is verified only
when a caller asks for it. **A hard error, not a warning**: every committed number is
conditional on this file, and a study that continues on data it cannot identify produces
results nobody can interpret, including its own author later.

This is strictly stronger than the row and fraud counts already asserted. Those catch a
re-release, a truncated download or the test split by mistake; they cannot catch a file of the
same shape with different contents -- re-encoded floats, repaired text encoding, rows reordered
within a timestamp -- and each of those moves the numbers while passing every other check
([D-127](decisions.md)).

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
| In this repository | **Not committed, and not reconstructible from this repository.** There is no fetch script and nothing here loads it. The archive was downloaded by hand on 2026-09-02 and staged as `datasets/creditcardfraud_ulb.zip`, which is **after** the analysis was frozen, so E15 still did not run and the arm stays withdrawn under amendment A9. Obtain it from the source above if you want to reproduce the stress case. |
| Role | Stress case only, and **withdrawn** by amendment A9. The figure previously quoted here -- roughly 98 calibration frauds against the 99 the floor `(1/alpha) - 1` requires -- did not follow from this protocol's own split fractions, and was withdrawn rather than recomputed because replacing it would assert a measurement never made. The degeneracy argument it was to support is made directly on IEEE-CIS instead. |

### 1.3 Sparkov

Not used in Phase I. Listed here so its absence is a decision on the record rather than an
oversight.

### 1.4 One measurement whose producer is not in the tree

`results/tables/mps_seed_spread.csv` records eight full-scale tensor-network fits at bond
dimension 16 -- four seeds at contraction width 1 and four at width 128 -- and is the measured
evidence behind [D-038](decisions.md)'s choice of the sequential contraction: at width 128 two
of four fits ended well below the trained band, at ROC AUC 0.684 and 0.564 against 0.782--0.801
for the six that trained; at width 1 none did.

The wording here used to be "never left chance", which the table does not support -- 0.684 is
not chance. It also claims a mechanism this file has no evidence for: only final values were
recorded for these eight fits, so whether they failed to learn or learned and diverged is
unknown. The two divergent fits in the *seed* sweep were re-run with their trajectories
captured and did the second ([D-121](decisions.md)); nothing licenses carrying that finding
across to these.

The variant of `run_seed_sweep.py` that produced it is not in the repository, so `make reproduce`
cannot regenerate it. It is retained rather than deleted because the results section asserts the
reason for that design choice and this file is what supports it, and it is named here rather
than left silent because a committed table nothing rewrites passes `freeze.py --check`
trivially. Reproducing it costs eight GPU fits. `tests/test_repo_hygiene.py` carries it as the
single *exemption* to the rule that every committed table has a producer.

Exemption is not the same as absence, and this file used to read as though it were. Two further
tables have no producer either -- `coverage_by_arm.csv` and `coverage_by_arm_seeds.csv` -- but
they are recorded in the same test as defects rather than exemptions, under
`TABLES_WHOSE_PRODUCER_IS_MISSING`, because unlike this one they should have had a producer all
along. They were invisible for the whole study because the guard searched the corpus for the
file name and three scripts *read* `coverage_by_arm.csv`, which a substring match cannot tell
from a write.

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
in `MANIFEST.sha256.json`. Every number quoted in the two PDFs is bound to a table row in
[claims.yaml](claims.yaml) and recomputed by `scripts/check_claims.py`; figures quoted only in
the markdown record -- protocol amendments, decision entries -- carry their measurement inline
instead, because `claims.yaml` gates what the submission asserts rather than everything the
repository has ever written down.

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
experiment runs. Three of its nine checks may skip, and only for a reason the run proves on the
host: the two Aer checks (S1, S5) when `qiskit-aer` is absent, since it lives in the
`gpu-crosscheck` extra that `make venv` does not install, and the ULB split feasibility (S7)
when that file is not extracted where the check looks. On this machine eight pass and S7 skips.
Any other skip is a failure.
