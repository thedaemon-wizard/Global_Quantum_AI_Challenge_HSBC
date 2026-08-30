# Environment and measured cost

The machine every number in this submission was produced on, and what each stage actually
cost. Recorded because the Phase 1 assessment scores feasibility on whether resource
requirements are realistic, and a resource estimate is worth more when it is a measurement.

Nothing here is extrapolated: every figure is read from the result table the run wrote. It is
*transcribed* from those tables rather than generated, which is a real weakness — the split
table below was wrong for three of its four rows until 2026-08-30, because a Markdown table is
not a bound claim and no gate can see it. Figures that also appear in the proposal are bound
there; treat `results/tables/` as authoritative wherever the two differ.

---

## 1. Hardware

| | |
|---|---|
| CPU | Intel Core i5-13600K, 13th generation |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB, driver 580.105.08 |
| Memory | 125 GB |
| OS | AlmaLinux 9.7 |

One workstation, one GPU. No cluster, no cloud, and no QPU in the loop — both quantum arms
run on simulators, which the challenge statement permits explicitly and does not penalise.

## 2. Software

| | |
|---|---|
| Python | 3.12.11, in `.venv` |
| PyTorch | 2.13.0+cu130, CUDA 13.0 |
| XGBoost | 3.4.1 |
| Qiskit | 2.5.2 |
| scikit-learn | 1.9.0 |
| NumPy / pandas / SciPy | 2.5.2 / 3.0.5 / 1.18.1 |

Versions are pinned in `pyproject.toml`, with one deliberate exception: **PyTorch is pinned in
the `Makefile`**, at `torch==2.13.0` from the CUDA 13.0 index. The default PyPI build carries
no `sm_120` kernels, so it imports cleanly and then cannot use this GPU. Declaring it as an
ordinary dependency would let pip resolve it from PyPI and undo that -- and a torch that
imports but cannot use the GPU is a worse failure than one that is absent, because it is
silent. It is declared as an optional `torch` extra so the requirement appears in the package
metadata, and `make venv` installs it from the correct index first. Installing the extra
directly from PyPI reproduces the problem it exists to document.

`qiskit-aer` and `qiskit-aer-gpu-cu11` are an optional extra, installed by `make venv-gpu`.
They bring NVIDIA's proprietary cuStateVec binary into the environment, which is a licence
consequence rather than a convenience — see [NOTICE](../NOTICE). Only the E11 parity
cross-check uses them; no scientific figure depends on that path.

## 3. Data volume

The four blocks of the temporal arm, from `splits.csv`:

| Block | Rows | Fraud | Days | Role |
|---|---|---|---|---|
| $D_{\mathrm{train}}$ | 356,216 | 12,039 | 0–100 | fits the classical scorer |
| $D_{\mathrm{band}}$ | 58,326 | 2,561 | 101–119 | the only data that may set band edges or any threshold |
| $D_{\mathrm{cal}}$ | 60,464 | 2,121 | 120–140 | certifies the composite rule |
| $D_{\mathrm{test}}$ | 115,534 | 3,942 | 141–181 | read once |

The four sum to 590,540, which is the dataset. **Corrected 2026-08-30**: this table previously
read 363,817 / 59,054 / 60,464 / 118,108 — three of the four taken from the wrong arms
(`card_disjoint,train` and `stratified,band`/`test`), summing to 601,443, or 10,903 rows that do
not exist. Only $D_{\mathrm{cal}}$ was right. It also disagreed with the proposal, which prints
356,216 from the claim gate; the gate never saw this file because a Markdown table is not a
bound claim.

## 4. Measured cost per stage

| Stage | Scale | Wall clock | Device |
|---|---|---|---|
| Tuned GBDT, full feature set | 431 features, 356,216 train rows | **15.9 – 18.8 s** | GPU |
| Tuned GBDT, in-band | 8 features, 2,916 band rows | **0.28 s** | GPU |
| Tensor network, in-band | 8 sites, $\chi \in \{4, 8, 16, 32\}$ | **0.69 – 1.91 s** per fit | GPU |
| Tensor network, full scale | 431 sites, 356,216 rows, 30 epochs | **2,747 – 3,145 s** per fit | GPU |
| Full-scale seed sweep | 16 fits, 4 bond dimensions $\times$ 4 seeds | **13.0 GPU-hours** total | GPU |
| Quantum kernel screens | 120 configurations, 300 stratified rows each | **0.041 – 0.708 s** per Gram matrix, **17.5 s** total | CPU |
| Conformal calibration | a sort and a grid scan over $D_{\mathrm{cal}}$ | seconds | CPU |

**The shape of this table is the feasibility argument.** The classical core and the certificate
are cheap — the component that carries the guarantee costs seconds. The expensive item is the
quantum-inspired arm, and its cost is dominated by chain length rather than by capacity:
per-step cost is flat across the bond dimension against the 64-fold arithmetic span a $\chi^2$
model predicts, because a 431-site chain is bound by the launch overhead of its sequential
contractions. So a resource estimate for a larger deployment scales with the feature count and
not with the ansatz.

## 5. Inference latency, against the authorisation budget

From `latency.csv`. The challenge statement puts the whole authorisation flow at 100–300 ms with
about 130 ms of network response, leaving roughly **170 ms** issuer-side, and names *latency vs.
complexity* as a bottleneck. Per single authorisation, batch size 1:

| Component | Serving profile | p50 | p99 |
|---|---|---|---|
| Classical scorer, 431 features | 1-thread CPU | **0.084 ms** | 0.320 ms |
| In-band re-scorer, 8 features | 1-thread CPU | 0.074 ms | 0.286 ms |
| Quantum kernel (screened out) | 1-thread CPU | **96.577 ms** | 129.080 ms |
| Classical scorer, 431 features | all-core CPU | 19.854 ms | 28.763 ms |
| Classical scorer, 431 features | GPU | 19.456 ms | 37.233 ms |

**The serving profile matters more than the model, by three orders of magnitude.** XGBoost
defaults its thread count to the core count. On a one-row payload the OpenMP barrier costs about
19 ms on these 20 cores while the prediction it synchronises costs about 0.05 ms:

| `nthread` | 1 | 4 | 20 (default) |
|---|---|---|---|
| single-row p50 | 0.051 ms | 0.051 ms | 19.33 ms |

The all-core and GPU rows are kept in the table as the documented trap, not as a recommendation.
A per-request scorer wants one thread — concurrency in an authorisation path comes from serving
many transactions at once, not from splitting one across cores. Reading the default figure as a
model cost was the error [D-063](decisions.md) records.

**What this does and does not bound.** It bounds *model* cost on this workstation, warm, in one
process, through `Booster.inplace_predict`. It is not production latency: a deployed scorer sits
behind a service boundary with serialisation, feature retrieval and network hops that dominate
everything above, on the issuer's hardware rather than this one.

**The feasibility consequence.** The classical core is not the latency risk — it uses 0.19 % of
the residual budget at its tail. The kernel is: at 96.6 ms it costs about 1,148× a classical
score, and its tail takes 75.9 % of the residual budget. It fits, and only because the band
rations it to a few percent of traffic. Batching helps the classical path by a further order of
magnitude (0.0038 ms per transaction at batch 1024) but is unavailable to a per-authorisation
decision, which is why batch 1 is the figure quoted.

## 6. Circuit structure, for near-term hardware feasibility

From `circuits.csv`, measured by decomposing every screened encoding to a portable
`rz, sx, x, cx` basis with all-to-all connectivity:

| | Range across the 20 screened encodings |
|---|---|
| Qubits | 1 – 8 |
| Logical depth | 4 – 31 |
| Transpiled depth | 7 – 35 |
| Two-qubit gates | 0 – 28 |

These are shallow. A device with limited connectivity would be deeper, never shallower, and no
figure is quoted against a named coupling map because no hardware run informs any number here.
What stopped the kernel arm was the a-priori screens, not the hardware.

## 7. Reproduction cost

`make reproduce` runs the pipeline end to end **except** the full-scale tensor-network sweep,
which is 13 GPU-hours against seconds for everything else and is therefore its own target,
`make seedsweep`. Everything `reproduce` does run is under a minute of compute plus the data
load. A reviewer wanting to check the certificate rather than the tensor-network arm can run
`make baseline conformal` and be done in the time it takes to read the parquet files.

`make check` — the gates, not the science — runs in seconds and is what verifies that every
number in the documents still resolves to the table it came from.
