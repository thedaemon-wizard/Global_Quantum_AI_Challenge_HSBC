# Decision log

Dated decisions, the alternatives rejected, and — where it applies — the retraction. An
entry is never deleted. When a decision is superseded the original stays and carries a
pointer, because the fact that a thing was once believed is part of the record.

---

## Round 1 — Environment and pinning (2026-08-28)

### D-001 Python 3.12 through the virtual environment's interpreter, never `python3`

On this machine `python3` is 3.9, and a bare `python3.12` resolves to a system interpreter
whose `torch` has no `sm_120`. Every Makefile target therefore invokes `.venv/bin/python`
directly and the bootstrap uses `/usr/bin/python3.12` explicitly.
`tests/test_repo_hygiene.py` forbids a bare `python3` anywhere in the repository.

### D-002 `torch==2.13.0+cu130`, installed before the project

Verified against the PyTorch cu130 index on 2026-08-28: `torch-2.13.0+cu130` publishes a
`cp312-cp312-manylinux_2_28_x86_64` wheel, and 2.13.0 is the newest version present on that
index. Installed from `--index-url https://download.pytorch.org/whl/cu130` before the
project itself, because the default PyPI build does not carry `sm_120`.

Confirmed on this host: `torch.cuda.get_arch_list()` ends `['sm_90', 'sm_100', 'sm_120']`.

### D-003 Six inherited pins were stale and were updated

The sibling Airbus project supplied the engineering scaffolding and its pins were carried
over as a starting point. Checked against the PyPI JSON API on 2026-08-28, six had moved:

| Package | Inherited | Now pinned |
|---|---|---|
| `torch` | 2.12.1+cu130 | 2.13.0+cu130 |
| `numpy` | 2.5.1 | 2.5.2 |
| `scipy` | 1.18.0 | 1.18.1 |
| `matplotlib` | 3.11.0 | 3.11.1 |
| `pypdf` | 6.16.1 | 6.16.2 |
| `ruff` | 0.15.20 | 0.16.5 |

Every pin in `pyproject.toml` was additionally checked for a `cp312` x86_64 manylinux wheel
so that no dependency arrives as a source build.

### D-004 `pandas==3.0.5`, defended by a gate rather than pinned backwards

**Rejected alternative:** pin `pandas<3` (the newest 2.x is 2.3.3, 2025-09-29).

Under copy-on-write — the only mode in pandas 3 — `df['a'][mask] = v` is a silent no-op, and
`SettingWithCopyWarning` has been removed, so nothing is emitted at all. The frame is simply
unchanged and every number downstream is quietly wrong. IEEE-CIS feature engineering is
heavy on exactly this pattern.

Pinning backwards would trade a current stack for an eleven-month-old one to avoid a hazard
that is mechanically detectable. Smoke check S6 reproduces the silent no-op on this host and
verifies that the AST gate flags it, and `tests/test_repo_hygiene.py` runs the same gate
over the whole repository. This is the project's stated discipline — fail loudly, do not
tolerate a silent fallback — applied to a dependency rather than to our own code.

Measured on this host, 2026-08-28: *"pandas 3.0.5: silent no-op confirmed; AST gate flags bad
and passes good."*

The fallback, if the gate ever proves insufficient, is `pandas==2.3.3` (cp312 wheel
confirmed present).

### D-005 sQUlearn is not adopted

**Rejected alternative:** use sQUlearn 0.11.2 for scikit-learn-native quantum kernels.

Its published metadata requires `qiskit-algorithms>=0.3.0`. That package's own README states
that it is no longer officially supported by IBM. Adopting sQUlearn would therefore pull a
publicly disowned dependency into a submission whose argument is governance, along with
`qiskit-ibm-runtime`, `bayesian-optimization`, `mapomatic` and `dill`.

`qiskit-machine-learning` 0.9.1 supplies `FidelityStatevectorKernel` directly with none of
that. Smoke check S0 asserts that `qiskit-algorithms` is absent, transitively included.

### D-006 The production kernel path is `FidelityStatevectorKernel`; Aer GPU is a cross-check

Two facts pull in the same direction.

**Licensing.** `qiskit-aer-gpu-cu11` depends on `cuquantum-cu11`, so installing Aer's GPU
build brings NVIDIA's cuStateVec binary into the environment. That binary is proprietary
under the NVIDIA Software License Agreement: royalty-free and commercially usable, but
no-modification, no-derivative-works, with liability capped at USD 10.00. It cannot be
described as open source. An earlier draft of this project's plan simultaneously excluded
cuQuantum "on licence grounds" and used Aer GPU — a self-contradiction.

**Measurement.** `FidelityStatevectorKernel` evaluates the Gram matrix pairwise in Python,
so a GPU does not accelerate it. Measured on this host: 8 qubits, n = 1000 costs 9.73 s,
about 19.5 µs per pair, scaling quadratically.

So the kernel path that produces reported numbers is pure numpy with no Aer and no NVIDIA
binary, and Aer is confined to the optional `gpu-crosscheck` extra used only for the
four-way simulator agreement check. The GPU is spent where it is genuinely first-class:
XGBoost (`sm_120` is in its compiled architecture list) and the tensor-network arm.

`NOTICE` section 4 records exactly what installing the extra adds.

### D-007 Aer's GPU wheel must be installed *last*

`qiskit-aer` and `qiskit-aer-gpu-cu11` are separate distributions that both install the same
`qiskit_aer` package. Resolved together in one `pip install`, the CPU build landed second and
overwrote the GPU binary. `AerSimulator().available_devices()` then returned `('CPU',)` — no
error, no warning, and a "GPU" cross-check that would silently have been a second CPU run.

`make venv-gpu` therefore reinstalls the GPU wheel with `--force-reinstall --no-deps` after
the resolution and asserts `'GPU' in available_devices()` before returning.

### D-008 LightGBM runs on CPU

Its PyPI wheel ships the OpenCL backend only; the CUDA backend requires a source build with
`USE_CUDA=ON`, for which no Blackwell validation could be found. At 590,540 rows by ~400
float32 columns the CPU path is comfortable on this machine, and GBDT GPU speed-ups are
modest at that scale. Putting a source build on an 18-day critical path is not warranted.

### D-009 The GPU memory witness must live outside the interpreter — retraction of a check

Smoke check S1 asserts that Aer's GPU path really executes on the device. Its first
implementation polled `nvidia-smi` between `job.done()` calls; that never sampled during
execution and reported a peak of 1189 MiB, which the check correctly read as a CPU
fallback — a false alarm.

The second implementation sampled `torch.cuda.mem_get_info()` from a Python thread every
2 ms. It reported the same 1189 MiB. Instrumenting the thread showed why: it received
**exactly two ticks across a 2.71 s run**, because Aer holds the GIL for the duration of the
C++ execution. An in-process Python witness cannot observe an Aer allocation.

The check now starts a streaming `nvidia-smi` subprocess before the run and terminates it
after, and additionally reads Aer's own result metadata, which names the device it
dispatched to. Both witnesses are required: metadata is authoritative but self-reported,
memory is external but samplable only from outside the process.

Measured on this host, 2026-08-28: *"28q statevector in 2.75s; Aer reports device=GPU
cuStateVec=True; device memory 629 → 5565 MiB (+4936, 4096 expected)."*

This entry is kept in full, including the two wrong implementations, because a reader
evaluating the GPU claim should be able to see how it was established.

### D-010 Conformal and statistics modules are written from the primary literature

**Rejected alternative:** port the corresponding modules from the QIntern 2026 Team A
repository, which implements comparable procedures and which the author co-wrote.

That repository publishes **no licence file** and has multiple contributors. Absent a licence
grant the material is all-rights-reserved and cannot be redistributed under Apache-2.0.
Independently, Challenge Terms and Conditions section 3 makes the absence of third-party
proprietary material a representation and warranty, so this is a contractual matter and not
only a copyright one.

The affected modules are short and implement published formulas, so they are written from
the sources instead, with each docstring naming the paper whose result it implements
(`NOTICE` section 2 lists the mapping). Describing the author's own contribution to that
project in the team-capability section remains unaffected; what is prohibited is copying
code and reproducing unpublished co-authored figures.

### D-011 Smoke checks may not be skipped except for three stated reasons

A skipped check reads like a passing one at a glance. `scripts/smoke.py` therefore treats a
skip as a failure unless the check is in an explicit allow-list: S1 and S5 exercise Aer,
which the default install deliberately omits (D-006), and S7 needs a dataset that
`make data` fetches separately. Any other skip fails the run.
