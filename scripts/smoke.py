#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Environment assertions that must hold before any number in this study is trusted.

Nine checks, each one written because the corresponding assumption is either known to have
been wrong recently or is unverifiable from package metadata alone.  They run before the
campaign, not after, because every one of them can silently corrupt a result rather than
raise: a GPU that quietly falls back to CPU still returns an answer, a chained pandas
assignment under copy-on-write returns no error at all, and an off-by-one in a conformal
quantile inflates the abstention rate without ever failing a test.

The house rule is that a skip is a failure unless the reason is provably true on this host.
Three checks may skip, and only for the reasons their own docstrings state: S1 and S5 both
exercise Aer, which lives in the ``gpu-crosscheck`` extra that ``make venv`` deliberately
does not install (it pulls a proprietary NVIDIA binary; see NOTICE section 4), and S7 needs
the ULB dataset, which is not obtainable from this repository.  The set is ``OPTIONAL``
below.  Nothing else may be skipped, and there is no flag that relaxes any of this: this
docstring described a ``--allow-optional`` that the parser never accepted, and named one
optional check where the code has always had three.

Usage
-----
    .venv/bin/python scripts/smoke.py            # all nine; S1, S5, S7 skip if unavailable
    .venv/bin/python scripts/smoke.py --only S3  # one check, for iterating on a failure
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata as md
import json
import os
import sys
import textwrap
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from hsbcfraud.paths import display_path

REPO = Path(__file__).resolve().parents[1]

# The only checks permitted to skip, and only for reasons stated in the docstrings below.
# S1 and S5 both exercise Aer, which lives in the `gpu-crosscheck` extra that `make venv`
# deliberately does not install (it pulls a proprietary NVIDIA binary; see NOTICE section
# 4).  S7 needs the ULB dataset, which is not obtainable from this repository.  Nothing
# else may skip.
OPTIONAL = {"S1", "S5", "S7"}


class SmokeFailure(AssertionError):
    """A check that did not hold.  The message is the report."""


class SmokeSkip(Exception):
    """A check whose precondition is provably absent on this host."""


@dataclass(frozen=True)
class Check:
    ident: str
    subject: str
    run: Callable[[], str]


# --------------------------------------------------------------------------------------
# S0  Dependency resolution
# --------------------------------------------------------------------------------------
def s0_dependency_resolution() -> str:
    """Every pin resolves, and no unsupported package arrived transitively.

    The specific hazard is ``qiskit-algorithms``.  Its own README states that it is no
    longer officially supported by IBM, and it is easy to acquire without asking for it:
    sQUlearn, for instance, hard-requires it.  A dependency an upstream maintainer has
    publicly disowned has no place in a submission whose argument is governance, so its
    absence is asserted rather than assumed.
    """
    required = {
        "qiskit": "2.5.2",
        "qiskit-machine-learning": "0.9.1",
        "amazon-braket-sdk": "1.126.0",
        "amazon-braket-default-simulator": "1.40.1",
        "mapie": "1.5.0",
        "crepes": "0.9.1",
        "xgboost": "3.4.1",
        "lightgbm": "4.7.0",
        "scikit-learn": "1.9.0",
        "shap": "0.52.0",
        "numpy": "2.5.2",
        "scipy": "1.18.1",
        "pandas": "3.0.5",
        "matplotlib": "3.11.1",
        "pydantic": "2.13.4",
        "kagglehub": "1.0.2",
    }
    wrong = []
    for name, want in required.items():
        try:
            got = md.version(name)
        except md.PackageNotFoundError:
            wrong.append(f"{name}: not installed (want {want})")
            continue
        if got != want:
            wrong.append(f"{name}: {got} installed, {want} pinned")
    if wrong:
        raise SmokeFailure("dependency pins not honoured:\n  " + "\n  ".join(wrong))

    installed = {d.metadata["Name"].lower() for d in md.distributions() if d.metadata["Name"]}
    for banned, why in [
        ("qiskit-algorithms", "IBM states it is no longer officially supported"),
        ("torchcp", "LGPL-3.0; excluded on SBOM-review grounds, see docs/decisions.md"),
    ]:
        if banned in installed:
            raise SmokeFailure(f"{banned} is installed but must not be: {why}")

    torch_v = md.version("torch")
    if "+cu130" not in torch_v:
        raise SmokeFailure(
            f"torch is {torch_v}; the cu130 build is required for sm_120. "
            "Reinstall from https://download.pytorch.org/whl/cu130"
        )
    return f"{len(required)} pins exact; torch {torch_v}; no disowned dependency present"


# --------------------------------------------------------------------------------------
# S1  Aer GPU on Blackwell (optional)
# --------------------------------------------------------------------------------------
def s1_aer_gpu() -> str:
    """The GPU statevector path really executes on the device.

    Published analysis of the Aer wheels' compiled architecture lists says sm_120 is
    absent, which would make this impossible.  It nevertheless works here, because the
    CUDA 11.8 PTX in the cu11 wheel is forward-JITted by the 580 driver.  Since the
    literature and the machine disagree, the machine is asked directly -- and asked in a
    way a silent CPU fallback cannot pass, by allocating a statevector too large to be
    mistaken for anything else and reading the device's own memory counter.
    """
    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise SmokeSkip(f"qiskit-aer not installed (gpu-crosscheck extra): {exc}") from exc

    if "GPU" not in AerSimulator().available_devices():
        raise SmokeSkip("Aer reports no GPU device on this host")

    import shutil
    import subprocess

    from qiskit import QuantumCircuit, transpile

    if shutil.which("nvidia-smi") is None:
        raise SmokeSkip("nvidia-smi is not on PATH, so device memory cannot be witnessed")

    n = 28  # 2**28 complex128 = 4.3 GB: far beyond anything a CPU fallback would allocate
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.save_statevector()

    sim = AerSimulator(method="statevector", device="GPU", cuStateVec_enable=True)
    tqc = transpile(qc, sim)

    # Two independent witnesses, because either alone is weak.  Aer's result metadata names
    # the device it actually dispatched to, which is authoritative but self-reported.
    # Device memory is external corroboration -- and it must be collected from OUTSIDE this
    # interpreter.  Aer holds the GIL for the duration of the C++ execution: a Python
    # sampler thread polling every 2 ms was measured to receive exactly two ticks across a
    # 2.7 s run, reporting a peak of 1188 MiB for an allocation an external 1 Hz sampler had
    # already observed at 5399 MiB.  So a streaming nvidia-smi runs alongside, with no
    # per-sample spawn cost.
    baseline = _device_memory_mib()
    sampler = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "100"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    start = time.perf_counter()
    try:
        result = sim.run(tqc).result()
    finally:
        elapsed = time.perf_counter() - start
        sampler.terminate()
        try:
            stream, _ = sampler.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            sampler.kill()
            stream = ""
    peak = max(
        (int(line) for line in stream.split() if line.isdigit()),
        default=baseline,
    )

    meta = result.results[0].metadata
    device = meta.get("device")
    if device != "GPU":
        raise SmokeFailure(
            f"Aer dispatched to {device!r}, not 'GPU'. A silent fallback occurred: every "
            "'GPU' timing downstream would actually be a CPU timing."
        )
    if not meta.get("cuStateVec_enable"):
        raise SmokeFailure(
            "Aer ran on GPU but without cuStateVec; the requested path was not taken"
        )

    rise = peak - baseline
    expected_mib = (2**n * 16) / (1024 * 1024)  # complex128 statevector
    if rise < 0.5 * expected_mib:
        raise SmokeFailure(
            f"Aer reported device=GPU but device memory rose by only {rise} MiB against "
            f"the {expected_mib:.0f} MiB a {n}-qubit complex128 statevector needs"
        )
    return (
        f"{n}q statevector in {elapsed:.2f}s; Aer reports device={device} cuStateVec=True; "
        f"device memory {baseline} -> {peak} MiB (+{rise}, {expected_mib:.0f} expected)"
    )


def _device_memory_mib() -> int:
    """Currently used device memory, in MiB, as the driver reports it."""
    import subprocess

    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip().splitlines()[0])


# --------------------------------------------------------------------------------------
# S2  XGBoost on sm_120
# --------------------------------------------------------------------------------------
def s2_xgboost_cuda() -> str:
    """``device='cuda'`` trains, and agrees with the CPU fit.

    XGBoost's CUDA 13 build lists compute capability 120, so sm_120 should be first-class.
    "Should be" is not a measurement, and the failure mode on Blackwell across the wider
    ecosystem has been a runtime "no kernel image is available" rather than an install
    error.  Agreement with the CPU fit is checked too: a GPU that runs but computes
    something different is worse than one that refuses.
    """
    import xgboost as xgb
    from sklearn.datasets import make_classification
    from sklearn.metrics import roc_auc_score

    x, y = make_classification(
        n_samples=20000, n_features=32, n_informative=12, weights=[0.97], random_state=0
    )
    params = dict(n_estimators=64, max_depth=5, learning_rate=0.2, tree_method="hist", verbosity=0)
    scores = {}
    for device in ("cuda", "cpu"):
        model = xgb.XGBClassifier(device=device, random_state=0, **params)
        model.fit(x, y)
        scores[device] = roc_auc_score(y, model.predict_proba(x)[:, 1])

    gap = abs(scores["cuda"] - scores["cpu"])
    if gap > 5e-3:
        raise SmokeFailure(
            f"GPU and CPU fits disagree: AUC {scores['cuda']:.6f} vs {scores['cpu']:.6f} "
            f"(gap {gap:.2e}); the GPU path is not computing the same model"
        )
    return f"xgboost {xgb.__version__} device=cuda AUC {scores['cuda']:.4f}, CPU gap {gap:.2e}"


# --------------------------------------------------------------------------------------
# S3  SHAP against this XGBoost
# --------------------------------------------------------------------------------------
def s3_shap_treeexplainer() -> str:
    """TreeExplainer runs on an XGBoost 3.4 model and satisfies local accuracy.

    XGBoost 3.0 changed ``base_score`` from a float to a JSON list, which broke SHAP's
    tree loader for roughly eight months.  SHAP 0.52.0 carries the fix but predates
    XGBoost 3.4.1, so the pair has not necessarily been exercised together upstream.  The
    challenge statement names SHAP explicitly under explainability, so a late discovery
    here would cost a scored criterion.

    Local accuracy -- that attributions plus the base value reconstruct the margin -- is
    checked rather than assumed, because a loader that half-works can return plausible
    numbers that do not sum.
    """
    import numpy as np
    import shap
    import xgboost as xgb
    from sklearn.datasets import make_classification

    x, y = make_classification(
        n_samples=4000, n_features=16, n_informative=8, weights=[0.97], random_state=1
    )
    model = xgb.XGBClassifier(
        n_estimators=48, max_depth=4, tree_method="hist", verbosity=0, random_state=1
    )
    model.fit(x, y)

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x[:256])
    margin = model.predict(x[:256], output_margin=True)
    reconstructed = values.sum(axis=1) + explainer.expected_value
    gap = float(np.abs(reconstructed - margin).max())
    if gap > 1e-3:
        raise SmokeFailure(
            f"SHAP local accuracy violated: max |sum(phi) + base - margin| = {gap:.3e}. "
            "Attributions do not reconstruct the model output."
        )
    return f"shap {shap.__version__} x xgboost {xgb.__version__}, max efficiency gap {gap:.2e}"


# --------------------------------------------------------------------------------------
# S4  Braket LocalSimulator without AWS
# --------------------------------------------------------------------------------------
def s4_braket_local_offline() -> str:
    """``LocalSimulator`` runs with no credentials, no config file and no region.

    The challenge asks participants to use Amazon Braket.  The local simulator satisfies
    that at zero cost, but only if it genuinely runs in-process -- and the SDK hard-depends
    on boto3, so "it does not call AWS" is a claim worth testing rather than repeating.
    Every AWS environment variable is cleared and HOME is redirected away from any
    ``~/.aws`` before the import, so credential discovery has nothing to find.
    """
    import tempfile

    saved = {k: v for k, v in os.environ.items() if k.startswith("AWS_") or k == "HOME"}
    try:
        for key in list(os.environ):
            if key.startswith("AWS_"):
                del os.environ[key]
        with tempfile.TemporaryDirectory() as empty_home:
            os.environ["HOME"] = empty_home
            from braket.circuits import Circuit
            from braket.devices import LocalSimulator

            device = LocalSimulator("braket_sv")
            circuit = Circuit().h(0).cnot(0, 1)
            result = device.run(circuit, shots=256).result()
            counts = result.measurement_counts
    finally:
        for key in [k for k in os.environ if k.startswith("AWS_")]:
            del os.environ[key]
        os.environ.update(saved)

    if set(counts) - {"00", "11"}:
        raise SmokeFailure(f"Bell state produced impossible outcomes: {dict(counts)}")
    return f"braket_sv ran offline with no credentials; Bell counts {dict(counts)}"


# --------------------------------------------------------------------------------------
# S5  Aer against this Qiskit
# --------------------------------------------------------------------------------------
def s5_aer_qiskit_pair() -> str:
    """Aer 0.17.2 actually works against Qiskit 2.5.x.

    Aer declares an unbounded ``qiskit>=1.1.0`` while its release notes claim validation
    only to Qiskit 2.1, and no Aer release has been cut in roughly eleven months.  pip
    will therefore happily assemble a combination nobody upstream has tested.  Since Aer
    is used only for the cross-simulator agreement check, a failure here is survivable --
    but it must be known, not discovered mid-campaign.
    """
    try:
        import qiskit_aer
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise SmokeSkip(f"qiskit-aer not installed (gpu-crosscheck extra): {exc}") from exc

    import numpy as np
    import qiskit
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector

    qc = QuantumCircuit(4)
    qc.h(range(4))
    for i in range(3):
        qc.cx(i, i + 1)
    qc.rz(0.37, 2)

    exact = Statevector(qc).data
    aer = AerSimulator(method="statevector")
    probe = qc.copy()
    probe.save_statevector()
    got = np.asarray(aer.run(transpile(probe, aer)).result().get_statevector())

    overlap = abs(np.vdot(exact, got))
    if abs(overlap - 1.0) > 1e-9:
        raise SmokeFailure(
            f"Aer statevector disagrees with qiskit.quantum_info: |<exact|aer>| = {overlap:.12f}"
        )
    return f"qiskit-aer {qiskit_aer.__version__} agrees with qiskit {qiskit.__version__} to 1e-9"


# --------------------------------------------------------------------------------------
# S6  pandas chained assignment
# --------------------------------------------------------------------------------------
def s6_pandas_chained_assignment() -> str:
    """Confirm the pandas 3 hazard is real, and that the AST gate catches it.

    Under copy-on-write -- the only mode in pandas 3 -- ``df['a'][mask] = v`` is a no-op.
    ``SettingWithCopyWarning`` was removed, so nothing is emitted: the frame is simply
    unchanged and every downstream number is quietly wrong.  This project stays on the
    current pandas and defends with a gate instead of pinning backwards, so the gate is
    tested here on a known-bad fragment.

    If this check fails, the fallback recorded in the plan is ``pandas==2.3.3``.
    """
    import warnings

    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    before = df["a"].tolist()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # chained-assignment-exempt: this IS the fixture the gate must catch
            df["a"][df["b"] > 20] = 999
        except Exception:
            silent_noop = False
        else:
            silent_noop = df["a"].tolist() == before

    bad = "df['a'][df['b'] > 20] = 999"
    if not _has_chained_assignment(bad):
        raise SmokeFailure(f"the AST gate failed to flag a known-bad fragment: {bad}")
    good = "df.loc[df['b'] > 20, 'a'] = 999"
    if _has_chained_assignment(good):
        raise SmokeFailure(f"the AST gate flagged correct code: {good}")

    verdict = (
        "silent no-op confirmed" if silent_noop else "pandas raised rather than silently passing"
    )
    return f"pandas {pd.__version__}: {verdict}; AST gate flags bad and passes good"


def _has_chained_assignment(source: str) -> bool:
    """Detect ``frame[...][...] = value``, the pattern that silently does nothing.

    Assigning to a subscript whose own value is itself a subscript means the first
    subscript produced a temporary, and under copy-on-write the write lands on that
    temporary.  Chained ``.loc``/``.iloc`` accessors are the same shape and are caught too.
    """
    for node in ast.walk(ast.parse(textwrap.dedent(source))):
        if not isinstance(node, ast.Assign | ast.AugAssign):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Subscript):
                inner = target.value
                if isinstance(inner, ast.Subscript):
                    return True
                if (
                    isinstance(inner, ast.Attribute)
                    and inner.attr in {"loc", "iloc", "at", "iat"}
                    and isinstance(inner.value, ast.Subscript)
                ):
                    return True
    return False


# --------------------------------------------------------------------------------------
# S7  ULB temporal split feasibility
# --------------------------------------------------------------------------------------
def s7_ulb_temporal_feasible() -> str:
    """Does ULB's two-day window leave enough fraud in each block to calibrate?

    The Mastercard/Oxford Quantum Circuits team abandoned a temporal split on this dataset
    because folds came out with zero fraud.  If that reproduces here, ULB is reported as a
    random-CV stress case and the reason is stated -- which is a stronger position than
    quietly using random CV and hoping the question is not asked.

    Skipped, not failed, when the file is absent: this check is about the data, and
    The dataset is not fetched by anything here; see docs/PROVENANCE.md section 1.2.
    """
    import pandas as pd

    path = _find_ulb()
    if path is None:
        raise SmokeSkip(
            "ULB creditcard.csv not present, and no script here fetches it "
            "(docs/PROVENANCE.md 1.2). E15 was not run for this reason."
        )

    df = pd.read_csv(path, usecols=["Time", "Class"]).sort_values("Time", kind="stable")
    df = df.drop_duplicates()
    n = len(df)
    cuts = [0, int(0.60 * n), int(0.70 * n), int(0.80 * n), n]
    names = ["train", "band", "cal", "test"]
    counts = {
        name: int(df.iloc[cuts[i] : cuts[i + 1]]["Class"].sum()) for i, name in enumerate(names)
    }
    verdict = "feasible" if min(counts.values()) >= 20 else "NOT feasible"
    return f"ULB temporal blocks fraud counts {counts} -> {verdict} (reported either way)"


def _find_ulb() -> Path | None:
    for candidate in (
        REPO / "datasets" / "creditcard.csv",
        REPO / "datasets" / "ulb" / "creditcard.csv",
    ):
        if candidate.exists():
            return candidate
    return None


# --------------------------------------------------------------------------------------
# S8  MAPIE's conformal quantile
# --------------------------------------------------------------------------------------
def s8_mapie_quantile() -> str:
    """MAPIE's split-conformal order statistic against the textbook definition.

    Releases up to and including 1.5.0 select the next-higher order statistic, which makes
    prediction sets conservative.  Conservative is safe for coverage and wrong for us: the
    certified abstention budget would be inflated by an artefact rather than by the data.
    The reference implementation here is the definition itself, so the direction and size
    of any discrepancy are measured rather than inferred from a changelog.
    """
    import math

    import numpy as np

    rng = np.random.default_rng(0)
    disagreements = []
    for n in (5, 20, 99, 100, 200, 999):
        scores = np.sort(rng.uniform(size=n))
        for alpha in (1e-2, 5e-3, 2e-3, 1e-3, 0.1):
            k = math.ceil((1.0 - alpha) * (n + 1))
            reference = math.inf if k > n else float(scores[k - 1])
            # MAPIE's released form: quantile of level ((n+1)(1-alpha))/n, method="higher".
            level = ((n + 1) * (1.0 - alpha)) / n
            mapie_like = (
                math.inf
                if level > 1.0
                else float(np.quantile(scores, level, method="higher"))
            )
            both_infinite = math.isinf(reference) and math.isinf(mapie_like)
            if not both_infinite and abs(reference - mapie_like) > 1e-12:
                disagreements.append((n, alpha, reference, mapie_like))

    conservative = all(m > r for _, _, r, m in disagreements)
    note = "all conservative" if conservative else "SOME ANTI-CONSERVATIVE -- investigate"
    if disagreements and not conservative:
        raise SmokeFailure(
            f"MAPIE-form quantile is anti-conservative in {len(disagreements)} cases; "
            "an under-covering certificate is not survivable"
        )
    return (
        f"{len(disagreements)} disagreements with ceil((n+1)(1-alpha)) over the pinned "
        f"alpha grid, {note}; conformal/split.py uses the reference form"
    )


CHECKS: tuple[Check, ...] = (
    Check("S0", "dependency resolution", s0_dependency_resolution),
    Check("S1", "Aer GPU on sm_120", s1_aer_gpu),
    Check("S2", "XGBoost device=cuda", s2_xgboost_cuda),
    Check("S3", "SHAP TreeExplainer", s3_shap_treeexplainer),
    Check("S4", "Braket LocalSimulator offline", s4_braket_local_offline),
    Check("S5", "Aer against Qiskit 2.5", s5_aer_qiskit_pair),
    Check("S6", "pandas chained assignment", s6_pandas_chained_assignment),
    Check("S7", "ULB temporal split feasibility", s7_ulb_temporal_feasible),
    Check("S8", "MAPIE conformal quantile", s8_mapie_quantile),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", action="append", help="run only these check ids, e.g. --only S3")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "results" / "tables" / "smoke.json",
        help="where to record the outcome",
    )
    args = parser.parse_args(argv)

    selected = [c for c in CHECKS if not args.only or c.ident in set(args.only)]
    if args.only and not selected:
        parser.error(f"no check matches {args.only}; known ids: {[c.ident for c in CHECKS]}")

    record: dict[str, dict[str, str]] = {}
    failed: list[str] = []
    for check in selected:
        start = time.perf_counter()
        try:
            detail = check.run()
        except SmokeSkip as exc:
            if check.ident not in OPTIONAL:
                failed.append(check.ident)
                status, detail = "FAIL", f"skipped, but {check.ident} may not be skipped: {exc}"
            else:
                status, detail = "SKIP", str(exc)
        except SmokeFailure as exc:
            failed.append(check.ident)
            status, detail = "FAIL", str(exc)
        except Exception as exc:
            failed.append(check.ident)
            status, detail = "FAIL", f"{type(exc).__name__}: {exc}"
        else:
            status = "PASS"
        elapsed = time.perf_counter() - start
        record[check.ident] = {"subject": check.subject, "status": status, "detail": detail}
        marker = {"PASS": "ok  ", "SKIP": "skip", "FAIL": "FAIL"}[status]
        print(f"[{marker}] {check.ident}  {check.subject}  ({elapsed:.1f}s)")
        for line in str(detail).splitlines():
            print(f"         {line}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if failed:
        print(f"\n{len(failed)} check(s) failed: {', '.join(failed)}", file=sys.stderr)
        print(
            "Fix the environment or change the design; do not proceed past a red smoke.",
            file=sys.stderr,
        )
        return 1
    print(f"\nAll {len(selected)} check(s) accounted for. Record: {display_path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
