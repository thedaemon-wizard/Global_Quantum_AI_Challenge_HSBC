# SPDX-License-Identifier: Apache-2.0
"""A-priori screens that decide whether a quantum kernel can help, before spending compute.

These are gates, not diagnostics.  The pre-registration commits to running the quantum arm
only on feature maps that pass, and to reporting a rejection as the result if none does.

Why screen at all
-----------------
Two independent lines of evidence say that an unscreened fidelity kernel on tabular data is
very unlikely to help, and that the reason is visible in the Gram matrix before any
classifier is fitted.

Thanasilp, Wang, Cerezo and Holmes (*Nature Communications* 15:5200, 2024) show exponential
concentration: as qubit count grows the off-diagonal kernel entries collapse toward a
constant, and the model's predictions become independent of its input.  Kakavand, Strohmeyer
and Schlotter (arXiv:2604.18837, 2026) measured the consequence across nine tabular datasets
and 8,400 SVM fits -- no significant quantum-classical difference anywhere -- and located it
in the eigenspectrum: a usable kernel sits in a middle range of effective rank, while quantum
kernels land either near-uniform (concentrated) or near-rank-one.

The escape from concentration is bandwidth tuning.  But Slattery et al. (*Physical Review A*
107:062417, 2023) and Florez-Ablan, Roth and Schnabel (arXiv:2503.05602, 2025) show that the
bandwidth restoring conditioning also drives the kernel toward a radial basis function.  So
conditioning alone is not evidence of anything: a perfectly conditioned quantum kernel that
is an RBF kernel in disguise offers nothing a classical pipeline does not already have.

The two-dimensional criterion
-----------------------------
A map must be **simultaneously** well conditioned and distinguishable from an RBF.  Neither
axis alone is informative, which is why they are evaluated together:

* effective-rank ratio inside a stipulated band -- see the note below on where the classical
  kernels on this data actually sit;
* correlation with the best-fitting RBF kernel below a threshold.

Measured on isotropic uniform data during design, no bandwidth satisfied both at once for an
8-qubit ZZ map: at natural bandwidth the effective-rank ratio was 0.979 with RBF correlation
0.032, and tightening the bandwidth to reach the classical range drove the correlation
monotonically to 0.712.  Whether real fraud features behave the same way is what this module
answers on the band block.

Huang's geometric difference is computed and reported alongside these two, but it is a
diagnostic and not a gate: no ``passes_*`` field consults it.  It was pre-registered as a gate
and never implemented as one; see docs/protocol.md Amendment A6, which also records that the
omission runs in the direction that flatters the rejection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ScreenResult",
    "effective_rank_ratio",
    "geometric_difference",
    "rbf_correlation",
    "screen_kernel",
]


def effective_rank_ratio(gram: np.ndarray) -> float:
    """Exponential of the spectral entropy, normalised by matrix size.

    A value near 1 means the eigenvalues are near-uniform: every direction carries the same
    weight, which is what a concentrated kernel looks like and which leaves a classifier
    nothing to key on.  A value near 0 means one direction dominates, which is a rank-one
    kernel and equally useless.  The gate's band is stipulated at [0.01, 0.35] rather than
    measured: on the 300 screening rows the RBF at the natural bandwidth (gamma = 1/8) sits at
    0.0039 and would itself be rejected, and the RBF family only enters the band above
    gamma = 3.2.  Every candidate that passes conditioning is therefore better conditioned
    than the classical reference, not merely comparable to it.
    """
    eig = np.linalg.eigvalsh(np.asarray(gram, dtype=float))
    eig = np.clip(eig, 0.0, None)
    total = eig.sum()
    if total <= 0:
        return float("nan")
    p = eig / total
    p = p[p > 0]
    entropy = float(-(p * np.log(p)).sum())
    return float(np.exp(entropy) / gram.shape[0])


def top_eigenvalue_share(gram: np.ndarray) -> float:
    """Share of spectral mass in the leading eigenvalue."""
    eig = np.linalg.eigvalsh(np.asarray(gram, dtype=float))
    eig = np.clip(eig, 0.0, None)
    total = eig.sum()
    return float(eig[-1] / total) if total > 0 else float("nan")


def rbf_correlation(
    gram: np.ndarray, x: np.ndarray, gammas: np.ndarray | None = None
) -> tuple[float, float]:
    """Largest absolute off-diagonal correlation with any RBF kernel on the same points.

    Returns ``(correlation, gamma)``.  The search over ``gamma`` is what makes this a fair
    test: a quantum kernel is only distinguishable from the RBF family if it fails to match
    *every* member, not merely the one member somebody happened to try.

    Only off-diagonal entries are compared.  Both kernels have unit diagonal by construction,
    so including it would add a block of perfectly correlated values and inflate the result.
    """
    from sklearn.metrics.pairwise import rbf_kernel

    if gammas is None:
        gammas = np.logspace(-3.0, 1.0, 25)
    off = ~np.eye(gram.shape[0], dtype=bool)
    quantum = np.asarray(gram, dtype=float)[off]

    # Tested once, before the gamma sweep, because it is a property of the candidate and not
    # of any RBF it is compared against.  A constant off-diagonal is exponential
    # concentration taken to its limit -- the positive result this screen exists to produce.
    # Skipping such a candidate inside the loop instead would leave `best` at its (0.0, nan)
    # initial value, and 0.0 passes the distinctness gate, so an information-free kernel
    # would be certified distinguishable from every RBF.  It is unmeasurable here, and it
    # fails: `nan <= rbf_correlation_max` is False.
    if quantum.std() == 0:
        return (float("nan"), float("nan"))

    best: tuple[float, float] | None = None
    for gamma in gammas:
        classical = rbf_kernel(x, gamma=float(gamma))[off]
        if classical.std() == 0:
            # Unmeasurable at this gamma, not uncorrelated: a constant RBF off-diagonal leaves
            # `corrcoef` nothing to divide by.  Three equidistant points do it at every gamma.
            continue
        r = abs(float(np.corrcoef(quantum, classical)[0, 1]))
        if best is None or r > best[0]:
            best = (r, float(gamma))
    if best is None:
        # Not one gamma was measurable, so no comparison ran.  Seeding `best` at 0.0 instead
        # returned the strongest possible distinctness for a test that never happened, and 0.0
        # passes the gate -- the same fail-open as the constant-Gram case above, reached from
        # the classical side.  Unmeasurable is reported as unmeasurable, and it fails.
        return (float("nan"), float("nan"))
    return best


def geometric_difference(
    k_classical: np.ndarray, k_quantum: np.ndarray, *, reg: float = 1e-6
) -> float:
    """Huang et al.'s geometric difference ``g(K_C || K_Q)``.

    From "Power of data in quantum machine learning", *Nature Communications* 12:2631, 2021:
    a small value means every function the quantum kernel can express is already expressible
    by the classical one at comparable cost, so no advantage is possible on this data
    regardless of how the downstream classifier is fitted.

    Computed as ``sqrt( || sqrt(K_Q) K_C^{-1} sqrt(K_Q) ||_inf )`` with ridge regularisation,
    since the Gram matrices here are near-singular by construction.
    """
    kc = np.asarray(k_classical, dtype=float)
    kq = np.asarray(k_quantum, dtype=float)
    n = kc.shape[0]
    kc_reg = kc + reg * n * np.eye(n)

    eig_q, vec_q = np.linalg.eigh(kq)
    sqrt_kq = vec_q @ np.diag(np.sqrt(np.clip(eig_q, 0.0, None))) @ vec_q.T
    middle = sqrt_kq @ np.linalg.solve(kc_reg, sqrt_kq)
    return float(np.sqrt(np.abs(np.linalg.eigvalsh(middle)).max()))


@dataclass(frozen=True)
class ScreenResult:
    """Everything the gate decided, and why."""

    name: str
    n_qubits: int
    bandwidth: float
    entanglement: str
    n_samples: int
    effective_rank: float
    top_eigenvalue: float
    off_diagonal_mean: float
    rbf_correlation: float
    rbf_gamma: float
    geometric_difference: float
    passes_conditioning: bool
    passes_distinctness: bool

    @property
    def passes(self) -> bool:
        return self.passes_conditioning and self.passes_distinctness

    def summary(self) -> str:
        verdict = "PASS" if self.passes else "reject"
        reason = ""
        if not self.passes_conditioning:
            reason = f" (effective rank {self.effective_rank:.3f} outside the usable band)"
        elif not self.passes_distinctness:
            reason = (
                f" (correlation {self.rbf_correlation:.3f} with RBF gamma="
                f"{self.rbf_gamma:.3g})"
            )
        return f"{verdict}{reason}"


def screen_kernel(
    gram: np.ndarray,
    x: np.ndarray,
    *,
    name: str,
    n_qubits: int,
    bandwidth: float,
    entanglement: str,
    classical_gram: np.ndarray | None = None,
    effective_rank_min: float,
    effective_rank_max: float,
    rbf_correlation_max: float,
) -> ScreenResult:
    """Apply both gates to one candidate feature map."""
    gram = np.asarray(gram, dtype=float)
    off = ~np.eye(gram.shape[0], dtype=bool)
    er = effective_rank_ratio(gram)
    corr, gamma = rbf_correlation(gram, x)
    gd = (
        geometric_difference(classical_gram, gram)
        if classical_gram is not None
        else float("nan")
    )
    return ScreenResult(
        name=name,
        n_qubits=n_qubits,
        bandwidth=bandwidth,
        entanglement=entanglement,
        n_samples=int(gram.shape[0]),
        effective_rank=er,
        top_eigenvalue=top_eigenvalue_share(gram),
        off_diagonal_mean=float(gram[off].mean()),
        rbf_correlation=corr,
        rbf_gamma=gamma,
        geometric_difference=gd,
        passes_conditioning=bool(effective_rank_min <= er <= effective_rank_max),
        passes_distinctness=bool(corr <= rbf_correlation_max),
    )
