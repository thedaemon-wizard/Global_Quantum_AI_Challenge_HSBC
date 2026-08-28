# SPDX-License-Identifier: Apache-2.0
"""Declarative, validated configuration for the measurement campaign.

Which alpha levels are certified, where the temporal blocks are cut, which feature maps
are screened -- these are data, not code.  Keeping them in YAML and validating them here
buys three things that matter for a study whose entire claim is a certified risk level:

* a run is fully described by a file that can be hashed into the manifest, so a table and
  the configuration that produced it travel together;
* a typo becomes a loud error naming the offending key rather than a silently different
  experiment (every model sets ``extra="forbid"``);
* the pre-registered quantities live in one hashable place, so ``docs/protocol.md`` can
  cite the file rather than restate the numbers and drift from them.

The distinction between this module and :mod:`docs.protocol` matters.  The protocol is the
*commitment*: alpha grid, delta, the geometric weight rho, the null hypotheses, the
stopping rule.  Changing any of those after the data have been seen invalidates the
Learn-then-Test guarantee, so those fields are additionally covered by a hash check in
``scripts/check_protocol.py``.  Everything else here is ordinary configuration.

Error boundary
--------------
Constructing a model directly raises pydantic's ``ValidationError``, which already names
the offending field.  :func:`load_config` wraps that in :class:`ConfigError` and adds the
file name, because when the input came from a file the first thing you need to know is
*which* file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "BandConfig",
    "ConfigError",
    "ExperimentConfig",
    "QuantumConfig",
    "RiskControlConfig",
    "SplitConfig",
    "load_config",
]

Fraction = Annotated[float, Field(gt=0.0, lt=1.0)]
Probability = Annotated[float, Field(gt=0.0, le=1.0)]
Qubits = Annotated[int, Field(ge=2, le=16)]


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or inconsistent."""


class _Strict(BaseModel):
    """Base with unknown keys rejected, so a typo cannot silently change a run."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SplitConfig(_Strict):
    """The four-block temporal split, and the two control arms it is measured against.

    The blocks are contiguous and time-ordered, and each has exactly one job:

    ``train``  fits the classical scorer and nothing else;
    ``band``   is the only data that may see the band edges or the in-band scorer's
               hyperparameters -- freezing it here is what makes the band a fixed
               measurable predicate rather than a data-dependent selection event;
    ``cal``    certifies the *composite* rule under Learn-then-Test;
    ``test``   is reported once, on the full imbalanced fold, and never resampled.

    Two control arms run alongside on the same proportions.  ``stratified`` reshuffles at
    random, which measures how much of any degradation is temporal rather than sampling
    noise.  ``card_disjoint`` partitions by the card key instead of by time, because the
    IEEE-CIS label rule propagates a chargeback to later transactions on linked entities,
    so a purely temporal cut does not separate entities -- it only changes the direction
    of the overlap.
    """

    train: Fraction = 0.60
    band: Fraction = 0.10
    cal: Fraction = 0.10
    test: Fraction = 0.20
    # Cuts land on day boundaries so a diurnal cycle is never divided between two blocks.
    # TransactionDT is seconds from an undisclosed epoch, so "day" here means a bucket of
    # 86400 seconds counted from the file's own minimum, not a calendar day.
    snap_to_day_boundary: bool = True
    control_arms: list[Literal["stratified", "card_disjoint"]] = ["stratified", "card_disjoint"]
    # The entity key used both for the card-disjoint arm and for block bootstrap. card1 is
    # a coarse card attribute rather than a card identifier; the measured median is 4
    # transactions per value, so blocks are small but the clustering is real.
    entity_key: str = "card1"
    seeds: list[int] = [20260828, 20260829, 20260830, 20260831, 20260901]

    @model_validator(mode="after")
    def _fractions_sum_to_one(self) -> SplitConfig:
        total = self.train + self.band + self.cal + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split fractions must sum to 1.0, got {total!r}")
        return self


class BandConfig(_Strict):
    """How the abstention band -- which is also the quantum re-ranking set -- is chosen.

    ``traffic_budget`` fixes the *volume* routed into the band, not the score cutoffs.
    That choice is deliberate and has to be stated: under drift, a band defined by fixed
    score cutoffs changes the volume it routes, while a band defined by fixed quantiles
    changes the risk it contains.  Holding volume constant keeps the operational cost
    (the number of step-up authentications per day) predictable, which is the quantity a
    bank actually budgets; the drift in risk content is then measured rather than assumed.
    """

    # 0.035 is the smallest budget at which the band-conditional certificate is reachable
    # at alpha = 1e-2 with a 10 % calibration block; see docs/protocol.md amendment A1.
    traffic_budget: Fraction = 0.05
    # Candidate budgets swept on the band block only.  The test fold sees exactly one.
    budget_grid: list[Fraction] = [0.02, 0.035, 0.05, 0.10]
    definition: Literal["fixed_quantile", "fixed_score"] = "fixed_quantile"


class RiskControlConfig(_Strict):
    """Pre-registered Learn-then-Test parameters.

    ``alpha_grid`` is an order of magnitude tighter than a textbook conformal sweep
    because the certified quantity is the rate at which legitimate customers are declined.
    An issuer's all-in decline rate sits well under one percent, so certifying alpha = 0.05
    would certify something no bank would operate.

    ``rho`` is the geometric weight of the non-exchangeable robustness variant (Barber
    et al. 2023, eq. 11).  The theorem requires the weights to be fixed and not fitted on
    the calibration set, which is precisely why it is pinned here and hashed rather than
    tuned.
    """

    alpha_grid: list[Probability] = [1e-2, 5e-3, 2e-3, 1e-3]
    # Decision-threshold grid size.  Holm corrects over (grid points x risks), so a larger
    # grid costs power directly: 41 points needs a ~30 % larger band budget than 11 points
    # to certify the same alpha.  Eleven is the smallest grid that still resolves the
    # budget-alpha frontier.
    n_lambda: int = 11
    # PAC confidence: P(risk <= alpha) >= 1 - delta.
    delta: Probability = 0.05
    # Both risks are certified. Controlling only the false-decline side would leave the
    # side that PSD2 caps, and that the loss reserve is held against, uncertified.
    risks: list[Literal["false_positive_rate", "recall"]] = ["false_positive_rate", "recall"]
    # Reported as a frontier rather than certified at one blind value; see A3.
    recall_floor: Probability = 0.60
    recall_floor_grid: list[Probability] = [0.45, 0.50, 0.55, 0.60]
    fwer_method: Literal["bonferroni", "bonferroni_holm"] = "bonferroni_holm"
    rho: Probability = 0.999
    # Central band level for the exact Beta-Binomial coverage check.
    coverage_band_level: Probability = 0.99


class QuantumConfig(_Strict):
    """The screened quantum arm.

    ``screen_*`` are gates, not diagnostics: if a candidate feature map fails them on the
    band block, the arm is not run and the rejection is the reported result.  The
    thresholds bracket the region a classical RBF kernel occupies on the same data, which
    is where a kernel has to sit to be usable at all -- outside it the Gram matrix is
    either near-rank-one or near-uniform.
    """

    qubits: list[Qubits] = [4, 6, 8]
    feature_maps: list[Literal["zz", "z", "dense_angle"]] = ["zz", "z", "dense_angle"]
    reps: int = 2
    entanglement: list[Literal["linear", "full", "none"]] = ["linear", "none"]
    bandwidths: list[float] = [1.0, 0.5, 0.25, 0.125, 0.0625]
    # Classical controls evaluated under kernel-swap discipline: identical features,
    # identical solver, identical hyperparameter budget, only the kernel changes.
    classical_kernels: list[Literal["rbf", "laplacian", "poly"]] = ["rbf", "laplacian", "poly"]
    screen_effective_rank_max: float = 0.35
    screen_effective_rank_min: float = 0.01
    screen_rbf_correlation_max: float = 0.60
    max_band_samples: int = 8000


class ExperimentConfig(_Strict):
    """Top-level campaign configuration."""

    seed: int = 20260828
    split: SplitConfig = SplitConfig()
    band: BandConfig = BandConfig()
    risk: RiskControlConfig = RiskControlConfig()
    quantum: QuantumConfig = QuantumConfig()
    # PSD2 SCA-RTS Annex reference fraud rates for remote card-based payments, keyed by
    # exemption threshold value in euro. These are VALUE-weighted rates on a rolling
    # 90-day window (Commission Delegated Regulation (EU) 2018/389, Articles 18-19), and
    # they govern eligibility to exempt a transaction from strong customer authentication
    # -- not the authorisation decline threshold. docs/REGULATORY_SOURCES.md holds the
    # official locator.
    psd2_reference_rates: dict[str, float] = {"100": 0.0013, "250": 0.0006, "500": 0.0001}
    psd2_window_days: int = 90
    # The operating point is selected by decline-rate budget, not by the PSD2 rates: on a
    # fraud-enriched benchmark those portfolio-level ceilings are reachable only at absurd
    # decline rates (94.7 % for the loosest tier, measured). See docs/protocol.md amendment
    # A2. Published all-in card-not-present decline rates run about 0.5-3 % of volume.
    # 5 %, not 2 %: below about 5 % the outer threshold catches too little fraud for the
    # concentration bound to establish any recall floor. See docs/protocol.md A3.
    decline_rate_budget: Probability = 0.05
    decline_rate_sensitivity: list[Probability] = [0.02, 0.05, 0.10]


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    """Load and validate a configuration file, or return the committed defaults.

    ``load_config(None)`` returns the defaults, which *are* the values that produced the
    tables in ``results/tables/``.
    """
    if path is None:
        return ExperimentConfig()
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ConfigError(f"cannot read configuration {p}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{p} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{p} must contain a mapping at the top level, got {type(raw).__name__}")
    try:
        return ExperimentConfig(**raw)
    except ValueError as exc:
        raise ConfigError(f"{p} is not a valid configuration: {exc}") from exc
