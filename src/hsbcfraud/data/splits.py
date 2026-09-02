# SPDX-License-Identifier: Apache-2.0
"""The four-block temporal split, its two control arms, and the test-fold access counter.

Why four blocks and not three
-----------------------------
There are two distinct ways exchangeability fails in this architecture, and conflating
them is the error this module exists to prevent.

*Temporal.*  The calibration block precedes the test block in time, so the joint law is not
exchangeable.  This is unavoidable and is measured rather than assumed away
(:mod:`hsbcfraud.conformal.exchange`).

*Selective.*  If a secondary scorer only re-ranks transactions inside a band, and the band
edges were estimated from the same data used to certify, then conditioning on band
membership is conditioning on a data-dependent event, and the finite-sample guarantee does
not apply.  This one is avoidable, and it is avoided **by construction**: the band is fixed
on its own block, ``D_band``, and never touched again.  Once frozen, membership is a fixed
measurable predicate, and filtering a sequence by a fixed predicate preserves
exchangeability of the retained subsequence.

So ``D_band`` exists solely to spend the selection budget somewhere that is not ``D_cal``.
A three-block split would save 10 % of the data and invalidate the certificate.

The consequence, stated because it is the next thing a careful reader asks: the guarantee
now lives on the band-conditional law, and drift inside a narrow score band straddling the
decision boundary is systematically worse than marginal drift.  That is measured too.

The control arms
----------------
``stratified`` reshuffles at random on the same proportions.  Comparing it against the
temporal arm decomposes any degradation into a temporal component and ordinary sampling
noise, instead of attributing all of it to time.

``card_disjoint`` partitions by entity.  It exists because the IEEE-CIS label rule
propagates a chargeback across linked accounts, so a temporal cut does not separate
entities -- measured on this file, 85.0 % of the ``card1`` values in the test block also
appear in the training block.  The card-disjoint arm is the only one of the three in which
an entity cannot appear on both sides.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from hsbcfraud.config import SplitConfig

__all__ = [
    "BLOCK_NAMES",
    "Blocks",
    "TestFoldGuard",
    "card_disjoint_blocks",
    "stratified_blocks",
    "temporal_blocks",
]

BLOCK_NAMES = ("train", "band", "cal", "test")
BlockName = Literal["train", "band", "cal", "test"]


class SplitError(ValueError):
    """Raised when a split cannot be formed as specified."""


@dataclass(frozen=True)
class Blocks:
    """Row-index arrays for the four blocks, plus the arm that produced them."""

    arm: str
    train: np.ndarray
    band: np.ndarray
    cal: np.ndarray
    test: np.ndarray

    def __getitem__(self, name: str) -> np.ndarray:
        # Checked against BLOCK_NAMES rather than by getattr: `arm` is a field too, so
        # blocks["arm"] used to return the arm string where the annotation promises an array,
        # and the mistake surfaced deep inside numpy rather than at the lookup.
        if name not in BLOCK_NAMES:
            raise KeyError(f"unknown block {name!r}; expected one of {BLOCK_NAMES}")
        return getattr(self, name)

    def sizes(self) -> dict[str, int]:
        return {name: int(self[name].size) for name in BLOCK_NAMES}

    def assert_disjoint(self, n_rows: int) -> None:
        """Every row lands in exactly one block.  A silent overlap is a silent leak."""
        stacked = np.concatenate([self[name] for name in BLOCK_NAMES])
        if stacked.size != n_rows:
            raise SplitError(f"blocks cover {stacked.size:,} rows, expected {n_rows:,}")
        if np.unique(stacked).size != n_rows:
            raise SplitError("blocks overlap; a row appears in more than one block")


def _cut_points(n: int, cfg: SplitConfig) -> list[int]:
    train = round(cfg.train * n)
    band = round(cfg.band * n)
    cal = round(cfg.cal * n)
    return [0, train, train + band, train + band + cal, n]


def temporal_blocks(day: pd.Series, cfg: SplitConfig) -> Blocks:
    """Contiguous time-ordered blocks, with cuts snapped to day boundaries.

    Snapping matters for a reason that is easy to miss: transaction volume has a strong
    diurnal cycle, so a cut placed mid-day puts the busy hours of one day in one block and
    the quiet hours in the next, which shows up as a spurious distribution shift between
    blocks.  Snapping trades exact proportions for blocks that differ only by when they
    happened.

    ``day`` must already be sorted, which :func:`hsbcfraud.data.ieee_cis.load_ieee_cis`
    guarantees by refusing to load a file whose ``TransactionDT`` is not monotone.
    """
    values = day.to_numpy()
    n = values.size
    raw = _cut_points(n, cfg)

    if cfg.snap_to_day_boundary:
        snapped = [0]
        for cut in raw[1:-1]:
            # Move forward to the first row of the next day, so a day is never divided.
            target_day = values[min(cut, n - 1)]
            boundary = int(np.searchsorted(values, target_day + 1, side="left"))
            snapped.append(boundary)
        snapped.append(n)
        cuts = snapped
    else:
        cuts = raw

    if any(b <= a for a, b in itertools.pairwise(cuts)):
        raise SplitError(f"snapping produced an empty block; cut points {cuts}")

    index = np.arange(n)
    return Blocks(
        arm="temporal",
        train=index[cuts[0] : cuts[1]],
        band=index[cuts[1] : cuts[2]],
        cal=index[cuts[2] : cuts[3]],
        test=index[cuts[3] : cuts[4]],
    )


def stratified_blocks(label: pd.Series, cfg: SplitConfig, seed: int) -> Blocks:
    """Class-stratified random blocks on the same proportions, as a control arm.

    Stratification is on the label so that every block holds a comparable number of frauds;
    an unstratified shuffle at a 3.5 % base rate would put visibly different fraud counts in
    the two 10 % blocks purely by chance and confound the comparison this arm exists for.
    """
    rng = np.random.default_rng(seed)
    values = label.to_numpy()
    parts: dict[str, list[np.ndarray]] = {name: [] for name in BLOCK_NAMES}
    for class_value in np.unique(values):
        members = np.flatnonzero(values == class_value)
        rng.shuffle(members)
        cuts = _cut_points(members.size, cfg)
        for i, name in enumerate(BLOCK_NAMES):
            parts[name].append(members[cuts[i] : cuts[i + 1]])
    return Blocks(arm="stratified", **{k: np.sort(np.concatenate(v)) for k, v in parts.items()})


def card_disjoint_blocks(entity: pd.Series, cfg: SplitConfig, seed: int) -> Blocks:
    """Blocks in which no entity appears on both sides.

    Entities are assigned whole to blocks, in random order, filling each block until it
    reaches its target share of *rows*.  Because entity sizes are heavily skewed -- measured
    on this file, a median of 4 transactions per ``card1`` value against a 95th percentile
    of 107 -- balancing on entity count instead would produce wildly uneven blocks.

    Missing entity values are assigned to their own singleton groups: a null card attribute
    is not evidence that two rows share a card.
    """
    codes = entity.astype("object").where(entity.notna(), None)
    groups: dict[object, list[int]] = {}
    for position, value in enumerate(codes.to_numpy()):
        key = value if value is not None else ("__missing__", position)
        groups.setdefault(key, []).append(position)

    keys = list(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    n = len(entity)
    targets = {
        "train": cfg.train * n,
        "band": cfg.band * n,
        "cal": cfg.cal * n,
        "test": cfg.test * n,
    }
    filled: dict[str, list[int]] = {name: [] for name in BLOCK_NAMES}
    counts = dict.fromkeys(BLOCK_NAMES, 0)
    for key in keys:
        # Whichever block is furthest below its target takes the next entity.  Greedy, but
        # deterministic given the seed, and the resulting block sizes are reported.
        name = max(BLOCK_NAMES, key=lambda b: targets[b] - counts[b])
        rows = groups[key]
        filled[name].extend(rows)
        counts[name] += len(rows)

    empty = [name for name in BLOCK_NAMES if not filled[name]]
    if empty:
        raise SplitError(f"card-disjoint split left blocks empty: {empty}")
    return Blocks(arm="card_disjoint", **{k: np.sort(np.asarray(v)) for k, v in filled.items()})


class TestFoldGuard:
    """Enforces the single-evaluation rule in code rather than by discipline.

    The pre-registration commits to evaluating ``D_test`` exactly once, for the single
    configuration selected on ``D_band`` and certified on ``D_cal``.  That commitment is
    worthless if nothing checks it, and it is very easy to violate by accident -- a sweep
    that "just peeks" at test is the most common way a held-out fold stops being held out.

    The ledger is persisted, so it survives process restarts and a reviewer can read it.
    Requesting the test fold for a configuration that has already been evaluated is fine
    (re-running the same thing changes nothing); requesting it for a *different* one raises.

    Each entry records the authorised configuration hash **and how many times it has been
    requested**.  The count exists because the documents claimed a reviewer could read one and
    an earlier version stored only the hash, so the file said nothing about how often the fold
    had been touched.  A repeat request for the same configuration is legitimate and is
    counted rather than suppressed: it is the number a reader wants when asking whether a
    held-out fold stayed held out.

    An entry carried over from the hash-only form keeps ``evaluations: null`` for good, and
    counts forward under ``evaluations_since_migration`` instead.  The two cannot be merged:
    the authorisations that happened before the counter existed were not recorded anywhere,
    so any total would be a guess, and this is the one artefact whose whole purpose is to
    answer "how often was the held-out fold touched" without guessing.
    """

    def __init__(self, ledger: Path) -> None:
        self.ledger = ledger
        self._seen: dict[str, dict[str, object]] = {}
        if ledger.exists():
            raw = json.loads(ledger.read_text(encoding="utf-8"))
            # Migrate the hash-only form written before the count existed.  Its evaluation
            # count is unknown, not zero, and is recorded as such rather than invented.
            self._seen = {
                dataset: (
                    entry
                    if isinstance(entry, dict)
                    else {"configuration": entry, "evaluations": None}
                )
                for dataset, entry in raw.items()
            }

    def evaluations(self, dataset: str) -> int | None:
        """How many times this fold has been authorised, or None when that is not knowable.

        None is never zero.  It covers no entry at all, and an entry migrated from the
        hash-only form, which keeps ``evaluations: null`` for good and counts forward under
        ``evaluations_since_migration``.  A caller that wants the migrated fold's live count
        has to read that field by name; this accessor declines to return a total it would have
        to guess at, which is the whole point of keeping the two counters apart.
        """
        entry = self._seen.get(dataset)
        return None if entry is None else entry.get("evaluations")

    def authorise(self, dataset: str, configuration_hash: str) -> None:
        entry = self._seen.get(dataset)
        previous = None if entry is None else entry.get("configuration")
        if previous is not None and previous != configuration_hash:
            raise SplitError(
                f"the {dataset} test fold has already been evaluated for configuration "
                f"{previous[:12]}, and a second evaluation was requested for "
                f"{configuration_hash[:12]}. The pre-registration (docs/protocol.md section "
                f"2.2) permits one. Run sweeps on D_band or D_cal. To start a genuinely new "
                f"campaign, delete {self.ledger} and record why in docs/decisions.md."
            )
        count = None if entry is None else entry.get("evaluations")
        record: dict[str, object] = {"configuration": configuration_hash}
        if entry is not None and count is None:
            # The entry exists but carries no count, which means it came through the
            # hash-only migration above.  Writing 1 here would silently convert "authorised
            # an unrecorded number of times before the counter existed" into "authorised
            # once" -- the invention the migration refuses to make, in the one file a
            # reviewer reads to find out whether the held-out fold stayed held out.  What is
            # known is how many authorisations have happened since, so that is what is
            # counted, under its own name.
            since = entry.get("evaluations_since_migration")
            record["evaluations"] = None
            record["evaluations_since_migration"] = 1 if since is None else int(since) + 1
        else:
            record["evaluations"] = 1 if count is None else int(count) + 1
        self._seen[dataset] = record
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.ledger.write_text(json.dumps(self._seen, indent=2, sort_keys=True) + "\n", "utf-8")
