#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Enforce the pre-registration, which until now was only asserted.

``docs/protocol.md`` opened by saying that this script hashes the protocol together with the
configuration and refuses to run an experiment if either changed without a dated entry in
``docs/decisions.md``.  That sentence was written before the script existed, which is exactly
the kind of documentation-versus-reality gap this project's own discipline is supposed to
catch.  It now exists -- and the sentence has been corrected, because what is hashed is the
guarantee-bearing configuration, not the protocol's prose.

A second gap of the same kind: both documents named ``configs/default.yaml`` while nothing
wrote to it and ``load_config(None)`` returned the dataclass defaults.  The hash was always
taken from those live defaults, so enforcement was real; the file now exists as a readable
record generated from them, and ``--verify-dump`` fails if it drifts.

What is actually enforced
-------------------------
Learn-then-Test controls the family-wise error rate over a **finite grid fixed in advance**.
Enlarging the grid, loosening ``alpha``, or moving ``delta`` after seeing the risks inflates
the error rate the procedure is supposed to bound.  The non-exchangeable variant separately
requires ``rho`` to be fixed rather than fitted.  So the fields below are the ones whose
silent movement would void a guarantee, and they are hashed:

* every alpha grid (unconditional, band-conditional, false-negative), delta, the
  decision-grid size, the FWER method, rho
* the band traffic-budget grid and the recall-floor grid
* the split fractions and the decline budget

Fields that do not enter a guarantee -- output paths, seeds used for model fitting, plotting
choices -- are deliberately outside the hash, because locking them would make the mechanism
annoying enough to be bypassed, and a bypassed gate protects nothing.

Amendments are legitimate
-------------------------
Protocol amendments were made during this study, each derived from block sizes, from
``D_band`` alone, or from the record of what was executed -- never from ``D_cal`` risks or
from ``D_test``.  The gate therefore does not
forbid change; it forbids **undocumented** change.  A new hash is accepted once
``docs/protocol.md`` contains a matching amendment heading and ``docs/decisions.md`` has grown.

    .venv/bin/python scripts/check_protocol.py            # verify
    .venv/bin/python scripts/check_protocol.py --freeze    # record the current state
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hsbcfraud.analysis.records import AMENDMENT, numbered_ids
from hsbcfraud.config import load_config
from hsbcfraud.paths import display_path

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "docs" / "protocol.lock.json"

# Exactly the fields whose movement would void a guarantee.
GUARANTEE_FIELDS = (
    ("risk", "alpha_grid"),
    ("risk", "delta"),
    ("risk", "n_lambda"),
    ("risk", "fwer_method"),
    ("risk", "rho"),
    ("risk", "alpha_band_grid"),
    ("risk", "alpha_fn_grid"),
    ("risk", "recall_floor_grid"),
    ("risk", "coverage_band_level"),
    ("band", "budget_grid"),
    ("split", "train"),
    ("split", "band"),
    ("split", "cal"),
    ("split", "test"),
)


def guarantee_state(config_path: Path | None) -> dict[str, object]:
    """The subset of configuration that the guarantees depend on."""
    cfg = load_config(config_path)
    dumped = cfg.model_dump()
    state: dict[str, object] = {}
    for section, field in GUARANTEE_FIELDS:
        state[f"{section}.{field}"] = dumped[section][field]
    state["decline_rate_budget"] = dumped["decline_rate_budget"]
    return state


def digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def amendment_headings(protocol: Path) -> list[str]:
    """Amendment headings in docs/protocol.md, in order.

    The pattern lives in ``analysis.records`` because the same count is written to
    ``decision_log.csv`` for the proposal to quote, and two copies of it would be two things
    to keep in step.
    """
    entries = numbered_ids(protocol.read_text(encoding="utf-8"), AMENDMENT)
    return [identifier for identifier, _ in entries]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--freeze", action="store_true", help="record the current state as the lock"
    )
    parser.add_argument("--protocol", type=Path, default=REPO / "docs" / "protocol.md")
    parser.add_argument("--decisions", type=Path, default=REPO / "docs" / "decisions.md")
    parser.add_argument(
        "--verify-dump",
        action="store_true",
        help="also check configs/default.yaml still matches the committed defaults",
    )
    args = parser.parse_args(argv)

    if args.verify_dump:
        from dump_config import render

        dump = REPO / "configs" / "default.yaml"
        if not dump.exists():
            print(
                f"{display_path(dump)} is missing; run scripts/dump_config.py",
                file=sys.stderr,
            )
            return 1
        if dump.read_text(encoding="utf-8") != render(None):
            print(
                f"{display_path(dump)} has drifted from the defaults in "
                "src/hsbcfraud/config.py. The dataclass is the source of truth; regenerate "
                "with scripts/dump_config.py.",
                file=sys.stderr,
            )
            return 1
        print(f"{display_path(dump)} matches the committed defaults.")

    state = guarantee_state(args.config)
    state_hash = digest(state)
    amendments = amendment_headings(args.protocol)
    decisions_lines = len(args.decisions.read_text(encoding="utf-8").splitlines())

    current = {
        "guarantee_state": state,
        "guarantee_hash": state_hash,
        "amendments": amendments,
        "decisions_lines": decisions_lines,
    }

    if args.freeze:
        LEDGER.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Frozen. Guarantee hash {state_hash[:16]}, amendments {amendments}")
        return 0

    if not LEDGER.exists():
        print(
            f"{display_path(LEDGER)} does not exist. Run with --freeze once to record the "
            "pre-registered state, then commit it.",
            file=sys.stderr,
        )
        return 1

    locked = json.loads(LEDGER.read_text(encoding="utf-8"))
    if locked["guarantee_hash"] == state_hash:
        print(f"Protocol unchanged. Guarantee hash {state_hash[:16]}, amendments {amendments}")
        # The hash covers the guarantee-bearing fields and nothing else, so an amendment that
        # changes no grid leaves it identical.  Five amendments and 1,800 decision lines
        # accumulated behind a matching hash and this file reported "unchanged" every time,
        # which left the ledger a reviewer reads saying four amendments against an actual nine.
        # Not a failure -- no guarantee moved -- but not something to keep silent about either.
        stale = []
        if locked["amendments"] != amendments:
            stale.append(f"amendments {locked['amendments']} recorded, {amendments} present")
        if locked["decisions_lines"] != decisions_lines:
            stale.append(
                f"{locked['decisions_lines']} decision lines recorded, {decisions_lines} present"
            )
        if stale:
            print(
                f"  {display_path(LEDGER)} is behind the documents it records: "
                + "; ".join(stale)
                + ".\n  The guarantee has not moved, so this is bookkeeping: re-freeze with "
                "--freeze.",
                file=sys.stderr,
            )
        return 0

    # The state moved.  That is allowed, but only alongside a documented amendment.
    new_amendments = [a for a in amendments if a not in locked["amendments"]]
    grew = decisions_lines > locked["decisions_lines"]

    print("Pre-registered state has changed:", file=sys.stderr)
    for key, value in state.items():
        if locked["guarantee_state"].get(key) != value:
            print(f"  {key}: {locked['guarantee_state'].get(key)!r} -> {value!r}", file=sys.stderr)

    if new_amendments and grew:
        print(
            f"\nAccompanied by amendment(s) {new_amendments} in docs/protocol.md and "
            f"{decisions_lines - locked['decisions_lines']} new lines in docs/decisions.md. "
            f"Re-freeze with --freeze to accept.",
            file=sys.stderr,
        )
        return 2

    missing = []
    if not new_amendments:
        missing.append("a new '## Amendment A<n>' section in docs/protocol.md")
    if not grew:
        missing.append("a new entry in docs/decisions.md")
    print(
        "\nThis change is NOT documented. Learn-then-Test controls the family-wise error rate "
        "over a grid fixed in advance; moving alpha, delta, the grid size or rho after seeing "
        "the data voids the guarantee. Add " + " and ".join(missing) + ", then re-freeze.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
