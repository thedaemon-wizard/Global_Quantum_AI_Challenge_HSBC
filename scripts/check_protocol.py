#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Enforce the pre-registration, which until now was only asserted.

``docs/protocol.md`` opens by saying that this script hashes the protocol together with the
configuration and refuses to run an experiment if either changed without a dated entry in
``docs/decisions.md``.  That sentence was written before the script existed, which is exactly
the kind of documentation-versus-reality gap this project's own discipline is supposed to
catch.  It now exists.

What is actually enforced
-------------------------
Learn-then-Test controls the family-wise error rate over a **finite grid fixed in advance**.
Enlarging the grid, loosening ``alpha``, or moving ``delta`` after seeing the risks inflates
the error rate the procedure is supposed to bound.  The non-exchangeable variant separately
requires ``rho`` to be fixed rather than fitted.  So the fields below are the ones whose
silent movement would void a guarantee, and they are hashed:

* the alpha grid, delta, the decision-grid size, the FWER method, rho
* the band traffic-budget grid and the recall-floor grid
* the split fractions and the decline budget

Fields that do not enter a guarantee -- output paths, seeds used for model fitting, plotting
choices -- are deliberately outside the hash, because locking them would make the mechanism
annoying enough to be bypassed, and a bypassed gate protects nothing.

Amendments are legitimate
-------------------------
Three protocol amendments were made during this study, each derived from block sizes or from
``D_band`` alone, never from ``D_cal`` risks or from ``D_test``.  The gate therefore does not
forbid change; it forbids **undocumented** change.  A new hash is accepted once
``docs/protocol.md`` contains a matching amendment heading and ``docs/decisions.md`` has grown.

    .venv/bin/python scripts/check_protocol.py            # verify
    .venv/bin/python scripts/check_protocol.py --freeze    # record the current state
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hsbcfraud.config import load_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "docs" / "protocol.lock.json"

# Exactly the fields whose movement would void a guarantee.
GUARANTEE_FIELDS = (
    ("risk", "alpha_grid"),
    ("risk", "delta"),
    ("risk", "n_lambda"),
    ("risk", "fwer_method"),
    ("risk", "rho"),
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
    """Amendment headings in docs/protocol.md, in order."""
    text = protocol.read_text(encoding="utf-8")
    return re.findall(r"^##\s+Amendment\s+(A\d+)\b", text, flags=re.MULTILINE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--freeze", action="store_true", help="record the current state as the lock")
    parser.add_argument("--protocol", type=Path, default=REPO / "docs" / "protocol.md")
    parser.add_argument("--decisions", type=Path, default=REPO / "docs" / "decisions.md")
    args = parser.parse_args(argv)

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
            f"{LEDGER.relative_to(REPO)} does not exist. Run with --freeze once to record the "
            "pre-registered state, then commit it.",
            file=sys.stderr,
        )
        return 1

    locked = json.loads(LEDGER.read_text(encoding="utf-8"))
    if locked["guarantee_hash"] == state_hash:
        print(f"Protocol unchanged. Guarantee hash {state_hash[:16]}, amendments {amendments}")
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
