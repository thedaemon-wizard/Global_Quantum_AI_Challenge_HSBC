#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Count the project's record documents from the documents, rather than from memory.

Two numbers the submission quotes about its own process are properties of a Markdown file and
not of any run: how many decision entries ``docs/decisions.md`` holds, and how many amendments
``docs/protocol.md`` carries.  Both were typed by hand, and both drifted.

The decision count drifted across four documents -- the README said twenty-nine, then
thirty-eight; both PDF sources said thirty-one; the log itself held forty-one.  Deriving it
here fixed that, and then it drifted again in a subtler way: the table was generated once, two
entries were appended later, and nothing regenerated it, so the shipped PDFs printed 61 against
an actual 63 while every gate passed.  ``make derived`` runs this before the claim gate, and
``test_decision_log_matches_the_decisions_document`` asserts the table still matches its source.

The amendment count drifted in the shipped appendix, which said four against seven.  It lives
in the same table because it fails the same way, is fixed by the same regeneration, and is
covered by the same test.

    .venv/bin/python scripts/summarise_decisions.py

Writes ``results/tables/decision_log.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hsbcfraud.analysis.records import AMENDMENT, DECISION, numbered_ids, sequence_problems
from hsbcfraud.paths import display_path

REPO = Path(__file__).resolve().parents[1]


def read_ids(path: Path, pattern, expected: str) -> list[tuple[str, int]]:
    """Every identifier in file order, refusing to report zero for a changed format.

    A silent zero is the dangerous outcome: it is a number, it flows into a claim, and it
    reads as "this document is empty" rather than "this parser no longer matches".
    """
    entries = numbered_ids(path.read_text(encoding="utf-8"), pattern)
    if not entries:
        raise SystemExit(
            f"no {expected} headings matched in {display_path(path)}. If the heading format "
            "changed, this script must change with it rather than silently report zero."
        )
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--decisions", type=Path, default=REPO / "docs" / "decisions.md")
    parser.add_argument("--protocol", type=Path, default=REPO / "docs" / "protocol.md")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "tables")
    args = parser.parse_args(argv)

    decisions = read_ids(args.decisions, DECISION, '"### D-NNN"')
    amendments = read_ids(args.protocol, AMENDMENT, '"## Amendment ANNN"')

    problems = sequence_problems(decisions) + sequence_problems(amendments)

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "decision_log.csv"
    pd.DataFrame(
        [{"decision_entries": len(decisions), "protocol_amendments": len(amendments)}]
    ).to_csv(target, index=False)

    print(
        f"{len(decisions)} decision entries in {display_path(args.decisions)}, "
        f"{decisions[0][0]} to {decisions[-1][0]}"
    )
    print(
        f"{len(amendments)} amendments in {display_path(args.protocol)}, "
        f"{amendments[0][0]} to {amendments[-1][0]}"
    )
    print(f"Wrote {display_path(target)}")

    if problems:
        print("\nThe sequence is not clean:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
