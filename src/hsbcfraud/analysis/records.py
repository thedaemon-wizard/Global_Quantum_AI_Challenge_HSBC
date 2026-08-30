# SPDX-License-Identifier: Apache-2.0
"""Counts derived from the project's record documents.

Two numbers in the submission are properties of a Markdown file rather than of a run: how many
decision entries ``docs/decisions.md`` holds, and how many amendments ``docs/protocol.md``
carries.  Both were typed by hand at some point and both went stale -- the decision count
across four documents, the amendment count in the shipped appendix, which said four against
seven.

They are parsed here rather than in the two scripts that need them, because the parsing is the
same operation twice and because a pure function over text is testable without a filesystem --
the same shape as ``analysis/citations.py``.

The identifiers are captured, not just counted.  A duplicate or a gap leaves a count looking
entirely plausible while the record has lost an entry, and that is the failure a bare ``len``
cannot see.
"""

from __future__ import annotations

import re

# Entry headings look like "### D-017 The split arm inflates average precision".
DECISION = re.compile(r"^### (D-(\d+))\b", re.MULTILINE)
# Amendment headings look like "## Amendment A5 -- the pre-registered test was wrong".
AMENDMENT = re.compile(r"^##\s+Amendment\s+(A(\d+))\b", re.MULTILINE)


def numbered_ids(text: str, pattern: re.Pattern[str]) -> list[tuple[str, int]]:
    """Every identifier the pattern matches, in file order, as ``(identifier, number)``.

    The pattern must expose the full identifier as group 1 and its integer as group 2.
    """
    return [(match.group(1), int(match.group(2))) for match in pattern.finditer(text)]


def sequence_problems(entries: list[tuple[str, int]]) -> list[str]:
    """Duplicate or non-consecutive identifiers, each described in full.

    Returns an empty list for a clean sequence.  A gap is reported rather than repaired: a
    retired entry should be marked superseded in place, not deleted, or the log stops being a
    record of what was decided when.
    """
    problems: list[str] = []
    seen: dict[int, int] = {}
    for position, (identifier, number) in enumerate(entries, start=1):
        if number in seen:
            problems.append(
                f"{identifier} appears twice, at entries {seen[number]} and {position}"
            )
        seen[number] = position

    if not seen:
        return problems

    numbers = sorted(seen)
    missing = [n for n in range(numbers[0], numbers[-1] + 1) if n not in seen]
    if missing:
        problems.append(
            "gaps in the sequence: "
            + ", ".join(f"{n:03d}" for n in missing)
            + ". A retired entry should be marked superseded in place, not deleted, or the "
            "log stops being a record of what was decided when."
        )
    return problems
