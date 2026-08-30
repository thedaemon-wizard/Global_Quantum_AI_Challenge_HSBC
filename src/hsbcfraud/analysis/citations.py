# SPDX-License-Identifier: Apache-2.0
"""Resolve citation-shaped strings in the documents against the reference list.

Four documents state that this project runs a citation gate; for most of the project's life it
did not, and a reference could be cited in prose without appearing in ``docs/REFERENCES.md``
or, worse, could cite an identifier that had been renumbered out from under it.

The logic is kept here rather than in the script so that it is a pure function of text --
``(documents, reference text) -> misses`` -- and can be tested without a filesystem, the same
shape as the literal scan in ``scripts/check_pdf.py``.

Two citation forms are recognised, because both appear in the corpus:

``[CP-4]`` or ``CP-4``   an entry in the reference list, declared there as ``**[CP-4]**``.
``arXiv:2505.06419``     a preprint identifier, which must appear in the reference list too.

Anything else -- a bare author name, a journal volume -- is out of scope. A gate that tried to
resolve free-text citations would produce false positives on ordinary prose, and a gate with
false positives is switched off rather than satisfied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Declaration form in the reference list: the identifier in bold brackets at the start of a
# line.  Matching the declaration rather than any occurrence is what stops an entry from
# "defining" itself by being mentioned in another entry's annotation.
DECLARATION = re.compile(r"^\*\*\[([A-Z]{2}-\d+)\]\*\*", re.MULTILINE)

# Usage form: bracketed or bare.  The identifier shape is two capitals, a hyphen and digits,
# which does not collide with anything in this corpus's prose.
IDENTIFIER = re.compile(r"\[?\b([A-Z]{2}-\d+)\b\]?")

# arXiv identifiers, in either the modern or the legacy-with-version form.
ARXIV = re.compile(r"arXiv:(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)

# Identifier-shaped strings that are not citations.  Each is a real token in this corpus that
# the pattern above would otherwise flag.  Kept explicit and small; a growing list here would
# mean the pattern is wrong rather than that the corpus is unusual.
NOT_CITATIONS = frozenset(
    {
        "SR-26",  # a supervisory letter is written SR 26-2, without the hyphen this shape needs
    }
)


@dataclass(frozen=True)
class Miss:
    """One citation that does not resolve to a reference entry."""

    document: str
    line: int
    citation: str
    context: str

    def __str__(self) -> str:
        return f"{self.document}:{self.line}: {self.citation} does not resolve ({self.context})"


def declared_identifiers(reference_text: str) -> set[str]:
    """Entry identifiers the reference list declares."""
    return set(DECLARATION.findall(reference_text))


def declared_arxiv(reference_text: str) -> set[str]:
    """arXiv identifiers the reference list carries, normalised without any version suffix."""
    return {match.lower() for match in ARXIV.findall(reference_text)}


def find_misses(documents: dict[str, str], reference_text: str) -> list[Miss]:
    """Every citation in ``documents`` that the reference list does not resolve.

    ``documents`` maps a display name to the file's text.  The reference list is not scanned
    against itself: an entry may legitimately mention a neighbour.
    """
    identifiers = declared_identifiers(reference_text)
    arxiv = declared_arxiv(reference_text)
    if not identifiers:
        raise ValueError(
            "the reference list declares no entries; expected lines starting '**[XX-N]**'. "
            "If the declaration format changed, this module must change with it rather than "
            "silently reporting that every citation resolves."
        )

    misses: list[Miss] = []
    for name, text in documents.items():
        for number, line in enumerate(text.splitlines(), start=1):
            for found in IDENTIFIER.findall(line):
                if found in NOT_CITATIONS or found in identifiers:
                    continue
                misses.append(Miss(name, number, found, line.strip()[:70]))
            for found in ARXIV.findall(line):
                if found.lower() not in arxiv:
                    misses.append(Miss(name, number, f"arXiv:{found}", line.strip()[:70]))
    return misses
