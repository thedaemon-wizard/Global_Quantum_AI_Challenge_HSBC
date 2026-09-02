# SPDX-License-Identifier: Apache-2.0
"""The citation gate, pinned in both directions.

Four documents in this repository stated that a citation gate existed before one did. A gate
whose negative case is untested is the same defect in a different place, so every test here
has a matching case that must pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hsbcfraud.analysis.citations import declared_arxiv, declared_identifiers, find_misses

REPO = Path(__file__).resolve().parents[1]

REFERENCES = """
# References

**[CP-1]** Vovk, Gammerman and Shafer. Something. arXiv:2110.01052.
Annotation mentioning CP-2 in passing.

**[QM-10]** Faryad. Something else. arXiv:2608.15718.
"""


# ------------------------------------------------------------------------------- parsing

def test_only_bold_bracketed_line_starts_declare_an_entry() -> None:
    """An entry mentioned inside another entry's annotation does not thereby exist."""
    assert declared_identifiers(REFERENCES) == {"CP-1", "QM-10"}


def test_arxiv_identifiers_are_collected_from_the_whole_file() -> None:
    assert declared_arxiv(REFERENCES) == {"2110.01052", "2608.15718"}


def test_an_empty_reference_list_is_an_error_not_a_pass() -> None:
    """Silently reporting success on an unparsable file is the worst possible failure here."""
    with pytest.raises(ValueError, match="declares no entries"):
        find_misses({"a.md": "see [CP-1]"}, "# References\n\nnothing declared\n")


# ------------------------------------------------------------------------------ resolution

@pytest.mark.parametrize(
    "line",
    [
        "As established in [CP-1], the threshold is an order statistic.",
        "Following CP-1 and QM-10.",
        "The comparator is arXiv:2608.15718.",
        "Versioned identifiers resolve too: arXiv:2110.01052v3.",
    ],
)
def test_a_declared_citation_resolves(line) -> None:
    assert find_misses({"a.md": line}, REFERENCES) == []


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("As shown in [ZZ-99].", "ZZ-99"),
        ("Undeclared preprint arXiv:9999.99999.", "arXiv:9999.99999"),
        ("A renumbered entry [CP-14] that no longer exists.", "CP-14"),
    ],
)
def test_an_undeclared_citation_is_caught(line, expected) -> None:
    misses = find_misses({"a.md": line}, REFERENCES)
    assert len(misses) == 1
    assert misses[0].citation == expected


def test_the_miss_reports_document_and_line() -> None:
    """A finding a reader cannot locate is a finding they will not act on."""
    misses = find_misses({"doc.tex": "line one\nline two\nsee [ZZ-99]\n"}, REFERENCES)
    assert misses[0].document == "doc.tex"
    assert misses[0].line == 3


def test_ordinary_prose_is_not_a_citation() -> None:
    """A gate with false positives gets switched off rather than satisfied."""
    prose = "The PSD2 SCA-RTS applies. IEEE-CIS has 431 features. Apache-2.0 licensed.\n"
    assert find_misses({"a.md": prose}, REFERENCES) == []


# -------------------------------------------------------------------------- the real corpus

def test_every_citation_in_the_repository_resolves() -> None:
    """Every Markdown document under ``docs/``, not a list of four.

    A test named after the whole repository covered four documents of sixteen, and was a
    strict subset of the production gate it shadows -- ``CITED_DOCUMENTS`` in
    ``scripts/check_claims.py`` already reaches three more. ``docs/FACTCHECK_LOG.md`` carries
    ten citation-shaped tokens and was under neither. Globbing makes a new document checked by
    existing rather than by being remembered.

    ``REFERENCES.md`` is the one exclusion: it is the declaration, and an entry may
    legitimately name a neighbouring one in its annotation.
    """
    documents = {
        str(path.relative_to(REPO)): path.read_text(encoding="utf-8")
        for pattern in ("README.md", "docs/*.md", "submission/content/*.tex")
        for path in sorted(REPO.glob(pattern))
        if path.name != "REFERENCES.md"
    }
    assert documents, "no documents found to check"
    references = (REPO / "docs" / "REFERENCES.md").read_text(encoding="utf-8")
    misses = find_misses(documents, references)
    assert misses == [], "\n".join(str(m) for m in misses)
