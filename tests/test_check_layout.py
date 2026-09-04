# SPDX-License-Identifier: Apache-2.0
"""``scripts/check_layout.py``, pinned in both directions.

A table's number must not print on one page while its rows print on the next.  ``check_pdf.py``
asserts the page count and the type size, and a document satisfies both while doing exactly
that -- which is what shipped in the sibling Airbus submission on 2026-09-04, with a caption as
the last line of page 4 and its header and four data rows opening page 5.  Guidelines 6.1
screens format for eligibility before any scoring, so the defect is not recoverable afterwards.

This document has never had it: every table here is a real ``\\begin{table}[tb]`` float, which
LaTeX sets as one unbreakable box.  The check is here so that remains true rather than because
it is currently false, and the synthetic cases below are what show the gate can still fire --
running it against a compliant document proves nothing about whether it works.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_check_layout():
    spec = importlib.util.spec_from_file_location(
        "check_layout", REPO / "scripts" / "check_layout.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_layout"] = module
    spec.loader.exec_module(module)
    return module


check_layout = _load_check_layout()


@pytest.mark.parametrize("document", ["proposal", "appendix"])
def test_the_built_document_has_no_split_caption_table_or_heading(document: str) -> None:
    pdf = REPO / "submission" / f"{document}.pdf"
    tex = REPO / "submission" / f"{document}.tex"
    if not pdf.exists():
        pytest.skip(f"{pdf.name} is not built")
    completed = subprocess.run(
        [sys.executable, "scripts/check_layout.py", str(pdf), str(tex)],
        capture_output=True, text=True, cwd=REPO, check=False,
    )
    assert completed.returncode == 0, (
        f"{document}.pdf has a layout defect:\n" + completed.stdout + completed.stderr
    )


def test_a_caption_separated_from_its_body_is_caught() -> None:
    """The shape of the real defect: caption on page 4, body opening page 5."""
    findings = check_layout._check_table_captions(
        "x.pdf",
        [{"page": 4, "y": 97.5, "kind": "table", "number": 1, "text": "Table 1: x"}],
        [{"top_page": 5, "top_y": 790.5, "bot_page": 5, "bot_y": 720.4}],
        {}, 36.0,
    )
    assert [code for code, _ in findings] == ["C1"]
    assert "page 5" in findings[0][1], "the message must say where the body went"


def test_an_intact_table_is_not_reported() -> None:
    """The other direction: a gate that fires on compliant documents gets switched off."""
    findings = check_layout._check_table_captions(
        "x.pdf",
        [{"page": 5, "y": 782.5, "kind": "table", "number": 1, "text": "Table 1: x"}],
        [{"top_page": 5, "top_y": 777.5, "bot_page": 5, "bot_y": 700.0}],
        {}, 36.0,
    )
    assert findings == []


def test_a_body_that_straddles_or_overflows_is_caught() -> None:
    straddle = check_layout._check_table_bodies(
        "x.pdf", [{"top_page": 2, "top_y": 100.0, "bot_page": 3, "bot_y": 700.0}], {}, 28.0
    )
    assert [code for code, _ in straddle] == ["C2"]
    overflow = check_layout._check_table_bodies(
        "x.pdf", [{"top_page": 2, "top_y": 100.0, "bot_page": 2, "bot_y": 12.0}], {}, 28.0
    )
    assert [code for code, _ in overflow] == ["C2"]


def test_the_folio_is_identified_by_content_not_by_height() -> None:
    """A y threshold that clears the page number also discards real text.

    This document carries body text down to y=29.9 on page 2, below any threshold that would
    clear a folio at y=21.1 -- which is why the rule is content-based.
    """
    assert check_layout.is_folio({"text": "2", "y": 21.1}, 2, [{"y": 21.1}])
    assert not check_layout.is_folio({"text": "and then", "y": 29.9}, 2, [{"y": 21.1}])
