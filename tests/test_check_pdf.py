# SPDX-License-Identifier: Apache-2.0
"""The source scans in ``scripts/check_pdf.py``, pinned in both directions.

A gate is only useful if it fails on the defect and passes on the compliant document, and this
one has now been wrong in both directions.  It read the horizontal entry of the text matrix and
declared 37 % of a compliant body undersized; it matched the version in ``Apache-2.0`` as a
measurement; and it matched the ``06`` in ``\\input{content/06-hybrid}``, which failed the whole
build on the proposal's own filenames.  Every test here fixes one of those, and each asserts
the negative case as well, because a check that cannot pass gets disabled rather than satisfied.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_check_pdf():
    """Import ``scripts/check_pdf.py``, which is a script rather than a package module."""
    spec = importlib.util.spec_from_file_location("check_pdf", REPO / "scripts" / "check_pdf.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_pdf"] = module
    spec.loader.exec_module(module)
    return module


check_pdf = _load_check_pdf()


def _scan(tmp_path: Path, text: str, check) -> list[str]:
    source = tmp_path / "fragment.tex"
    source.write_text(text, encoding="utf-8")
    return check([source])


# --------------------------------------------------------------------- structural arguments

@pytest.mark.parametrize(
    "line",
    [
        r"\input{content/06-hybrid}",
        r"\include{content/07-team}",
        r"\author{Global Quantum + AI Challenge 2026 --- HSBC track}",
        r"\label{tab:mps4}",
        r"\documentclass[10pt,a4paper]{article}",
        r"\renewcommand{\topfraction}{0.9}",
        r"\setlength{\parskip}{0.5em}",
        r"\Cited{all effect sizes below 0.013 ARI}",
        r"\Record{Yale Peaked Hackathon 2026, \#13 of 550}",
    ],
)
def test_identifiers_and_declared_externals_are_not_measurements(tmp_path, line) -> None:
    """Digits belonging to a filename, a key, or another work's result are not claims."""
    assert _scan(tmp_path, line, check_pdf.check_literals) == []


def test_a_declared_external_wrapped_across_lines_is_still_stripped(tmp_path) -> None:
    """The verdict must not depend on where the source happens to wrap.

    The scan was line-based, so a ``\\Record{...}`` opening on one line and closing on the next
    was stripped on the first and leaked its argument onto the second.
    """
    wrapped = (
        "\\Record{a GPU backend change cutting allocation\n"
        "  and host-device copies by two thirds}.\n"
    )
    assert _scan(tmp_path, wrapped, check_pdf.check_literals) == []
    assert _scan(tmp_path, wrapped, check_pdf.check_number_words) == []


def test_line_numbers_survive_multiline_stripping(tmp_path) -> None:
    """Blanking a match must preserve newlines, or every later finding is misattributed."""
    text = "\\Record{one\ntwo}\n\nthe ratio is 0.7314\n"
    problems = _scan(tmp_path, text, check_pdf.check_literals)
    assert len(problems) == 1
    assert ":4:" in problems[0], problems


# ------------------------------------------------------------------------ numeric literals

def test_a_measured_literal_is_still_caught(tmp_path) -> None:
    """The negative control: the check must still find a typed measurement."""
    problems = _scan(
        tmp_path, "The ratio reaches 1.494 at the loosest level.\n", check_pdf.check_literals
    )
    assert len(problems) == 1
    assert "1.494" in problems[0]


def test_a_claim_macro_is_not_a_literal(tmp_path) -> None:
    assert _scan(tmp_path, r"The ratio reaches \ClaimTemporalRatio{} there." + "\n",
                 check_pdf.check_literals) == []


def test_a_comment_is_not_scanned(tmp_path) -> None:
    assert _scan(tmp_path, "% an earlier draft said 1.494 here\n", check_pdf.check_literals) == []


# --------------------------------------------------------------------------- number words

@pytest.mark.parametrize(
    "phrase",
    [
        "a thousandfold change in FLOPs",
        "the sixty-four-fold spread",
        "roughly eight times the evaluation set",
        "seed noise is nearly three times the signal",
        "cutting copies by two thirds",
        "outside on four of five seeds",
        "a one-in-eight failure to train",
    ],
)
def test_a_figure_spelled_as_a_word_is_caught(tmp_path, phrase) -> None:
    """Spelling a measurement out defeats the literal scan completely."""
    assert _scan(tmp_path, phrase + "\n", check_pdf.check_number_words) != []


@pytest.mark.parametrize(
    "phrase",
    [
        "an issuer has three actions, not two: approve, decline, review",
        "Two breaks must not be conflated",
        "the one we bound, and the cost of that choice",
        "a third latent hazard was removed",
        "one of them was never gated",
        "half the time, on a calibrated system",
    ],
)
def test_ordinary_counting_prose_is_left_alone(tmp_path, phrase) -> None:
    """A blanket ban is unworkable: the body counts in words about ninety times, legitimately.

    Only two constructions are always doing measurement work -- a multiplier and a proportion
    of a stated total -- and everything else has to pass, or the check gets switched off.
    """
    assert _scan(tmp_path, phrase + "\n", check_pdf.check_number_words) == []


def test_the_submission_sources_pass_both_scans() -> None:
    """The built artefact itself, so a regression cannot land without a test failing."""
    sources = sorted((REPO / "submission" / "content").glob("*.tex"))
    assert sources, "no proposal sources found"
    assert check_pdf.check_literals(sources) == []
    assert check_pdf.check_number_words(sources) == []


# ------------------------------------------------------------------------ overfull lines

def _log_with(tmp_path: Path, body: str) -> Path:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"")
    pdf.with_suffix(".log").write_text(body, encoding="utf-8")
    return pdf


def test_a_badly_overfull_line_is_caught(tmp_path) -> None:
    """The defect this check was written for: 146 pt of overhang clipped a URL off the page."""
    pdf = _log_with(
        tmp_path, "Overfull \\hbox (146.4248pt too wide) in paragraph at lines 4--8\n"
    )
    problems = check_pdf.check_overfull(pdf)
    assert len(problems) == 1
    assert "146 pt" in problems[0]
    assert "4-8" in problems[0]


def test_ordinary_typesetting_slack_is_not_a_defect(tmp_path) -> None:
    """A long inline equation overhangs by a few points and stays entirely readable.

    A gate that fired on those would be switched off rather than satisfied.
    """
    pdf = _log_with(
        tmp_path, "Overfull \\hbox (46.9pt too wide) in paragraph at lines 26--32\n"
    )
    assert check_pdf.check_overfull(pdf) == []


def test_a_clean_log_passes(tmp_path) -> None:
    assert check_pdf.check_overfull(_log_with(tmp_path, "no warnings here\n")) == []


def test_a_missing_log_is_reported_rather_than_passed(tmp_path) -> None:
    """Silently passing when the evidence is absent is the worst behaviour for a gate."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"")
    problems = check_pdf.check_overfull(pdf)
    assert len(problems) == 1
    assert "missing" in problems[0]


def test_the_built_proposal_has_no_clipped_lines() -> None:
    """The artefact itself, so a regression cannot land without a test failing."""
    proposal = REPO / "submission" / "proposal.pdf"
    if not proposal.exists():
        pytest.skip("proposal.pdf is not built")
    assert check_pdf.check_overfull(proposal) == []
