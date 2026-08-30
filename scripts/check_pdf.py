#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Assert the submission's format constraints against the built PDF.

Page count, paper size and minimum font size are portal requirements.  Checking them by
looking at the document is unreliable: a page count changes when a table reflows, and a
\\small nested inside a footnote can drop below the floor without being visible at reading
size.  These are therefore build assertions, read out of the PDF itself.

One further check has nothing to do with the portal.  ``--source`` files are scanned for
numeric literals, because every figure in this submission is supposed to arrive through a
``\\Claim`` macro generated from ``docs/claims.yaml``.  A literal in the body means a number
was typed rather than measured, which is the defect class that produced four prose-versus-table
disagreements in this project.

    .venv/bin/python scripts/check_pdf.py submission/proposal.pdf --max-pages 5 \\
        --paper a4 --min-font 10 --source submission/content/03-results.tex

Exit 0 if every assertion holds, 1 otherwise.  ``--report-only`` prints the same findings and
exits 0, for the fitting loop where an over-length draft must still be produced.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

from pypdf import PdfReader

from hsbcfraud.paths import display_path

REPO = Path(__file__).resolve().parents[1]

# How far a line may exceed the text block before it is treated as a defect rather than as
# typesetting slack.  A long inline equation routinely overhangs by a few points and remains
# entirely readable.
#
# This was 60 pt, chosen to catch a portfolio URL that overhung by 146 pt and lost its last two
# words off the page edge.  A threshold set to catch a catastrophe does not enforce quality: at
# 60 pt it certified an inline equation overhanging by 47 pt, which put text 15.5 mm into an
# 18 mm margin on page 4 and was found by a human looking at the PDF, not by this file.  The
# comment above says "a few points", so the number now says a few points.
OVERFULL_TOLERANCE_PT = 12.0

# The text block of the built documents, in points: A4 at an 18 mm margin, matching
# submission/preamble.tex.  Used by check_margins, which measures the artefact rather than
# LaTeX's complaint about it.
A4_WIDTH_PT = 595.276
MARGIN_PT = 18.0 / 25.4 * 72.0

# Slack allowed when comparing measured glyph positions to that block.  Extracted x positions
# are the origin of a text run, not its right edge, and italic correction and kerning move them
# by a point or so; 2 pt keeps ordinary typesetting from registering as a breach.
MARGIN_SLACK_PT = 2.0
OVERFULL = re.compile(
    r"^Overfull \\hbox \(([\d.]+)pt too wide\) in paragraph at lines (\d+)--(\d+)",
    re.MULTILINE,
)

# Points, at 72 per inch.  A4 is 210 x 297 mm.
PAPER_SIZES = {"a4": (595.276, 841.890), "letter": (612.0, 792.0)}
PAPER_TOLERANCE = 2.0

# Share of characters allowed below the font floor.  Mathematical sub- and superscripts are
# legitimately smaller; a body or table that is genuinely undersized is not this rare.
SMALL_TEXT_SHARE = 0.08

# Numbers that are structural rather than measured, and so may appear literally.
ALLOWED_LITERALS = {
    "0", "1", "2", "3", "4", "5", "10", "12", "20", "60", "100",  # section/percent scaffolding
    "0.001", "0.002", "0.005", "0.01", "0.010", "0.05", "0.10", "0.45",  # pre-registered levels
    "0.60",  # the RBF-distinctness screen threshold, fixed in configs/default.yaml
    "90",  # the PSD2 SCA-RTS rolling window, in days
    "431",  # sites in the full-scale chain, i.e. the feature count
    "64",  # SUPPORT_ROWS in scripts/measure_latency.py: an assumption the kernel timing prices
}
# A digit run that is not part of an identifier.  The trailing lookahead stops mid-number
# matches; the leading one excludes both word characters and a preceding "letter-hyphen",
# which is what makes "Apache-2.0" and "3-D Secure" identifiers rather than figures.
#
# `\.\w` in the trailing lookahead rather than a bare `.`: a full stop ending a sentence is not
# part of the number, and excluding it hid every figure written at the end of one -- "0.89." and
# "0.5000." both shipped unbound while this file reported nothing unaccounted for.  A full stop
# followed by a word character still is part of the token, which keeps "section 7.A" and a
# three-part version out.
LITERAL = re.compile(r"(?<![\w.\\])(?<![A-Za-z]-)(\d+\.\d+|\d+)(?![\w]|\.\w)")

# Commands whose braced argument is an identifier or document metadata rather than prose.
# Their digits belong to a filename, a cross-reference key or the challenge year, none of
# which can resolve to a table.  Stripping the command name alone is not enough: an earlier
# version removed "\input" but left "{content/06-hybrid}", and adding two sections to the
# proposal then failed the whole build on six of its own filenames.
STRUCTURAL_COMMANDS = (
    "input", "include", "includegraphics", "label", "ref", "eqref", "cite",
    "bibliography", "title", "author", "date", "usepackage", "documentclass",
    # Typesetting configuration.  A float fraction or a margin is a layout constant, not a
    # measurement, and it can never resolve to a table.  These take two braced arguments,
    # which is why the pattern below allows more than one.
    "renewcommand", "setlength", "addtolength", "setcounter",
)
# Commands that declare a figure as belonging to another work: a cited paper's result and
# the author's own biographical record.  Neither can resolve to a table in this repository,
# so both are declared in the source and removed before scanning.
DECLARED_EXTERNAL = ("Cited", "Record")

# One braced group, allowing a single level of nesting inside it.  A flat `[^{}]*` was not
# enough: `\author{... {\normalsize \url{...}} ...}` nests, the group failed to match, and the
# scanner then reported the challenge year in the title block as an unbound measurement.  That
# is the fourth wrong verdict from this checker and the third caused by arguments rather than
# by the commands themselves, so the pattern is widened rather than the case excused.
#
# One level, not arbitrary depth: regex cannot balance braces in general, and a document that
# needs two levels inside a structural argument is one worth looking at by hand.
#
# A command's arguments must also sit on ONE source line.  The scan is line-by-line so that a
# failure can name a line, and a brace opened on one line and closed on another is beyond any
# pattern applied to a single line.  That is a real limit, not a defect to work around in the
# document: a structural command whose argument spans lines should be joined.
_BRACED = r"\{(?:[^{}]|\{[^{}]*\})*\}"

# A command from either group together with its optional and braced arguments.
ARGUMENT_BEARING = re.compile(
    r"\\(?:" + "|".join((*DECLARED_EXTERNAL, *STRUCTURAL_COMMANDS)) + r")\b"
    r"\s*(?:\[[^\]]*\])?\s*(?:" + _BRACED + r"\s*){0,2}"
)

# Number words, composed from the parts English builds them from rather than enumerated, so
# that compounds like "sixty-four" and "twenty-five" fall out instead of needing entries.
_NUMBER_WORD = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|"
    r"sixty|seventy|eighty|ninety|hundred|thousand|million|billion|dozen)"
)
_COMPOUND = rf"{_NUMBER_WORD}(?:[-\s]{_NUMBER_WORD})*"

# "sixty-four-fold", "thousandfold", "eight times", "two thirds", "twice".  The separator is
# optional before "fold" because English writes it closed for the round scales.
MULTIPLIER = re.compile(
    rf"\b(?:{_COMPOUND}[-\s]?(?:fold|times)|{_COMPOUND}[-\s](?:thirds|quarters|halves))\b",
    re.IGNORECASE,
)
# "four of five", "one-in-eight", "two of sixteen": a count against a stated total.
PROPORTION = re.compile(
    rf"\b{_COMPOUND}[-\s](?:of|in)[-\s]{_COMPOUND}\b", re.IGNORECASE
)


def check_pages(reader: PdfReader, limit: int) -> list[str]:
    n = len(reader.pages)
    if n > limit:
        return [f"page count {n} exceeds the limit of {limit}"]
    print(f"  pages       {n} of {limit}")
    return []


def check_paper(reader: PdfReader, name: str) -> list[str]:
    expected = PAPER_SIZES[name]
    problems = []
    for index, page in enumerate(reader.pages, start=1):
        box = page.mediabox
        actual = (float(box.width), float(box.height))
        if any(abs(a - e) > PAPER_TOLERANCE for a, e in zip(actual, expected, strict=True)):
            problems.append(
                f"page {index} is {actual[0]:.1f} x {actual[1]:.1f} pt, "
                f"not {name.upper()} ({expected[0]:.1f} x {expected[1]:.1f})"
            )
    if not problems:
        print(f"  paper       {name.upper()} on all {len(reader.pages)} pages")
    return problems


def check_fonts(reader: PdfReader, floor: float) -> list[str]:
    """Smallest rendered font size, gathered by instrumenting the text extractor.

    pypdf exposes font size through a visitor rather than as a property, so the sizes are
    collected during extraction.  The declared size is not the rendered one: it is scaled by
    the current transformation matrix and the text matrix, and a document that sets a 10 pt
    font inside a 0.9 scale renders at 9 pt.  Both are applied here.

    Sizes at or below zero are skipped -- they come from degenerate matrices belonging to
    rules and other non-glyph operators.
    """
    smallest: list[tuple[float, int]] = []

    def visit(text, cm, tm, font_dict, font_size) -> None:  # pypdf callback signature
        if not text or not text.strip() or font_size is None:
            return
        # Index 3, the vertical scale, not index 0.  A PDF text matrix is [a b c d e f] and
        # glyph height -- which is what a font size is -- follows d.  Reading a instead
        # measures horizontal expansion, and microtype routinely stretches glyphs by a couple
        # of percent to improve justification: an earlier version of this check reported a
        # compliant 10 pt body as a spread of 9.8, 9.9, 10.1 and 10.2 pt sizes and declared
        # 37 % of the document under-sized.  The document was fine; the check was wrong.
        scale = 1.0
        for matrix in (cm, tm):
            if matrix and len(matrix) > 3 and abs(float(matrix[3])) > 0:
                scale *= abs(float(matrix[3]))
        size = abs(float(font_size)) * scale
        if size > 0.01:
            smallest.append((size, len(text.strip())))

    for page in reader.pages:
        page.extract_text(visitor_text=visit)

    if not smallest:
        return ["no text found in the PDF; the font-size check could not run"]

    # The absolute minimum is the wrong statistic.  LaTeX derives script sizes from the base
    # size -- scriptstyle at 0.7x, scriptscriptstyle at 0.5x -- so a compliant 10 pt document
    # renders subscripts at 7 pt and nested subscripts at 5 pt.  Failing on those would make
    # the check unpassable for any document containing mathematics, and a check that cannot
    # pass gets disabled rather than satisfied.
    #
    # What the portal's floor means is body text, so that is what is measured: the size
    # carrying the most characters.  The minimum is still reported, and a floor breach is
    # raised only when a substantial share of the document sits below it, which is what an
    # actually-undersized body or table would look like.
    weighted: dict[float, int] = {}
    for size, length in smallest:
        weighted[round(size, 1)] = weighted.get(round(size, 1), 0) + length

    total = sum(weighted.values())
    body = max(weighted, key=lambda key: weighted[key])
    below = sum(count for size, count in weighted.items() if size < floor - 0.05)
    share = below / total

    problems = []
    if body < floor - 0.05:
        problems.append(
            f"body text renders at {body:.2f} pt, below the {floor} pt floor "
            f"({weighted[body] / total:.0%} of characters)"
        )
    if share > SMALL_TEXT_SHARE:
        problems.append(
            f"{share:.0%} of characters render below {floor} pt, above the "
            f"{SMALL_TEXT_SHARE:.0%} allowance for mathematical scripts; check for an "
            "undersized table or a \\footnotesize block"
        )
    if not problems:
        print(
            f"  font        body {body:.2f} pt, floor {floor}; "
            f"{share:.1%} of characters below it (scripts), smallest "
            f"{min(s for s, _ in smallest):.2f} pt"
        )
    return problems



def check_margins(pdf: Path, slack: float = MARGIN_SLACK_PT) -> list[str]:
    """Text that actually falls outside the text block, measured page by page.

    Every other check in this file reads what LaTeX *said*.  This one reads what the reader
    *gets*: the rightmost glyph position on each page, against the right edge of the text
    block.  The difference is not academic.  An inline equation on page 4 overhung by 47 pt and
    put text 15.5 mm into an 18 mm margin, 6.9 pt from the paper edge; the log-based check saw
    it, compared it to a tolerance set for a worse case, and passed the document.  Meanwhile a
    31 pt overfull box in the centred title block puts no text past the margin at all and is
    not a defect.  Reading the log cannot tell those apart.  Measuring can.

    Only the right edge is checked.  Overhang to the left would need a negative-indent bug
    rather than an unbreakable box, and nothing in this document tree can produce one.
    """
    import pypdf

    reader = pypdf.PdfReader(pdf)
    problems = []
    worst = 0.0
    for number, page in enumerate(reader.pages, start=1):
        limit = float(page.mediabox.width) - MARGIN_PT
        positions: list[float] = []

        def collect(text, _cm, matrix, _font, _size, sink=positions):
            if text.strip():
                sink.append(matrix[4])

        page.extract_text(visitor_text=collect)
        if not positions:
            continue
        rightmost = max(positions)
        worst = max(worst, rightmost - limit)
        if rightmost > limit + slack:
            problems.append(
                f"page {number}: text begins {rightmost - limit:.0f} pt past the right margin "
                f"({(rightmost - limit) / 72 * 25.4:.1f} mm), so it runs toward the paper edge"
            )
    if not problems:
        print(f"  margins     no text outside the text block; closest {worst:+.0f} pt")
    return problems


def check_overfull(pdf: Path, tolerance: float = OVERFULL_TOLERANCE_PT) -> list[str]:
    """Lines LaTeX could not fit, read from the build log beside the PDF.

    LaTeX reports these as warnings and carries on, so an overfull line ships silently. Small
    overhangs are normal; a large one means text has run past the margin and been clipped, and
    a reader sees a truncated URL or a missing word rather than an error.
    """
    log = pdf.with_suffix(".log")
    if not log.exists():
        return [f"{display_path(log)} is missing; the overfull-line check could not run"]

    problems = []
    worst = 0.0
    count = 0
    for match in OVERFULL.finditer(log.read_text(encoding="utf-8", errors="replace")):
        width, first, last = float(match.group(1)), match.group(2), match.group(3)
        count += 1
        worst = max(worst, width)
        if width > tolerance:
            problems.append(
                f"a line overhangs the text block by {width:.0f} pt at source lines "
                f"{first}-{last}; text that far past the margin is clipped from the page"
            )
    if not problems:
        print(
            f"  overfull    {count} line(s) overhang, worst {worst:.0f} pt, "
            f"tolerance {tolerance:.0f} pt"
        )
    return problems


def _blank_out(match: re.Match[str]) -> str:
    """Replace a match with blanks, keeping its newlines so line numbers do not shift."""
    return "".join("\n" if character == "\n" else " " for character in match.group(0))


def _strip_markup(text: str) -> str:
    """Remove the LaTeX that legitimately carries digits, leaving the prose to scan.

    Applied to the **whole document**, not line by line.  A ``\\Record{...}`` or ``\\Cited{...}``
    wrapped across two source lines would otherwise be stripped on its first line and leak its
    argument onto the second, which made the check's verdict depend on where the text happened
    to wrap.  Matches are blanked rather than deleted so that every line keeps its number.

    Argument-bearing commands go first because they are the more specific pattern: removing
    the bare control sequence first would leave the filename in ``\\input{content/06-hybrid}``
    behind as free-standing prose.
    """
    cleaned = ARGUMENT_BEARING.sub(_blank_out, text)
    # Remaining control sequences -- \ClaimFoo, \section, \textbf -- and typeset lengths.
    cleaned = re.sub(r"\\[A-Za-z]+", " ", cleaned)
    return re.sub(r"\d+(mm|pt|cm|em|ex|in|\\%)", " ", cleaned)


def _scan_sources(sources: list[Path], find: Callable[[str], Iterator[str]]) -> list[str]:
    """Apply a line-level finder to every non-comment line of every source.

    Shared by the literal and number-word checks so that both see exactly the same view of
    the document; a divergence there would let a figure hide from one check by satisfying the
    other's idea of what counts as markup.
    """
    problems = []
    for path in sources:
        if not path.exists():
            problems.append(f"source {path} does not exist")
            continue
        source = path.read_text(encoding="utf-8")
        # Stripped once, over the whole file, so a command wrapped across lines is handled.
        # _strip_markup preserves newlines, so these two lists stay aligned by index.
        for number, (raw, cleaned) in enumerate(
            zip(source.splitlines(), _strip_markup(source).splitlines(), strict=True), start=1
        ):
            if raw.strip().startswith("%"):
                continue
            for complaint in find(cleaned):
                problems.append(
                    f"{display_path(path)}:{number}: {complaint} ({raw.strip()[:60]})"
                )
    return problems


def check_literals(sources: list[Path]) -> list[str]:
    """Numeric literals in the body, which should have arrived through a Claim macro."""

    def find(cleaned: str) -> Iterator[str]:
        for match in LITERAL.finditer(cleaned):
            if match.group(1) not in ALLOWED_LITERALS:
                yield (
                    f"numeric literal {match.group(1)!r} -- should this be a \\Claim macro?"
                )

    problems = _scan_sources(sources, find)
    if not problems and sources:
        print(f"  literals    none unaccounted for across {len(sources)} source file(s)")
    return problems


def check_number_words(sources: list[Path]) -> list[str]:
    """Figures spelled as words, which the literal scan cannot see.

    Spelling a measurement out defeats ``LITERAL`` completely, and the submission accumulated
    several that way: a "thousandfold" change in FLOPs where the measured grid spans
    sixty-four, and "roughly eight times" the evaluation set where no table gives eight.

    A blanket ban is not workable -- the body legitimately says "three actions, not two" and
    "one of five" ninety-odd times -- so this targets the two constructions that are always
    doing measurement work rather than counting in prose:

    * a **multiplier**: "sixty-four-fold", "thousandfold", "eight times", "two thirds";
    * a **proportion of a stated total**: "four of five", "one-in-eight".

    Both must arrive through a ``\\Claim`` macro, because both are ratios of measured
    quantities and both change when the tables change.
    """

    def find(cleaned: str) -> Iterator[str]:
        for pattern, kind in ((MULTIPLIER, "multiplier"), (PROPORTION, "proportion")):
            for match in pattern.finditer(cleaned):
                yield (
                    f"{kind} written as words, {match.group(0).strip()!r} -- "
                    "spell it with a \\Claim macro so it tracks the table"
                )

    problems = _scan_sources(sources, find)
    if not problems and sources:
        print(f"  number words none doing measurement work across {len(sources)} source file(s)")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--max-pages", type=int, required=True)
    parser.add_argument("--paper", choices=sorted(PAPER_SIZES), default="a4")
    parser.add_argument("--min-font", type=float, default=10.0)
    parser.add_argument("--source", type=Path, action="append", default=[])
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)

    if not args.pdf.exists():
        print(f"{args.pdf} does not exist; build it first", file=sys.stderr)
        return 1

    print(display_path(args.pdf))
    reader = PdfReader(str(args.pdf))
    problems = [
        *check_pages(reader, args.max_pages),
        *check_paper(reader, args.paper),
        *check_fonts(reader, args.min_font),
        *check_overfull(args.pdf),
        *check_margins(args.pdf),
        *check_literals(args.source),
        *check_number_words(args.source),
    ]

    if not problems:
        print("  all format assertions hold")
        return 0

    label = "REPORT" if args.report_only else "FAIL"
    print(f"\n{label}: {len(problems)} problem(s)", file=sys.stderr)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    return 0 if args.report_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
