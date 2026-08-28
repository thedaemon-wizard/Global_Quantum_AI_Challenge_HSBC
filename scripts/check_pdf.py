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
from pathlib import Path

from pypdf import PdfReader

REPO = Path(__file__).resolve().parents[1]

# Points, at 72 per inch.  A4 is 210 x 297 mm.
PAPER_SIZES = {"a4": (595.276, 841.890), "letter": (612.0, 792.0)}
PAPER_TOLERANCE = 2.0

# Share of characters allowed below the font floor.  Mathematical sub- and superscripts are
# legitimately smaller; a body or table that is genuinely undersized is not this rare.
SMALL_TEXT_SHARE = 0.08

# Numbers that are structural rather than measured, and so may appear literally.
ALLOWED_LITERALS = {
    "0", "1", "2", "3", "4", "5", "10", "12", "20", "60", "100",  # section/percent scaffolding
    "0.001", "0.002", "0.005", "0.010", "0.05", "0.10", "0.45",  # pre-registered grid levels
    "0.60",  # the RBF-distinctness screen threshold, fixed in configs/default.yaml
}
LITERAL = re.compile(r"(?<![\w.\\])(\d+\.\d+|\d+)(?![\w.])")


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


def _display(path: Path) -> str:
    """Repository-relative path when possible, absolute otherwise.

    Sources arrive from the Makefile as relative paths, so `relative_to(REPO)` raises on
    them; an error formatter must never be the thing that crashes the check.
    """
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO))
    except ValueError:
        return str(path)


def check_literals(sources: list[Path]) -> list[str]:
    """Numeric literals in the body, which should have arrived through a Claim macro."""
    problems = []
    for path in sources:
        if not path.exists():
            problems.append(f"source {path} does not exist")
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("%"):
                continue
            # Strip LaTeX control sequences first: \ClaimFoo, \section, lengths like 10mm,
            # and label/ref arguments all legitimately contain digits.
            cleaned = re.sub(r"\\[A-Za-z]+", " ", stripped)
            cleaned = re.sub(r"\d+(mm|pt|cm|em|ex|in|\\%)", " ", cleaned)
            for match in LITERAL.finditer(cleaned):
                if match.group(1) not in ALLOWED_LITERALS:
                    problems.append(
                        f"{_display(path)}:{number}: numeric literal "
                        f"{match.group(1)!r} -- should this be a \\Claim macro? "
                        f"({stripped[:60]})"
                    )
    if not problems and sources:
        print(f"  literals    none unaccounted for across {len(sources)} source file(s)")
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

    print(_display(args.pdf))
    reader = PdfReader(str(args.pdf))
    problems = [
        *check_pages(reader, args.max_pages),
        *check_paper(reader, args.paper),
        *check_fonts(reader, args.min_font),
        *check_literals(args.source),
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
