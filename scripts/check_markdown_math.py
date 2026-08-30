#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Assert that the mathematics in the markdown documents survives GitHub's renderer.

GitHub was the one published surface in this repository with no gate on it, and it shipped a
broken display equation in README section 4.2: "Missing or unrecognized delimiter for \\Bigl".

Every rule below was measured against GitHub's own ``POST /markdown`` endpoint and then against
MathJax, which is what GitHub renders math with.  The error string above is MathJax's wording
for ``\\Bigl{``; KaTeX answers "Expected group as argument to '\\Bigr'" for the same input,
which is how the renderer was identified.  Four transformations matter, and they differ by
where the mathematics is written:

``$...$`` and ``$$...$$``
    A backslash before ASCII punctuation is **removed**.  ``\\Bigl\\{`` arrives as ``\\Bigl{``
    and fails outright; ``\\,`` and ``\\;`` arrive as a literal comma and semicolon and render
    *without* an error, which is worse, because nothing flags it.

``$...$`` only
    A raw ``<`` or ``>`` is escaped **twice**: the renderer receives the five characters
    ``&lt;`` and answers "Misplaced &".  Inside ``$$`` or a fence it is escaped once, decodes
    back to ``<``, and is fine.

a fenced ``math`` block
    Delivered byte for byte -- ``\\;`` ``\\,`` ``\\{`` ``\\}`` all survive.  One exception: a
    row separator at the end of a line gains an extra backslash, so it must start the next line
    instead.

any ``$`` delimiter
    It is not a delimiter at all if it touches a letter or digit on the outside.  ``$\\Delta$AP``
    and ``AP$\\Delta$`` and ``$\\Delta$5`` render as literal text; ``$\\Delta$ AP`` and
    ``$\\Delta$,`` are mathematics.

Display mathematics therefore belongs in a fence, where it can keep its natural spelling.
Inline mathematics has no fence available -- a fence cannot live in a table cell -- so it must
use ``\\lbrace`` ``\\rbrace`` ``\\lt`` ``\\gt`` and no escaped punctuation at all.  And an
inline span has to open and close on one source line: across a line break GitHub emits the
literal text and no math node.

    .venv/bin/python scripts/check_markdown_math.py
    .venv/bin/python scripts/check_markdown_math.py --render DIR   # DIR resolves mathjax-full

Exit 0 if every assertion holds, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from hsbcfraud.paths import display_path

REPO = Path(__file__).resolve().parents[1]

SKIP_DIRECTORIES = frozenset({".git", ".venv", "node_modules", "__pycache__", "datasets",
                              ".pytest_cache"})

# The characters CommonMark treats as escapable.  A backslash before any of them is dropped on
# the way to the renderer; a backslash before a backslash is not, which is why the pair is
# excluded here rather than removed from the set.
ESCAPABLE = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
ESCAPE = re.compile(r"\\([" + re.escape(ESCAPABLE) + r"])")
SAFE_ESCAPE = "\\"

# What to write instead, in a span that has no fence available.  Each is the same glyph or the
# same width, spelled with an alphabetic macro name that no markdown pass touches.
REPLACEMENTS = {
    "\\{": "\\lbrace", "\\}": "\\rbrace",
    "\\,": "\\thinspace", "\\;": "\\thickspace", "\\:": "\\medspace",
}
COMPARISON = {"<": "\\lt", ">": "\\gt"}


@dataclass(frozen=True)
class MathSpan:
    """One expression GitHub will render, and how it was written."""

    path: str
    line: int
    expression: str
    display: bool
    fenced: bool
    inert: bool = False

    @property
    def where(self) -> str:
        return f"{self.path}:{self.line}"


def mangle(expression: str) -> str:
    """The expression as GitHub delivers it from a dollar-delimited span."""
    return ESCAPE.sub(lambda m: m.group(0) if m.group(1) == SAFE_ESCAPE else m.group(1),
                      expression)


def extract(text: str) -> Iterator[MathSpan]:
    """Every math span in one document, with the form it was written in.

    Code fences other than ``math`` and inline code spans are skipped: GitHub renders no
    mathematics inside either, and treating their contents as math would report every shell
    prompt in the reproduction section as a defect.
    """
    lines = text.split("\n")
    fence: tuple[str, str] | None = None
    fence_start = 0
    buffered: list[str] = []
    dollar_start: int | None = None

    for number, line in enumerate(lines, start=1):
        marker = re.match(r"^\s*(`{3,}|~{3,})\s*(\w*)", line)
        if fence is not None:
            if marker and marker.group(1).startswith(fence[0]):
                if fence[1] == "math":
                    yield MathSpan("", fence_start, "\n".join(buffered), True, True)
                fence, buffered = None, []
            else:
                buffered.append(line)
            continue
        if marker:
            fence, fence_start, buffered = (marker.group(1), marker.group(2)), number, []
            continue
        if dollar_start is not None:
            if re.fullmatch(r"\s*\$\$\s*", line):
                yield MathSpan("", dollar_start, "\n".join(buffered), True, False)
                dollar_start, buffered = None, []
            else:
                buffered.append(line)
            continue
        if re.fullmatch(r"\s*\$\$\s*", line):
            dollar_start, buffered = number, []
            continue

        bare = re.sub(r"`[^`]*`", lambda m: " " * len(m.group(0)), line)
        for match in re.finditer(r"\$\$(.+?)\$\$", bare):
            yield MathSpan("", number, match.group(1), True, False)
        # A delimiter touching an alphanumeric on the outside is not a delimiter: measured on
        # GitHub, `$\Delta$AP` and `AP$\Delta$` and `$\Delta$5` all render as literal text
        # while `$\Delta$ AP` and `$\Delta$,` are mathematics. Such a span is yielded anyway,
        # with `inert` set, so the checker can report it rather than quietly agreeing with
        # GitHub that the author did not mean mathematics.
        inline = re.sub(r"\$\$.+?\$\$", "", bare)
        for match in re.finditer(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", inline):
            before = inline[match.start() - 1] if match.start() else ""
            after = inline[match.end()] if match.end() < len(inline) else ""
            inert = before.isalnum() or after.isalnum()
            yield MathSpan("", number, match.group(1), False, False, inert)


def markdown_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.md")
                  if not SKIP_DIRECTORIES & set(p.relative_to(root).parts))


def collect(root: Path) -> list[MathSpan]:
    """Every math span in the repository's markdown, in file then line order."""
    spans: list[MathSpan] = []
    for path in markdown_files(root):
        where = display_path(path)
        for span in extract(path.read_text(encoding="utf-8")):
            # A span with no macro is a pair of dollar signs in prose, not mathematics.
            if re.search(r"\\[a-zA-Z]", span.expression):
                spans.append(MathSpan(where, span.line, span.expression, span.display,
                                      span.fenced, span.inert))
    return spans


def check_spans(spans: list[MathSpan]) -> list[str]:
    """The three transformations above, as assertions about how each span is written."""
    problems = []
    for span in spans:
        if span.inert:
            problems.append(
                f"{span.where}: a $ delimiter touching a letter or digit is not a delimiter; "
                f"GitHub renders this span as literal text. Separate it with a space, or bring "
                f"the adjacent text inside the mathematics"
            )
            continue
        if span.fenced:
            # A row separator at end of line gains a backslash inside a fence.
            if re.search(r"\\\\[ \t]*(\n|$)", span.expression):
                problems.append(
                    f"{span.where}: a row separator ends a line inside a fence, where it gains "
                    f"an extra backslash; start the next line with it instead"
                )
            continue

        eaten = sorted({m.group(0) for m in ESCAPE.finditer(span.expression)
                        if m.group(1) != SAFE_ESCAPE})
        if eaten:
            advice = ", ".join(f"{e} -> {REPLACEMENTS[e]}" for e in eaten if e in REPLACEMENTS)
            problems.append(
                f"{span.where}: {' '.join(eaten)} loses its backslash outside a fence"
                + (f"; write {advice}, or move the block into a ```math fence" if advice
                   else "; spell it with an alphabetic macro name")
            )
        if not span.display:
            raw = sorted({c for c in COMPARISON if c in span.expression})
            if raw:
                advice = ", ".join(f"{c} -> {COMPARISON[c]}" for c in raw)
                problems.append(
                    f"{span.where}: {' '.join(raw)} inside inline math reaches the renderer as "
                    f"an HTML entity and fails with 'Misplaced &'; write {advice}"
                )
    return problems


def check_unbalanced(root: Path) -> list[str]:
    """An inline span that crosses a line break is not mathematics at all."""
    problems = []
    for path in markdown_files(root):
        in_code = False
        for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
            if re.match(r"^\s*(`{3,}|~{3,})", line):
                in_code = not in_code
                continue
            if in_code:
                continue
            bare = re.sub(r"`[^`]*`", lambda m: " " * len(m.group(0)), line)
            if re.sub(r"\$\$.+?\$\$", "", bare).count("$") % 2:
                problems.append(
                    f"{display_path(path)}:{number}: an inline span opens and does not close on "
                    f"this line; GitHub emits the literal text and no math node"
                )
    return problems


# MathJax is the only oracle for whether an expression parses at all: no structural rule would
# catch an undefined macro. It needs node and a resolvable mathjax-full, so it is opt-in and
# says plainly when it did not run rather than passing quietly.
RENDER_SCRIPT = """
import { mathjax } from 'mathjax-full/js/mathjax.js'
import { TeX } from 'mathjax-full/js/input/tex.js'
import { SVG } from 'mathjax-full/js/output/svg.js'
import { liteAdaptor } from 'mathjax-full/js/adaptors/liteAdaptor.js'
import { RegisterHTMLHandler } from 'mathjax-full/js/handlers/html.js'
import { AllPackages } from 'mathjax-full/js/input/tex/AllPackages.js'
const adaptor = liteAdaptor()
RegisterHTMLHandler(adaptor)
const doc = mathjax.document('', {
  InputJax: new TeX({ packages: AllPackages }), OutputJax: new SVG({ fontCache: 'none' }),
})
const out = []
for (const s of JSON.parse(process.argv[1])) {
  // MathJax does not throw on a bad macro: it renders a red box carrying the message, so the
  // output has to be inspected. Silence is not success.
  const svg = adaptor.innerHTML(doc.convert(s.delivered, { display: s.display }))
  const error = svg.match(/data-mjx-error="([^"]*)"/)
  if (error) out.push({ ...s, problem: error[1] })
}
console.log(JSON.stringify(out))
"""


def check_render(spans: list[MathSpan], mathjax_dir: Path) -> list[str]:
    """Every expression parses, as the renderer will receive it."""
    payload = json.dumps([
        {"path": s.path, "line": s.line, "display": s.display,
         "delivered": s.expression if s.fenced else mangle(s.expression)}
        for s in spans if not s.inert
    ])
    result = subprocess.run(
        ["node", "--input-type=module", "-e", RENDER_SCRIPT, payload],
        cwd=mathjax_dir, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        detail = stderr.splitlines()[-1] if stderr else "no output"
        return [f"the renderer could not run in {display_path(mathjax_dir)}: {detail}"]
    return [f"{f['path']}:{f['line']}: {f['problem']}" for f in json.loads(result.stdout)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument(
        "--render", type=Path, metavar="MATHJAX_DIR",
        help="directory from which 'mathjax-full' resolves; enables the parse check",
    )
    args = parser.parse_args(argv)

    spans = collect(args.root)
    problems = check_spans(spans) + check_unbalanced(args.root)
    fenced = sum(1 for s in spans if s.fenced)
    print(f"{len(spans)} math span(s) in {len(markdown_files(args.root))} document(s), "
          f"{fenced} in a fence")

    if args.render is not None:
        if shutil.which("node") is None:
            problems.append("--render was requested and node is not on PATH")
        else:
            problems.extend(check_render(spans, args.render))
    else:
        print("  render      not run; pass --render MATHJAX_DIR to parse every expression")

    if problems:
        print(f"\nFAIL: {len(problems)} problem(s)", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print("  written     nothing GitHub would strip, escape twice, or split")
    if args.render is not None:
        print("  render      every expression parses as the renderer receives it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
