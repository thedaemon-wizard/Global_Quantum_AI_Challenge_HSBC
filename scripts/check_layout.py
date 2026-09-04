#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
r"""Refuse a built PDF in which a caption, a table body or a heading is broken across a page.

Submission Guidelines 6.1 screens format for eligibility *before* any scoring, so a table whose
number sits at the foot of one page while its rows open the next is not a cosmetic complaint.
It is read as carelessness by the first person to open the file, and it cannot be argued away
afterwards.  On 2026-09-04 exactly that shipped: Table 1's caption was the last line of page 4
and its header and four data rows began page 5.

Four classes, all hard failures:

  C0  the PDF and the .tex sources disagree about how many captions or headings exist.
      Not a layout defect in itself; it means the other checks are inspecting a document the
      sources no longer describe, so it fails rather than silently passing on a subset.
  C1  a table caption is on a different page from its body.
  C2  a table body straddles a page break, or runs off the foot of a page.
  C3  a heading is stranded at the foot of a page with no body text beneath it.
  C4  a figure caption is on a different page from its image.

Four properties were measured rather than assumed, and each is load-bearing:

* **Rules are stroked paths, never filled rectangles.**  Every rule in every document here is
  drawn with ``m``/``l``/``S``.  A detector that scans for ``re`` followed by ``f`` -- the
  obvious first guess, and the one tried first here -- finds *zero* rules and passes
  everything.  Booktabs stroke weights separate them: 0.797 pt for ``\toprule`` and
  ``\bottomrule`` against 0.498 pt for ``\midrule``, so a body is bounded by a *pair* of heavy
  rules and an odd count of them is itself a C2.
* **pypdf hands the visitor ``cm`` and ``tm`` separately.**  Using ``tm[5]`` raw misplaces
  everything drawn inside a Form XObject; the baseline is the translation of ``tm x cm``.
* **Caption direction is read from the source, never assumed.**  Tables here caption *above*
  their body and figures caption *below* theirs, so a detector with one hard-coded direction is
  wrong about half the objects.  :func:`tex_floats` compares the line number of the caption
  against that of the first body anchor and derives the direction per object.
* **The folio is identified by content, not by a y threshold.**  A threshold that clears a page
  number at y=21 also discards real body text, which in the HSBC proposal reaches y=29.9.

Captions are matched anchored and with a font-size guard, which is what stops the 27 in-text
mentions of "Table N" in these documents from being read as captions.  A caption is paired to a
body by *proximity*, not by index: a deferred float and an inline block can reach the page in an
order that does not match caption numbering, and index pairing would then blame the wrong
object.  C1 tests page identity alone and is deliberately threshold-free, because LaTeX cannot
insert prose between a caption and its body -- only a page break can separate them -- while a
"no lines between them" rule instead false-positives on every multi-line caption.

    .venv/bin/python scripts/check_layout.py submission/proposal.pdf submission/proposal.tex
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from pypdf import PdfReader
from pypdf.generic import ContentStream

IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

#: Booktabs stroke weights.  ``\toprule`` and ``\bottomrule`` share one weight and ``\midrule``
#: is lighter, which is what lets a body be bounded without parsing the table.
TOPRULE_WIDTH = 0.797
MIDRULE_WIDTH = 0.498
RULE_WIDTH_TOLERANCE = 0.06

#: Anchored, and requiring the number and the colon.  Unanchored matching reads "Appendix
#: Table 2" in running prose as a caption.
CAPTION_RE = re.compile(r"^(Table|Figure)\s+(\d+)\s*[:.]")

SECTION_RE = re.compile(
    r"\\(section|subsection|subsubsection|paragraph|subparagraph)(\*?)\s*(?:\[[^\]]*\])?\s*\{"
)
INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
CAPTION_CMD_RE = re.compile(r"\\caption(?:of\s*\{(?P<kind>[a-z]+)\})?\s*(?:\[[^\]]*\])?\s*\{")
LABEL_RE = re.compile(r"\\label\s*\{([^}]+)\}")
BEGIN_RE = re.compile(r"\\begin\s*\{([a-zA-Z*]+)\}")
END_RE = re.compile(r"\\end\s*\{([a-zA-Z*]+)\}")
GRAPHIC_RE = re.compile(r"\\includegraphics|\\begin\s*\{tikzpicture\}")
TABULAR_RE = re.compile(r"\\begin\s*\{(tabular[x*]?|longtable|tabu)\}")

#: A generated table macro on a line of its own -- ``\TableReynoldsSweep``.  The bodies of
#: three of these tables arrive this way, so a scanner looking only for ``\begin{tabular}``
#: would find no body and derive the caption direction wrongly.
GENERATED_TABLE_RE = re.compile(r"^\s*\\(Table[A-Za-z]+)\s*(?:\{\})?\s*$")

PAINT_OPS = {"S", "s", "f", "F", "f*", "B", "B*", "b", "b*"}
FOLIO_RE = re.compile(r"^[0-9ivxlcIVXLC]{1,6}$")

#: Environments that can carry a caption and a body.  ``keptwithcaption`` is this project's
#: unbreakable table environment, defined in ``submission/preamble.tex``; it is listed here
#: rather than passed in so that running the script by hand needs no arguments.  A project
#: naming its environment differently extends the set with ``--caption-env``.
CAPTION_ENVS = ("figure", "figure*", "table", "table*", "center", "keptwithcaption")


def matrix_multiply(m, n):
    """``m`` then ``n``, in the PDF's [a b c d e f] convention."""
    a, b, c, d, e, f = m
    A, B, C, D, E, F = n
    return (a * A + b * C, a * B + b * D,
            c * A + d * C, c * B + d * D,
            e * A + f * C + E, e * B + f * D + F)


def transform(m, x, y):
    a, b, c, d, e, f = m
    return (a * x + c * y + e, b * x + d * y + f)


def normalise(text):
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def is_folio(line, page_number, page_lines_):
    """Is this text line the page number?

    Identified by *content* rather than by a y threshold: the HSBC proposal carries real body
    text down to y=29.9, below any threshold that would clear a folio at y=21.1.
    """
    text = line["text"].strip()
    if not FOLIO_RE.match(text):
        return False
    if text == str(page_number):
        return True
    return line["y"] <= min(other["y"] for other in page_lines_) + 0.5


def page_lines(page):
    """Text lines with their composed baseline, sorted down the page."""
    fragments = []

    def visit(text, cm, tm, font_dict, font_size):
        if not text.strip():
            return
        placed = matrix_multiply(tuple(float(v) for v in tm), tuple(float(v) for v in cm))
        fragments.append((placed[5], placed[4], float(font_size or 0.0),
                          tuple(float(v) for v in cm) == IDENTITY, text))

    page.extract_text(visitor_text=visit)

    buckets = {}
    for y, x, size, identity, text in fragments:
        buckets.setdefault(round(y, 1), []).append((x, size, identity, text))

    lines = []
    for y in sorted(buckets, reverse=True):
        items = sorted(buckets[y], key=lambda item: item[0])
        lines.append({
            "y": y,
            "x0": items[0][0],
            "size": max(item[1] for item in items),
            "text": re.sub(r"\s+", " ", "".join(item[3] for item in items)).strip(),
            "identity": all(item[2] for item in items),
        })
    return lines


def page_graphics(page, reader):
    """Horizontal rules and ink extents, from the page's own operators.

    Returns ``(rules, marks)``.  ``marks`` are bounding boxes of everything painted, including
    placed XObjects, and are what a figure's caption is tested against.
    """
    resources = page.get("/Resources", {}) or {}
    xobjects = resources.get("/XObject")
    xobjects = xobjects.get_object() if xobjects is not None else {}
    contents = page.get_contents()
    if contents is None:
        return [], []

    ctm = (1, 0, 0, 1, 0, 0)
    line_width = 1.0
    stack = []
    path = []
    rules = []
    marks = []

    for operands, op in ContentStream(contents, reader).operations:
        name = op.decode() if isinstance(op, bytes) else op
        try:
            if name == "q":
                stack.append((ctm, line_width))
            elif name == "Q":
                if stack:
                    ctm, line_width = stack.pop()
            elif name == "cm":
                ctm = matrix_multiply(tuple(float(v) for v in operands), ctm)
            elif name == "w":
                line_width = float(operands[0])
            elif name in ("m", "l"):
                path.append(transform(ctm, float(operands[0]), float(operands[1])))
            elif name == "c":
                for i in (0, 2, 4):
                    path.append(transform(ctm, float(operands[i]), float(operands[i + 1])))
            elif name in ("v", "y"):
                for i in (0, 2):
                    path.append(transform(ctm, float(operands[i]), float(operands[i + 1])))
            elif name == "re":
                x, y, w, h = (float(v) for v in operands[:4])
                path += [transform(ctm, x, y), transform(ctm, x + w, y),
                         transform(ctm, x + w, y + h), transform(ctm, x, y + h)]
            elif name in PAINT_OPS:
                if path:
                    xs = [p[0] for p in path]
                    ys = [p[1] for p in path]
                    marks.append((min(xs), min(ys), max(xs), max(ys)))
                    horizontal = max(ys) - min(ys) < 0.3 and max(xs) - min(xs) > 20
                    if name in ("S", "s") and horizontal:
                        scale = (abs(ctm[0]) + abs(ctm[3])) / 2 or 1.0
                        rules.append({
                            "y": (min(ys) + max(ys)) / 2,
                            "x0": min(xs),
                            "x1": max(xs),
                            "width": line_width * scale,
                        })
                path = []
            elif name == "n":
                path = []
            elif name == "Do":
                placed = _xobject_box(xobjects, operands[0], ctm)
                if placed is not None:
                    marks.append(placed)
        except (ValueError, TypeError, IndexError):
            # A malformed operand should not blind the whole check; drop the path and go on.
            path = []
    return rules, marks


def _xobject_box(xobjects, name, ctm):
    obj = xobjects.get(name)
    if obj is None:
        return None
    obj = obj.get_object()
    subtype = obj.get("/Subtype")
    if subtype == "/Form" and obj.get("/BBox"):
        box = [float(v) for v in obj["/BBox"]]
        corners = [transform(ctm, box[0], box[1]), transform(ctm, box[2], box[1]),
                   transform(ctm, box[2], box[3]), transform(ctm, box[0], box[3])]
    elif subtype == "/Image":
        corners = [transform(ctm, 0, 0), transform(ctm, 1, 0),
                   transform(ctm, 1, 1), transform(ctm, 0, 1)]
    else:
        return None
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def read_pdf(path):
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        rules, marks = page_graphics(page, reader)
        pages.append({
            "lines": page_lines(page),
            "rules": rules,
            "marks": marks,
            "height": float(page.mediabox.height),
        })
    return pages


def body_size(pages):
    """The dominant type size, weighted by characters -- the body, not a caption or a label."""
    counter = Counter()
    for page in pages:
        for line in page["lines"]:
            counter[round(line["size"], 1)] += len(line["text"])
    return counter.most_common(1)[0][0] if counter else 10.0


def expand(root, seen=None, base=None):
    """Every source line of the document, following ``\\input`` and ``\\include``."""
    seen = seen if seen is not None else set()
    path = Path(root).resolve()
    if not path.suffix:
        path = path.with_suffix(".tex")
    if path in seen or not path.exists():
        return []
    seen.add(path)
    if base is None:
        base = path.parent

    lines = []
    source = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for number, raw in enumerate(source, 1):
        line = re.sub(r"(?<!\\)%.*$", "", raw)
        match = INPUT_RE.search(line)
        if not match:
            lines.append((path, number, line))
            continue
        child = None
        for candidate in (base / match.group(1), path.parent / match.group(1)):
            candidate = candidate if candidate.suffix else candidate.with_suffix(".tex")
            if candidate.exists():
                child = candidate
                break
        if child is not None:
            lines.extend(expand(child, seen, base))
    return lines


def brace_argument(text, start):
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif text[i] == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
    return text[start + 1:]


def strip_tex(text):
    text = re.sub(r"\\[a-zA-Z]+\s*\*?", " ", text)
    return re.sub(r"\s+", " ", re.sub(r"[{}$~\\]", " ", text)).strip()


def tex_floats(root, envs=CAPTION_ENVS):
    """Every captioned object in the sources, with the direction its caption sits."""
    units = []
    stack = []
    for path, number, text in expand(root):
        for match in BEGIN_RE.finditer(text):
            if match.group(1) in envs:
                stack.append({"env": match.group(1), "file": path, "start": number, "items": []})
        for match in CAPTION_CMD_RE.finditer(text):
            if stack:
                stack[-1]["items"].append(("caption", match.group("kind"), number))
        if GRAPHIC_RE.search(text) and stack:
            stack[-1]["items"].append(("graphic", None, number))
        if (TABULAR_RE.search(text) or GENERATED_TABLE_RE.match(text)) and stack:
            stack[-1]["items"].append(("tabular", None, number))
        for match in LABEL_RE.finditer(text):
            if stack:
                stack[-1]["items"].append(("label", match.group(1), number))
        for match in END_RE.finditer(text):
            if stack and stack[-1]["env"] == match.group(1):
                unit = stack.pop()
                unit["end"] = number
                if any(item[0] == "caption" for item in unit["items"]):
                    units.append(unit)

    resolved = []
    sequence = {"figure": 0, "table": 0}
    for unit in units:
        captions = [i for i in unit["items"] if i[0] == "caption"]
        bodies = [i for i in unit["items"] if i[0] in ("graphic", "tabular")]
        kind = captions[0][1] or _infer_kind(unit["env"], bodies)
        if kind not in ("figure", "table"):
            kind = "figure"
        sequence[kind] += 1
        labels = [i[1] for i in unit["items"] if i[0] == "label"]
        direction = None
        if captions and bodies:
            direction = "above" if captions[0][2] < bodies[0][2] else "below"
        resolved.append({
            "kind": kind,
            "number": sequence[kind],
            "env": unit["env"],
            "label": labels[0] if labels else None,
            "is_float": unit["env"].startswith(("figure", "table")),
            "direction": direction,
            "file": str(unit["file"]),
            "start": unit["start"],
            "end": unit["end"],
        })
    return resolved


def _infer_kind(env, bodies):
    if env.startswith("figure"):
        return "figure"
    if env.startswith("table"):
        return "table"
    return "table" if any(item[0] == "tabular" for item in bodies) else "figure"


def tex_headings(root):
    headings = []
    for path, number, line in expand(root):
        for match in SECTION_RE.finditer(line):
            text = strip_tex(brace_argument(line, match.end() - 1))
            if text:
                headings.append({
                    "level": match.group(1),
                    "starred": bool(match.group(2)),
                    "text": text,
                    "file": str(path),
                    "line": number,
                })
    return headings


def pdf_captions(pages, size):
    """Caption lines, excluding in-text cross-references and drawing labels."""
    captions = []
    for index, page in enumerate(pages, 1):
        for line in page["lines"]:
            match = CAPTION_RE.match(line["text"])
            if match and abs(line["size"] - size) < 0.6:
                captions.append({
                    "page": index,
                    "y": line["y"],
                    "kind": match.group(1).lower(),
                    "number": int(match.group(2)),
                    "text": line["text"],
                })
    return captions


def table_bodies(pages):
    """Each table body as the pair of heavy rules bounding it, in document order."""
    heavy = []
    for index, page in enumerate(pages, 1):
        for rule in sorted(page["rules"], key=lambda r: -r["y"]):
            if abs(rule["width"] - TOPRULE_WIDTH) < RULE_WIDTH_TOLERANCE:
                heavy.append((index, rule))

    bodies = []
    for i in range(0, len(heavy) - 1, 2):
        (top_page, top), (bottom_page, bottom) = heavy[i], heavy[i + 1]
        bodies.append({
            "top_page": top_page,
            "top_y": top["y"],
            "bot_page": bottom_page,
            "bot_y": bottom["y"],
        })
    return bodies, len(heavy) % 2 == 1


def locate_headings(pages, headings, size):
    """Match each source heading to the line that renders it, in document order."""
    flat = [(index, line) for index, page in enumerate(pages, 1) for line in page["lines"]]
    located = []
    cursor = 0
    for heading in headings:
        target = normalise(heading["text"])[:26]
        hit = None
        for k in range(cursor, len(flat)):
            index, line = flat[k]
            if line["size"] < size - 0.6:
                continue
            candidate = normalise(line["text"])
            at = candidate.find(target) if target else -1
            if at != -1 and at <= 6:
                hit = (index, line, candidate)
                cursor = k + 1
                break
        if hit is None:
            located.append({**heading, "page": None, "y": None, "pdftext": None, "runin": False})
            continue
        index, line, candidate = hit
        # A run-in heading shares its line with the paragraph it introduces, so "nothing below
        # it on this page" is not a defect: the text is beside it, not under it.
        runin = len(candidate) > len(target) * 1.35 + 8
        located.append({**heading, "page": index, "y": line["y"],
                        "pdftext": line["text"], "runin": runin})
    return located


def check(pdf_path, tex_path, max_gap=36.0, min_after=2, folio_y=28.0, envs=CAPTION_ENVS):
    pages = read_pdf(pdf_path)
    size = body_size(pages)
    captions = pdf_captions(pages, size)
    bodies, odd_rule_count = table_bodies(pages)
    floats = tex_floats(tex_path, envs)
    by_number = {(f["kind"], f["number"]): f for f in floats}
    name = Path(pdf_path).name
    findings = []

    for kind in ("table", "figure"):
        in_tex = sum(1 for f in floats if f["kind"] == kind)
        in_pdf = sum(1 for c in captions if c["kind"] == kind)
        if in_tex != in_pdf:
            findings.append(("C0", f"{name}: {kind} captions: {in_pdf} in PDF, {in_tex} in .tex"))

    tables = sorted((c for c in captions if c["kind"] == "table"), key=lambda c: c["number"])
    if len(tables) != len(bodies):
        findings.append(("C0", f"{name}: {len(tables)} table captions but "
                               f"{len(bodies)} booktabs bodies"))
    if odd_rule_count:
        findings.append(("C2", f"{name}: odd number of heavy rules -- a table body is missing "
                               f"its closing rule, which is what a body split across a page "
                               f"leaves behind"))

    findings += _check_table_bodies(name, bodies, by_number, folio_y)
    findings += _check_table_captions(name, tables, bodies, by_number, max_gap)
    findings += _check_figure_captions(name, captions, pages, by_number, size, max_gap)
    findings += _check_headings(name, pages, tex_path, size, min_after)
    return findings


def _source_of(entry):
    return f"{entry['file']}:{entry['start']}" if entry else "?"


def _check_table_bodies(name, bodies, by_number, folio_y):
    findings = []
    for index, body in enumerate(bodies):
        where = _source_of(by_number.get(("table", index + 1)))
        if body["top_page"] != body["bot_page"]:
            findings.append(("C2", f"{name}: table body #{index + 1} straddles pages "
                                   f"{body['top_page']}->{body['bot_page']} "
                                   f"(top rule y={body['top_y']:.1f}, bottom rule "
                                   f"y={body['bot_y']:.1f}). Source {where}"))
        elif body["bot_y"] < folio_y:
            findings.append(("C2", f"{name}: table body #{index + 1} overflows the foot of page "
                                   f"{body['bot_page']} (bottom rule at y={body['bot_y']:.1f}). "
                                   f"Source {where}"))
    return findings


def _check_table_captions(name, tables, bodies, by_number, max_gap):
    findings = []
    for index, caption in enumerate(tables):
        entry = by_number.get(("table", caption["number"]))
        where = _source_of(entry)
        direction = (entry or {}).get("direction") or "above"
        if direction == "above":
            near = [b for b in bodies
                    if b["top_page"] == caption["page"] and b["top_y"] < caption["y"]]
            body = min(near, key=lambda b: caption["y"] - b["top_y"]) if near else None
        else:
            near = [b for b in bodies
                    if b["bot_page"] == caption["page"] and b["bot_y"] > caption["y"]]
            body = min(near, key=lambda b: b["bot_y"] - caption["y"]) if near else None

        if body is None:
            positional = bodies[index] if index < len(bodies) else None
            elsewhere = ""
            if positional and positional["top_page"] != caption["page"]:
                elsewhere = f" Its body appears to start on page {positional['top_page']}."
            findings.append(("C1", f"{name}: Table {caption['number']} caption is on page "
                                   f"{caption['page']} (y={caption['y']:.1f}) with no table body "
                                   f"{direction} it on that page.{elsewhere} Source {where}"))
            continue

        edge = body["top_y"] if direction == "above" else body["bot_y"]
        gap = abs(caption["y"] - edge)
        if gap > max_gap * 3:
            findings.append(("C1", f"{name}: Table {caption['number']} caption is {gap:.1f} pt "
                                   f"from its body on page {caption['page']}. Source {where}"))
    return findings


def _check_figure_captions(name, captions, pages, by_number, size, max_gap):
    findings = []
    figures = sorted((c for c in captions if c["kind"] == "figure"), key=lambda c: c["number"])
    for caption in figures:
        page = pages[caption["page"] - 1]
        entry = by_number.get(("figure", caption["number"]))
        where = _source_of(entry)
        direction = (entry or {}).get("direction") or "below"
        # A TikZ figure places no XObject, so its ink has to be recovered from painted marks
        # and from the small type of its own node labels.
        if direction == "below":
            ink = ([m[1] for m in page["marks"] if m[1] > caption["y"]]
                   + [ln["y"] for ln in page["lines"]
                      if ln["y"] > caption["y"] and ln["size"] < size - 1.0])
        else:
            ink = ([m[3] for m in page["marks"] if m[3] < caption["y"]]
                   + [ln["y"] for ln in page["lines"]
                      if ln["y"] < caption["y"] and ln["size"] < size - 1.0])
        if not ink:
            findings.append(("C4", f"{name}: Figure {caption['number']} caption on page "
                                   f"{caption['page']} (y={caption['y']:.1f}) has no graphical "
                                   f"content {direction} it on that page. Source {where}"))
            continue
        nearest = min(ink) if direction == "below" else max(ink)
        gap = abs(nearest - caption["y"])
        if gap > max_gap:
            findings.append(("C4", f"{name}: Figure {caption['number']} caption on page "
                                   f"{caption['page']} is {gap:.1f} pt from the nearest graphics "
                                   f"(limit {max_gap:.0f} pt). Source {where}"))
    return findings


def _check_headings(name, pages, tex_path, size, min_after):
    findings = []
    for heading in locate_headings(pages, tex_headings(tex_path), size):
        if heading["page"] is None:
            findings.append(("C0", f"{name}: heading not found in PDF: "
                                   f"\\{heading['level']} {{{heading['text'][:48]}}} "
                                   f"({heading['file']}:{heading['line']})"))
            continue
        if heading["runin"] or heading["level"] in ("paragraph", "subparagraph"):
            continue
        page = pages[heading["page"] - 1]
        after = [ln for ln in page["lines"]
                 if ln["y"] < heading["y"] - 1.0
                 and abs(ln["size"] - size) < 0.6
                 and not is_folio(ln, heading["page"], page["lines"])]
        # A heading followed immediately by a float is not stranded: the float is its content.
        floats_below = [m for m in page["marks"] if m[3] < heading["y"]]
        if len(after) < min_after and not floats_below:
            findings.append(("C3", f"{name}: heading '{heading['pdftext'][:44]}' at the foot of "
                                   f"page {heading['page']} (y={heading['y']:.1f}) is followed by "
                                   f"{len(after)} line(s) of body text; {min_after} required. "
                                   f"({heading['file']}:{heading['line']})"))
    return findings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdf", type=Path)
    parser.add_argument("tex", type=Path, nargs="?",
                        help="root .tex of the document; defaults to the PDF's name")
    parser.add_argument("--max-gap", type=float, default=36.0)
    parser.add_argument("--min-lines-after-heading", type=int, default=2)
    parser.add_argument("--only", action="append", choices=["C0", "C1", "C2", "C3", "C4"])
    parser.add_argument("--caption-env", action="append", default=[],
                        help="extra environment that may carry a caption and a body")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    tex = args.tex or args.pdf.with_suffix(".tex")
    if not args.pdf.exists():
        print(f"check_layout: no such PDF: {args.pdf}", file=sys.stderr)
        return 3
    if not tex.exists():
        print(f"check_layout: no such .tex: {tex}", file=sys.stderr)
        return 3

    try:
        findings = check(str(args.pdf), str(tex), args.max_gap, args.min_lines_after_heading,
                         envs=tuple(CAPTION_ENVS) + tuple(args.caption_env))
    except Exception as error:
        # Broad on purpose: a crash inside the detector must exit non-zero, not read as a
        # pass.  Exit 3 separates "could not check" from exit 1, "checked and found bad".
        print(f"check_layout: {type(error).__name__}: {error}", file=sys.stderr)
        return 3

    if args.only:
        findings = [f for f in findings if f[0] in args.only]
    for code, message in findings:
        print(f"{code} {message}", file=sys.stderr)
    if not findings and not args.quiet:
        print(f"check_layout: {args.pdf.name}: no layout defects")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
