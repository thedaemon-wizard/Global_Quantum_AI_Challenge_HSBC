#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Generate the reference cross-check: which reference each part of the repository uses.

``docs/REFERENCES.md`` says this document records "which module or table uses each one and
what it was checked against". It did not exist. Writing it by hand would have made it stale
within a round -- references get renumbered, modules get refactored, and a hand-maintained map
drifts exactly like the hand-maintained decision count did.

So it is generated. Each entry is searched for by three independent handles:

* its **identifier**, ``[CP-4]`` or ``CP-4``;
* its **arXiv identifier**, if it has one;
* its **first author's surname**, which is how the source modules actually cite -- the
  docstrings name people rather than reference keys.

An entry found by none of the three is reported as uncited, which is a finding rather than an
error: a reference list may legitimately carry background and exclusion records. What it must
not do is carry them silently.

    .venv/bin/python scripts/make_crosscheck.py

Writes ``docs/REFERENCE_CROSSCHECK.md``.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

from hsbcfraud.analysis.citations import DECLARATION
from hsbcfraud.paths import display_path

REPO = Path(__file__).resolve().parents[1]

# Where a reference can legitimately be used.  The reference list itself is excluded, and so is
# this script's own output.
SEARCH_GLOBS = (
    "src/hsbcfraud/**/*.py",
    "scripts/*.py",
    "tests/*.py",
    "docs/protocol.md",
    "docs/decisions.md",
    "docs/PROVENANCE.md",
    "docs/guarantee.md",
    "docs/claims.yaml",
    "docs/tables.yaml",
    "submission/content/*.tex",
    "README.md",
    "NOTICE",
    "Makefile",
    "pyproject.toml",
)

# Files that are ABOUT the citation machinery rather than users of it.  They name identifiers
# as format examples, and counting those as uses would report the reference list as fully
# cited by the code that parses it.
TOOLING = frozenset(
    {
        "scripts/make_crosscheck.py",
        "src/hsbcfraud/analysis/citations.py",
        "tests/test_citations.py",
    }
)

ARXIV = re.compile(r"arXiv:(\d{4}\.\d{4,5})", re.IGNORECASE)
# The first surname in an entry, which is what a docstring cites.  Entries start with the
# identifier, then the author list; institutional authors are handled by the fallback below.
SURNAME = re.compile(r"^\*\*\[[A-Z]{2}-\d+\]\*\*\s+([A-Z][A-Za-z'À-ſ-]+)")
# Sections whose entries are deliberately not used: they record what was NOT relied upon.
EXCLUSION_HEADING = re.compile(r"^##\s+\d*\.?\s*.*not\s+relied\s+upon", re.IGNORECASE)

# Surnames too common to search on without false positives.  Each is checked by identifier and
# arXiv id instead.  Kept explicit and short; a long list would mean the heuristic is wrong.
AMBIGUOUS_SURNAMES = frozenset({"Board", "Commission", "Regulation", "Financial", "Prudential"})


@dataclass
class Entry:
    """One reference and everywhere the repository reaches it from."""

    identifier: str
    section: str
    title: str
    arxiv: str | None
    surname: str | None
    by_identifier: list[str] = field(default_factory=list)
    by_arxiv: list[str] = field(default_factory=list)
    by_surname: list[str] = field(default_factory=list)

    @property
    def used(self) -> bool:
        return bool(self.by_identifier or self.by_arxiv or self.by_surname)

    @property
    def sites(self) -> list[str]:
        seen: dict[str, None] = {}
        for site in (*self.by_identifier, *self.by_arxiv, *self.by_surname):
            seen.setdefault(site, None)
        return list(seen)


def parse_entries(text: str) -> list[Entry]:
    """Every declared entry, with the section heading it sits under."""
    entries: list[Entry] = []
    section = "(no section)"
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("## "):
            section = line.lstrip("# ").strip()
        match = DECLARATION.match(line)
        if not match:
            continue
        # The entry runs to the next blank line OR the next declaration, whichever comes
        # first.  Section 6 writes its entries on consecutive lines with no blank between
        # them, so stopping only at a blank line swallowed the following entry's identifier
        # and reported it as this one's title.
        body = [line]
        for following in lines[index + 1 :]:
            if not following.strip() or DECLARATION.match(following):
                break
            body.append(following)
        blob = " ".join(body)
        arxiv = ARXIV.search(blob)
        surname = SURNAME.match(line)
        name = surname.group(1) if surname else None
        entries.append(
            Entry(
                identifier=match.group(1),
                section=section,
                title=_title(blob),
                arxiv=arxiv.group(1) if arxiv else None,
                surname=None if name in AMBIGUOUS_SURNAMES else name,
            )
        )
    return entries


def _title(blob: str) -> str:
    """The entry's title, however it is marked up.

    Three forms appear in this corpus: a quoted title, an italicised book or journal title,
    and neither -- an instrument named in running text.  Splitting on the first full stop
    fails on all of them, because author lists are written with initials.
    """
    # Strip the bold identifier first: "**[CP-1]**" contains an italic-looking "*[CP-1]*".
    stripped = re.sub(r"^\*\*\[[A-Z]{2}-\d+\]\*\*\s*", "", blob)
    quoted = re.search(r'"([^"]{4,150})"', stripped)
    if quoted:
        return quoted.group(1).replace("\n", " ")
    # An italic run is a title only near the start.  Later ones are journal names or ordinary
    # emphasis inside the annotation -- RG-1 emphasises "value-weighted" three lines down, and
    # reporting that as the regulation's title would be worse than reporting no title.
    italic = re.search(r"\*([^*]{4,150})\*(\s*\d)?", stripped[:120])
    # An italic run followed immediately by a number is a journal and a volume, not a title.
    # FR-4 is "*Technologies* 14(4):212" and has no article title at all in the reference
    # list, which is a gap in the entry rather than something to paper over here.
    if italic and not italic.group(2):
        return italic.group(1).strip()
    # No title markup: take the leading clause, but never break inside an initial such as
    # "Vovk, V." -- a full stop after a single capital is part of a name.
    clause = re.split(r"(?<![A-Z])\.\s", stripped)[0]
    return clause[:110].strip().rstrip(",")


def locate(entries: list[Entry], repo: Path) -> None:
    """Fill in every site each entry is reached from."""
    files = sorted(
        {path for pattern in SEARCH_GLOBS for path in repo.glob(pattern) if path.is_file()}
    )
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        name = str(path.relative_to(repo))
        if name in TOOLING:
            continue
        lowered = text.lower()
        for entry in entries:
            if re.search(rf"\[?\b{re.escape(entry.identifier)}\b\]?", text):
                entry.by_identifier.append(name)
            if entry.arxiv and entry.arxiv in text:
                entry.by_arxiv.append(name)
            if entry.surname and re.search(rf"\b{re.escape(entry.surname.lower())}\b", lowered):
                entry.by_surname.append(name)


def render(entries: list[Entry], exclusion_sections: set[str]) -> str:
    """The cross-check document."""
    used = [e for e in entries if e.used]
    background = [e for e in entries if not e.used and e.section in exclusion_sections]
    uncited = [e for e in entries if not e.used and e.section not in exclusion_sections]

    lines = [
        "# Reference cross-check",
        "",
        "Which part of the repository reaches each reference, and how. **Generated by**",
        "`scripts/make_crosscheck.py` — do not edit; a hand-maintained map of this kind drifts",
        "within a round, which is why the file `REFERENCES.md` promised did not exist for so long.",
        "",
        "Each entry is searched for by three independent handles: its identifier, its arXiv",
        "identifier, and its first author's surname. Source modules cite people rather than",
        "reference keys, so the surname is usually the handle that finds them.",
        "",
        f"**{len(entries)} entries. {len(used)} reached from the repository, "
        f"{len(background)} recorded as deliberately not relied upon, "
        f"{len(uncited)} cited nowhere.**",
        "",
        "---",
        "",
        "## Reached from the repository",
        "",
        "| Entry | Subject | Reached from |",
        "|---|---|---|",
    ]
    for entry in used:
        sites = ", ".join(f"`{s}`" for s in entry.sites[:6])
        if len(entry.sites) > 6:
            sites += f" and {len(entry.sites) - 6} more"
        lines.append(f"| **{entry.identifier}** | {entry.title} | {sites} |")

    lines += [
        "",
        "## Cited nowhere",
        "",
    ]
    if uncited:
        lines += [
            "These are declared in the reference list and reached by nothing in the repository.",
            "That is not automatically a defect — a reference can establish context for a claim",
            "made in prose — but each one is a citation a reader cannot follow to a use, so it is",
            "listed rather than left to be discovered.",
            "",
            "| Entry | Subject |",
            "|---|---|",
        ]
        lines += [f"| **{e.identifier}** | {e.title} |" for e in uncited]
    else:
        lines.append("None. Every declared entry is reached from at least one file.")

    if background:
        lines += [
            "",
            "## Deliberately not relied upon",
            "",
            "Entries in a section that exists to record exclusions. Being cited nowhere is their",
            "purpose.",
            "",
            "| Entry | Subject |",
            "|---|---|",
        ]
        lines += [f"| **{e.identifier}** | {e.title} |" for e in background]

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--references", type=Path, default=REPO / "docs" / "REFERENCES.md")
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "REFERENCE_CROSSCHECK.md")
    args = parser.parse_args(argv)

    text = args.references.read_text(encoding="utf-8")
    entries = parse_entries(text)
    if not entries:
        raise SystemExit(
            f"no entries parsed from {display_path(args.references)}; expected lines starting "
            "'**[XX-N]**'. If the format changed, this script must change with it rather than "
            "writing an empty map."
        )
    exclusions = {
        line.lstrip("# ").strip()
        for line in text.splitlines()
        if EXCLUSION_HEADING.match(line)
    }
    locate(entries, REPO)
    args.out.write_text(render(entries, exclusions), encoding="utf-8")

    used = sum(1 for e in entries if e.used)
    print(f"{len(entries)} entries, {used} reached from the repository")
    for entry in entries:
        if not entry.used:
            marker = "background" if entry.section in exclusions else "CITED NOWHERE"
            print(f"  {entry.identifier:8s} {marker}")
    print(f"  Wrote {display_path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
