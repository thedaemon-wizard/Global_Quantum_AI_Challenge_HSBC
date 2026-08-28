#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Stage exactly the files the portal accepts, and refuse to stage a stale one.

Assembling by hand is where a verified repository turns into an unverified upload: the PDF
that gets attached is the one that happened to be in the directory, not necessarily the one
the gates passed.  This copies from the working tree into ``submission/portal/`` only after
re-checking that each artefact matches the manifest written by ``scripts/freeze.py``.

    .venv/bin/python scripts/assemble_submission.py

Exit 0 when the staged set is complete and current, 1 otherwise.  Nothing is copied on
failure, because a partially staged directory is worse than an empty one -- it looks ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "MANIFEST.sha256.json"

# What is uploaded, and under what name.  The portal sees these names, so they carry the
# track and the artefact rather than the repository's internal layout.
STAGED: tuple[tuple[str, str], ...] = (
    ("submission/proposal.pdf", "HSBC-proposal.pdf"),
    ("submission/appendix.pdf", "HSBC-appendix.pdf"),
    ("README.md", "README.md"),
    ("docs/protocol.md", "protocol.md"),
    ("docs/decisions.md", "decisions.md"),
)

# Artefacts whose hash must still match the manifest.  Documents that are expected to move
# between a freeze and an upload are staged but not hash-checked; the PDFs are not among them.
HASH_CHECKED = ("submission/proposal.pdf", "submission/appendix.pdf")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "submission" / "portal")
    args = parser.parse_args(argv)

    problems: list[str] = []
    for source, _ in STAGED:
        if not (REPO / source).exists():
            problems.append(f"{source} does not exist; run `make pdf` first")

    if MANIFEST.exists():
        recorded = json.loads(MANIFEST.read_text(encoding="utf-8")).get("scientific", {})
        for source in HASH_CHECKED:
            path = REPO / source
            if not path.exists():
                continue
            if source not in recorded:
                problems.append(f"{source} is not in the manifest; run `make freeze`")
            elif recorded[source] != digest(path):
                problems.append(
                    f"{source} differs from the manifest -- it was rebuilt after the freeze. "
                    "Run `make check` and re-freeze before staging."
                )
    else:
        problems.append("MANIFEST.sha256.json does not exist; run `make freeze`")

    if problems:
        print(f"Not staging. {len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    for source, name in STAGED:
        shutil.copy2(REPO / source, args.out / name)
        print(f"  {name:24s} <- {source}")

    print(f"\nStaged {len(STAGED)} file(s) in {args.out.relative_to(REPO)}.")
    print("Re-verify the portal URLs in a browser before uploading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
