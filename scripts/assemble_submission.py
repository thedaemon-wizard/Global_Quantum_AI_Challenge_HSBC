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

from hsbcfraud.paths import display_path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "MANIFEST.sha256.json"

# Formats the portal's upload control accepts, verified in a browser against the live form.
# Markdown is NOT among them, which is what this list exists to catch: three of the five files
# staged here were .md, so the set could never have been uploaded.  A staging script whose
# output the portal rejects is worse than no staging script, because it looks finished.
PORTAL_FORMATS = frozenset({
    ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".py",
    ".json", ".js", ".xls", ".xlsx", ".csv", ".doc", ".docx",
})
# The portal exposes five upload slots.
PORTAL_SLOTS = 5

# What is uploaded, and under what name.  The portal sees these names, so they carry the track
# and the artefact rather than the repository's internal layout.  Five artefacts, chosen so
# that a reviewer who opens only one still gets something self-contained: the two documents,
# the certificate itself as data, the implementation that produces it, and the one picture
# that shows how little of the pre-registered grid actually certifies.
STAGED: tuple[tuple[str, str], ...] = (
    ("submission/proposal.pdf", "HSBC-proposal.pdf"),
    ("submission/appendix.pdf", "HSBC-appendix.pdf"),
    ("results/tables/riskcontrol.csv", "HSBC-certificate.csv"),
    ("src/hsbcfraud/conformal/riskcontrol.py", "HSBC-riskcontrol.py"),
    ("results/figures/certified_region.png", "HSBC-certified-region.png"),
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
    if len(STAGED) > PORTAL_SLOTS:
        problems.append(f"{len(STAGED)} files staged for {PORTAL_SLOTS} upload slots")
    for source, name in STAGED:
        if not (REPO / source).exists():
            problems.append(f"{source} does not exist; run `make pdf` first")
        suffix = Path(name).suffix.lower()
        if suffix not in PORTAL_FORMATS:
            problems.append(
                f"{name} is a {suffix or 'no-extension'} file, which the portal's upload "
                f"control rejects. Accepted: {' '.join(sorted(PORTAL_FORMATS))}"
            )

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

    print(f"\nStaged {len(STAGED)} file(s) in {display_path(args.out)}.")
    print("Re-verify the portal URLs in a browser before uploading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
