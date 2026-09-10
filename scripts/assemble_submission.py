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

# Phase 1 Submission Guidelines section 5: "File size must not exceed 20 MB."  Nothing in this
# repository checked that, and the sentence is ambiguous twice over, so both ambiguities are
# resolved toward the stricter reading rather than left to the person doing the upload:
#
#  * Per file or over the staged set?  Only the per-file bound holds under both readings, so
#    that is what fails the run; the total is printed beside it so a human can see the other.
#  * MB or MiB?  20,000,000 bytes is the smaller number, so a set that passes here also passes
#    a portal counting in MiB.  Sizes are reported in the same unit they are judged in.
#
# predictions.csv grows with the held-out block and is deliberately unrounded, so it is the one
# staged file that can walk into this limit.
BYTES_PER_MB = 1_000_000
MAX_UPLOAD_BYTES = 20 * BYTES_PER_MB

# What is uploaded, and under what name.  The portal sees these names, so they carry the track
# and the artefact rather than the repository's internal layout.  Five artefacts, chosen so
# that a reviewer who opens only one still gets something self-contained: the two documents,
# the per-transaction output the challenge statement names as its first expected outcome, the
# implementation that produces the certificate, and the one picture that shows how little of the
# pre-registered grid actually certifies.
#
# The certificate table held this slot and was dropped for the predictions, which the challenge
# statement names as an expected output and the table is not.  The justification here used to
# read that every certified row was printed in the appendix; none of the five is, in either PDF
# -- both carry only the aggregate "5 of 48" and the margin.  What is true is that the figure
# plots the full 48-cell grid and the table itself is committed as
# results/tables/riskcontrol.csv, so a reviewer can still reach every cell.
STAGED: tuple[tuple[str, str], ...] = (
    ("submission/proposal.pdf", "HSBC-proposal.pdf"),
    ("submission/appendix.pdf", "HSBC-appendix.pdf"),
    ("results/tables/predictions.csv", "HSBC-predictions.csv"),
    ("src/hsbcfraud/conformal/riskcontrol.py", "HSBC-riskcontrol.py"),
    ("results/figures/certified_region.png", "HSBC-certified-region.png"),
)

# Artefacts whose hash must still match the manifest.
#
# This was the two PDFs alone, under a comment saying documents expected to move between a
# freeze and an upload are staged but not hash-checked.  That reasoning does not describe a
# results table: `predictions.csv` and `certified_region.png` are both **scientific** members of
# the manifest, so re-running `export_predictions.py` after `make freeze` let this script stage
# a changed file and exit 0 while `freeze.py --check` on the same tree exited 1.  The upload set
# is the one artefact a reviewer actually receives, and it was being checked less strictly than
# the repository it comes from.
#
# `riskcontrol.py` is deliberately absent: it is source, the manifest covers neither `src/` nor
# any Python file, and listing it here would silently check nothing.
HASH_CHECKED = (
    "submission/proposal.pdf",
    "submission/appendix.pdf",
    "results/tables/predictions.csv",
    "results/figures/certified_region.png",
)


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
    staged_bytes = 0
    largest_bytes = 0
    for source, name in STAGED:
        path = REPO / source
        if not path.exists():
            problems.append(f"{source} does not exist; run `make pdf` first")
        else:
            size = path.stat().st_size
            staged_bytes += size
            largest_bytes = max(largest_bytes, size)
            if size > MAX_UPLOAD_BYTES:
                problems.append(
                    f"{name} is {size / BYTES_PER_MB:.1f} MB, over the "
                    f"{MAX_UPLOAD_BYTES / BYTES_PER_MB:.0f} MB the guidelines allow. Truncating "
                    "the held-out block would change what is reported, so drop the file from "
                    "STAGED and record why instead."
                )
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

    # `--out` is emptied before staging, so it must not be a directory that contains anything
    # this script is about to read.  The default is `submission/portal`, and `--out submission`
    # is one keystroke away: that would delete `submission/proposal.pdf` and
    # `submission/appendix.pdf` -- the two files whose hashes were verified four lines above --
    # and then fail trying to copy them.  Neither is recoverable without a rebuild.
    out = args.out.resolve()
    for source, _name in STAGED:
        origin = (REPO / source).resolve()
        if out == origin or out in origin.parents:
            print(
                f"REFUSING to stage into {display_path(args.out)}: it contains {source}, which "
                f"this script copies from, and staging empties the directory first.\n"
                f"  Pass a directory that holds nothing but the upload set "
                f"(the default is {display_path(REPO / 'submission' / 'portal')}).",
                file=sys.stderr,
            )
            return 1

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    for source, name in STAGED:
        shutil.copy2(REPO / source, args.out / name)
        print(f"  {name:24s} <- {source}")

    print(
        f"\nStaged {len(STAGED)} file(s) in {display_path(args.out)}: "
        f"{staged_bytes / BYTES_PER_MB:.1f} MB in total, largest "
        f"{largest_bytes / BYTES_PER_MB:.1f} MB, against a "
        f"{MAX_UPLOAD_BYTES / BYTES_PER_MB:.0f} MB cap read per file."
    )
    print("Re-verify the portal URLs in a browser before uploading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
