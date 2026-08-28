#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Hash the artefacts a reviewer receives, so a second run can be compared against the first.

Reproducibility claims are cheap to make and hard to check.  This writes a SHA-256 manifest
over the files that carry the science, and ``--check`` re-hashes them and reports what moved.

Files are grouped, because "the output changed" is not one question:

* **scientific** -- results tables and the built PDFs.  These *must* be identical across runs
  on the same inputs.  A difference here is either a real change in a measurement or a
  reproducibility failure, and either way it needs an explanation.
* **specification** -- the protocol, the decision log, the claim and table bindings, the
  configuration.  These change deliberately and are expected to; the manifest records what
  they were when the tables were produced, so a reviewer can tell whether a result predates
  an amendment.

A third category is deliberately absent.  Intermediate run artefacts under ``results/runs/``
are not hashed: they are large, they are not committed, and they are derived.

    .venv/bin/python scripts/freeze.py           # write the manifest
    .venv/bin/python scripts/freeze.py --check   # compare against it

Exit 0 if the scientific class is unchanged, 1 otherwise.  Movement in the specification
class is reported but does not fail, because amending the protocol is legitimate and
``scripts/check_protocol.py`` is what governs it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "MANIFEST.sha256.json"

SCIENTIFIC = (
    "results/tables/*.csv",
    "results/tables/*.json",
    "results/figures/*",
    "submission/proposal.pdf",
    "submission/appendix.pdf",
)
SPECIFICATION = (
    "docs/protocol.md",
    "docs/protocol.lock.json",
    "docs/decisions.md",
    "docs/claims.yaml",
    "docs/tables.yaml",
    "docs/REFERENCES.md",
    "configs/*.yaml",
)


def digest(path: Path) -> str:
    """SHA-256 of a file, read in chunks so a large figure does not need to fit in memory."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(patterns: tuple[str, ...]) -> dict[str, str]:
    found: dict[str, str] = {}
    for pattern in patterns:
        for path in sorted(REPO.glob(pattern)):
            if path.is_file():
                found[path.relative_to(REPO).as_posix()] = digest(path)
    return found


def compare(name: str, recorded: dict[str, str], current: dict[str, str]) -> list[str]:
    """Differences in one class, as human-readable lines."""
    problems = []
    for path in sorted(set(recorded) | set(current)):
        before, after = recorded.get(path), current.get(path)
        if before == after:
            continue
        if before is None:
            problems.append(f"{name}: {path} is new since the manifest was written")
        elif after is None:
            problems.append(f"{name}: {path} was in the manifest and is now missing")
        else:
            problems.append(f"{name}: {path} changed ({before[:12]} -> {after[:12]})")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    args = parser.parse_args(argv)

    current = {
        "scientific": collect(SCIENTIFIC),
        "specification": collect(SPECIFICATION),
    }
    counts = {k: len(v) for k, v in current.items()}

    if not args.check:
        MANIFEST.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {MANIFEST.relative_to(REPO)}")
        for name, count in counts.items():
            print(f"  {name:14s} {count} file(s)")
        return 0

    if not MANIFEST.exists():
        print(f"{MANIFEST.relative_to(REPO)} does not exist; run without --check", file=sys.stderr)
        return 1

    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    scientific = compare("scientific", recorded.get("scientific", {}), current["scientific"])
    specification = compare(
        "specification", recorded.get("specification", {}), current["specification"]
    )

    for line in specification:
        print(f"  note  {line}")
    if specification:
        print(
            "  (specification movement is legitimate; scripts/check_protocol.py governs "
            "whether it needed an amendment)"
        )

    if scientific:
        print(
            f"\n{len(scientific)} scientific artefact(s) differ from the manifest:",
            file=sys.stderr,
        )
        for line in scientific:
            print(f"  {line}", file=sys.stderr)
        print(
            "\nEither a measurement genuinely changed -- in which case re-freeze and say why "
            "in docs/decisions.md -- or the run is not reproducible.",
            file=sys.stderr,
        )
        return 1

    print(f"All {counts['scientific']} scientific artefacts match the manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
