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

from hsbcfraud.paths import display_path

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
    # Derived from `docs/decisions.md` rather than from any run.  It lands under
    # `results/tables/`, so the scientific glob above swept it up and every added decision
    # entry was reported as a *scientific* artefact changing -- the same message a corrupted
    # measurement produces, from a file whose own docstring says it counts documents and "not
    # of any run".
    #
    # Worse, `make check` regenerates it before verifying: `check` depends on `claims`, which
    # depends on `derived`, which runs `summarise_decisions.py`.  So adding a decision made
    # `make check` fail against a manifest the same command had just invalidated, and the fix
    # looked like "re-freeze until it passes" -- which is exactly the habit a freeze exists to
    # prevent.  Classifying it with the document it projects makes the movement legitimate,
    # which it always was.  See D-128.
    "results/tables/decision_log.csv",
    # Derived from `screens.csv` by `summarise_screens.py`, which `make derived` runs before
    # `make check` compares the manifest -- the same regeneration-then-verify ordering that made
    # `decision_log.csv` report as a changed scientific artefact.  It measures nothing new.
    "results/tables/screen_bandwidth.csv",
    "results/tables/screen_distinct.csv",
    # The single-evaluation ledger.  Excluded from SCIENTIFIC below for the same reason and
    # listed here so it stays *tracked*: dropping it from both classes would leave the file that
    # records how many times the held-out fold was read as the one artefact under `results/`
    # that no manifest covers.
    "results/tables/test_access.json",
    "results/tables/smoke.json",
)

# Members of a SCIENTIFIC pattern that are bookkeeping rather than measurement.  Listed
# explicitly rather than pattern-matched: this exemption weakens a gate, so it should be
# impossible to widen by accident.
#
# `test_access.json` is the single-evaluation ledger.  `TestFoldGuard.authorise` *increments* a
# counter in it, and two scripts authorise, so any reproduction moves its bytes while nothing
# measured changes.  Under the SCIENTIFIC class that made `make check` report "a scientific
# artefact differs" -- with the alarming "either a measurement genuinely changed, or the run is
# not reproducible" text -- on every clean-room run, which is D-080's failure mode exactly: a
# check that fires every time teaches its reader to clear it without reading the list.  It is a
# record of what was done, not a measurement, so it belongs with `decision_log.csv`.
SPECIFICATION_UNDER_RESULTS = (
    "results/tables/decision_log.csv",
    "results/tables/screen_bandwidth.csv",
    "results/tables/screen_distinct.csv",
    "results/tables/test_access.json",
    # `smoke.json` records what S0-S8 asserted about *this machine* -- driver, device name, CUDA
    # build.  Under SCIENTIFIC it promised byte-identity across hosts, which it cannot keep by
    # construction; the same reasoning as `test_access.json` above.
    "results/tables/smoke.json",
)

# The two scientific artefacts that `make check` rebuilds before verifying them, because
# `check` depends on `pdf`.  They quote bound claims, so a specification change alone is enough
# to move their bytes -- adding a decision entry changes `\ClaimDecisionEntries` and therefore
# both PDFs.  That is expected, and it is not what "the run is not reproducible" means.
BUILT_DOCUMENTS = ("submission/proposal.pdf", "submission/appendix.pdf")


def digest(path: Path) -> str:
    """SHA-256 of a file, read in chunks so a large figure does not need to fit in memory."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(patterns: tuple[str, ...], *, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    excluded = set(exclude)
    found: dict[str, str] = {}
    for pattern in patterns:
        for path in sorted(REPO.glob(pattern)):
            relative = path.relative_to(REPO).as_posix()
            if path.is_file() and relative not in excluded:
                found[relative] = digest(path)
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
        "scientific": collect(SCIENTIFIC, exclude=SPECIFICATION_UNDER_RESULTS),
        "specification": collect(SPECIFICATION),
    }
    counts = {k: len(v) for k, v in current.items()}

    if not args.check:
        MANIFEST.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {display_path(MANIFEST)}")
        for name, count in counts.items():
            print(f"  {name:14s} {count} file(s)")
        return 0

    if not MANIFEST.exists():
        print(f"{display_path(MANIFEST)} does not exist; run without --check", file=sys.stderr)
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
        moved = {line.split(": ", 1)[1].split(" ", 1)[0] for line in scientific}
        rebuilt_only = moved <= set(BUILT_DOCUMENTS)
        if rebuilt_only and specification:
            # `check` depends on `pdf`, so it rebuilds these before comparing them.  Telling a
            # user their run may not be reproducible, when all that happened is that they added
            # a decision entry and this command recompiled the document quoting it, teaches
            # them to re-freeze until the message goes away -- which is the one habit a freeze
            # exists to prevent.
            print(
                "\nOnly the built documents moved, and a specification file they quote moved "
                "too, so this is the expected case: `make check` rebuilds the PDFs before "
                "verifying them.\nRe-freeze with `make freeze`. No measurement is implicated.",
                file=sys.stderr,
            )
        else:
            print(
                "\nEither a measurement genuinely changed -- in which case re-freeze and say "
                "why in docs/decisions.md -- or the run is not reproducible.",
                file=sys.stderr,
            )
        return 1

    print(f"All {counts['scientific']} scientific artefacts match the manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
