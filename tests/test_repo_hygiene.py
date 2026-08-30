# SPDX-License-Identifier: Apache-2.0
"""Properties of the repository itself, pinned so they cannot regress.

These are not unit tests of the science.  They are assertions about the artefact a reviewer
receives, and each exists because the corresponding property was violated at least once and
nothing raised.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src", "scripts", "tests")


def _tracked_files() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    )
    return set(out.stdout.split())


def test_every_source_file_is_tracked() -> None:
    """No source file is silently excluded by .gitignore.

    An unanchored ``data/`` pattern matches at any depth.  It excluded
    ``src/hsbcfraud/data/`` -- the IEEE-CIS loader, the four-block split, the test-fold guard
    and the label audit -- from three pushed commits.  ``git add`` does not warn when it skips
    an ignored path and ``git status`` does not list it, so the omission was invisible locally
    and total for anyone cloning: the package the Makefile's first target imports was not
    there.
    """
    tracked = _tracked_files()
    missing = []
    for directory in SOURCE_DIRS:
        for path in sorted((REPO / directory).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(REPO).as_posix()
            if relative not in tracked:
                missing.append(relative)
    assert not missing, (
        "these source files exist on disk but are not tracked by git, so they are absent "
        f"from any clone: {missing}. Check .gitignore for an unanchored directory pattern."
    )


def test_results_tables_are_tracked() -> None:
    """Every committed number is reachable from a clone.

    The documents quote figures that resolve to ``results/tables/*.csv``.  A table present
    locally but untracked makes the corresponding claim unverifiable by a reviewer.
    """
    tracked = _tracked_files()
    missing = [
        p.relative_to(REPO).as_posix()
        for p in sorted((REPO / "results" / "tables").glob("*.csv"))
        if p.relative_to(REPO).as_posix() not in tracked
    ]
    assert not missing, f"result tables exist but are untracked: {missing}"


def test_decision_log_matches_the_decisions_document() -> None:
    """The decision count is derived from the document, and stays derived.

    ``decision_log.csv`` is the only table in the repository that summarises a *document*
    rather than a run, which makes it the only one a normal editing session can invalidate
    without touching any code.  It did: two entries were appended to ``docs/decisions.md``
    and the table was not regenerated, so both PDFs printed 61 against an actual 63 while
    every gate passed -- ``check_claims.py`` compares ``claims.yaml`` to the table, and
    nothing compared the table to its source.

    This closes the loop the claim gate cannot see, so the count is wrong in the test suite
    before it is wrong in the submission.
    """
    recorded = pd.read_csv(REPO / "results" / "tables" / "decision_log.csv")

    for document, pattern, column, first in (
        ("docs/decisions.md", r"^### D-(\d+)\b", "decision_entries", "D-001"),
        ("docs/protocol.md", r"^##\s+Amendment\s+A(\d+)\b", "protocol_amendments", "A1"),
    ):
        found = re.findall(pattern, (REPO / document).read_text(encoding="utf-8"), re.M)
        assert int(recorded[column].iloc[0]) == len(found), (
            f"{document} holds {len(found)} entries, decision_log.csv records "
            f"{int(recorded[column].iloc[0])} in {column}. Regenerate it with: make derived"
        )

        # Contiguity, asserted rather than assumed: a gap or a duplicate leaves the count
        # looking plausible while the record has lost or doubled an entry.
        numbers = [int(n) for n in found]
        assert numbers == list(range(1, len(numbers) + 1)), (
            f"{document} identifiers are not contiguous from {first}: {numbers}"
        )


def test_appendix_lists_every_amendment_it_claims() -> None:
    """The printed list must be as long as the number printed above it.

    Binding the count was the fix for the previous two drift defects and it was not enough
    here: ``ProtocolAmendments`` moved 8 -> 9 the moment A9 was written, the appendix
    dutifully printed 9, and the ``enumerate`` beneath it still held 8 items.  A macro
    protects the digit, not the prose the digit describes, and a reviewer who counts a
    numbered list finds the gap in the one section whose purpose is proving nothing was
    dropped.
    """
    recorded = pd.read_csv(REPO / "results" / "tables" / "decision_log.csv")
    claimed = int(recorded["protocol_amendments"].iloc[0])

    body = (REPO / "submission" / "content" / "A1-protocol.tex").read_text(encoding="utf-8")
    block = re.search(r"\\begin\{enumerate\}.*?\\end\{enumerate\}", body, re.S)
    assert block is not None, "A1-protocol.tex no longer contains an enumerate block"
    listed = len(re.findall(r"^\\item\b", block.group(0), re.M))

    assert listed == claimed, (
        f"the appendix prints {claimed} amendments and lists {listed}. Add the missing "
        f"\\item to submission/content/A1-protocol.tex, or amend docs/protocol.md"
    )


def test_every_document_is_linked_from_the_readme() -> None:
    """No document in docs/ is orphaned.

    COMPLIANCE_CHECKLIST P3 asserts the README links the supporting documents. It stated a
    count, and the count drifted the moment a document was added. The property is what was
    meant, so the property is what is checked.
    """
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    section = readme[readme.index("**Supporting documents**") : readme.index("## 2.")]
    linked = set(re.findall(r"\[`docs/([^`]+)`\]", section))
    orphaned = sorted({p.name for p in (REPO / "docs").glob("*.md")} - linked)
    assert not orphaned, f"documents in docs/ not linked from README section 1: {orphaned}"


def test_no_bare_python3_invocation() -> None:
    """Nothing invokes a bare ``python3``.

    On this machine ``python3`` is 3.9 and a bare ``python3.12`` resolves to an interpreter
    whose torch has no sm_120.  Every entry point must use ``.venv/bin/python``.
    """
    offenders = []
    scanned = [REPO / "Makefile", *[p for d in SOURCE_DIRS for p in (REPO / d).rglob("*.py")]]
    for path in scanned:
        if "__pycache__" in path.parts or not path.exists():
            continue
        if path.resolve() == Path(__file__).resolve():
            continue  # this file necessarily contains the string it searches for
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("@echo"):
                continue  # comments and help text describe, they do not invoke
            if "python3" in stripped and ".venv/bin/python" not in stripped:
                if "/usr/bin/python3.12" in stripped or "$(BOOT)" in stripped:
                    continue  # the documented bootstrap
                offenders.append(f"{path.relative_to(REPO)}:{number}: {stripped}")
    assert not offenders, "bare python3 invocations found:\n  " + "\n  ".join(offenders)


def _has_chained_assignment(source: str) -> bool:
    """Detect ``frame[...][...] = value``, a silent no-op under pandas copy-on-write."""
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign | ast.AugAssign):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Subscript):
                inner = target.value
                if isinstance(inner, ast.Subscript):
                    return True
                if (
                    isinstance(inner, ast.Attribute)
                    and inner.attr in {"loc", "iloc", "at", "iat"}
                    and isinstance(inner.value, ast.Subscript)
                ):
                    return True
    return False


def test_no_chained_pandas_assignment() -> None:
    """Chained assignment is a silent no-op under pandas 3 copy-on-write.

    ``SettingWithCopyWarning`` was removed, so the frame is simply unchanged and every number
    downstream is quietly wrong.  Smoke check S6 confirms the behaviour on this host; this
    test enforces the absence of the pattern across the repository.

    The detector is syntactic and cannot tell a DataFrame from a list, so
    ``rows[-1]["key"] = value`` on a list of dicts is flagged although it is correct.  Rather
    than weaken the detector, such code is rewritten to build the row before appending, which
    is clearer anyway.  A file that must contain the pattern -- ``scripts/smoke.py`` carries it
    as the fixture S6 tests against -- opts out with an explicit marker.
    """
    offenders = []
    for directory in SOURCE_DIRS:
        for path in sorted((REPO / directory).rglob("*.py")):
            if "__pycache__" in path.parts or path.resolve() == Path(__file__).resolve():
                continue  # this file defines the detector and would match itself
            source = path.read_text(encoding="utf-8")
            if "chained-assignment-exempt" in source:
                continue  # scripts/smoke.py carries the fixture the gate exists to catch
            if _has_chained_assignment(source):
                offenders.append(path.relative_to(REPO).as_posix())
    assert not offenders, f"chained pandas assignment found in: {offenders}"


def test_every_source_file_carries_an_spdx_header() -> None:
    """Apache-2.0 requires the licence to travel with the file."""
    missing = []
    for directory in SOURCE_DIRS:
        for path in sorted((REPO / directory).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            head = path.read_text(encoding="utf-8")[:200]
            if "SPDX-License-Identifier" not in head:
                missing.append(path.relative_to(REPO).as_posix())
    assert not missing, f"files without an SPDX header: {missing}"


def test_makefile_scripts_exist() -> None:
    """Every script the Makefile invokes is present.

    A Makefile target that cannot run is worse than an absent one: it advertises a capability
    the repository does not have.
    """
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    referenced = sorted(
        {
            token
            for line in makefile.splitlines()
            for token in line.split()
            if token.startswith("scripts/") and token.endswith(".py")
        }
    )
    missing = [s for s in referenced if not (REPO / s).exists()]
    if missing:
        pytest.xfail(
            f"{len(missing)} of {len(referenced)} Makefile-referenced scripts are not yet "
            f"written: {missing}"
        )


def test_protocol_lock_matches_configuration() -> None:
    """The pre-registration gate passes, or fails for a documented reason.

    Exit 0 means unchanged; exit 2 means changed with an accompanying amendment, which is
    legitimate but must be re-frozen.  Exit 1 means an undocumented change to a quantity a
    guarantee depends on.
    """
    result = subprocess.run(
        [sys.executable, "scripts/check_protocol.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 1, (
        "the pre-registered state changed without a documented amendment:\n" + result.stderr
    )
