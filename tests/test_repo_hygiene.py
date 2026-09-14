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
    # An assertion, not pytest.xfail: xfail marks the test as expected-to-fail, which the
    # runner counts as a pass, so this check could not fail and had not been able to for as
    # long as it has existed. It happens to be satisfied -- every referenced script is
    # present -- which is exactly why nothing noticed.
    assert not missing, (
        f"{len(missing)} of {len(referenced)} scripts referenced by the Makefile do not "
        f"exist, so the target that names them cannot run: {missing}"
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


def test_markdown_math_survives_the_github_renderer() -> None:
    """No math span in any document relies on an escape GitHub removes.

    Three transformations, each measured against GitHub's own renderer. Outside a fenced
    ``math`` block the backslash is stripped from escaped ASCII punctuation, so ``\\Bigl\\{``
    arrives as ``\\Bigl{`` and fails outright while ``\\,`` and ``\\;`` arrive as a literal
    comma and semicolon and render *without* an error. Inside inline ``$...$`` a raw ``<`` is
    escaped twice and reaches the renderer as ``&lt;``. And an inline span that crosses a line
    break is not mathematics at all.

    Eleven expressions were affected when this was written, one of them the display equation in
    README section 4.2 that a reader reported. The property is checked here rather than only in
    ``make check`` because the published README was the one surface in this repository with no
    gate on it at all.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    import check_markdown_math

    spans = check_markdown_math.collect(REPO)
    assert spans, "no math found in any document; the extractor has stopped working"
    problems = check_markdown_math.check_spans(spans) + check_markdown_math.check_unbalanced(REPO)
    assert not problems, "\n".join(problems)


def test_exported_predictions_agree_with_the_held_out_validation() -> None:
    """The per-transaction file must not drift from the table the proposal quotes.

    ``predictions.csv`` is a portal deliverable and is derived rather than measured: the score
    comes from the frozen scorer output and the three thresholds from the certificate. That
    makes it exactly the kind of artefact that can silently stop agreeing with its source, and
    a reviewer holding the CSV and the appendix would be the one to notice.

    The comparison is on the legitimate subset because that is what H5 certifies; the CSV
    additionally carries fraudulent rows, which the certificate says nothing about.
    """
    tables = REPO / "results" / "tables"
    predictions = pd.read_csv(tables / "predictions.csv")
    riskcontrol = pd.read_csv(tables / "riskcontrol.csv")
    validation = pd.read_csv(tables / "h5_validation.csv")

    certified = riskcontrol[riskcontrol["certified"].astype(bool)]
    chosen = certified.sort_values(["alpha", "budget"], ascending=[True, False]).iloc[0]

    # The exported file must name the configuration it was built from.
    assert predictions["band_low"].nunique() == 1
    assert predictions["band_low"].iloc[0] == pytest.approx(chosen["band_lo"])
    assert predictions["band_high"].iloc[0] == pytest.approx(chosen["band_hi"])
    assert predictions["in_band_threshold"].iloc[0] == pytest.approx(chosen["selected_lambda"])

    row = validation[
        (validation["alpha"] == chosen["alpha"]) & (validation["band_budget"] == chosen["budget"])
    ]
    assert len(row) == 1, "the chosen configuration has no held-out validation row"

    in_band = predictions["decision"].isin(["step-up", "step-up-declined"])
    assert in_band.sum() >= int(row["n_legit_band_test"].iloc[0]), (
        "fewer in-band rows than the validation counts as legitimate alone"
    )
    # And an upper bound, because the line above admits everything above 10,021: a mask that
    # flagged all 115,534 rows satisfies it. The band budget comes from run_conformal.py and
    # the share from the exporter, so this is not a file compared against itself, which is why
    # operating_point.csv is not the reference -- export_predictions.py writes both.
    assert in_band.mean() <= float(chosen["budget"]), (
        f"{in_band.mean():.4f} of held-out traffic is in the band, above the certified "
        f"abstention budget of {float(chosen['budget']):.4f}"
    )
    assert (predictions["fraud_probability"].between(0.0, 1.0)).all(), (
        "the statement asks for a float in [0, 1]"
    )
    # The binary column has to be the decision, not a second opinion about it.
    terminal = predictions["decision"].isin(["decline", "step-up-declined"]).astype(int)
    assert (predictions["predicted_fraud"] == terminal).all()


# The protocol names the scripts that read the held-out block and marks which route through the
# guard. A regex over the sources cannot decide "reads the test block for a result" -- the
# string appears in figure code and in the smoke check too -- so the table is taken as the
# claim and verified against the code, which is the direction that matters: the appendix said
# three unguarded scripts against an actual four, a disclosure document under-reporting a
# disclosure, and it is the second time a count in that paragraph has drifted.
GUARD = "TestFoldGuard"
GUARD_ROW = re.compile(r"^\| `(\w+\.py)` \| [^|]+ \| (`authorise\(\)`|none) \|$", re.M)


def test_the_disclosure_names_the_right_number_of_unguarded_scripts() -> None:
    """Amendment A8's table and counts must match the tree they describe.

    A8 exists because the pre-registration claimed nothing reads the test fold outside the
    guard. Getting its arithmetic wrong turns a disclosure into a smaller version of the thing
    being disclosed.
    """
    protocol = (REPO / "docs" / "protocol.md").read_text(encoding="utf-8")
    rows = GUARD_ROW.findall(protocol)
    assert rows, "the guard table in docs/protocol.md no longer parses"

    unguarded = []
    for name, marker in rows:
        path = REPO / "scripts" / name
        assert path.exists(), f"the guard table names {name}, which does not exist"
        authorises = GUARD in path.read_text(encoding="utf-8")
        expected = marker == "`authorise()`"
        assert authorises == expected, (
            f"docs/protocol.md marks {name} as {marker} and the source says otherwise"
        )
        if not authorises:
            unguarded.append(name)

    assert f"{_WORDS[len(rows)]} scripts read the test block" in protocol.lower(), (
        f"the table lists {len(rows)} scripts and the sentence above it disagrees"
    )
    assert f"and {_WORDS[len(unguarded)]} of them do not go" in protocol.lower(), (
        f"{len(unguarded)} of them are unguarded and the sentence above the table disagrees"
    )

    appendix = (REPO / "submission" / "content" / "A1-protocol.tex").read_text(encoding="utf-8")
    assert f"of the {_WORDS[len(rows)]} scripts" in appendix, (
        "the appendix states a different number of test-block readers than the protocol table"
    )
    assert f"{_WORDS[len(unguarded)]} do so without the guard" in appendix


def test_the_disclosure_table_names_every_script_that_reads_the_held_out_block() -> None:
    """The table must be complete, not merely internally consistent.

    The count in that paragraph has now drifted three times, and each time the checks above
    passed: they verify the table against the sources it *names* and the sentence against the
    table, so a script missing from the table is invisible to both. Twice the missing scripts
    were added by this study after the amendment was written.

    A regex cannot decide "reads the test block for a result", which is why the exemption list
    is explicit rather than a pattern. What it can decide is that a script touching raw
    held-out rows appears somewhere in the disclosure, which is the property that kept failing.
    """
    protocol = (REPO / "docs" / "protocol.md").read_text(encoding="utf-8")
    listed = {name for name, _ in GUARD_ROW.findall(protocol)}

    missing = []
    for path in sorted((REPO / "scripts").glob("*.py")):
        if path.name in READS_A_COMMITTED_TABLE_NOT_THE_BLOCK or path.name in listed:
            continue
        source = path.read_text(encoding="utf-8")
        if RAW_TEST_READ.search(source):
            missing.append(path.name)

    assert not missing, (
        f"these scripts read the held-out block and are absent from amendment A8's table in "
        f"docs/protocol.md: {missing}. Add a row for each, correct the two counts in the "
        f"sentence above the table and in A1-protocol.tex, and say in A8 that the count moved."
    )


_WORDS = {
    2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
}

# Scripts that filter on a `block` column of a *committed table* rather than reading held-out
# rows. They belong in no disclosure: reading `baselines.csv` is reading a published result.
# Listed explicitly because the exemption weakens the completeness check below, and a pattern
# would quietly absorb a real reader that happened to match it.
READS_A_COMMITTED_TABLE_NOT_THE_BLOCK = frozenset(
    {"summarise_split_arms.py", "summarise_mps_lift.py", "make_figures.py", "smoke.py",
     "check_protocol.py", "run_power.py"}
)

# How a raw read of the held-out rows looks, as opposed to a filter on a published table.
RAW_TEST_READ = re.compile(
    "|".join(
        [
            r'blocks\["test"\]',
            r'block"\]\s*==\s*"test"',
            r'REPORTED_BLOCK\s*=\s*"test"',
        ]
    )
)


# Committed tables whose producer is deliberately not in the tree, with the reason. Anything
# else must be regenerable, because `freeze.py --check` cannot detect a stale table that no
# target rewrites: a file nothing touches trivially matches its own hash. Four tables sat in
# that blind spot, two of them backing tables printed in the proposal.
TABLES_WITHOUT_A_PRODUCER = {
    "mps_seed_spread.csv": (
        "a contraction-width comparison at chi=16, eight GPU fits, whose producing variant of "
        "run_seed_sweep.py is not in the tree. Retained because it is the measured evidence "
        "behind D-038's choice of the sequential contraction; see docs/PROVENANCE.md."
    ),
}


# Committed tables that no script writes although one should. Kept apart from the exemption
# above because these are a defect rather than a decision, and recorded as expected failures
# rather than exempted so that the run summary carries them: `xfail(strict=True)` fails the
# suite the moment a producer appears, which is what forces the entry back out again.
# Empty, and kept rather than deleted.  It held the two by-arm coverage tables, which were
# committed by hand and which nothing rewrote -- so `freeze.py --check` passed them trivially,
# because a file no script regenerates can never differ from its own hash.  Both now have a
# producer in `scripts/run_coverage_arms.py`, and it reproduces them byte for byte.  The strict
# xfail is what forced that to be visible instead of quietly true.
TABLES_WHOSE_PRODUCER_IS_MISSING: dict[str, str] = {}


def _tables_written_by_a_script() -> set[str]:
    """Basenames of the CSV files something in ``scripts/`` or ``src/`` actually writes.

    A *mention* is not a producer. The first version of this check searched the concatenated
    source for the file name, and a module docstring naming a table satisfied it: two tables
    sat in that blind spot for the whole study, one of them printed in the proposal. Reading
    the syntax tree instead means only a string that reaches a ``to_csv`` call counts, either
    directly or through the variable the path was assigned to.
    """

    def csv_names(node: ast.AST) -> set[str]:
        return {
            child.value.rsplit("/", 1)[-1]
            for child in ast.walk(node)
            if isinstance(child, ast.Constant)
            and isinstance(child.value, str)
            and child.value.endswith(".csv")
        }

    written: set[str] = set()
    for directory in ("scripts", "src"):
        for path in sorted((REPO / directory).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            # Two passes, because the destination is often bound above the write and `ast.walk`
            # does not promise to reach the assignment first.
            bound = {
                node.targets[0].id: csv_names(node.value)
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and csv_names(node.value)
            }
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "to_csv"
                ):
                    continue
                for arg in node.args:
                    written |= csv_names(arg)
                    if isinstance(arg, ast.Name):
                        written |= bound.get(arg.id, set())
    return written


def test_every_committed_table_has_a_producer() -> None:
    """Some script must write each results table, or it is not reproducible.

    The manifest check compares a table to its own recorded hash, which a table nothing
    rewrites always passes. Reproducibility is the property that a target regenerates it, and
    that is what this asserts.
    """
    written = _tables_written_by_a_script()
    recorded = set(TABLES_WITHOUT_A_PRODUCER) | set(TABLES_WHOSE_PRODUCER_IS_MISSING)
    orphaned = sorted(
        path.name
        for path in sorted((REPO / "results" / "tables").glob("*.csv"))
        if path.name not in written and path.name not in recorded
    )
    assert not orphaned, (
        f"these committed tables are written by no script, so `make reproduce` carries them "
        f"forward rather than regenerating them: {orphaned}. Write the producer, or record the "
        f"reason in TABLES_WITHOUT_A_PRODUCER."
    )

    stale = sorted(
        name for name in recorded if not (REPO / "results" / "tables" / name).exists()
    )
    assert not stale, f"the exemption list names tables that no longer exist: {stale}"


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(name, marks=pytest.mark.xfail(strict=True, reason=reason))
        for name, reason in sorted(TABLES_WHOSE_PRODUCER_IS_MISSING.items())
    ],
)
def test_a_table_recorded_as_producerless_has_gained_a_producer(name: str) -> None:
    """The recorded defects, asserted so they clear themselves.

    This fails while the gap is open, which is the honest state, and fails the other way once
    the producer exists -- at which point the entry above is what has to go.
    """
    assert name in _tables_written_by_a_script(), (
        f"results/tables/{name} is still written by no script. Write the producer and delete "
        f"its entry from TABLES_WHOSE_PRODUCER_IS_MISSING."
    )


# Mermaid renders client-side on GitHub with a configuration this repository does not control.
# `htmlLabels` may be disabled, in which case an HTML tag in a node label is shown literally
# rather than applied, and a diagram that looked right locally reads as markup to a reviewer.
# `<br/>` is exempt: Mermaid handles it natively as a line break in both configurations.
MERMAID_SAFE_TAG = re.compile(r"<(?!br\s*/?>)[a-zA-Z]")
MERMAID_ENTITY = re.compile(r"&#\d+;|&[a-zA-Z]+;")
MERMAID_UNQUOTED_LABEL = re.compile(r'\[(?![\s]*")[^\]]*[<>][^\]]*\]')


def _mermaid_blocks() -> list[tuple[str, str]]:
    blocks = []
    for path in sorted(REPO.glob("**/*.md")):
        if ".venv" in path.parts or ".git" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"```mermaid\n(.*?)\n```", text, re.S):
            blocks.append((str(path.relative_to(REPO)), match.group(1)))
    return blocks


def test_mermaid_labels_survive_a_renderer_without_html_labels() -> None:
    for name, block in _mermaid_blocks():
        tag = MERMAID_SAFE_TAG.search(block)
        assert tag is None, (
            f"{name}: mermaid label carries the HTML tag at offset {tag.start()}; it renders "
            f"literally when htmlLabels is off. Use plain text and <br/>."
        )
        entity = MERMAID_ENTITY.search(block)
        assert entity is None, (
            f"{name}: mermaid label carries the HTML entity {entity.group(0)!r}, which is not "
            f"decoded when htmlLabels is off. Write the character or spell the symbol out."
        )
        unquoted = MERMAID_UNQUOTED_LABEL.search(block)
        assert unquoted is None, (
            f"{name}: mermaid label {unquoted.group(0)!r} contains an angle bracket outside "
            f"quotes, which the parser reads as syntax."
        )


def test_mermaid_node_numbers_agree_with_the_split_table() -> None:
    """The diagram prints block sizes.  They must come from the same table as everything else."""
    blocks = [b for _, b in _mermaid_blocks() if "Temporal split" in b]
    assert blocks, "the pipeline diagram is missing from the README"
    splits = pd.read_csv(REPO / "results" / "tables" / "splits.csv")
    temporal = splits[splits["arm"] == "temporal"]
    for _, row in temporal.iterrows():
        printed = f"{int(row['n_rows']):,} rows"
        assert any(printed in b for b in blocks), (
            f"the diagram does not print '{printed}' for the {row['block']} block; "
            f"splits.csv is the source of truth"
        )
    total = f"{int(temporal['n_rows'].sum()):,} transactions"
    assert any(total in b for b in blocks), (
        f"the diagram's dataset total must be the sum of the four blocks, {total}"
    )


def test_baseline_defaults_reproduce_the_committed_table() -> None:
    """`make reproduce` passes no flags, so the defaults are what rebuild the table.

    They did not.  The committed `baselines.csv` is one model over three arms; the script
    defaulted to three models over one arm, so `make reproduce` overwrote the table with a
    different shape and `summarise_split_arms.py` then halted on the two missing arms.  The
    repository's own reproduction command could not rebuild the table behind the largest
    effect it reports, and nothing raised, because a default is not exercised by any test that
    passes explicit flags.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_baselines", REPO / "scripts" / "run_baselines.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    committed = pd.read_csv(REPO / "results" / "tables" / "baselines.csv")

    from hsbcfraud.config import load_config

    cfg = load_config(None)
    default_arms = [module.REPORTED_ARM, *cfg.split.control_arms]
    assert sorted(committed["arm"].unique()) == sorted(default_arms), (
        f"baselines.csv covers {sorted(committed['arm'].unique())} but the default arms are "
        f"{sorted(default_arms)}; `make reproduce` would write a table of a different shape"
    )
    assert sorted(committed["model"].unique()) == ["xgboost"], (
        "baselines.csv is single-model; if that changes, the --models default must change too"
    )


def test_the_history_reads_as_one_author() -> None:
    """A sole-author submission whose history shows three contributors invites the wrong question.

    Three author strings accumulated over 58 commits -- two name variants on one address, one
    of them a typo for the GitHub handle, plus a second address.  All are the same person, and
    `.mailmap` is the supported way to say so without rewriting a single commit.

    This asserts the canonical view rather than the raw one: `git log --format=%an` still
    returns all three, which is correct, because the history is evidence and must not be
    edited to look tidier than it was.
    """
    out = subprocess.run(
        ["git", "shortlog", "-sne", "--all"], cwd=REPO, capture_output=True, text=True, check=True
    )
    authors = [line.split("\t", 1)[1].strip() for line in out.stdout.splitlines() if line.strip()]
    assert len(authors) == 1, (
        f"`git shortlog -sne` reports {len(authors)} authors: {authors}. Add the new commit "
        f"identity to .mailmap; do not rewrite history to fix this."
    )


def test_the_derived_decision_log_is_not_a_scientific_artefact() -> None:
    """It projects `docs/decisions.md`, so it must be classified with the document it projects.

    It lands under `results/tables/`, which the scientific glob sweeps, so every added decision
    entry used to be reported as a *scientific* artefact changing -- the same message a
    corrupted measurement produces. `make check` also regenerates it before verifying, because
    `check` depends on `claims` depends on `derived`, so the command invalidated the manifest
    it then checked. Its own producer's docstring says it counts documents and "not of any
    run".
    """
    sys.path.insert(0, str(REPO / "scripts"))
    from freeze import SCIENTIFIC, SPECIFICATION, SPECIFICATION_UNDER_RESULTS

    log = "results/tables/decision_log.csv"
    assert log in SPECIFICATION, "the derived log must be frozen with the document it projects"
    assert log in SPECIFICATION_UNDER_RESULTS, "it must also be excluded from the scientific glob"
    assert any(pattern.startswith("results/tables/") for pattern in SCIENTIFIC), (
        "this test is only meaningful while a scientific pattern still covers results/tables"
    )


def test_the_built_documents_are_named_so_a_rebuild_is_diagnosed_correctly() -> None:
    """`make check` rebuilds the PDFs before verifying them, and must say so when they move.

    A specification change alone moves their bytes -- adding a decision entry changes
    `ClaimDecisionEntries` and therefore both PDFs. Reporting that as "the run is not
    reproducible" teaches whoever hits it to re-freeze until the message stops, which is the
    one habit a freeze exists to prevent.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    from freeze import BUILT_DOCUMENTS, SCIENTIFIC

    assert set(BUILT_DOCUMENTS) <= set(SCIENTIFIC), (
        "every built document must also be a frozen scientific artefact, or the special-case "
        "message would describe files the check never compares"
    )


# Ranges that must not appear in a tracked text file.  Deliberately narrow: this repository
# legitimately carries em dashes, section signs, Greek letters, ceilings and comparison
# operators, and a surname regex whose class runs to U+017F.  Banning non-ASCII wholesale would
# reject all of that and the gate would be deleted within a day.
FORBIDDEN_RANGES = (
    (0x3000, 0x303F, "CJK punctuation"),
    (0x3040, 0x30FF, "kana"),
    (0x4E00, 0x9FFF, "CJK ideographs"),
    (0xFF00, 0xFFEF, "fullwidth forms"),
    (0x2600, 0x27BF, "miscellaneous symbols and dingbats"),
    (0x2B00, 0x2BFF, "miscellaneous symbols and arrows"),
    (0xFE0F, 0xFE0F, "emoji variation selector"),
    (0x1F000, 0x1FAFF, "emoji"),
)

TEXT_SUFFIXES = {".md", ".tex", ".yaml", ".py", ".toml", ".json", ".txt", ".cfg"}


def _forbidden_character(text: str) -> tuple[int, str, str] | None:
    for index, character in enumerate(text):
        point = ord(character)
        for low, high, label in FORBIDDEN_RANGES:
            if low <= point <= high:
                return index, f"U+{point:04X}", label
    return None


def test_no_japanese_or_emoji_survives_in_a_tracked_text_file() -> None:
    """The repository goes public, and two conventions here were enforced only by hand.

    `COMPLIANCE_CHECKLIST.md` P6 forbids emoji and its evidence column said "scanned" -- a
    statement about one afternoon, not a property of the tree. The status marker was written
    in Japanese throughout for the same reason: nothing checked. Both were fixed by sweeping
    28 occurrences of one string and ten of one glyph, and a sweep with no gate behind it comes
    back.

    The ranges are narrow on purpose. Em dashes, section signs, Greek letters, ceilings and
    accented surnames are all legitimate and all non-ASCII; a gate that rejected them would be
    removed rather than obeyed.
    """
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()

    offences = []
    for name in tracked:
        path = REPO / name
        if path.suffix not in TEXT_SUFFIXES or not path.is_file():
            continue
        found = _forbidden_character(path.read_text(encoding="utf-8", errors="ignore"))
        if found is not None:
            _index, code, label = found
            offences.append(f"{name}: {code} ({label})")

    assert not offences, (
        "tracked text files carry characters the project forbids: "
        + "; ".join(offences)
        + ". Use an English status marker such as 'Needs confirmation', and a word such as "
        "'Note:' in place of a warning glyph."
    )


def test_no_tracked_file_points_at_the_private_planning_directory() -> None:
    """The planning notes are confidential and this repository becomes public.

    `COMPLIANCE_CHECKLIST.md` P10 asserts that no content, filename or citation from that
    directory appears here, and its evidence was "scanned across every tracked file" -- the
    same past-action evidence that let P6 decay while ten forbidden glyphs accumulated behind
    it.

    The check is on the directory path rather than on the note filenames deliberately. Naming
    the files here would put the very strings the rule protects into a public repository, so
    the gate would leak exactly what it guards. The path is a standard tooling directory name
    and reveals nothing.

    A clone cannot pull the notes in by accident -- they sit outside this working tree, in a
    parent that is not a git repository -- so what remains is a person pasting a path or a
    quotation, which is what this catches.
    """
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()

    # Assembled rather than written as a literal, because the gate scans every tracked file
    # including this one: a literal here would make the check fail on itself, which is how a
    # self-referential rule ends up being deleted rather than obeyed.
    needle = "." + "claude" + "/"
    offenders = []
    for name in tracked:
        path = REPO / name
        if path.suffix not in TEXT_SUFFIXES or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for number, line in enumerate(text.splitlines(), 1):
            if needle in line:
                offenders.append(f"{name}:{number}")

    assert not offenders, (
        f"tracked files reference the private planning directory: {offenders}. Remove the "
        f"reference; if a path must stay ignored, put the rule in .git/info/exclude, which is "
        f"not published."
    )


def test_the_method_figure_is_generated_and_reachable() -> None:
    """The opening figure must be produced by the build and referenced where it is claimed.

    No test asserted anything about a figure before this one, which is how the README came to
    embed `overview.png` while `architecture.png` -- the figure the proposal actually shipped --
    was referenced from a single documentation file and from nowhere a reviewer would look.

    Numbers inside it are substituted from `splits.csv` and `mps_lift.csv` at draw time and
    `substitute()` raises on an unreplaced token, so this checks reachability rather than
    re-deriving values a gate already owns.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    from make_figures import METHOD_STAGES, method_figure

    for suffix in ("png", "pdf"):
        assert (REPO / "results" / "figures" / f"method.{suffix}").is_file(), (
            f"method.{suffix} is missing; run `make figures`"
        )

    assert method_figure in _figure_builders(), (
        "method_figure is not in make_figures.main's builder tuple, so `make figures` would "
        "leave a stale file on disk"
    )

    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "results/figures/method.png" in readme, "the README no longer embeds the method figure"
    proposal = (REPO / "submission" / "content" / "02-method.tex").read_text(encoding="utf-8")
    assert "method.pdf" in proposal, "the proposal no longer includes the method figure"

    # The band is where the quantum arms enter, so it must be one of the shaded stages or the
    # figure's dashed arrows would point at an unshaded box and say nothing.
    shaded = [name for name, _detail, fill in METHOD_STAGES if fill != "#dfe6ee"]
    assert any("band" in name for name in shaded), (
        f"the band is no longer drawn as guarantee-carrying; shaded stages are {shaded}"
    )


def _figure_builders() -> tuple:
    """The builders `make_figures.main` actually iterates, read from its source.

    Read rather than imported because `main` runs them; importing the tuple would mean
    executing the figure build inside the test suite.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    import make_figures

    source = (REPO / "scripts" / "make_figures.py").read_text(encoding="utf-8")
    listed = re.search(r"for builder in \(([^)]+)\):", source, re.S)
    assert listed, "make_figures.main no longer iterates a builder tuple"
    return tuple(
        getattr(make_figures, name.strip())
        for name in listed.group(1).replace("\n", " ").split(",")
        if name.strip()
    )


# Width of a stage box and of a dead-end box, as fractions of the 100-unit x-range the method
# figure draws in. Kept beside the test rather than imported, so a change to the figure's
# geometry has to be made deliberately in two places instead of silently widening the check.
METHOD_STAGE_BOX_UNITS = 13.2
METHOD_DEAD_BOX_UNITS = 29.5


def test_no_method_figure_label_overflows_its_box() -> None:
    """Every label must fit the rectangle drawn around it.

    Two did not, and both were found by eye rather than by any check: "temporal split" ran
    0.960 in against a 0.904 in box, and "tensor network: no improvement" ran 1.860 against
    1.850. A figure whose text crosses its own border reads as carelessness in the one
    artefact a reviewer looks at before reading anything.

    Measured through matplotlib's renderer at the sizes the figure actually uses, so this
    catches a label that grows, a font size that rises, and a box that narrows.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(REPO / "scripts"))
    from make_figures import METHOD_DEAD_ENDS, METHOD_STAGES, TEXT_WIDTH_IN

    figure = plt.figure(figsize=(TEXT_WIDTH_IN, 1.10))
    renderer = figure.canvas.get_renderer()

    def widest(text: str, size: float, weight: str) -> float:
        return max(
            figure.text(0, 0, line, fontsize=size, fontweight=weight)
            .get_window_extent(renderer=renderer)
            .width
            / figure.dpi
            for line in text.split("\n")
        )

    stage_box = METHOD_STAGE_BOX_UNITS / 100 * TEXT_WIDTH_IN
    dead_box = METHOD_DEAD_BOX_UNITS / 100 * TEXT_WIDTH_IN
    placeholders = {"<rows>": "590,540", "<first>": "0", "<last>": "181"}

    too_wide = []
    for name, detail, _fill in METHOD_STAGES:
        for text, size, weight in ((name, 8.8, "bold"), (detail, 7.2, "normal")):
            for token, value in placeholders.items():
                text = text.replace(token, value)
            measured = widest(text, size, weight)
            if measured > stage_box:
                too_wide.append(f"{text!r} is {measured:.3f} in against a {stage_box:.3f} in box")
    for label in METHOD_DEAD_ENDS:
        measured = widest(label, 7.0, "bold")
        if measured > dead_box:
            too_wide.append(f"{label!r} is {measured:.3f} in against a {dead_box:.3f} in box")
    plt.close(figure)

    assert not too_wide, "labels overflow their boxes in the method figure: " + "; ".join(too_wide)


def test_the_method_figure_stays_within_the_height_it_replaced() -> None:
    """It sits on a page with no spare line, so its height is a hard constraint.

    The split figure it replaced was 1.264 in after the tight bounding box. The first draft of
    the replacement came out at 1.400 and pushed the proposal to seven pages, which is a
    failure a page-count gate catches only after a full LaTeX rebuild.
    """
    pypdf = pytest.importorskip("pypdf")
    figure = REPO / "results" / "figures" / "method.pdf"
    if not figure.is_file():
        pytest.skip("method.pdf is not built; run `make figures`")
    height = float(pypdf.PdfReader(str(figure)).pages[0].mediabox.height) / 72
    assert height <= 1.264, (
        f"the method figure is {height:.3f} in tall against the 1.264 in it replaced; page 2 "
        f"has no spare line, so a taller figure costs a page"
    )


# A tripwire, not a measured optimum.  The README is 471 lines today and its job is to be read
# end to end; the failure mode this guards is growth by accretion, where sections are appended
# until nobody reads past the first screen and the `docs/` split quietly stops being maintained.
# The headroom is deliberate: a ceiling that fires on the next honest paragraph gets raised
# rather than obeyed, which is how a gate becomes a formality.
README_MAX_LINES = 600


def test_the_readme_stays_short_enough_to_be_read() -> None:
    """The README's length is a standing requirement with no check behind it until now.

    `COMPLIANCE_CHECKLIST.md` P3 requires supporting documents to be linked rather than inlined
    so the README stays readable.  That is a property of the *documents*, and it held while the
    README itself grew past four hundred lines.  Nothing measured the thing the requirement is
    actually about.

    The remedy when this fires is to move a section into `docs/` and link it, which is what P3
    already asks for -- not to compress prose until the ceiling is met.
    """
    lines = (REPO / "README.md").read_text(encoding="utf-8").splitlines()
    assert len(lines) <= README_MAX_LINES, (
        f"README.md is {len(lines)} lines, over the {README_MAX_LINES}-line ceiling. "
        f"Move a section into docs/ and link it from README section 1; every .md in docs/ is "
        f"already required to be linked there, so the split costs nothing to navigate."
    )


def test_the_kaggle_rules_acceptance_is_recorded_as_an_attestation() -> None:
    """Acceptance is the author's word, and must stay labelled as the author's word.

    The competition rules gate what this dataset may be used for, so a reviewer will want to
    know they were accepted.  Nothing in this repository can evidence that -- it happened in a
    browser -- and the risk is not that the row goes missing but that it quietly firms up into
    a claim of verification during a later edit.

    So this pins three things: that the row exists, that it still says *attested*, and that the
    two rule quotations it sits beside remain attributed to a URL a reviewer can open. The last
    matters because the quotations are independently checkable and the attestation is not;
    letting them blur together would borrow the stronger fact's standing for the weaker one.
    """
    text = (REPO / "docs" / "PROVENANCE.md").read_text(encoding="utf-8")

    assert "| Rules accepted |" in text, (
        "PROVENANCE.md section 1.1 no longer records whether the competition rules were accepted"
    )
    assert "Attested by the author, not verifiable from this repository" in text, (
        "the rules-acceptance row no longer marks itself as an attestation; if it has become "
        "verifiable, say what verifies it rather than dropping the qualifier"
    )
    assert "https://www.kaggle.com/competitions/ieee-fraud-detection/rules" in text, (
        "the rules URL is gone, so the 7.A and 7.B quotations are no longer checkable"
    )
    for section in ("non-commercial purposes only", "transmit, duplicate, publish, redistribute"):
        assert section in text, f"the competition-rules quotation lost {section!r}"


def test_every_decision_cross_reference_resolves_on_github() -> None:
    """`#d-094` is not the anchor GitHub generates for `### D-094 Two teammates' ...`.

    GitHub slugs a heading from its **whole text**, so the decision log's own cross-references --
    written as short `#d-NNN` fragments -- resolve only where an explicit `<a id>` was placed by
    hand. Twenty-four had one and a hundred and nineteen did not, so eighteen links inside
    `decisions.md` landed at the top of a four-thousand-line file instead of at the entry they
    named. Nothing failed, because no gate read links.

    Every heading now carries an anchor, and this pins both halves: that the anchors keep pace
    with the headings, and that no reference points at one that does not exist.
    """
    text = (REPO / "docs" / "decisions.md").read_text(encoding="utf-8")
    headings = {name.lower() for name in re.findall(r"^### (D-\d+)\b", text, re.M)}
    anchors = set(re.findall(r'<a id="(d-\d+)"></a>', text))

    missing = sorted(headings - anchors)
    assert not missing, (
        f"{len(missing)} decision heading(s) carry no explicit anchor, so a `#{missing[0]}` link "
        f"would silently land at the top of the file: {missing[:8]}. Add `<a id=\"d-nnn\"></a>` "
        f"on the line above the heading."
    )

    referenced = {frag for _, frag in re.findall(r"\]\(([^)\s]*)#(d-\d+)\)", text)}
    dangling = sorted(referenced - anchors)
    assert not dangling, f"decision reference(s) point at anchors that do not exist: {dangling}"


def test_no_module_under_src_is_orphaned() -> None:
    """A module nothing imports is read by a reviewer as live code, and it is not.

    `src/hsbcfraud/telemetry.py` was 685 lines with **zero importers and zero tests**, describing
    an append-only run-record design that `progress.py` implements differently -- two
    implementations of one subsystem, one of them dead, including two `format_duration`s that
    disagree on non-finite input. Nothing referenced it from the README, `docs/`, or the
    Makefile, so nothing would ever have failed.

    Both it and the unused `progress.reporting` helper were removed. This is what stops the next
    one: every module under `src/` must be imported somewhere in the tree, so a subsystem that
    stops being used has to be deleted deliberately rather than left to be mistaken for shipped
    work.
    """
    package = REPO / "src" / "hsbcfraud"
    modules = {
        path.relative_to(package).with_suffix("").as_posix().replace("/", ".")
        for path in package.rglob("*.py")
        if path.name != "__init__.py"
    }

    imported: set[str] = set()
    for directory in SOURCE_DIRS:
        for path in (REPO / directory).rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:  # pragma: no cover - a syntax error is another test's problem
                continue
            for node in ast.walk(tree):
                # Both spellings reach a module, and only one puts its name in `node.module`.
                # `from hsbcfraud.metrics import f` names it there; `from hsbcfraud import
                # metrics` names it in the aliases, and reading only the first flagged a module
                # two scripts import on every run. A gate that accuses live code is worse than
                # no gate, because it gets deleted rather than obeyed.
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.removeprefix("hsbcfraud."))
                    for alias in node.names:
                        imported.add(f"{node.module}.{alias.name}".removeprefix("hsbcfraud."))
                        imported.add(alias.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imported.add(alias.name.removeprefix("hsbcfraud."))

    orphaned = sorted(m for m in modules if m not in imported)
    assert not orphaned, (
        f"{len(orphaned)} module(s) under src/hsbcfraud are imported nowhere in src/, scripts/ "
        f"or tests/: {orphaned}. Delete them, or wire them in. A module that ships without a "
        f"caller reads as live code to anyone auditing this repository."
    )


def test_the_readme_reference_count_matches_the_reference_file() -> None:
    """README section 8.1 counts `REFERENCES.md`; nothing made the two move together.

    The count drifted to 57 against an actual 60 because three entries were added on
    2026-09-14 and `REFERENCE_CROSSCHECK.md` followed -- it is generated -- while the README
    did not, because no gate binds a README number to anything.  D-141 recorded that hole and
    closed it for the Beta-Binomial worked example only; this is the same class of defect at a
    different number, found by an audit rather than by a check.

    Binding the two here is cheaper than a general README ledger and catches the specific
    failure that actually occurred: adding a reference without renumbering the prose.
    """
    references = (REPO / "docs" / "REFERENCES.md").read_text(encoding="utf-8")
    entries = re.findall(r"^\*\*\[[A-Z]+-[0-9]+\]\*\*", references, re.M)
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    stated = re.search(r"\| (\d+) numbered entries,", readme)
    assert stated, "README section 8.1 no longer states a reference count in the expected form"
    assert int(stated.group(1)) == len(entries), (
        f"README.md says {stated.group(1)} numbered entries; docs/REFERENCES.md has "
        f"{len(entries)}. Adding a reference means updating the count in README section 8.1."
    )


def test_the_readme_length_narration_is_current() -> None:
    """Two files narrate the README's line count, and both had gone stale at 469 against 471.

    The ceiling itself never fired -- 471 is well under 600 -- so the gate stayed green while
    the sentences describing it became wrong.  A number in prose beside a passing test reads as
    checked, which is worse than no number at all.
    """
    actual = len((REPO / "README.md").read_text(encoding="utf-8").splitlines())
    for relative, pattern in (
        ("docs/COMPLIANCE_CHECKLIST.md", r"(\d+) lines against a 600-line ceiling"),
        ("tests/test_repo_hygiene.py", r"The README is (\d+) lines today"),
    ):
        found = re.search(pattern, (REPO / relative).read_text(encoding="utf-8"))
        assert found, f"{relative} no longer narrates the README length in the expected form"
        assert int(found.group(1)) == actual, (
            f"{relative} says {found.group(1)} lines; README.md has {actual}."
        )

