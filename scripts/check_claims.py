#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Recompute every number quoted in prose from the table it came from.

Prose and tables have disagreed four times in this project, each time caught only by someone
reading carefully.  Reading does not scale to a five-page proposal and does not survive a
late edit, so every quoted number is bound to a table row in ``docs/claims.yaml`` and this
script recomputes it.

Exit codes: 0 all claims verified; 1 at least one mismatch, missing table or ambiguous
selector.  There is no partial success -- a proposal containing one wrong number is wrong.

Two further gates share this entry point because they answer the same question -- does a
statement in the documents resolve to something the repository can produce?

``--citations``  every citation-shaped string in the documents resolves to an entry in
                 ``docs/REFERENCES.md``.  Four documents claimed this gate existed before it
                 did.
``--unused``     every claim defined in ``docs/claims.yaml`` is used somewhere.  An unused
                 claim is either dead weight or a number that lost its home when the prose
                 changed, and one of them was a figure a decision entry had retracted.

    .venv/bin/python scripts/check_claims.py
    .venv/bin/python scripts/check_claims.py --citations
    .venv/bin/python scripts/check_claims.py --unused
    .venv/bin/python scripts/check_claims.py --update   # rewrite values from the tables
"""

from __future__ import annotations

import argparse
import contextlib
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from hsbcfraud.analysis.citations import find_misses
from hsbcfraud.paths import display_path

REPO = Path(__file__).resolve().parents[1]

# Documents whose citations must resolve.  The reference list itself is excluded: an entry may
# legitimately mention a neighbouring one in its annotation.
#
# docs/REFERENCE_IMPLEMENTATION.md is here because it is nothing but citations -- twenty
# identifiers mapped to the files that implement them -- and it was ungated, so renumbering an
# entry in REFERENCES.md would have broken all twenty at once without failing anything.
CITED_DOCUMENTS = (
    "README.md",
    "docs/RESULTS.md",
    "docs/protocol.md",
    "docs/decisions.md",
    "docs/PROVENANCE.md",
    "docs/COMPLIANCE_CHECKLIST.md",
    "docs/SUBMISSION_CHECKLIST.md",
    "docs/REFERENCE_IMPLEMENTATION.md",
)
# LaTeX sources, where a claim is consumed as a \Claim macro.
CLAIM_MACRO_CONSUMERS = ("submission/content/*.tex",)
# Markdown documents, which have no macro mechanism and quote the value as text instead.
# Scanning them for the value serves two purposes: a claim quoted only in prose is not dead,
# and a prose figure that has drifted from its claim stops matching and is reported.
#
# docs/decisions.md is deliberately NOT here. It is a historical log and quotes superseded
# figures on purpose, so matching against it would mark a retracted claim as live -- which is
# the exact failure this check exists to catch.
# docs/RESULTS.md is here because the README's results section moved into it: a list of
# documents to scan is a list that goes stale the moment one is added, and this one did --
# splitting the README out reported a live figure as a dead claim within the same hour.
CLAIM_TEXT_CONSUMERS = (
    "README.md",
    "docs/RESULTS.md",
    "docs/PROVENANCE.md",
    "docs/protocol.md",
)


class ClaimError(Exception):
    """A claim could not be resolved to exactly one number."""


def select_rows(frame: pd.DataFrame, selectors: dict[str, Any]) -> pd.DataFrame:
    """Rows matching every selector.

    Numeric selectors are compared with a tolerance rather than for equality: a bond
    dimension written as ``32`` in YAML meets ``32.0`` in a CSV whose column also holds
    NaN for the baseline row, and exact comparison silently returns nothing.
    """
    mask = pd.Series(True, index=frame.index)
    for column, wanted in selectors.items():
        if column not in frame.columns:
            raise ClaimError(f"selector column {column!r} is not in the table")
        series = frame[column]
        if isinstance(wanted, (int, float)) and not isinstance(wanted, bool):
            mask &= (pd.to_numeric(series, errors="coerce") - float(wanted)).abs() < 1e-9
        else:
            mask &= series.astype(str) == str(wanted)
    return frame[mask]


def resolve_derived(claim: dict[str, Any], resolved: dict[str, float]) -> float:
    """A claim computed from other claims rather than read from a table.

    Several figures in the prose are ratios of two measured quantities -- seed spread against
    capacity spread, cross-family standard error against within-family, the sample-size factor
    implied by a minimum detectable effect, a kernel's share of a latency budget.  Typing those
    by hand puts a number in the document that does not move when its inputs do, which is the
    exact failure this file exists to prevent: the proposal claimed a "thousandfold" FLOP span
    for a grid that spans sixty-four.

    Derived claims may only reference table-backed claims, not other derived ones.  One level
    keeps the resolution order trivial -- position in the file does not matter -- and keeps the
    dependency legible without tracing a chain.

    The referenced values are the ones the document prints, ``scale`` already applied, so a
    ratio of two percentages is the ratio a reader would compute from the page.
    """
    spec = claim["derived"]
    op = spec["op"]
    if op != "ratio":
        raise ClaimError(f"unknown derived op {op!r}")
    left, right = spec["of"]
    for key in (left, right):
        if key not in resolved:
            raise ClaimError(
                f"derived claim references {key!r}, which did not resolve to a "
                "table-backed claim"
            )
    if resolved[right] == 0:
        raise ClaimError(f"derived claim divides by {right!r}, which resolved to zero")
    ratio = (resolved[left] / resolved[right]) ** float(spec.get("power", 1))
    # `scale` exists so a share can be stated as a percentage without the percentage itself
    # being typed anywhere.  A hand-typed 87.8 does not move when its two inputs do, which is
    # the whole reason this file exists.
    return ratio * float(spec.get("scale", 1))


def resolve(claim: dict[str, Any]) -> float:
    """The number this claim asserts, recomputed from its source table."""
    path = REPO / claim["source"]
    if not path.exists():
        raise ClaimError(f"source table {claim['source']} does not exist")
    frame = pd.read_csv(path)

    reduce_spec = claim.get("reduce")
    if reduce_spec is not None:
        # A reduction may be restricted to a subset of rows.  Without it a range over a
        # multi-arm table silently mixes arms, which is the kind of quiet category error this
        # file exists to prevent.
        where = reduce_spec.get("where_equals")
        if where:
            frame = select_rows(frame, where)
            if frame.empty:
                raise ClaimError(f"no rows match {where}")
        op = reduce_spec["op"]
        if op == "sum":
            return float(frame[reduce_spec["column"]].astype(bool).sum())
        if op == "sum_and":
            columns = reduce_spec["columns"]
            combined = frame[columns[0]].astype(bool)
            for extra in columns[1:]:
                combined &= frame[extra].astype(bool)
            return float(combined.sum())
        if op == "min_margin":
            # Smallest gap between two columns, over rows passing a boolean filter.  Used
            # for the certificate margin, where the quantity of interest is a distance the
            # table does not store as a column.
            #
            # The boolean filter is optional, as ``where_equals`` already is: a margin between
            # two columns that are constant down the table -- a budget and the part of it
            # already committed -- has no subset to restrict to.
            boolean_filter = reduce_spec.get("where")
            subset = frame[frame[boolean_filter].astype(bool)] if boolean_filter else frame
            left, right = reduce_spec["columns"]
            return float((subset[left] - subset[right]).min())
        if op == "sum_values":
            # Numeric sum, distinct from "sum" which counts truthy rows.
            return float(frame[reduce_spec["column"]].sum())
        if op == "median":
            return float(frame[reduce_spec["column"]].median())
        if op == "max_ratio":
            left, right = reduce_spec["columns"]
            return float((frame[left] / frame[right]).max())
        if op == "range":
            # Max minus min. The comparison that motivates it -- seed spread against
            # across-configuration spread -- is a statement about two ranges, so the range
            # itself has to be the bound quantity rather than a difference computed in prose.
            values = frame[reduce_spec["column"]]
            return float(values.max() - values.min())
        if op == "count":
            return float(len(frame))
        if op == "min":
            return float(frame[reduce_spec["column"]].min())
        if op == "max":
            return float(frame[reduce_spec["column"]].max())
        raise ClaimError(f"unknown reduce op {op!r}")

    rows = select_rows(frame, claim.get("select", {}))
    if len(rows) == 0:
        raise ClaimError(f"no row matches {claim.get('select', {})}")
    if len(rows) > 1:
        raise ClaimError(
            f"{len(rows)} rows match {claim.get('select', {})}; a claim must identify one"
        )
    column = claim["column"]
    if column not in rows.columns:
        raise ClaimError(f"column {column!r} is not in the table")
    return float(rows.iloc[0][column])


def apply_updates(path: Path, updates: list[tuple[str, object, float]]) -> str:
    """Rewrite only the ``value:`` lines that changed, leaving the rest of the file alone.

    Not a YAML round-trip.  ``yaml.safe_dump`` emits the data and discards everything else,
    which on the first version of this flag deleted all twenty-five comment lines explaining
    what the file is for and reformatted the remainder -- a convenience that silently damaged
    the artefact it exists to maintain.  The file is edited as text instead: find the block
    for a key, find its ``value:`` line, replace the number.

    Raises if a key or its value line cannot be located rather than writing a file that is
    partly updated, because a half-applied edit is worse than a refused one.
    """
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for key, _old, new in updates:
        start = next(
            (i for i, line in enumerate(lines) if line.strip() == f"- key: {key}"), None
        )
        if start is None:
            raise SystemExit(f"cannot locate a block for claim {key!r} in {path}")
        offset = next(
            (
                i
                for i in range(start + 1, len(lines))
                if lines[i].lstrip().startswith("value:")
                or lines[i].lstrip().startswith("- key:")
            ),
            None,
        )
        if offset is None or lines[offset].lstrip().startswith("- key:"):
            raise SystemExit(f"claim {key!r} has no value line to update")
        indent = lines[offset][: len(lines[offset]) - len(lines[offset].lstrip())]
        lines[offset] = f"{indent}value: {new}\n"
    return "".join(lines)


def check_citations() -> int:
    """Every citation-shaped string in the documents resolves to a reference entry."""
    references = REPO / "docs" / "REFERENCES.md"
    if not references.exists():
        print(f"{display_path(references)} does not exist", file=sys.stderr)
        return 1

    documents = {}
    for name in CITED_DOCUMENTS:
        path = REPO / name
        if path.exists():
            documents[name] = path.read_text(encoding="utf-8")
    for path in sorted((REPO / "submission" / "content").glob("*.tex")):
        documents[str(path.relative_to(REPO))] = path.read_text(encoding="utf-8")

    misses = find_misses(documents, references.read_text(encoding="utf-8"))
    print(f"{len(documents)} document(s) checked against {display_path(references)}")
    if misses:
        print(f"\n{len(misses)} citation(s) do not resolve:", file=sys.stderr)
        for miss in misses:
            print(f"  {miss}", file=sys.stderr)
        return 1
    print("  every citation resolves to a reference entry")
    return 0


def _value_appears(value: Any, text: str) -> bool:
    """Whether a claim's value, exactly as written, appears as a standalone number in ``text``.

    Bounded on both sides so that 0.05 does not match inside 0.058, and 5 does not match inside
    58,343.  A thousands separator is allowed on the left because prose writes large counts
    that way.
    """
    literal = re.escape(str(value))
    return re.search(rf"(?<![\d.]){literal}(?![\d.])", text) is not None or (
        re.search(rf"(?<![\d.]){re.escape(f'{int(value):,}')}(?![\d.])", text) is not None
        if str(value).isdigit()
        else False
    )


def check_unused(claims: list[dict[str, Any]]) -> int:
    """Claims defined in claims.yaml and reachable from no document.

    A claim counts as used if a document names its macro, *or* if a derived claim divides by
    it and that derived claim is itself used.  Missing the second case would report the inputs
    of every ratio as dead, which is how a correct check gets switched off.

    Reported, and fatal, because the last unused claim in this file was a figure a decision
    entry had already retracted and which the next edit could have reached for.
    """
    latex = "\n".join(
        path.read_text(encoding="utf-8")
        for pattern in CLAIM_MACRO_CONSUMERS
        for path in sorted(REPO.glob(pattern))
    )
    markdown = "\n".join(
        path.read_text(encoding="utf-8")
        for pattern in CLAIM_TEXT_CONSUMERS
        for path in sorted(REPO.glob(pattern))
    )
    keys = [c["key"] for c in claims]
    cited = {c["key"] for c in claims if f"Claim{c['key']}" in latex}
    # A markdown document quotes the value rather than the macro.  Matching the value exactly
    # as claims.yaml writes it is what makes this a check and not a guess: a README figure that
    # has drifted from its claim no longer matches, and the claim is reported as unused.
    cited.update(c["key"] for c in claims if _value_appears(c["value"], markdown))
    # One level of indirection, matching the one level resolve_derived permits.
    for claim in claims:
        if "derived" in claim and claim["key"] in cited:
            cited.update(claim["derived"]["of"])
    unused = [key for key in keys if key not in cited]
    print(f"{len(keys)} claim(s) defined, {len(keys) - len(unused)} used")
    if unused:
        print(f"\n{len(unused)} claim(s) are defined and used nowhere:", file=sys.stderr)
        for key in unused:
            print(f"  {key}", file=sys.stderr)
        return 1
    print("  every claim is used")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--claims", type=Path, default=REPO / "docs" / "claims.yaml")
    parser.add_argument(
        "--citations",
        action="store_true",
        help="check that every citation in the documents resolves to a reference entry",
    )
    parser.add_argument(
        "--unused",
        action="store_true",
        help="report claims defined in claims.yaml and used in no document",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite each value from its table instead of checking it",
    )
    args = parser.parse_args(argv)

    if args.citations:
        return check_citations()

    document = yaml.safe_load(args.claims.read_text(encoding="utf-8"))
    claims = document["claims"]

    if args.unused:
        return check_unused(claims)

    failures: list[str] = []
    pending: list[tuple[str, object, str]] = []
    updated = 0
    print(f"{len(claims)} claims in {display_path(args.claims)}\n")

    # Table-backed claims resolve first so that a derived claim can reference any of them
    # regardless of where it sits in the file.
    resolved: dict[str, float] = {}
    for claim in claims:
        if "derived" not in claim:
            # A failure here is reported by the main loop below, where it can be attributed
            # to its own claim rather than to whichever derived claim happened to need it.
            with contextlib.suppress(ClaimError, KeyError):
                resolved[claim["key"]] = resolve(claim) * float(claim.get("scale", 1.0))

    for claim in claims:
        key = claim["key"]
        try:
            actual = resolve_derived(claim, resolved) if "derived" in claim else resolve(claim)
        except ClaimError as error:
            failures.append(f"{key}: {error}")
            print(f"  UNRESOLVED  {key:32s} {error}")
            continue
        except KeyError as error:
            failures.append(f"{key}: missing field {error}")
            print(f"  MALFORMED   {key:32s} missing field {error}")
            continue

        actual *= float(claim.get("scale", 1.0))
        stated = float(claim["value"])
        tolerance = float(claim.get("tolerance", 0.0))
        # Round to the stated precision before comparing.  A claim of 1.494 against a
        # measured 1.4938 is correct at the precision it is printed; comparing raw would
        # reject every rounded figure in the document.
        agrees = abs(_round_like(actual, claim["value"]) - stated) <= tolerance

        if args.update and not agrees:
            # Formatted to the SAME number of decimals as the value it replaces.  Plain
            # rounding writes 0.02 where the claim said 0.0222, and _decimals then reads two
            # places instead of four on the next run, so every update would quietly loosen
            # the precision the claim is checked at.
            replacement = _format_like(actual, claim["value"])
            pending.append((key, claim["value"], replacement))
            updated += 1
            print(f"  UPDATED     {key:32s} {claim['value']} -> {replacement}")
        elif agrees:
            print(f"  ok          {key:32s} {stated}")
        else:
            failures.append(f"{key}: prose says {stated}, table gives {actual}")
            print(f"  MISMATCH    {key:32s} prose {stated}  table {actual}")

    if args.update:
        if pending:
            args.claims.write_text(apply_updates(args.claims, pending), encoding="utf-8")
        print(f"\nRewrote {updated} value(s) in {display_path(args.claims)}.")
        print("Review the diff: an updated claim means the prose around it may also be stale.")
        # An UNRESOLVED or MALFORMED claim is not rewritten and is not fixed by rewriting the
        # ones that did resolve, so it must still fail.  Returning 0 here let --update turn a
        # vanished source table into a clean exit.
        if failures:
            print(f"\n{len(failures)} claim(s) could not be resolved and were not rewritten:",
                  file=sys.stderr)
            for failure in failures:
                print(f"  {failure}", file=sys.stderr)
            return 1
        return 0

    if failures:
        print(f"\n{len(failures)} of {len(claims)} claims failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(f"\nAll {len(claims)} claims agree with their tables.")
    return 0


def _is_exponential(value: Any) -> bool:
    """Whether the claim is written in exponent notation, e.g. ``3.442e-15``."""
    return "e" in str(value).lower()


def _decimals(value: Any) -> int:
    """Decimal places in the value as written, so comparison happens at printed precision.

    For a value in exponent notation the significant digits are what is printed, not the
    decimal places: ``3.442e-15`` shows four significant figures and *fifteen* leading zeros,
    and reading "442e-15" as eighteen decimal places is meaningless.
    """
    text = str(value).lower()
    if _is_exponential(text):
        mantissa = text.split("e")[0]
        return len(mantissa.split(".")[1]) if "." in mantissa else 0
    return len(text.split(".")[1]) if "." in text else 0


def _round_like(value: float, template: Any) -> float:
    """``value`` rounded to the precision ``template`` is printed at."""
    places = _decimals(template)
    if _is_exponential(template):
        return float(f"{value:.{places}e}")
    return round(value, places)


def _format_like(value: float, template: Any) -> str:
    """``value`` rendered in the same notation and precision as ``template``.

    Formatting a value of 3.4e-15 with ``:.3f`` yields ``0.000``, which silently replaces a
    measurement with zero and then compares equal to it forever after. The notation has to
    follow the claim.
    """
    places = _decimals(template)
    if _is_exponential(template):
        return f"{value:.{places}e}"
    return f"{value:.{places}f}" if places else f"{round(value):d}"


if __name__ == "__main__":
    raise SystemExit(main())
