#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Recompute every number quoted in prose from the table it came from.

Prose and tables have disagreed four times in this project, each time caught only by someone
reading carefully.  Reading does not scale to a five-page proposal and does not survive a
late edit, so every quoted number is bound to a table row in ``docs/claims.yaml`` and this
script recomputes it.

Exit codes: 0 all claims verified; 1 at least one mismatch, missing table or ambiguous
selector.  There is no partial success -- a proposal containing one wrong number is wrong.

    .venv/bin/python scripts/check_claims.py
    .venv/bin/python scripts/check_claims.py --update   # rewrite values from the tables
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]


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


def resolve(claim: dict[str, Any]) -> float:
    """The number this claim asserts, recomputed from its source table."""
    path = REPO / claim["source"]
    if not path.exists():
        raise ClaimError(f"source table {claim['source']} does not exist")
    frame = pd.read_csv(path)

    reduce_spec = claim.get("reduce")
    if reduce_spec is not None:
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
            subset = frame[frame[reduce_spec["where"]].astype(bool)]
            left, right = reduce_spec["columns"]
            return float((subset[left] - subset[right]).min())
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--claims", type=Path, default=REPO / "docs" / "claims.yaml")
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite each value from its table instead of checking it",
    )
    args = parser.parse_args(argv)

    document = yaml.safe_load(args.claims.read_text(encoding="utf-8"))
    claims = document["claims"]

    failures: list[str] = []
    updated = 0
    print(f"{len(claims)} claims in {args.claims.relative_to(REPO)}\n")

    for claim in claims:
        key = claim["key"]
        try:
            actual = resolve(claim)
        except ClaimError as error:
            failures.append(f"{key}: {error}")
            print(f"  UNRESOLVED  {key:32s} {error}")
            continue
        except KeyError as error:
            failures.append(f"{key}: missing field {error}")
            print(f"  MALFORMED   {key:32s} missing field {error}")
            continue

        stated = float(claim["value"])
        tolerance = float(claim.get("tolerance", 0.0))
        # Round to the stated precision before comparing.  A claim of 1.494 against a
        # measured 1.4938 is correct at the precision it is printed; comparing raw would
        # reject every rounded figure in the document.
        agrees = abs(round(actual, _decimals(claim["value"])) - stated) <= tolerance

        if args.update and not agrees:
            claim["value"] = round(actual, _decimals(claim["value"]))
            updated += 1
            print(f"  UPDATED     {key:32s} {stated} -> {claim['value']}")
        elif agrees:
            print(f"  ok          {key:32s} {stated}")
        else:
            failures.append(f"{key}: prose says {stated}, table gives {actual}")
            print(f"  MISMATCH    {key:32s} prose {stated}  table {actual}")

    if args.update:
        args.claims.write_text(
            yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=100),
            encoding="utf-8",
        )
        print(f"\nRewrote {updated} value(s) into {args.claims.relative_to(REPO)}.")
        print("Review the diff: an updated claim means the prose around it may also be stale.")
        return 0

    if failures:
        print(f"\n{len(failures)} of {len(claims)} claims failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(f"\nAll {len(claims)} claims agree with their tables.")
    return 0


def _decimals(value: Any) -> int:
    """Decimal places in the value as written, so comparison happens at printed precision."""
    text = str(value)
    return len(text.split(".")[1]) if "." in text else 0


if __name__ == "__main__":
    raise SystemExit(main())
