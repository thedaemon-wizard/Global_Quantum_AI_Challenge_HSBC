#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Emit every measured number as a LaTeX macro, so the proposal cannot contain a typed one.

``docs/claims.yaml`` binds each quoted number to a table row; ``scripts/check_claims.py``
verifies the binding.  This script turns the same file into ``\\newcommand`` definitions.  A
number written directly into the ``.tex`` source is then a defect that review can see, because
every legitimate figure appears as ``\\ClaimSomething`` instead.

Values are emitted **as written in claims.yaml**, at the precision recorded there, so the
document and the checker agree by construction rather than by two independent roundings.

    .venv/bin/python scripts/make_tex.py --out submission/generated
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]

# LaTeX macro names may contain only letters, so digits in a key would silently produce an
# unusable macro.  Keys are validated rather than mangled: a key that cannot become a macro
# is an authoring error worth failing on.
ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


def format_value(value: Any) -> str:
    """The value exactly as claims.yaml records it, with a LaTeX minus for negatives.

    A bare ``-`` in text mode renders as a hyphen, which is wrong for a numeric difference
    and is the kind of typographic error that survives every proofread.
    """
    text = str(value)
    return f"$-${text[1:]}" if text.startswith("-") else text


LATEX_LABELS = {
    "temporal": "temporal",
    "stratified": "stratified",
    "card_disjoint": "card-disjoint",
    "mps": "MPS",
    "xgboost": "GBDT",
}


def escape(text: str) -> str:
    """Minimal LaTeX escaping for values that came out of a CSV."""
    return str(text).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def derive(name: str, row: pd.Series) -> str:
    """Columns that are a statement about a row rather than a field of it."""
    if name == "interval_verdict":
        inside = bool(row["finite_sample_ok"])
        verdict = "inside" if inside else "breached"
        return f"{verdict} ($[{row['band_low']:.0f},{row['band_high']:.0f}]$)"
    if name == "interval_verdict_rolling":
        # This table carries its verdict as a word rather than a boolean, and uses upper case
        # for the breaches.  Normalise rather than trusting the casing.
        verdict = str(row["verdict"]).lower()
        return f"{verdict} ($[{row['band_low']:.0f},{row['band_high']:.0f}]$)"
    if name == "confidence_interval":
        return f"$[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]$"
    raise SystemExit(f"unknown derived column {name!r}")


def render_cell(spec: dict[str, Any], row: pd.Series) -> str:
    if "derived" in spec:
        return derive(spec["derived"], row)
    value = row[spec["from"]]
    fmt = spec.get("format", "{}")
    if fmt == "label":
        return LATEX_LABELS.get(str(value), escape(value))
    if pd.isna(value):
        return "---"
    rendered = fmt.format(value)
    # A leading "-" in text mode renders as a hyphen, which is visibly shorter than a minus
    # and wrong for a signed difference.  Signed cells go into math mode; unsigned ones stay
    # in text so the table does not mix two digit shapes for no reason.
    return f"${rendered}$" if rendered[0] in "+-" else rendered


def select_row(frame: pd.DataFrame, selectors: dict[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, wanted in selectors.items():
        series = frame[column]
        if isinstance(wanted, (int, float)) and not isinstance(wanted, bool):
            mask &= (pd.to_numeric(series, errors="coerce") - float(wanted)).abs() < 1e-9
        else:
            mask &= series.astype(str) == str(wanted)
    matched = frame[mask]
    if len(matched) != 1:
        raise SystemExit(
            f"selector {selectors} matched {len(matched)} rows; a table row must identify one"
        )
    return matched.iloc[0]


def render_table(spec: dict[str, Any], repo: Path) -> str:
    path = repo / spec["source"]
    if not path.exists():
        raise SystemExit(f"table source {spec['source']} does not exist")
    frame = pd.read_csv(path)
    rows = (
        [select_row(frame, selectors) for selectors in spec["rows"]]
        if "rows" in spec
        else [frame.iloc[i] for i in range(len(frame))]
    )
    columns = spec["columns"]
    header = " & ".join(c["header"] for c in columns) + r" \\"
    body = [
        " & ".join(render_cell(c, row) for c in columns) + r" \\" for row in rows
    ]
    source = spec["source"].replace("results/tables/", "")
    return "\n".join(
        [
            f"% Generated from {source} by scripts/make_tex.py. Do not edit.",
            r"\begin{table}[t]\centering",
            f"\\caption{{{spec['caption'].strip()}}}",
            f"\\label{{{spec['label']}}}",
            f"\\begin{{tabular}}{{{spec['align']}}}",
            r"\toprule",
            header,
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--claims", type=Path, default=REPO / "docs" / "claims.yaml")
    parser.add_argument("--tables", type=Path, default=REPO / "docs" / "tables.yaml")
    parser.add_argument("--out", type=Path, default=REPO / "submission" / "generated")
    args = parser.parse_args(argv)

    document = yaml.safe_load(args.claims.read_text(encoding="utf-8"))
    claims = document["claims"]

    bad = [c["key"] for c in claims if not set(c["key"]) <= ALLOWED]
    if bad:
        raise SystemExit(
            f"these claim keys contain characters LaTeX cannot use in a macro name: {bad}. "
            "Use letters only."
        )

    lines = [
        "% Generated by scripts/make_tex.py from docs/claims.yaml. Do not edit.",
        "% Every number in the proposal comes from here; a literal in the .tex is a defect.",
        "",
    ]
    for claim in claims:
        source = claim["source"].replace("results/tables/", "")
        lines.append(f"% {claim['key']}: {source}")
        lines.append(f"\\newcommand{{\\Claim{claim['key']}}}{{{format_value(claim['value'])}}}")
    lines.append("")

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "claims.tex"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(claims)} macros to {target.relative_to(REPO)}")

    tables = yaml.safe_load(args.tables.read_text(encoding="utf-8"))["tables"]
    for spec in tables:
        rendered = render_table(spec, REPO)
        out = args.out / f"table-{spec['key']}.tex"
        out.write_text(rendered, encoding="utf-8")
        print(f"Wrote {out.relative_to(REPO)}")
    print(f"{len(tables)} table(s) generated from their source CSVs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
