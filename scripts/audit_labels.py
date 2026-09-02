#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""E2 -- test whether IEEE-CIS labels are censored at the end of the file.

Writes ``results/tables/label_censoring.csv`` (the 14-day bucket series) and
``results/tables/label_verdict.csv`` (the trend and two-proportion statistics).

If censorship were present, the final block -- the test fold, where the certified quantity
lives -- would carry the least trustworthy ``Y = 0`` labels in the dataset, and a maturity
buffer of up to 120 of the file's 182 days would be required. That is expensive enough to
be worth testing rather than assuming in either direction.

    .venv/bin/python scripts/audit_labels.py
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from hsbcfraud.data.ieee_cis import IEEE_CIS_ZIP, load_ieee_cis
from hsbcfraud.data.label_audit import audit_label_censoring
from hsbcfraud.paths import display_path
from hsbcfraud.progress import run_log

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", type=Path, default=REPO / "datasets" / IEEE_CIS_ZIP)
    parser.add_argument("--bucket-days", type=int, default=14)
    parser.add_argument("--out", type=Path, default=REPO / "results" / "tables")
    parser.add_argument("--runs", type=Path, default=REPO / "results" / "runs")
    args = parser.parse_args(argv)

    # Wrapped because the IEEE-CIS load is the expensive part and reported nothing: the
    # file is 590,540 rows with identity columns, and until it finished the script was
    # indistinguishable from a hang.  `run_log` writes to results/runs/ and to stdout,
    # so a run launched with its output redirected is still followable.
    with run_log("audit_labels", directory=args.runs) as run:
        run.info(f"loading {args.zip.name}")
        loaded = load_ieee_cis(args.zip, ["TransactionAmt"])
        run.info(f"{loaded.n_rows:,} rows over {loaded.span_days:.2f} days")
        verdict, buckets = audit_label_censoring(loaded.frame, bucket_days=args.bucket_days)
        run.info(f"censoring verdict: {verdict.summary()}")

    # Read off the verdict rather than typed: 120 comes from LABEL_WINDOW_DAYS, the same
    # constant that produced the trailing_days column written below, so the sentence and the
    # table cannot state different windows.
    print(
        f"IEEE-CIS spans {loaded.span_days:.2f} days; "
        f"label window is {verdict.trailing_days} days\n"
    )
    print(f"{'days':>12}  {'n':>8}  {'frauds':>7}  {'count rate':>11}  {'value rate':>11}")
    for _, row in buckets.iterrows():
        print(
            f"{int(row.bucket):5d}-{int(row.bucket_end):<6d} {int(row.n):8,d}  "
            f"{int(row.n_fraud):7,d}  {row.count_rate:11.4f}  {row.value_rate:11.4f}"
        )
    print(f"\n{verdict.summary()}")

    if verdict.censoring_detected:
        print(
            "\nCensoring detected. A maturity buffer is required and the estimand must be "
            "restated; see docs/protocol.md section 1.1."
        )
    else:
        print(
            "\nNo terminal decay. The final block is usable as the test fold without a "
            "maturity buffer. Label PROPAGATION across linked entities is a separate "
            "matter and is handled by the card-level block bootstrap (E1, E3)."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    buckets.to_csv(args.out / "label_censoring.csv", index=False)
    pd.DataFrame([asdict(verdict)]).to_csv(args.out / "label_verdict.csv", index=False)
    print(
        f"\nWrote {display_path(args.out / 'label_censoring.csv')} and "
        f"{display_path(args.out / 'label_verdict.csv')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
