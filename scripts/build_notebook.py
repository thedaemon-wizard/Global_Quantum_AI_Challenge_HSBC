#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""E14 -- execute the reviewer's walkthrough and commit the result as a notebook.

``notebooks/walkthrough.py`` traces the certificate against the committed tables and asserts
every figure the submission reports.  Running it takes about a second, but a reviewer browsing
the repository on GitHub should not have to clone it to see the result.  GitHub renders
``.ipynb`` inline, so an executed notebook shows every assertion and every value with no tooling
on the reader's side at all.

**The tooling is an optional extra, and that is the point.**  ``make venv`` installs ``.[dev]``;
jupytext and nbconvert are in ``.[notebook]`` instead, so the environment the clean-room
procedure builds is unchanged.  Reading the committed notebook needs nothing; only regenerating
it needs the extra.  That is the same arrangement ``parity.csv`` has with ``gpu-crosscheck``.

**Byte-stability is engineered, not hoped for.**  The walkthrough reads CSVs and prints numbers,
so cell *outputs* are deterministic.  Notebook *metadata* is not: ``language_info.version``
carries the interpreter's patch level and ``kernelspec`` the kernel's name, both of which vary by
machine.  Left alone that is the ``smoke.json`` problem -- an artefact promising a byte-identity
it cannot keep across hosts.  So the volatile fields are removed after execution and the
notebook is written through ``nbformat`` with a pinned version, which makes the file reproducible
anywhere the tables are the same.

    .venv/bin/python -m pip install -e '.[notebook]'
    .venv/bin/python scripts/build_notebook.py

Writes ``notebooks/walkthrough.ipynb``.  Exit 0 on success, 1 if the walkthrough's own assertions
fail -- in which case the notebook is *not* written, because a notebook showing a failed
walkthrough would be a worse artefact than none.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hsbcfraud.paths import display_path

REPO = Path(__file__).resolve().parents[1]

# Written into the notebook in place of whatever kernel happened to execute it.  A name, not a
# path: the value is what a reader's Jupyter will try to match, and pinning it keeps the file
# identical across machines whose kernels are registered under different display names.
KERNEL_NAME = "python3"
KERNEL_DISPLAY = "Python 3"

# Metadata keys removed after execution because they describe the machine rather than the
# result.  `version` is the interpreter patch level; the rest are nbconvert bookkeeping.
VOLATILE_LANGUAGE_KEYS = ("version",)


def _require_extra() -> tuple[object, object, object]:
    """Import the optional toolchain, or fail with the command that installs it.

    Deliberately not a soft skip.  A silent no-op here leaves a stale notebook committed and a
    green build, which is exactly the drift this repository keeps finding; the sync gate in
    `tests/test_repo_hygiene.py` would catch it eventually, but the failure should be here and
    immediate.
    """
    try:
        import jupytext
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"{error.name} is not installed. Regenerating the notebook needs the optional "
            f"toolchain, which is deliberately absent from `.[dev]` so that `make venv` and the "
            f"clean-room procedure stay unchanged:\n"
            f"    .venv/bin/python -m pip install -e '.[notebook]'\n"
            f"Reading the committed notebook needs none of this."
        ) from error
    return jupytext, nbformat, ExecutePreprocessor


def normalise(notebook: object) -> None:
    """Strip every field that describes the run rather than the computation.

    Three sources of churn, each of which makes the file differ between two runs that computed
    exactly the same thing, and all three were measured rather than guessed at:

    * ``metadata.language_info.version`` and ``metadata.kernelspec`` -- the interpreter patch
      level and the local kernel's registered name, so they vary by host.
    * ``cell.metadata.execution`` -- ``ExecutePreprocessor`` records four wall-clock timestamps
      per cell by default. On this notebook that was **52 timestamps**, and it is the single
      largest source of diff between runs.
    * ``cell.id`` -- nbformat 4.5 assigns a random identifier to every cell. Numbering them by
      position keeps them unique and valid while making them a function of the document.

    Without all three, `freeze.py` reports a scientific artefact changing on every rebuild when
    nothing measured has moved, which is precisely the alarm-fatigue failure D-080 is about.
    """
    metadata = notebook.metadata  # type: ignore[attr-defined]
    metadata["kernelspec"] = {
        "display_name": KERNEL_DISPLAY,
        "language": "python",
        "name": KERNEL_NAME,
    }
    language_info = metadata.get("language_info", {})
    for key in VOLATILE_LANGUAGE_KEYS:
        language_info.pop(key, None)
    metadata["language_info"] = language_info

    for position, cell in enumerate(notebook.cells):  # type: ignore[attr-defined]
        cell["id"] = f"cell-{position:03d}"
        cell.get("metadata", {}).pop("execution", None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=REPO / "notebooks" / "walkthrough.py")
    parser.add_argument("--out", type=Path, default=REPO / "notebooks" / "walkthrough.ipynb")
    args = parser.parse_args(argv)

    jupytext, nbformat, execute_preprocessor = _require_extra()

    notebook = jupytext.read(args.source, fmt="py:percent")
    print(f"Read {display_path(args.source)}: {len(notebook.cells)} cells.")

    runner = execute_preprocessor(timeout=600, kernel_name=KERNEL_NAME)
    try:
        runner.preprocess(notebook, {"metadata": {"path": str(args.source.parent)}})
    except Exception as error:
        print(
            f"\nThe walkthrough did not run to completion, so no notebook was written.\n"
            f"  {type(error).__name__}: {error}\n"
            f"Run `make walkthrough` to see the failure directly -- its assertions compare the "
            f"documents against results/tables/, so this usually means a table moved and a "
            f"document did not.",
            file=sys.stderr,
        )
        return 1

    normalise(notebook)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle, version=nbformat.NO_CONVERT)

    outputs = sum(len(cell.get("outputs", [])) for cell in notebook.cells)
    print(f"  executed, {outputs} output(s) captured")
    print(f"  Wrote {display_path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
