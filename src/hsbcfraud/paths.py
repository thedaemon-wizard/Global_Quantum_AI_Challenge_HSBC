# SPDX-License-Identifier: Apache-2.0
"""Path formatting for the command-line scripts.

Every script prints the files it wrote, and the readable form is the repository-relative one.
``Path.relative_to`` is the obvious way to produce it and the wrong one on its own: it raises
``ValueError`` when the path does not lie under the repository, which includes the ordinary
case of a relative argument such as ``--out submission/generated`` taken from the Makefile.

That turns a success message into a crash after the work is already done -- the artefact is on
disk and the script exits non-zero -- which is the most confusing failure a build can have.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def display_path(path: Path | str, *, root: Path = REPO) -> str:
    """``path`` relative to the repository when it lies inside it, unchanged otherwise.

    Resolving first is what makes a relative argument work: ``submission/generated`` resolves
    against the working directory, which for every entry point here is the repository root.
    """
    try:
        return str(Path(path).resolve().relative_to(root))
    except ValueError:
        return str(path)


def require_run_artefact(path: Path, *, produced_by: str) -> Path:
    """Return ``path`` if it exists, or exit naming the target that produces it.

    Everything under ``results/runs/`` is gitignored deliberately: the per-seed score files are
    large intermediates, and the repository commits the tables derived from them instead.  So a
    fresh clone does not carry them, and ten scripts read the same one.

    A bare ``FileNotFoundError`` on an absolute path to a file a reader has never heard of is
    the least useful thing a build can say, and it is what a clean-room checkout produced.
    There is no fallback to add here -- the score cannot be invented -- so the only improvement
    available is to turn the crash into an instruction.

    All ten readers now route through here; six did not when the clean-room checkout failed.
    The count is worth stating because the helper is only worth having if it is the single
    door: one unwrapped ``read_parquet`` reproduces the original failure verbatim, and no test
    enumerates the readers, so the next one added is the one that regresses this.
    """
    if path.exists():
        return path
    raise SystemExit(
        f"{display_path(path)} is missing.  It is an intermediate artefact and is not "
        f"committed; run `make {produced_by}` to produce it."
    )
