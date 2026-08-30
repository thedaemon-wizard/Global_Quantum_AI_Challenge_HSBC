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
