# SPDX-License-Identifier: Apache-2.0
"""The input file is the one thing this study cannot ship, so it must be identifiable.

The IEEE-CIS competition licence forbids redistributing the data.  Everything else here is
reproducible from the repository -- every table has a producer, every claim is bound, and the
manifest freezes 50 artefacts -- but all of it is conditional on one archive that a reader has
to fetch themselves from Kaggle.

``src/hsbcfraud/data/ieee_cis.py`` already asserted three structural properties on load: row
count, fraud count, column count.  Those catch a re-release, a truncated download, or the test
split loaded by mistake.  They cannot catch a file of the same shape holding different bytes,
and that is the case where every number silently changes and nothing complains.

A digest is not the data.  It redistributes nothing and it is the strongest identity statement
the licence permits, so these tests hold the project to actually having one.
"""

from __future__ import annotations

import re
import tempfile
import zipfile
from pathlib import Path

import pytest

from hsbcfraud.data.ieee_cis import (
    EXPECTED_MEMBER_DIGESTS,
    IEEE_CIS_ZIP,
    DatasetError,
    verify_member_digests,
)

REPO = Path(__file__).resolve().parents[1]


def test_every_member_the_loader_reads_has_a_recorded_digest() -> None:
    """A digest table that omits a member it reads gives false assurance.

    The loader verifies the transaction table always and the identity table when asked for
    them, so both must be present here or the check silently covers less than it appears to.
    """
    assert set(EXPECTED_MEMBER_DIGESTS) == {"train_transaction.csv", "train_identity.csv"}
    for member, digest in EXPECTED_MEMBER_DIGESTS.items():
        assert len(digest) == 64, f"{member} digest is not a SHA-256 hex string"
        assert set(digest) <= set("0123456789abcdef"), f"{member} digest is not lowercase hex"


def test_the_calibrated_digests_match_the_archive_on_this_machine() -> None:
    """The constants must be the ones the committed numbers were produced from.

    A digest that has drifted from the file is worse than no digest: it fails every correct
    download and teaches whoever hits it to delete the check.
    """
    archive = REPO / "datasets" / IEEE_CIS_ZIP
    if not archive.exists():
        pytest.skip("the archive is not on this machine; the digests cannot be confirmed here")
    computed = verify_member_digests(archive)
    assert computed == {member: EXPECTED_MEMBER_DIGESTS[member] for member in computed}


def test_a_tampered_member_is_refused() -> None:
    """The check must reject, not merely compute.

    Written against a synthetic archive rather than the real one so it runs anywhere and costs
    nothing: what is under test is the comparison, not the hash function.
    """
    with tempfile.TemporaryDirectory() as directory:
        forged = Path(directory) / IEEE_CIS_ZIP
        with zipfile.ZipFile(forged, "w") as archive:
            archive.writestr("train_transaction.csv", "TransactionID,isFraud\n1,0\n")
        with pytest.raises(DatasetError, match="not the file this study was calibrated"):
            verify_member_digests(forged, ["train_transaction.csv"])


def test_a_missing_member_names_what_the_archive_actually_holds() -> None:
    """The failure has to be diagnosable from its message alone.

    Someone hitting this has downloaded the wrong archive or an incomplete one, and the useful
    thing to tell them is what they do have -- not merely that a name was absent.
    """
    with tempfile.TemporaryDirectory() as directory:
        wrong = Path(directory) / IEEE_CIS_ZIP
        with zipfile.ZipFile(wrong, "w") as archive:
            archive.writestr("test_transaction.csv", "TransactionID\n1\n")
        with pytest.raises(DatasetError, match=re.escape("test_transaction.csv")):
            verify_member_digests(wrong, ["train_transaction.csv"])
