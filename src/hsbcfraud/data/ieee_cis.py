# SPDX-License-Identifier: Apache-2.0
"""Loading the IEEE-CIS Fraud Detection file, and what its label actually means.

The dataset is read directly out of the competition zip rather than an extracted copy, so
there is one artefact on disk whose hash pins the input.  ``datasets/`` is gitignored: the
data is distributed under Kaggle competition rules, not an open licence, and is never
committed.

The label
---------
``isFraud`` is not "this transaction was fraudulent".  Per the competition host (Kaggle
discussion 101203), a reported chargeback sets the flag, and the flag is then **propagated
to every later transaction sharing a user account, email address or billing address**; a
transaction is labelled 0 only when nothing is reported within 120 days.

Three consequences run through the whole study and are handled rather than noted:

* many ``isFraud == 1`` rows are transactions linked to a compromised card, not frauds, so
  a model fitted here is partly performing entity contamination detection;
* a purely temporal split does not separate entities.  It changes the direction of the
  overlap, and measured on this file 85.0 % of the ``card1`` values in the test block also
  occur in the training block.  Confidence intervals are therefore card-level block
  bootstrap, never row-level (see :mod:`hsbcfraud.stats`);
* the propagation rule is exactly what the competition's winning "UID" feature
  reconstructed.  That feature is deliberately not used here, and its ablation is reported:
  an issuer holds the true client identifier natively, so recovering it from de-identified
  columns measures the de-identification, not transferable headroom.

The obvious fourth consequence -- that rows near the end of the file cannot have had their
120-day window elapse, so ``isFraud == 0`` would be immature exactly where the test block
sits -- is testable, and :mod:`hsbcfraud.data.label_audit` tests it.  On this file it does
not hold: the fraud rate shows no terminal decay.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = [
    "IEEE_CIS_ZIP",
    "SECONDS_PER_DAY",
    "IeeeCisFrame",
    "load_ieee_cis",
    "read_ieee_cis_columns",
]

SECONDS_PER_DAY = 86_400
IEEE_CIS_ZIP = "ieee-fraud-detection.zip"
_TRANSACTION_MEMBER = "train_transaction.csv"
_IDENTITY_MEMBER = "train_identity.csv"

# Asserted on load.  These are properties of the published file, and a mismatch means the
# archive is not the one this study was calibrated against -- a re-release, a partial
# download, or the test split by mistake.  Failing here is much cheaper than discovering it
# in a certified risk level.
EXPECTED_ROWS = 590_540
EXPECTED_FRAUDS = 20_663
EXPECTED_COLUMNS = 394


class DatasetError(RuntimeError):
    """Raised when the archive is missing, malformed, or not the expected release."""


@dataclass(frozen=True)
class IeeeCisFrame:
    """A loaded IEEE-CIS transaction table and the facts asserted about it.

    ``day`` is a 86,400-second bucket counted from the file's own minimum
    ``TransactionDT``.  It is not a calendar day: the reference epoch is undisclosed, so
    the only defensible statement is "days since the first transaction in this file".
    """

    frame: pd.DataFrame
    n_rows: int
    n_frauds: int
    n_columns: int
    span_days: float

    @property
    def fraud_rate(self) -> float:
        return self.n_frauds / self.n_rows


def read_ieee_cis_columns(
    zip_path: Path, columns: list[str] | None = None, *, member: str = _TRANSACTION_MEMBER
) -> pd.DataFrame:
    """Read selected columns straight from the archive, without extracting it.

    Reading a subset matters: the full transaction table is 683 MB of CSV and most of this
    study's stages need six columns.  Passing ``columns=None`` reads all 394.
    """
    if not zip_path.exists():
        raise DatasetError(
            f"{zip_path} not found. The IEEE-CIS archive must be downloaded from "
            "kaggle.com/competitions/ieee-fraud-detection/data after accepting the "
            "competition rules in a browser; the API returns 403 until then."
        )
    with zipfile.ZipFile(zip_path) as archive:
        if member not in archive.namelist():
            raise DatasetError(
                f"{zip_path} does not contain {member}; found {sorted(archive.namelist())}"
            )
        with archive.open(member) as handle:
            return pd.read_csv(handle, usecols=columns)


def load_ieee_cis(
    zip_path: Path, columns: list[str] | None = None, *, with_identity: bool = False
) -> IeeeCisFrame:
    """Load the training transactions, add a ``day`` column, and assert the file's identity.

    ``with_identity`` left-joins ``train_identity.csv`` on ``TransactionID``.  Only about a
    quarter of transactions carry identity rows, so the join introduces missingness that is
    itself informative; the join is not performed unless asked for.
    """
    # TransactionID joins the requested columns only when the identity merge needs it as a
    # key.  Adding it unconditionally would change what an explicit column list means, and
    # leaving it out made `load_ieee_cis(zip, ["card1"], with_identity=True)` die on a bare
    # pandas KeyError raised from inside the merge -- the one failure in this module that did
    # not arrive as a DatasetError naming what was wrong.
    required = {"TransactionDT", "isFraud"} | ({"TransactionID"} if with_identity else set())
    needed = None if columns is None else sorted(required | set(columns))
    frame = read_ieee_cis_columns(zip_path, needed)

    n_rows = len(frame)
    if n_rows != EXPECTED_ROWS:
        raise DatasetError(f"expected {EXPECTED_ROWS:,} training rows, read {n_rows:,}")
    n_frauds = int(frame["isFraud"].sum())
    if n_frauds != EXPECTED_FRAUDS:
        raise DatasetError(f"expected {EXPECTED_FRAUDS:,} frauds, read {n_frauds:,}")

    if not frame["TransactionDT"].is_monotonic_increasing:
        # The file ships in time order, which is why a temporal split is a prefix cut.  If
        # that ever stops holding, sorting here would silently change what "the first 60 %"
        # means relative to the committed tables, so it is refused instead.
        raise DatasetError("TransactionDT is not monotonically increasing; the file changed")

    # The column-count assertion belongs here, on the transaction table alone, and not after
    # the identity join: joining adds 40 identity columns, and checking afterwards would
    # either fail on a correct file or force the expected count to encode whether a join
    # happened, which is two facts in one number.
    n_columns = len(frame.columns)
    if columns is None and n_columns != EXPECTED_COLUMNS:
        raise DatasetError(f"expected {EXPECTED_COLUMNS} transaction columns, read {n_columns}")

    origin = int(frame["TransactionDT"].min())
    span_days = float(
        (frame["TransactionDT"].max() - frame["TransactionDT"].min()) / SECONDS_PER_DAY
    )

    if with_identity:
        identity = read_ieee_cis_columns(zip_path, None, member=_IDENTITY_MEMBER)
        frame = frame.merge(identity, on="TransactionID", how="left")

    # Assigned after the join so the copy that de-fragments the frame happens once.
    frame = frame.assign(day=((frame["TransactionDT"] - origin) // SECONDS_PER_DAY).astype("int32"))

    return IeeeCisFrame(
        frame=frame,
        n_rows=n_rows,
        n_frauds=n_frauds,
        n_columns=n_columns,
        span_days=span_days,
    )
