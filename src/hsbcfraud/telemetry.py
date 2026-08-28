# SPDX-License-Identifier: Apache-2.0
"""An append-only record of a long fit, and the human stream projected from it.

The artefact comes first
------------------------
One process writes one file: ``results/runs/telemetry/<run_id>.jsonl``.  One JSON object per
line, every line terminated and flushed before the next unit of work starts, so the file is a
complete account of the run at every instant a reader might open it.  A four-job bond-dimension
sweep is one file, not four, because the question "where did the sweep get to" is about the
sweep.  Nesting is carried in fields (``scope``, ``job``), not in the filesystem.

Everything a person sees on the terminal is :func:`render` applied to one of those objects.
There is no second formatting path.  That is the whole design: replaying the file afterwards
reproduces the live stream byte for byte, which means the log a reviewer reads and the log the
operator watched cannot disagree.

Three measured failures motivated it, and each one lands somewhere specific.

**Forty-six minutes of silence.**  Python gives a redirected stdout an 8,192-byte block
buffer, and a job emitting one line per epoch never fills it, so nothing appears until the
process exits.  Measured on this machine: a plain ``print`` loop had written zero bytes after
2.5 seconds; the same loop with ``flush=True`` had written 122.  Every write here is flushed.
``python -u`` and ``PYTHONUNBUFFERED`` also work and are both rejected: they restore the silent
failure the moment somebody launches the script without them.

**A twenty-minute run that died with no trajectory.**  ``fit_mps`` recorded loss per epoch,
so a non-finite loss at step 14,000 left thirty numbers to diagnose it with.  Here every step
is a line in the file, carrying the loss, the pre-clip gradient norm and the realised learning
rate, so "when did it turn" is an ``awk`` expression over an artefact that already exists
rather than a twenty-minute re-run.  Note what this makes unnecessary: an in-memory ring
buffer of recent steps, flushed on abort, is only worth building when there is no durable
per-step record.  There is one, so there is no ring buffer.

**No estimate.**  A scope reports seconds remaining for itself and for the run.  It also
reports *how* it got the run estimate, in ``clock.eta_basis``, because an extrapolation whose
assumption is invisible is worse than none.

Why this is not in ``results/tables/``
--------------------------------------
``scripts/freeze.py`` hashes ``results/tables/*.csv`` and ``*.json`` in its ``scientific``
class and requires them to be bit-identical across runs; ``make check`` fails otherwise, and a
file that appears after the manifest was written is a hard error.  A telemetry record contains
wall-clock timestamps and elapsed seconds, so it would fail that check on every single run and
train everyone to ignore the gate.  ``results/runs/`` is the directory that exists for exactly
this -- gitignored, unhashed, "large, not committed, derived" in ``freeze.py``'s own words --
and ``results/runs/telemetry/`` needs no change to ``freeze.py``, to ``.gitignore`` or to the
manifest.  The one committable consequence is a boolean ``telemetry`` column in the results
row, so the timing table can state whether instrumentation was on; the *path* must not go in
the CSV, because it contains a timestamp and would move the hash.

What is deterministic and what is not
-------------------------------------
Every non-deterministic quantity lives under the single key ``clock``.  Nothing else in a
record depends on when the run happened or how fast the machine was.  So

    jq -c 'del(.clock)' a.jsonl > a.det

is the diffable projection of a run, and two runs of the same seed on the same hardware differ
only where the arithmetic differs.  This is a property the format guarantees structurally, not
a convention: :meth:`Scope.advance` has no way to put a duration anywhere else.

Non-finite floats are written as the strings ``"nan"``, ``"inf"`` and ``"-inf"``.  Bare ``NaN``
is what :mod:`json` emits by default and it is not JSON -- ``jq`` rejects the line -- which
would corrupt the record at the exact moment it matters most.  ``allow_nan=False`` is set so a
leak raises here rather than producing an unparseable file.

What this module does not do
----------------------------
It holds no thresholds and no opinion about health.  Whether a gradient norm is alarming is a
statement about the model, and it belongs beside the model, so ``fit_mps`` decides and calls
:meth:`Scope.note`.  This module transports, timestamps, formats and flushes.

It also never terminates the run it is observing.  A caller that miscounts its own scopes gets
a loud warning in both streams and a null run estimate, not an exception forty minutes in.
Aborting is the caller's decision and is made on the caller's evidence.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import TextIO

__all__ = [
    "SCHEMA_VERSION",
    "Scope",
    "TelemetryLog",
    "format_duration",
    "read_records",
    "render",
]

# Bumped when a field changes meaning or disappears.  Every record carries it, so a reader
# that predates a change fails on the version rather than on a missing key.
SCHEMA_VERSION = 1

# Units excluded from the rate estimate.  Measured on the full-scale fit from a cold process:
# step 0 costs 0.7391 s against a 0.1424 s steady state, and step 1 is already within 5 % of
# steady state.  Dropping one step gives a usable estimate after three; dropping a whole epoch
# would discard 3.3 % of the run to remove a 0.025 % bias, which is not a trade worth making.
WARMUP_UNITS = 1

# Kinds that are always shown, whatever the throttle says.  A landmark suppressed because a
# step happened to be shown two seconds earlier is a landmark lost.
_ALWAYS_SHOWN = frozenset({"run_open", "run_close", "scope_open", "scope_close", "scope_fail",
                           "mark", "note"})

_MARKERS = {
    "run_open": "run  ",
    "run_close": "done ",
    "scope_open": "open ",
    "scope_close": "close",
    "scope_fail": "FAIL ",
    "unit": "     ",
    "mark": "mark ",
    "note": "note ",
}


def format_duration(seconds: float | None) -> str:
    """Human-readable duration, or ``--`` when there is nothing to report.

    Deliberately not :class:`datetime.timedelta`, whose string form carries microseconds.  An
    estimate built from a handful of samples has no business displaying six decimal places.
    """
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--"
    whole = round(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _iso() -> str:
    """UTC to the millisecond.  Millisecond because a step takes 156 ms."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _encode(value: object) -> object:
    """Convert one value to something :mod:`json` accepts, refusing to guess.

    Torch tensors, numpy scalars and numpy arrays all expose ``tolist``, so one branch covers
    them; note that calling it on a CUDA tensor forces a host synchronisation, which is why
    hot loops should read their scalars at the call site where that cost is visible.  Anything
    else that is not already a JSON type raises, because a telemetry module that coerces an
    unexpected object into ``str`` produces a record nobody can parse and nobody noticed.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # Written as strings so the line stays parseable.  Losing the distinction between
        # nan and inf would discard the most informative byte in a divergence record.
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _encode(tolist())
    raise TypeError(
        f"cannot record a value of type {type(value).__name__}; pass a JSON scalar, a list, "
        "or an object with .tolist()"
    )


def _format_value(value: object) -> str:
    # No thousand separators on a metric integer.  ``i`` and ``n`` are counts by construction
    # and get them in the progress line; an arbitrary metric may be a seed or an epoch index,
    # and "seed 20,260,828" is wrong in a way that is hard to unsee.
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}" if value == 0 or abs(value) >= 1e-3 else f"{value:.3e}"
    return str(value)


def render(record: Mapping[str, object], *, width: int = 0) -> str | None:
    """Project one record onto one line of human-readable text.

    Takes a mapping rather than a class so that the same function serves the writer and a
    reader parsing the file back with :func:`read_records`.  That is what makes a replay
    identical to the live stream instead of merely similar.

    Returns ``None`` for a record with no human projection, which at schema 1 never happens
    but is the extension point for a record kind added later that only a tool should read.
    """
    kind = str(record.get("kind", ""))
    marker = _MARKERS.get(kind)
    if marker is None:
        return None
    scope = str(record.get("scope", ""))
    head = f"{marker} {scope:<{width}}" if width else f"{marker} {scope}".rstrip()
    clock = record.get("clock")
    clock = clock if isinstance(clock, Mapping) else {}
    metrics = record.get("m")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    tail = "  ".join(f"{key} {_format_value(item)}" for key, item in metrics.items())

    if kind == "run_open":
        return (
            f"{marker} {record.get('run')}  pid {record.get('pid')}  "
            f"{record.get('scopes_planned')} scopes planned"
            + (f"  {tail}" if tail else "")
        )
    if kind == "run_close":
        return (
            f"{marker} {record.get('run')}  "
            f"{format_duration(_number(clock.get('mono_s')))} total  "
            f"{record.get('scopes_closed')}/{record.get('scopes_planned')} scopes closed"
            + (f"  {tail}" if tail else "")
        )
    if kind == "scope_open":
        return (
            f"{head}  job {record.get('job')}/{record.get('jobs')}  "
            f"{_count(record.get('n'))} {record.get('unit')}s" + (f"  {tail}" if tail else "")
        )
    if kind == "scope_close":
        return (
            f"{head}  {format_duration(_number(record.get('elapsed_s')))}  "
            f"{_count(record.get('i'))}/{_count(record.get('n'))} {record.get('unit')}s"
            + (f"  {tail}" if tail else "")
        )
    if kind == "scope_fail":
        return f"{head}  {record.get('error_type')}: {record.get('error')}"
    if kind == "note":
        level = str(record.get("level", "info"))
        prefix = "warn " if level == "warn" else marker
        return f"{prefix} {scope + '  ' if scope else ''}{record.get('message')}"
    if kind == "mark":
        position = ""
        if record.get("mark_index") is not None:
            position = f" {record.get('mark_index')}/{record.get('mark_total')}"
        return f"{head}  {record.get('mark')}{position}" + (f"  {tail}" if tail else "")

    # kind == "unit": the progress line proper.
    return (
        f"{head}  {_count(record.get('i'))}/{_count(record.get('n'))}  "
        f"{format_duration(_number(clock.get('mono_s'))):>7s} in  "
        f"{format_duration(_number(clock.get('eta_scope_s'))):>7s} left  "
        f"(run {format_duration(_number(clock.get('eta_run_s')))})"
        + (f"  {tail}" if tail else "")
    )


def _count(value: object) -> str:
    return f"{value:,}" if isinstance(value, int) else str(value)


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def read_records(path: Path) -> Iterator[dict[str, object]]:
    """Parse a telemetry file, tolerating exactly one thing: a run still in progress.

    A final line with no terminator means the writer is between ``write`` and ``flush``, which
    is a property of reading an append-only file live and not a corruption.  Every other
    malformed line raises, because a telemetry reader that skips what it cannot understand
    reports a shorter run than actually happened.
    """
    with path.open(encoding="utf-8") as handle:
        pending: str | None = None
        for number, line in enumerate(handle, start=1):
            if pending is not None:
                raise ValueError(f"{path}:{number - 1}: unterminated line before end of file")
            if not line.endswith("\n"):
                pending = line
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: not a JSON object: {exc}") from exc


class TelemetryLog:
    """The file, the run identity, and the throttle that governs the human stream.

    ``stream`` gets the projection of every record, subject to ``min_interval_s`` for unit
    records only.  It defaults to stdout because that is where this repository's scripts put
    their narrative, but note the hazard that creates: this class flushes and a bare ``print``
    does not, so mixing the two on a redirected stdout reorders the file.  Scripts should send
    their narrative through :meth:`say` rather than ``print``, which also has the effect that
    the ``.jsonl`` becomes a complete account of the run rather than of the fitting only.

    ``scopes`` is how many scopes the run intends to open.  Declaring it buys a run-level
    estimate; not declaring it writes ``null`` for that estimate rather than inventing one.
    """

    def __init__(
        self,
        path: Path,
        *,
        run: str,
        stream: TextIO | None = sys.stdout,
        min_interval_s: float = 10.0,
        scopes: int | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        if min_interval_s < 0:
            raise ValueError(f"min_interval_s must not be negative, got {min_interval_s}")
        if scopes is not None and scopes < 1:
            raise ValueError(f"scopes must be at least 1, got {scopes}")

        self.run = run
        self.path = path
        self.scopes = scopes
        self.origin = time.perf_counter()
        self._stream = stream
        self._interactive = bool(stream is not None and _is_interactive(stream))
        self._min_interval = min_interval_s
        self._last_emit = float("-inf")
        self._seq = 0
        self._opened = 0
        self._closed = 0
        self._durations: list[float] = []
        self._width = 0
        self._pending_newline = False

        path.parent.mkdir(parents=True, exist_ok=True)
        # Append rather than truncate: the run identity makes collision practically
        # impossible, and if a path is ever reused by mistake the right outcome is a longer
        # file to puzzle over, not a destroyed record.  newline="\n" so the bytes are the
        # same everywhere.
        self._handle = path.open("a", encoding="utf-8", newline="\n")
        self._write(
            "run_open",
            pid=os.getpid(),
            argv=list(sys.argv),
            scopes_planned=scopes,
            path=str(path),
            m=dict(context or {}),
        )

    @classmethod
    def open_run(
        cls, directory: Path, label: str, **kwargs: object
    ) -> TelemetryLog:
        """Open a log at ``<directory>/<timestamp>-<label>-<pid>.jsonl``.

        The identity carries the pid because two sweeps started in the same second is a thing
        that happens, and because it makes attaching a profiler to a stalled run a copy rather
        than a search.
        """
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run = f"{stamp}-{label}-{os.getpid()}"
        return cls(directory / f"{run}.jsonl", run=run, **kwargs)  # type: ignore[arg-type]

    def scope(
        self, name: str, *, unit: str, total: int, **context: object
    ) -> Scope:
        """Open one job: a fit, a bootstrap, a Gram matrix.  Use it as a context manager."""
        if total < 1:
            raise ValueError(f"total must be at least 1, got {total}")
        self._opened += 1
        self._width = max(self._width, len(name))
        if self.scopes is not None and self._opened > self.scopes:
            self.note(
                f"scope {self._opened} was opened but only {self.scopes} were planned; the "
                "run estimate is suppressed rather than extrapolated",
                level="warn",
            )
        return Scope(self, name, unit=unit, total=total, index=self._opened, context=context)

    def say(self, message: str, **fields: object) -> None:
        """Narrative: what is about to happen, what was just written.  Same path as progress."""
        self.note(message, **fields)

    def note(self, message: str, *, level: str = "info", scope: str = "", **fields: object) -> None:
        """A warning or a landmark that is not a unit of work.  Goes to both destinations.

        Both, always.  A warning that reaches only the machine record is a warning nobody
        reads, and one that reaches only the terminal is gone when the terminal is.
        """
        self._write("note", scope=scope, level=level, message=message, m=fields)

    def artefact_path(self, name: str) -> Path:
        """A sibling path for something too large to be a field, such as a state snapshot.

        Path policy stays here rather than in the library that writes the snapshot, so that
        ``src/hsbcfraud/`` still decides nothing about where output lands.
        """
        return self.path.with_name(f"{self.run}.{name}")

    def close(self, **summary: object) -> None:
        """Write the run footer and close the file.  Idempotent."""
        if self._handle is None:
            return
        self._write("run_close", scopes_closed=self._closed, scopes_planned=self.scopes, m=summary)
        self._handle.close()
        self._handle = None  # type: ignore[assignment]
        if self._stream is not None and self._pending_newline:
            self._stream.write("\n")
            self._stream.flush()

    def __enter__(self) -> TelemetryLog:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc is not None:
            self.note(f"{type(exc).__name__}: {exc}", level="warn")
        self.close()

    def _eta_run(self, eta_scope: float | None, projected_scope: float | None
                 ) -> tuple[float | None, str]:
        """Seconds left in the whole run, and the basis on which that was claimed.

        Extrapolating across jobs is only defensible when jobs cost about the same, and here
        they measurably do: per-step cost at 431 sites is 0.153, 0.156, 0.147 and 0.149 s for
        bond dimensions 4, 8, 16 and 32, against the 1x/4x/16x/64x a chi^2 model predicts,
        because the chain is bound by the launch overhead of 431 sequential contractions and
        not by their arithmetic (docs/decisions.md D-032).  A chi^2 extrapolation from the
        chi=4 job predicts 284,740 s against a true 13,400.  So: measure, never scale.  The
        assumption is named in every record, and if a future sweep varies something that does
        change cost, ``eta_basis`` is where a reader will see the claim being made.
        """
        if self.scopes is None:
            return None, "undeclared"
        remaining = self.scopes - self._closed - 1
        if remaining < 0:
            return None, "overrun"
        if eta_scope is None:
            return None, "warming"
        if self._durations:
            per_scope = sum(self._durations) / len(self._durations)
            return eta_scope + per_scope * remaining, "scope_mean"
        if projected_scope is None:
            return None, "warming"
        return eta_scope + projected_scope * remaining, "current_scope"

    def _finish_scope(self, duration: float) -> None:
        self._durations.append(duration)
        self._closed += 1

    def _write(self, kind: str, *, m: Mapping[str, object] | None = None, **fields: object) -> None:
        """Assemble, encode, append, flush, project.  The only writer in the module."""
        if self._handle is None:
            raise RuntimeError(f"telemetry log {self.path} is closed")
        self._seq += 1
        record: dict[str, object] = {
            "schema": SCHEMA_VERSION,
            "run": self.run,
            "seq": self._seq,
            "kind": kind,
            **{key: _encode(value) for key, value in fields.items()},
        }
        record.setdefault("clock", {"iso": _iso(), "mono_s": round(self._elapsed(), 3)})
        if m:
            record["m"] = {str(key): _encode(value) for key, value in m.items()}
        # sort_keys so two runs produce byte-comparable lines; allow_nan=False so a
        # non-finite float that escaped _encode raises here instead of writing invalid JSON.
        self._handle.write(
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        )
        self._handle.flush()
        self._emit(kind, record)

    def _elapsed(self) -> float:
        return time.perf_counter() - self.origin

    def _emit(self, kind: str, record: Mapping[str, object]) -> None:
        if self._stream is None:
            return
        now = time.perf_counter()
        if kind not in _ALWAYS_SHOWN:
            if now - self._last_emit < self._min_interval:
                return
            self._last_emit = now
        line = render(record, width=self._width)
        if line is None:
            return
        if self._interactive and kind == "unit":
            self._stream.write(f"\r{line}\x1b[K")
            self._pending_newline = True
        else:
            # One newline-terminated line.  A carriage return in a redirected file is one
            # 1.0 MB line that wc -l reports as zero and tail -n cannot read at all.
            prefix = "\n" if self._pending_newline else ""
            self._stream.write(f"{prefix}{line}\n")
            self._pending_newline = False
        self._stream.flush()


class Scope:
    """One job with a known number of units: a fit, a bootstrap, a Gram matrix.

    Constructed by :meth:`TelemetryLog.scope`.  The unit is whatever the caller counts --
    ``step``, ``round``, ``resample``, ``pair`` -- and appears in every record so a line is
    self-describing without reference to the header.

    Use it as a context manager.  Leaving the block by exception writes a ``scope_fail``
    record naming the exception before the exception propagates, so a run that dies still
    leaves a record that says it died and where.
    """

    def __init__(
        self,
        log: TelemetryLog,
        name: str,
        *,
        unit: str,
        total: int,
        index: int,
        context: Mapping[str, object],
    ) -> None:
        self.log = log
        self.name = name
        self.unit = unit
        self.total = total
        self.index = index
        self._start = time.perf_counter()
        self._last = self._start
        self._warm_start: float | None = None
        self._done = 0
        self._closed = False
        log._write(
            "scope_open",
            scope=name,
            job=index,
            jobs=log.scopes,
            unit=unit,
            n=total,
            m=dict(context),
        )

    @property
    def done(self) -> int:
        return self._done

    def advance(self, **metrics: object) -> None:
        """Record one completed unit of work and whatever was measured while doing it.

        Called once per unit, unconditionally.  Sampling happens on the way out to the
        terminal, never on the way into the file: the file is the artefact that has to answer
        "which step was it" afterwards, and a file that saw one step in fifty cannot.
        """
        now = time.perf_counter()
        self._done += 1
        if self._done == WARMUP_UNITS:
            self._warm_start = now

        rate: float | None = None
        eta_scope: float | None = None
        projected: float | None = None
        timed = self._done - WARMUP_UNITS
        if self._warm_start is not None and timed >= 1:
            # The mean, not the median.  Remaining time is a sum, and n * sample_mean is the
            # unbiased estimator of a sum; a median discards contention stalls, which are
            # real seconds that recur.  Measured on this fit the median runs 3.3 % short.
            mean = (now - self._warm_start) / timed
            if mean > 0:
                rate = 1.0 / mean
            eta_scope = mean * max(0, self.total - self._done)
            projected = mean * self.total
        eta_run, basis = self.log._eta_run(eta_scope, projected)

        clock = {
            "iso": _iso(),
            "mono_s": round(now - self.log.origin, 3),
            "unit_s": round(now - self._last, 4),
            "rate_per_s": round(rate, 4) if rate is not None else None,
            "eta_scope_s": round(eta_scope, 1) if eta_scope is not None else None,
            "eta_run_s": round(eta_run, 1) if eta_run is not None else None,
            "eta_basis": basis,
        }
        self._last = now
        self.log._write(
            "unit",
            scope=self.name,
            job=self.index,
            jobs=self.log.scopes,
            unit=self.unit,
            i=self._done,
            n=self.total,
            clock=clock,
            m=metrics,
        )

    def mark(self, name: str, *, index: int | None = None, total: int | None = None,
             **metrics: object) -> None:
        """A landmark inside the scope: an epoch boundary, a monitor evaluation.

        Always shown, never throttled.  This is where a per-epoch discriminative metric goes,
        and it is the one line that closes the failure this project has hit twice -- a loss
        that fell steadily while the model scored AUC 0.5000 -- which no amount of loss
        history can close on its own.
        """
        self.log._write(
            "mark",
            scope=self.name,
            job=self.index,
            unit=self.unit,
            i=self._done,
            n=self.total,
            mark=name,
            mark_index=index,
            mark_total=total,
            m=metrics,
        )

    def note(self, message: str, *, level: str = "info", **fields: object) -> None:
        """A warning from the caller's own health checks, tagged with this scope."""
        self.log.note(message, level=level, scope=self.name, **fields)

    def close(self, **summary: object) -> None:
        """Write the scope footer.  Idempotent, and called for you by ``with``."""
        if self._closed:
            return
        self._closed = True
        elapsed = time.perf_counter() - self._start
        self.log._finish_scope(elapsed)
        self.log._write(
            "scope_close",
            scope=self.name,
            job=self.index,
            jobs=self.log.scopes,
            unit=self.unit,
            i=self._done,
            n=self.total,
            elapsed_s=round(elapsed, 3),
            m=summary,
        )

    def fail(self, exc: BaseException, **fields: object) -> None:
        """Record that this scope died, and why, before the exception leaves the block."""
        if self._closed:
            return
        self._closed = True
        elapsed = time.perf_counter() - self._start
        self.log._write(
            "scope_fail",
            scope=self.name,
            job=self.index,
            unit=self.unit,
            i=self._done,
            n=self.total,
            elapsed_s=round(elapsed, 3),
            error_type=type(exc).__name__,
            error=str(exc),
            m=fields,
        )

    def __enter__(self) -> Scope:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc is not None:
            self.fail(exc)
        else:
            self.close()


def _is_interactive(stream: TextIO) -> bool:
    """Whether a carriage return will be interpreted or land in a file as a literal byte.

    ``isatty`` is precisely that question.  It raises on a closed file and is absent from some
    substituted streams, and in both cases the answer is no.
    """
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False
