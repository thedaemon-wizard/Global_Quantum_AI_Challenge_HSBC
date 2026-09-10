# SPDX-License-Identifier: Apache-2.0
"""Progress and telemetry for long-running fits.

Written because a full-scale training run produced no output for forty-six minutes and then
raised, and neither the silence nor the failure could be diagnosed without re-running.

Three defects motivate the design, and each maps to a decision below.

**Silence.**  Python buffers stdout when it is not a terminal.  A run launched as
``python scripts/run_mps.py > log`` therefore shows nothing until the buffer fills, which for
a job emitting one line per epoch means nothing until it ends.  The only way to tell the run
was alive was to attach a profiler to the process.  Every write here is flushed.

**No trajectory.**  An earlier run trained for twenty minutes and then reported a non-finite
loss.  Nothing recorded *when* the loss began to diverge, so the cause had to be reproduced at
reduced scale.  A per-step record now exists, and :class:`DivergenceWatch` reports the turn
while the run is still going rather than after it dies.

**No estimate.**  A four-job sweep at forty-six minutes a job gave no indication of remaining
time.  :class:`ProgressReporter` estimates within a job and :class:`SweepTimer` across one.

Two conventions this module obeys, both taken from the surrounding code rather than invented:

* ``src/hsbcfraud/`` does not print.  The library returns data and the scripts report it.  So
  nothing here writes anywhere unless a caller passes a destination, and ``fit_mps`` takes a
  reporter rather than constructing one.
* Telemetry belongs in ``results/runs/``, never ``results/tables/``.  ``scripts/freeze.py``
  hashes everything under ``results/tables/`` in its ``scientific`` class and requires it to
  be bit-identical across runs; a record containing wall-clock timestamps would fail
  ``make check`` every time.  ``results/runs/`` is gitignored and deliberately unhashed.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import sys
import time
from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

__all__ = [
    "DivergenceWarning",
    "DivergenceWatch",
    "ProgressReporter",
    "RunLog",
    "SweepTimer",
    "Tick",
    "format_duration",
    "run_log",
]

# Ticks ignored when estimating the rate.  The first pass through a CUDA model pays kernel
# compilation and allocator warmup and is not representative of the rest; on the full-scale
# matrix-product-state fit the first epoch runs materially slower than the median.  One is
# enough: the cost is flat from the second tick onwards.
WARMUP_TICKS = 1

# Recent tick durations kept for the estimate.  Long enough to be stable, short enough to
# follow a genuine change in cost, such as a learning-rate schedule altering step time or
# another process taking the GPU.
RATE_WINDOW = 20


def format_duration(seconds: float) -> str:
    """Human-readable duration.

    Deliberately not ``datetime.timedelta``: its string form carries microseconds, which is
    false precision for an estimate built from a handful of samples.
    """
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = round(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


@dataclass(frozen=True)
class Tick:
    """One unit of progress, with everything needed to reconstruct the run afterwards."""

    task: str
    index: int
    total: int
    elapsed: float
    duration: float
    rate: float
    remaining: float
    metrics: Mapping[str, float] = field(default_factory=dict)

    @property
    def fraction(self) -> float:
        return self.index / self.total if self.total else 0.0

    def as_record(self) -> dict[str, object]:
        """The JSONL form.  Flat, because a flat record is greppable."""
        record: dict[str, object] = {
            "task": self.task,
            "index": self.index,
            "total": self.total,
            "elapsed_s": round(self.elapsed, 3),
            "duration_s": round(self.duration, 4),
            "rate_per_s": round(self.rate, 6),
            "remaining_s": round(self.remaining, 1) if math.isfinite(self.remaining) else None,
        }
        record.update({key: _plain(value) for key, value in self.metrics.items()})
        return record

    def summary(self, *, width: int = 0) -> str:
        """One line for a human, with the metrics in the order the caller supplied them."""
        head = f"{self.task:<{width}}" if width else self.task
        parts = [
            f"{head} {self.index:>{len(str(self.total))}d}/{self.total}",
            f"{format_duration(self.elapsed):>7s} elapsed",
            f"{format_duration(self.remaining):>7s} left",
        ]
        parts.extend(f"{key} {_format_metric(value)}" for key, value in self.metrics.items())
        return "  ".join(parts)


def _plain(value: object) -> object:
    """Convert a metric to something ``json`` accepts, without guessing.

    Torch tensors and numpy scalars both expose ``item``; anything else that is not already a
    JSON scalar is a caller error rather than something to coerce silently.
    """
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(
        f"metric of type {type(value).__name__} cannot be recorded; pass a scalar, or call "
        ".item() at the call site so the synchronisation point is visible there"
    )


def _format_metric(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}" if abs(value) >= 1e-3 or value == 0 else f"{value:.3e}"
    return str(value)


class ProgressReporter:
    """Report progress of a task with a known number of units.

    A reporter is a sink, not a policy: it does not decide when to tick.  The caller calls
    :meth:`tick` once per unit and passes whatever metrics it has.

    ``stream`` receives one line per tick when it is not a terminal, and a single rewritten
    line when it is.  The distinction matters: a carriage-return progress bar written into a
    redirected file produces one enormous unreadable line, which is worse than no output.

    ``log_path`` receives a JSON object per tick.  It is opened at construction so an
    unwritable path fails immediately rather than forty minutes in, and it is flushed on every
    write so the file can be read by another process while the run continues.

    Both destinations are optional and independent.  With neither, the reporter still measures
    and returns :class:`Tick`, which is what makes it safe to construct inside a library.
    """

    def __init__(
        self,
        task: str,
        total: int,
        *,
        stream: TextIO | None = None,
        log_path: Path | None = None,
        logger: logging.Logger | None = None,
        context: Mapping[str, object] | None = None,
        label_width: int = 0,
        every: int = 1,
    ) -> None:
        if total < 1:
            raise ValueError(f"total must be at least 1, got {total}")
        if every < 1:
            raise ValueError(f"every must be at least 1, got {every}")

        self.task = task
        self.total = total
        self.every = every
        self.context = dict(context or {})
        self._stream = stream
        self._logger = logger
        self._label_width = label_width
        self._start = time.perf_counter()
        self._last = self._start
        self._durations: deque[float] = deque(maxlen=RATE_WINDOW)
        self._ticks = 0
        self._last_index = 0
        self._interactive = bool(stream is not None and stream.isatty())

        self._log: TextIO | None = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = log_path.open("w", encoding="utf-8")
            self._write_log({"event": "start", "task": task, "total": total, **self.context})

    def tick(self, index: int | None = None, **metrics: object) -> Tick:
        """Record one completed unit and return what was recorded."""
        now = time.perf_counter()
        duration = now - self._last
        self._last = now
        self._ticks += 1
        index = self._ticks if index is None else index

        # Cost per UNIT, not per tick.  A caller that ticks every hundredth resample advances
        # the index by a hundred each time, and dividing by that stride is what keeps the
        # estimate right: without it the bootstrap predicted 1m11s for work that took one
        # second, because it multiplied a per-hundred duration by the number of remaining
        # resamples rather than the number of remaining ticks.
        advanced = max(1, index - self._last_index)
        self._last_index = index
        # Warmup ticks are timed and recorded but excluded from the rate, so the estimate is
        # not poisoned by kernel compilation on the first pass.
        if self._ticks > WARMUP_TICKS:
            self._durations.append(duration / advanced)
        per_unit = statistics.median(self._durations) if self._durations else duration / advanced
        remaining = per_unit * max(0, self.total - index)

        tick = Tick(
            task=self.task,
            index=index,
            total=self.total,
            elapsed=now - self._start,
            duration=duration,
            rate=1.0 / per_unit if per_unit > 0 else math.inf,
            remaining=remaining,
            metrics={key: _plain(value) for key, value in metrics.items()},
        )
        self._write_log(tick.as_record())
        if index % self.every == 0 or index >= self.total:
            self._emit(tick)
        return tick

    def note(self, message: str, **fields: object) -> None:
        """Record something that is not a unit of progress: a warning, a phase change.

        Goes to both destinations, because a warning that appears only in the machine record
        is a warning nobody reads.
        """
        self._write_log(
            {
                "event": "note",
                "message": message,
                **{key: _plain(value) for key, value in fields.items()},
            }
        )
        if self._logger is not None:
            self._logger.warning(message)
        if self._stream is not None:
            prefix = "\n" if self._interactive else ""
            self._stream.write(f"{prefix}  {message}\n")
            self._stream.flush()

    def trace(self, record: Mapping[str, object]) -> None:
        """Record a step-resolution row.  Goes to the log only, never to the stream.

        A failure dump is hundreds of rows; putting them on stdout would bury the message
        that says why they are there.
        """
        self._write_log({"event": "trace", **{k: _plain(v) for k, v in record.items()}})

    def close(self, **summary: object) -> None:
        """Finish the task.  Safe to call twice; the second call does nothing."""
        if self._stream is not None and self._interactive:
            self._stream.write("\n")
            self._stream.flush()
        if self._log is not None:
            self._write_log(
                {
                    "event": "end",
                    "task": self.task,
                    "elapsed_s": round(time.perf_counter() - self._start, 3),
                    "ticks": self._ticks,
                    **{key: _plain(value) for key, value in summary.items()},
                }
            )
            self._log.close()
            self._log = None

    def __enter__(self) -> ProgressReporter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        # Closed even when the body raised, so a diverged run still leaves a complete record.
        self.close()

    def _emit(self, tick: Tick) -> None:
        line = tick.summary(width=self._label_width)
        if self._logger is not None:
            # Timestamped and durable, and it lands in the same file as any warning the run
            # emits, so a post-mortem reads one file rather than correlating two.
            self._logger.info(line)
        if self._stream is None:
            return
        if self._interactive:
            self._stream.write(f"\r{line}\x1b[K")
        else:
            self._stream.write(f"  {line}\n")
        self._stream.flush()

    def _write_log(self, record: Mapping[str, object]) -> None:
        if self._log is None:
            return
        self._log.write(json.dumps(record, sort_keys=False) + "\n")
        # Flushed rather than buffered: the point of the file is that it can be read while
        # the run is still going.
        self._log.flush()


class SweepTimer:
    """Remaining-time estimate across a sequence of similar jobs.

    Extrapolating from finished jobs to unfinished ones is only defensible when the jobs cost
    about the same.  For the bond-dimension sweep they do, and that is a measurement rather
    than an assumption: at 431 sites the per-step cost is 139.5, 148.8, 161.4 and 158.8 ms for
    bond dimensions 4, 8, 16 and 32, against the 1x/4x/16x/64x a ``chi^2`` model predicts.  The
    chain is bound by the launch overhead of 431 sequential contractions, not by their
    arithmetic.  See ``docs/decisions.md`` D-032.

    The estimate is therefore the median of completed job durations times the number left.  If
    that assumption is ever violated -- a sweep over something that does change cost -- this
    class will mislead, so :meth:`remaining` reports the spread it is working from and callers
    should say what they are sweeping.
    """

    def __init__(self, total_jobs: int) -> None:
        if total_jobs < 1:
            raise ValueError(f"total_jobs must be at least 1, got {total_jobs}")
        self.total_jobs = total_jobs
        self._durations: list[float] = []
        self._start = time.perf_counter()

    def record(self, duration: float) -> None:
        self._durations.append(duration)

    @property
    def completed(self) -> int:
        return len(self._durations)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def remaining(self) -> float:
        """Estimated seconds left, or ``inf`` before any job has finished."""
        if not self._durations:
            return math.inf
        return statistics.median(self._durations) * (self.total_jobs - self.completed)

    def summary(self) -> str:
        if not self._durations:
            return f"job 1 of {self.total_jobs}, no estimate yet"
        spread = ""
        if len(self._durations) > 1:
            spread = f" (jobs so far {min(self._durations):.0f}-{max(self._durations):.0f}s)"
        return (
            f"{self.completed} of {self.total_jobs} jobs done, "
            f"{format_duration(self.elapsed)} elapsed, "
            f"{format_duration(self.remaining())} left{spread}"
        )


class DivergenceWarning(RuntimeWarning):
    """A run is showing the signature of an impending non-finite loss."""


class DivergenceWatch:
    """Report the turn towards divergence while the run can still be stopped.

    The failure this exists for: a loss that falls for twenty minutes and then becomes
    non-finite, with nothing in between to say when it turned.  Two signals are tracked, both
    already computed by the training loop.

    **Gradient norm.**  ``torch.nn.utils.clip_grad_norm_`` returns the total norm *before*
    clipping, and the loop was discarding it.  It moves before the loss does, because clipping
    masks the effect on the loss for a while.

    **Loss trend.**  Used when no gradient norm is available -- a boosted-tree callback, for
    instance.  Comparing the mean of the second half of a window against the first detects
    exponential growth ``exp(t / tau)`` only when ``tau < (window / 2) / ln(multiple)``, because
    the ratio of the two half-means is exactly ``exp((window / 2) / tau)`` and does not grow as
    the run progresses.  This is not a tuning detail: with the gradient window of 50 and a
    multiple of 1.5 the limit is 62 steps, and a simulated divergence with ``tau = 90`` was
    never detected at all.  The loss window is therefore separate and larger -- 200 steps,
    giving a limit of 247 -- and a caller changing either value can work out what it will and
    will not see.

    The test is on *persistence*, not magnitude, and that choice is a measurement rather than
    a preference.  An earlier version compared the latest gradient norm against its running
    median and warned above a multiple of it.  On simulated traces carrying the heavy-tailed
    spikes real training produces -- rare batches with norms eight to twenty-five times
    typical -- that test warned on **40 of 40 healthy runs** at multiples of 5, 10 and 20.
    Raising the multiple to 40 or 80 bought silence on healthy runs at the cost of firing 228
    and 314 steps *after* the loss had already doubled, which is not a warning.  A single
    spike cannot move a median, so comparing the median of a recent window against the
    long-run median separates the two cleanly: at a multiple of 2 it gave 0 of 40 false
    alarms and 117 steps of warning before the loss doubled.

    It warns; it does not stop.  Abandoning a forty-six-minute job is the caller's decision,
    and a watchdog that halts on a heuristic would eventually discard a good run.

    **This has been validated on synthetic traces only, and it does not transfer.**  Measured
    across sixteen real full-scale fits, two of which failed to train: the detector warned on
    ten of sixteen, catching one of the two failures and firing on nine of the fourteen healthy
    runs -- 50 per cent sensitivity at a 64 per cent false-positive rate.

    The cause is not a threshold.  It is the premise.  The design assumes a failing run shows
    *elevated* gradients; on this model the two that failed had a median gradient norm of 0.81
    against 10.69 for the healthy ones.  The failure mode is the run going **quiet**, not loud,
    and no threshold on elevation can detect a signal that moves the other way.

    So the gradient path is retained for the failure it was built against -- a loss growing
    towards non-finite, which the pre-clip norm does lead -- and is **not** a detector for a
    fit that stops learning.  What separates those two groups is the evaluation metric: the
    failed fits had a falling AUC while every healthy one rose.  ``fit_mps`` takes an
    ``evaluate`` callback for that reason and every long-running caller must pass it.  With two
    positives in sixteen runs there is not enough evidence to calibrate an early rule, and
    tuning one on two examples would be fitting noise.  See ``docs/decisions.md`` D-040.
    """

    def __init__(
        self,
        *,
        window: int = 50,
        history: int = 2048,
        loss_window: int = 200,
        grad_norm_multiple: float = 2.0,
        loss_multiple: float = 1.5,
        min_observations: int = 200,
    ) -> None:
        if window < 2:
            raise ValueError(f"window must be at least 2, got {window}")
        if loss_window < 4:
            raise ValueError(f"loss_window must be at least 4, got {loss_window}")
        if history <= window:
            raise ValueError(
                f"history ({history}) must exceed window ({window}); the test compares a "
                "recent window against a longer baseline and is meaningless otherwise"
            )
        self.window = window
        self.loss_window = loss_window
        self.grad_norm_multiple = grad_norm_multiple
        self.loss_multiple = loss_multiple
        self.min_observations = min_observations
        self._recent_norms: deque[float] = deque(maxlen=window)
        self._history_norms: deque[float] = deque(maxlen=history)
        self._losses: deque[float] = deque(maxlen=loss_window)
        self._observations = 0
        self._reported = False

    @property
    def observations(self) -> int:
        return self._observations

    def recent_trace(self, limit: int = 200) -> list[dict[str, float]]:
        """The last steps, for dumping when a fit dies.

        Per-epoch records answer "which epoch did it turn in", which at 696 steps an epoch is
        not an answer.  The watch is already holding these deques for its own tests, so
        exposing them costs nothing and converts the post-mortem from a re-run into a read.

        The window is bounded, and that bound is a real limitation rather than a detail: a
        precursor beginning more than ``limit`` steps before the failure will not appear here,
        and nothing in the record says so.  The per-epoch gradient-norm maximum is the only
        coarser signal, and a maximum over an epoch cannot say where in the epoch it occurred.
        """
        losses = list(self._losses)[-limit:]
        norms = list(self._history_norms)[-len(losses) :] if self._history_norms else []
        first_step = self._observations - len(losses) + 1
        trace: list[dict[str, float]] = []
        for offset, loss in enumerate(losses):
            record: dict[str, float] = {"step": first_step + offset, "loss": loss}
            if offset < len(norms):
                record["grad_norm"] = norms[offset]
            trace.append(record)
        return trace

    @property
    def slowest_detectable_growth(self) -> float:
        """Largest exponential time constant, in steps, the loss test can see."""
        return (self.loss_window / 2) / math.log(self.loss_multiple)

    def observe(self, loss: float, grad_norm: float | None = None) -> str | None:
        """Record one step.  Returns a message the first time something looks wrong.

        Reports once.  A diverging run trips the test on every subsequent step, and a warning
        repeated twenty thousand times is noise that buries the one that mattered.
        """
        self._observations += 1
        self._losses.append(float(loss))
        if grad_norm is not None and math.isfinite(grad_norm):
            self._recent_norms.append(float(grad_norm))
            self._history_norms.append(float(grad_norm))

        if self._reported or self._observations < self.min_observations:
            return None

        message = self._check_grad_norm() or self._check_loss_trend()
        if message is not None:
            self._reported = True
        return message

    def _check_grad_norm(self) -> str | None:
        if len(self._recent_norms) < self.window or len(self._history_norms) <= self.window:
            return None
        baseline = statistics.median(self._history_norms)
        recent = statistics.median(self._recent_norms)
        if baseline <= 0 or recent <= self.grad_norm_multiple * baseline:
            return None
        return (
            f"gradient norm has been elevated for {self.window} steps: median {recent:.3g} "
            f"against a long-run median of {baseline:.3g} "
            f"({recent / baseline:.1f}x) at step {self._observations}"
        )

    def _check_loss_trend(self) -> str | None:
        if len(self._losses) < self.loss_window:
            return None
        half = self.loss_window // 2
        earlier = statistics.mean(list(self._losses)[:half])
        later = statistics.mean(list(self._losses)[half:])
        if not math.isfinite(earlier) or earlier <= 0:
            return None
        if later <= self.loss_multiple * earlier:
            return None
        return (
            f"loss rose from {earlier:.4g} to {later:.4g} across the last "
            f"{self.loss_window} steps, ending at step {self._observations}"
        )


@dataclass
class RunLog:
    """A named run with one durable text log and a place to put per-task telemetry.

    The text log is the thing a person tails; the JSONL files beside it are the thing a
    post-mortem parses.  Both live under ``results/runs/``, which is gitignored, because
    ``scripts/freeze.py`` requires everything under ``results/tables/`` to be bit-identical
    across runs and a timestamped log never is.
    """

    name: str
    directory: Path
    logger: logging.Logger
    stream: TextIO | None = None

    @property
    def path(self) -> Path:
        return self.directory / f"{self.name}.log"

    def reporter(self, task: str, total: int, **kwargs: object) -> ProgressReporter:
        """A reporter wired to this run: stdout, the text log, and its own JSONL."""
        return ProgressReporter(
            task,
            total,
            stream=self.stream,
            logger=self.logger,
            log_path=self.directory / f"{self.name}-{task.replace(' ', '_')}.jsonl",
            **kwargs,  # type: ignore[arg-type]
        )

    def info(self, message: str) -> None:
        self.logger.info(message)
        if self.stream is not None:
            self.stream.write(f"{message}\n")
            self.stream.flush()


@contextmanager
def run_log(
    name: str,
    *,
    directory: Path | None = None,
    stream: TextIO | None = sys.stdout,
) -> Iterator[RunLog]:
    """Give a script one durable log file, in one line.

    The reason this exists is narrow and was measured.  Progress reporting was added to the
    training loop and then not used: a seed sweep called ``fit_mps`` without a reporter and
    ran silently for between 350 and 2,765 seconds per seed, which is the same defect the
    telemetry was written to fix, reintroduced at the call site.  A helper that makes the
    wired-up path shorter than the silent one is the only version of this that holds.

    Warnings are captured into the same file.  A run that emits a ``PerformanceWarning`` and
    then behaves oddly is much easier to read when the two are interleaved in one place.
    """
    directory = directory or Path.cwd() / "results" / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"hsbcfraud.run.{name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.FileHandler(directory / f"{name}.log", mode="w", encoding="utf-8")
    # Date as well as time: the seed sweep runs for thirteen hours and a log that says only
    # "01:14:07" cannot be ordered once it has crossed midnight.
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s",
                                           datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.addHandler(handler)

    record = RunLog(name=name, directory=directory, logger=logger, stream=stream)
    record.info(f"run {name} started; log at {record.path}")
    try:
        yield record
    finally:
        record.info(f"run {name} finished")
        logger.removeHandler(handler)
        warnings_logger.removeHandler(handler)
        handler.close()
