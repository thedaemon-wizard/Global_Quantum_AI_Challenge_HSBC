# SPDX-License-Identifier: Apache-2.0
"""Behaviour of the progress and telemetry module.

Each test pins a property that was established by measurement rather than by preference, and
the docstring says which measurement.  The divergence thresholds in particular were chosen
after a first design failed: see ``DivergenceWatch`` and the tests below.
"""

from __future__ import annotations

import io
import json
import math

import numpy as np
import pytest

from hsbcfraud.progress import (
    DivergenceWatch,
    ProgressReporter,
    SweepTimer,
    format_duration,
)


def synthetic_trace(
    kind: str, seed: int, *, n: int = 5000, onset: int = 3000, tau: float = 90.0
) -> tuple[list[float], list[float]]:
    """A loss and gradient-norm trace, healthy or diverging.

    Healthy training carries rare legitimate gradient spikes -- hard minibatches produce
    norms eight to twenty-five times typical.  Modelling them is the point: a detector that
    cannot tolerate them warns on every real run, which is what the first version of
    ``DivergenceWatch`` did.
    """
    rng = np.random.default_rng(seed)
    loss: list[float] = []
    norms: list[float] = []
    for step in range(n):
        healthy = 0.70 * math.exp(-step / 1500) + 0.30
        spike = rng.uniform() < 0.004
        if kind == "healthy" or step < onset:
            loss.append(healthy * (1 + 0.04 * rng.standard_normal()))
            base = abs(0.8 + 0.25 * rng.standard_normal())
            norms.append(base * rng.uniform(8, 25) if spike else base)
        else:
            norms.append(abs(0.8 * math.exp((step - onset) / 120) + 0.25 * rng.standard_normal()))
            loss.append(0.34 if step < onset + 150 else 0.34 * math.exp((step - onset - 150) / tau))
    return loss, norms


def first_warning(watch: DivergenceWatch, trace: tuple[list[float], list[float]], *, norms: bool):
    loss, gradient = trace
    for index, (value, norm) in enumerate(zip(loss, gradient, strict=True)):
        if watch.observe(value, norm if norms else None) is not None:
            return index
    return None


def test_format_duration_has_no_false_precision() -> None:
    assert format_duration(0) == "0s"
    assert format_duration(95) == "1m35s"
    assert format_duration(3725) == "1h02m"
    assert format_duration(math.inf) == "unknown"
    assert format_duration(-1) == "unknown"


def test_reporter_writes_one_line_per_tick_to_a_file_destination() -> None:
    """A carriage-return progress bar in a redirected file is one unreadable line.

    The run that motivated this module was launched with stdout redirected, so the
    non-terminal path is the one that matters.
    """
    stream = io.StringIO()  # not a tty
    reporter = ProgressReporter("job", 3, stream=stream)
    for _ in range(3):
        reporter.tick(loss=0.5)
    reporter.close()
    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert len(lines) == 3
    assert "\r" not in stream.getvalue()
    assert "1/3" in lines[0] and "3/3" in lines[2]


def test_log_is_readable_while_the_run_continues(tmp_path) -> None:
    """The forty-six-minute silence was buffering. Every record is flushed as it is written."""
    path = tmp_path / "run.jsonl"
    reporter = ProgressReporter("job", 4, log_path=path)
    reporter.tick(loss=0.9)
    reporter.tick(loss=0.8)
    # Read from a separate handle while the reporter still holds its own open.
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["event"] == "start"
    assert [r["loss"] for r in records[1:]] == [0.9, 0.8]
    reporter.close()


def test_unwritable_log_path_fails_at_construction(tmp_path) -> None:
    """Fail in the first second, not forty minutes in."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        ProgressReporter("job", 2, log_path=blocker / "sub" / "run.jsonl")


def test_metrics_that_are_not_scalars_raise_rather_than_coerce() -> None:
    """Silently stringifying an unexpected value would record something meaningless."""
    reporter = ProgressReporter("job", 1)
    with pytest.raises(TypeError, match="cannot be recorded"):
        reporter.tick(loss=object())


def test_estimate_excludes_the_warmup_tick() -> None:
    """The first pass through a CUDA model pays kernel compilation and is not representative.

    Simulated here by making the first tick far slower than the rest and checking the
    estimate follows the rest rather than the mean.
    """
    reporter = ProgressReporter("job", 100)
    ticks = []
    for index in range(6):
        # Force the durations rather than sleeping: the property under test is which samples
        # enter the median, not the clock.
        reporter._last -= 10.0 if index == 0 else 0.1
        ticks.append(reporter.tick())
    # With the warmup excluded, the per-unit estimate is near 0.1 s, so the remaining 94 units
    # are about 9 s rather than the ~160 s a mean including the 10 s tick would give.
    assert ticks[-1].remaining < 30


def test_watch_does_not_warn_on_healthy_synthetic_runs_with_gradient_spikes() -> None:
    """Measured on SYNTHETIC traces: 0 of 40 healthy seeds warn.

    The qualifier is load-bearing and was added after this test certified the wrong thing.
    On sixteen real full-scale fits the same detector fired on nine of fourteen healthy runs.
    This test pins the synthetic behaviour, which is all it ever established; the real-data
    result is in DivergenceWatch's docstring and D-040.

    The first design compared the latest gradient norm against its running median. On these
    same traces it warned on 40 of 40 healthy runs at multiples of 5, 10 and 20, because
    legitimate spikes reach twenty-five times typical. Comparing medians instead makes a
    single spike invisible and sustained elevation obvious.
    """
    warned = sum(
        first_warning(DivergenceWatch(), synthetic_trace("healthy", seed), norms=True) is not None
        for seed in range(12)
    )
    assert warned == 0


def test_watch_warns_before_the_loss_has_visibly_degraded() -> None:
    """Measured: median warning 117 steps before the loss doubles.

    Clipping masks the effect of a bad region on the loss for a while, which is exactly why
    the pre-clip gradient norm is the earlier signal.
    """
    leads = []
    for seed in range(12):
        loss, norms = synthetic_trace("diverge", seed)
        index = first_warning(DivergenceWatch(), (loss, norms), norms=True)
        assert index is not None, f"seed {seed} diverged without a warning"
        reference = float(np.median(loss[2900:3000]))
        doubled = next((i for i in range(3000, len(loss)) if loss[i] > 2 * reference), None)
        assert doubled is not None
        leads.append(doubled - index)
    assert float(np.median(leads)) > 50


def test_watch_reports_once() -> None:
    """A diverging run trips the test on every later step; twenty thousand copies is noise."""
    loss, norms = synthetic_trace("diverge", 0)
    watch = DivergenceWatch()
    messages = [
        m for m in (watch.observe(v, g) for v, g in zip(loss, norms, strict=True)) if m is not None
    ]
    assert len(messages) == 1


def test_loss_only_sensitivity_matches_the_derived_limit() -> None:
    """The half-window ratio is exp((window/2)/tau) and does not grow with the run.

    So the test detects growth with time constant below ``(window/2)/ln(multiple)`` and never
    detects anything slower, however long it runs. The default gives 247 steps; this pins
    both sides of that boundary so a change to the window cannot silently blind the test.
    """
    watch = DivergenceWatch()
    assert 240 < watch.slowest_detectable_growth < 255

    detected = first_warning(DivergenceWatch(), synthetic_trace("diverge", 0, tau=150), norms=False)
    assert detected is not None, "growth well inside the limit should be detected"

    missed = first_warning(DivergenceWatch(), synthetic_trace("diverge", 0, tau=400), norms=False)
    assert missed is None, "growth beyond the limit is not detectable by this test, by design"


def test_sweep_timer_reports_no_estimate_before_a_job_finishes() -> None:
    timer = SweepTimer(4)
    assert timer.remaining() == math.inf
    assert "no estimate yet" in timer.summary()
    timer.record(100.0)
    timer.record(120.0)
    # Median of finished jobs times the number left; flat cost across the sweep is measured,
    # not assumed. See docs/decisions.md D-032.
    assert timer.remaining() == pytest.approx(220.0)
    assert "2 of 4" in timer.summary()


def test_reporter_closes_even_when_the_body_raises(tmp_path) -> None:
    """A diverged run must still leave a complete record."""
    path = tmp_path / "run.jsonl"
    with pytest.raises(RuntimeError), ProgressReporter("job", 2, log_path=path) as reporter:
        reporter.tick(loss=1.0)
        raise RuntimeError("diverged")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[-1]["event"] == "end"


def test_recent_trace_numbers_steps_from_the_end_of_the_run() -> None:
    """The failure dump must say which step each row is, not just give an ordered list.

    Per-epoch records answer "which epoch did it turn in", and at 696 steps an epoch that is
    not an answer.
    """
    watch = DivergenceWatch()
    for step in range(1, 501):
        watch.observe(loss=1.0 / step, grad_norm=float(step))
    trace = watch.recent_trace(limit=10)
    assert len(trace) == 10
    assert [row["step"] for row in trace] == list(range(491, 501))
    assert trace[-1]["loss"] == pytest.approx(1.0 / 500)
    assert trace[-1]["grad_norm"] == pytest.approx(500.0)


def test_recent_trace_is_bounded_by_what_the_watch_retains() -> None:
    """Asking for more than is held returns what is held rather than padding.

    This bound is a real limitation: a precursor starting earlier than the retained window
    does not appear, and nothing in the record says so.
    """
    watch = DivergenceWatch()
    for _ in range(50):
        watch.observe(loss=0.5, grad_norm=1.0)
    assert len(watch.recent_trace(limit=1000)) == 50


def test_trace_rows_go_to_the_log_and_not_to_the_stream(tmp_path) -> None:
    """A dump is hundreds of rows; on stdout it would bury the message explaining it."""
    path = tmp_path / "run.jsonl"
    stream = io.StringIO()
    reporter = ProgressReporter("job", 1, stream=stream, log_path=path)
    reporter.note("loss became non-finite at step 900")
    for step in range(3):
        reporter.trace({"step": 900 + step, "loss": float("inf")})
    reporter.close()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert sum(r.get("event") == "trace" for r in records) == 3
    assert "non-finite" in stream.getvalue()
    assert "trace" not in stream.getvalue()


def test_estimate_is_correct_when_ticking_at_a_stride() -> None:
    """A caller may tick every hundredth unit; the estimate must still be in units.

    The clustered bootstrap ticks once per hundred resamples because one tick per resample
    would dominate a loop whose body costs about 1.5 ms. Before this was handled, the reporter
    multiplied a per-hundred duration by the number of remaining *resamples* and predicted
    1m11s for work that finished in one second -- an estimate wrong by two orders of magnitude
    at exactly the point a reader decides whether to wait.
    """
    reporter = ProgressReporter("strided", 1000)
    ticks = []
    for index in range(100, 1001, 100):
        reporter._last -= 0.1  # a tenth of a second per hundred units
        ticks.append(reporter.tick(index))
    # Nine hundred units remain after the first tick at a millisecond each, so about 0.9 s.
    assert ticks[0].remaining < 2.0, f"estimated {ticks[0].remaining:.1f}s for about 0.9s of work"
    assert ticks[-1].remaining == pytest.approx(0.0, abs=1e-6)


def test_run_log_captures_progress_and_warnings_in_one_file(tmp_path) -> None:
    """A run that warns and then behaves oddly is read most easily as one interleaved file."""
    import warnings

    from hsbcfraud.progress import run_log

    with run_log("unit", directory=tmp_path, stream=None) as run:
        reporter = run.reporter("phase", 2)
        reporter.tick(loss=0.5)
        reporter.tick(loss=0.4)
        reporter.close()
        warnings.warn("something to record", RuntimeWarning, stacklevel=1)

    text = run.path.read_text(encoding="utf-8")
    assert "phase 1/2" in text and "phase 2/2" in text
    assert "loss 0.5" in text
    assert "something to record" in text
    # The machine-readable record sits beside the human one, not inside it.
    assert (tmp_path / "unit-phase.jsonl").exists()


def test_every_long_fit_script_supplies_an_evaluation_probe() -> None:
    """A training loop that reports only loss cannot see the failure mode it is watched for.

    Two silent defects in this classifier produced steadily falling loss with a test AUC of
    exactly 0.5000. ``fit_mps`` therefore takes an ``evaluate`` callback, and its docstring
    says why -- but ``scripts/run_seed_sweep.py`` was written without one and launched a
    thirteen-hour job that could not have noticed a degenerate fit until it ended. The
    omission was invisible because the log still looked healthy: elapsed, remaining, loss,
    learning rate and gradient norms all present.

    This pins the wiring rather than the intent. Any script that calls ``fit_mps`` for a job
    long enough to matter must pass ``evaluate``.
    """
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    offenders = []
    for path in sorted((repo / "scripts").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for call in re.finditer(r"fit_mps\(", source):
            # Take the balanced argument list following the call.
            depth, index = 0, call.end() - 1
            while index < len(source):
                depth += source[index] == "("
                depth -= source[index] == ")"
                if depth == 0:
                    break
                index += 1
            arguments = source[call.end() : index]
            if "evaluate=" not in arguments:
                line = source[: call.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(repo)}:{line}")
    assert not offenders, (
        "these fit_mps calls report loss without a ranking metric, so a fit that stops "
        f"depending on its input would look healthy: {offenders}"
    )
