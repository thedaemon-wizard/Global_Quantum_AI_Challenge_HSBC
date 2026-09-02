#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Full-scale tensor-network sweep with seed replication, so capacity can be read at all.

The committed sweep in ``mps_full.csv`` runs one seed per bond dimension. D-038 established
that this cannot support a statement about capacity: four seeds at a *fixed* chi = 16 spread
by 0.0188 in ROC AUC and 0.0724 in average precision, against 0.0186 and 0.0771 for the
committed spread *across* four bond dimensions.  The two are the same size, so the
per-configuration numbers were reading seed noise.  The proposal previously quoted the best of
those four draws, which is selection on a maximum from a distribution whose spread equals the
reported quantity.

This runs every bond dimension at four seeds, which is what turns the withdrawal into a
measurement: capacity dependence reported with intervals, or reported as absent with the
evidence to say so.

Contraction width is fixed at 1 -- the sequential fold, bit-identical to the implementation
that produced the committed tables.  D-038 also found that the reduction tree of D-036 does
not merely perturb the trajectory but destabilises it: at width 128 two of the same four seeds
failed to train at all.  The speedup is for exploration, not for numbers the study reports.

    .venv/bin/python scripts/run_seed_sweep.py

Roughly 52 minutes a job, 16 jobs.  Writes ``results/tables/mps_seed_sweep.csv`` incrementally
so a run that is interrupted still leaves usable rows, and logs to ``results/runs/`` where it
can be read while it goes.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from hsbcfraud.config import load_config
from hsbcfraud.data.ieee_cis import IEEE_CIS_ZIP, load_ieee_cis

# The sweep re-fits the same model as run_mps.py at other seeds, so it has to build its
# inputs the same way.  It carried a private copy of `prepare` that happened to agree; a copy
# that agrees is the one that hides the day it stops agreeing, and these rows are reported as
# the seed-stability evidence for the tensor-network arm.
from hsbcfraud.features.band import prepare
from hsbcfraud.paths import display_path, require_run_artefact
from hsbcfraud.progress import SweepTimer, run_log
from hsbcfraud.quantum.mps import MPSConfig, fit_mps

REPO = Path(__file__).resolve().parents[1]

# Fixed at the sequential fold. See the module docstring: the reduction tree is faster but
# measurably worse-conditioned, and these rows are reported.
CONTRACTION_WIDTH = 1


def display(path: Path) -> str:
    """Repository-relative when possible, absolute otherwise.

    An `--out` outside the tree is legitimate for a smoke run, and a path formatter must not
    be the thing that fails after fourteen hours of compute.
    """
    return display_path(path)


def require_idle_gpu(*, allow_shared: bool = False) -> bool:
    """Refuse to start if anything else holds the device.  Return whether it was contended.

    Not fastidiousness.  A contended run reports a fit time that is wrong by a factor of
    three -- observed on this host when a test suite ran in the same window -- and
    ``fit_seconds`` is a reported column.

    ``allow_shared`` exists for one case: a diagnostic run whose reported quantity is the loss
    trajectory rather than the timing.  It does not weaken the guard, because it cannot be used
    silently -- the caller gets ``True`` back and every row of the output carries a
    ``gpu_contended`` column, so a timing produced under contention can never later be read as
    if it were clean.
    """
    listing = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    # `> 0`, not `> 1`. This runs before any CUDA context is created, so this process does not
    # appear in the listing; `> 1` tolerated exactly one foreign process, which is the case the
    # docstring above says the guard exists to stop.
    if len(listing) > 0:
        if not allow_shared:
            raise SystemExit(
                f"{len(listing)} processes hold the GPU ({listing}); fit_seconds would be wrong"
            )
        return True
    return False


def progress_probe(x_eval: np.ndarray, y_eval: np.ndarray, n_rows: int, seed: int):
    """A per-epoch ranking metric on a stratified subsample of the evaluation block.

    Not optional decoration.  ``fit_mps`` reports loss, learning rate and gradient norms, and
    a falling loss is not evidence of learning: two silent defects in this classifier produced
    steadily decreasing loss with a test AUC of exactly 0.5000, and only a ranking metric
    distinguishes "training" from "the contraction stopped depending on the input".  A sweep
    that runs for thirteen hours without one cannot notice a job degenerating until it ends.

    Stratified because average precision on a 3.4 per cent positive rate is unstable under
    simple random sampling, and drawn once so the number moves between epochs only when the
    model does.  A subsample rather than the full block because scoring all 115,534 rows every
    epoch costs about a ninth of the epoch, against roughly two per cent for this.
    """
    if len(np.unique(y_eval)) < 2:
        raise ValueError("the evaluation block has one class; a ranking metric is undefined")
    if n_rows <= 0 or n_rows >= len(y_eval):
        rows = np.arange(len(y_eval))
    else:
        rng = np.random.default_rng(seed)
        positive = np.flatnonzero(y_eval == 1)
        negative = np.flatnonzero(y_eval == 0)
        take_positive = max(1, round(n_rows * len(positive) / len(y_eval)))
        rows = np.concatenate(
            [
                rng.choice(positive, size=min(take_positive, len(positive)), replace=False),
                rng.choice(
                    negative,
                    size=min(max(1, n_rows - take_positive), len(negative)),
                    replace=False,
                ),
            ]
        )
    x_probe, y_probe = x_eval[rows], y_eval[rows]

    def probe(model) -> dict[str, float]:
        scored = model.predict_proba(x_probe)
        return {
            "auc": float(roc_auc_score(y_probe, scored)),
            "ap": float(average_precision_score(y_probe, scored)),
        }

    return probe



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--zip", type=Path, default=REPO / "datasets" / IEEE_CIS_ZIP)
    parser.add_argument("--runs", type=Path, default=REPO / "results" / "runs")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "tables")
    parser.add_argument("--arm", default="temporal")
    parser.add_argument("--bonds", type=int, nargs="*", default=[4, 8, 16, 32])
    parser.add_argument(
        "--seeds", type=int, nargs="*", default=[20260828, 20260829, 20260830, 20260831]
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help=(
            "run even when another process holds the device.  Every row is then stamped "
            "gpu_contended=True and its fit_seconds must not be quoted as a timing."
        ),
    )
    parser.add_argument(
        "--eval-rows",
        type=int,
        default=20_000,
        help="stratified rows for the per-epoch ranking metric; 0 uses the whole block",
    )
    args = parser.parse_args(argv)

    contended = require_idle_gpu(allow_shared=args.allow_shared_gpu)
    cfg = load_config(args.config)
    scores = pd.read_parquet(
        require_run_artefact(
            args.runs / f"scores_{args.arm}_{cfg.split.seeds[0]}.parquet",
            produced_by="baseline",
        )
    )
    loaded = load_ieee_cis(args.zip, None, with_identity=True)
    frame = loaded.frame
    for column in frame.columns:
        if frame[column].dtype == "object" or str(frame[column].dtype) == "str":
            frame[column] = frame[column].astype("category").cat.codes.astype("int32")

    columns = [
        c
        for c in frame.select_dtypes(include=[np.number]).columns
        if c not in {"isFraud", "day", "TransactionDT", "TransactionID"}
    ]
    labels = frame["isFraud"].to_numpy()
    train_rows = scores[scores["block"] == "train"]["row"].to_numpy()
    test_rows = scores[scores["block"] == "test"]["row"].to_numpy()

    x_train, scaler = prepare(frame, train_rows, columns, None)
    x_test, _ = prepare(frame, test_rows, columns, scaler)
    y_train, y_test = labels[train_rows], labels[test_rows]

    probe = progress_probe(x_test, y_test, args.eval_rows, cfg.split.seeds[0])
    target = args.out / "mps_seed_sweep.csv"
    rows: list[dict] = []
    jobs = [(chi, seed) for chi in args.bonds for seed in args.seeds]

    # Extrapolation across jobs is defensible here and only here: at 431 sites the per-step
    # cost barely moves with the bond dimension (D-032), so the median finished job predicts
    # the unfinished ones. A thirteen-hour run that cannot say how much is left is a run
    # nobody watches.
    sweep = SweepTimer(len(jobs))

    with run_log(f"mps_seed_sweep_{args.arm}", directory=args.runs) as run:
        run.info(
            f"{len(jobs)} jobs: {len(args.bonds)} bond dimensions x {len(args.seeds)} seeds, "
            f"{x_train.shape[0]:,} train rows x {x_train.shape[1]} sites, "
            f"contraction width {CONTRACTION_WIDTH}"
        )
        for index, (chi, seed) in enumerate(jobs, start=1):
            started = time.perf_counter()
            reporter = run.reporter(f"chi{chi}_seed{seed}", args.epochs, label_width=20)
            with reporter:
                model, history = fit_mps(
                    x_train,
                    y_train,
                    MPSConfig(
                        bond_dimension=chi,
                        epochs=args.epochs,
                        seed=seed,
                        device="cuda",
                        contraction_chunk=CONTRACTION_WIDTH,
                    ),
                    reporter=reporter,
                    evaluate=probe,
                )
            elapsed = time.perf_counter() - started
            predicted = model.predict_proba(x_test)
            rows.append(
                {
                    "experiment": "full",
                    "bond_dimension": chi,
                    "seed": seed,
                    "contraction_width": CONTRACTION_WIDTH,
                    "n_features": x_train.shape[1],
                    "n_train": x_train.shape[0],
                    "n_eval": x_test.shape[0],
                    "roc_auc": float(roc_auc_score(y_test, predicted)),
                    "average_precision": float(average_precision_score(y_test, predicted)),
                    "final_loss": history[-1],
                    # The trajectory was computed and thrown away, and two runs in the
                    # committed sweep sit at a final loss of about ln 2 -- the value a model
                    # that never left its initialisation returns.  `final_loss` alone cannot
                    # distinguish "converged to a poor optimum" from "never moved", and that
                    # distinction decides whether the row is evidence about capacity or about
                    # the optimiser.  Three scalars separate them.
                    "loss_initial": history[0],
                    "loss_min": min(history),
                    "epoch_of_min": int(min(range(len(history)), key=history.__getitem__)) + 1,
                    "fit_seconds": elapsed,
                    "gpu_contended": contended,
                }
            )
            # Written after every job.  A fourteen-hour run that is interrupted at hour ten
            # should leave ten hours of usable rows, not nothing.
            pd.DataFrame(rows).to_csv(target, index=False)
            sweep.record(elapsed)
            run.info(
                f"  job {index}/{len(jobs)}  chi={chi} seed={seed}: "
                f"AUC {rows[-1]['roc_auc']:.4f}  AP {rows[-1]['average_precision']:.4f}  "
                f"loss {history[-1]:.4f}  {elapsed:.0f}s"
            )
            run.info(f"  sweep: {sweep.summary()}")

    frame_out = pd.DataFrame(rows)
    print(f"\nWrote {display(target)} ({len(frame_out)} rows)")
    print("\nSpread within each bond dimension against the spread across them:")
    within = frame_out.groupby("bond_dimension")["average_precision"].agg(
        lambda s: float(s.max() - s.min())
    )
    across = float(
        frame_out.groupby("bond_dimension")["average_precision"].mean().pipe(
            lambda s: s.max() - s.min()
        )
    )
    for chi, spread in within.items():
        print(f"  chi={chi:3d}  seed spread {spread:.4f}")
    print(f"  across bond dimensions, seed-averaged: {across:.4f}")
    print(
        "  capacity is resolvable only if the across-chi spread exceeds the within-chi one"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
