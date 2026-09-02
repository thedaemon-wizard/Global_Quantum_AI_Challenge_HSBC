#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""E13 -- per-transaction inference latency, against the authorisation budget.

The challenge statement puts the whole authorisation flow at 100 to 300 ms and reports about
130 ms of network response time, "leaving limited additional budget for issuer-side decisioning
and fraud scoring", and it names *latency versus complexity* as one of five bottlenecks: "more
expressive models often cannot meet millisecond latency requirements".

This project's answer to that bottleneck is structural rather than algorithmic. The expensive
component runs on the abstention band, which is a few percent of traffic, so its cost is paid on
a few percent of authorisations. That argument was made from volume arithmetic and never timed.
This times it.

**The serving profile dominates the model, by three orders of magnitude.**

That is this script's main finding, and it was nearly reported the other way round. XGBoost
defaults its thread count to the core count. On a one-row payload the OpenMP barrier costs about
19 ms on this 20-thread machine while the prediction it synchronises costs about 0.05 ms -- so the
default configuration is roughly 380 times slower than a single thread, and the penalty is
independent of device, feature count and batch size. Measured on a fitted booster, one row:

    nthread=1   0.051 ms      nthread=4   0.051 ms      nthread=20   19.33 ms

An earlier version of this script swept only device and batch size, and produced a table where
CPU and GPU were indistinguishable, a 431-feature model cost the same as an 8-feature one, and
batch 1 cost two thirds of batch 1024. Every one of those anomalies was the same barrier. Taken
at face value it would have put the classical scorer at tens of milliseconds against a residual
budget of about 170 ms -- attributing a threading default to model complexity, in a section whose
whole argument is about model cost. The profile is therefore swept as a first-class dimension and
the all-core row is kept in the table as the documented trap, not as the recommendation.

A per-request scorer wants one thread: concurrency in an authorisation path comes from serving
many transactions at once, not from splitting one across cores.

**What is measured, and what a reader should not read into it.**

Measured: wall-clock time to score one transaction, on this machine, warm, in a Python process,
batching as the pipeline batches, through ``Booster.inplace_predict``. That is the right quantity
for comparing the three components against each other and against a millisecond budget.

Not measured, and not claimed: production latency. A deployed scorer runs behind a service
boundary with serialisation, feature retrieval and network hops that dominate everything here,
and it runs on the issuer's hardware rather than this workstation. The figures below bound the
*model* cost, which is the part a model choice controls.

Percentiles are reported rather than means because a tail is what breaks a timeout. p99 on a
few hundred repetitions is itself noisy, so the repetition count is reported alongside.

    .venv/bin/python scripts/measure_latency.py

Writes ``results/tables/latency.csv``.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import Booster, XGBClassifier

from hsbcfraud.config import load_config
from hsbcfraud.data.ieee_cis import IEEE_CIS_ZIP, load_ieee_cis
from hsbcfraud.features.band import band_edges, prepare, rows_in_band, select_band_features
from hsbcfraud.paths import display_path
from hsbcfraud.progress import ProgressReporter
from hsbcfraud.quantum.featuremaps import build_feature_map
from hsbcfraud.quantum.kernel import fidelity_gram

REPO = Path(__file__).resolve().parents[1]

# The authorisation budget the challenge statement states, and the network share it attributes
# to the card network.  Both are quoted to give the measurement a scale, not used as thresholds.
AUTHORISATION_BUDGET_MS = 300.0
NETWORK_RESPONSE_MS = 130.0

# Batch sizes to time at.  A scorer called once per authorisation and a scorer called on a
# batch are different machines: per-row cost falls by an order of magnitude with batching, and
# quoting only one of them would either flatter or libel the design.
BATCH_SIZES = (1, 32, 1024)

# Serving profiles: (label, device, thread count).  Training is on the GPU either way; this is
# the *serving* path, and the profile turns out to matter far more than the model.
#
# The all-core profile is here to be measured, not to be recommended.  XGBoost defaults its
# thread count to the core count, and on a one-row payload that barrier costs three orders of
# magnitude more than the prediction it synchronises -- so an unconfigured scorer reports tens
# of milliseconds per authorisation and invites the reader to blame the model.  A per-request
# scorer wants one thread, because concurrency comes from serving many authorisations at once
# rather than from splitting one across cores.
#
# ``None`` means "leave the library default", which is what an unconfigured deployment gets.
PROFILES = (
    ("1-thread CPU", "cpu", 1),
    ("all-core CPU", "cpu", None),
    ("GPU", "cuda", None),
)

# Support-set size for the kernel: the Gram row a single authorisation would need.
SUPPORT_ROWS = 64

# Repetitions per configuration.  Enough that p99 means something without the sweep dominating
# the run; the count is written to the table so a reader can judge the tail themselves.
REPETITIONS = 200
# Untimed passes before measurement, to let allocator and kernel caches settle.
WARMUP = 20

PERCENTILES = (50, 95, 99)

# Fitted identically to the pipeline's scorers, so the timing prices the deployed model rather
# than a stand-in.  Kept in one place because both components use it.
SCORER_PARAMS = dict(
    n_estimators=400, max_depth=6, learning_rate=0.05, tree_method="hist",
    eval_metric="aucpr", subsample=0.8, colsample_bytree=0.8, verbosity=0,
)


def fit_scorer(x: np.ndarray, y: np.ndarray, seed: int) -> XGBClassifier:
    """Fit on the GPU, as the pipeline does.  Serving device is chosen later."""
    model = XGBClassifier(device="cuda", random_state=seed, **SCORER_PARAMS)
    model.fit(x, y)
    return model


def serve_as(model: XGBClassifier, device: str, threads: int | None) -> Booster:
    """A copy of the fitted booster configured for one serving profile.

    A copy rather than a mutation: the profiles are timed in the same process and one must not
    disturb another.  The trees are identical across profiles, so every difference in the
    timings is the serving configuration and nothing else.

    ``inplace_predict`` rather than ``predict_proba``: the sklearn wrapper's validation and
    ``DMatrix`` construction are not part of a deployed scorer, and including them would price
    an artefact of the API instead of the model.
    """
    booster = copy.deepcopy(model).set_params(device=device).get_booster()
    if threads is not None:
        booster.set_param({"nthread": threads})
    return booster


def time_calls(fn, payloads: list[np.ndarray], reporter: ProgressReporter | None) -> np.ndarray:
    """Wall-clock nanoseconds per call, one entry per repetition.

    ``perf_counter_ns`` rather than ``perf_counter``: at batch size 1 a scorer call is tens of
    microseconds and float seconds lose resolution where it matters most.
    """
    for payload in payloads[:WARMUP]:
        fn(payload)
    timings = np.empty(len(payloads), dtype=np.float64)
    for index, payload in enumerate(payloads):
        start = time.perf_counter_ns()
        fn(payload)
        timings[index] = time.perf_counter_ns() - start
        if reporter is not None:
            reporter.tick(index + 1)
    return timings


def summarise(
    name: str, stage: str, profile: str, batch: int, timings: np.ndarray, n_features: int
) -> dict:
    """One row of the table: per-call and per-transaction cost at the given batch size.

    The last column counts FEATURES, not rows.  It was named ``rows_scored`` and the call site
    has always passed ``source.shape[1]``, so the name said one thing and the value another in
    a table whose whole purpose is timing per transaction.
    """
    per_call_ms = timings / 1e6
    return {
        "component": name,
        "stage": stage,
        "profile": profile,
        "batch_size": batch,
        "repetitions": len(timings),
        "n_features": n_features,
        **{
            f"per_call_p{p}_ms": float(np.percentile(per_call_ms, p)) for p in PERCENTILES
        },
        **{
            f"per_transaction_p{p}_ms": float(np.percentile(per_call_ms, p)) / batch
            for p in PERCENTILES
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--zip", type=Path, default=REPO / "datasets" / IEEE_CIS_ZIP)
    parser.add_argument("--runs", type=Path, default=REPO / "results" / "runs")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "tables")
    parser.add_argument("--arm", default="temporal")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--band-features", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed = args.seed or cfg.split.seeds[0]

    scores = pd.read_parquet(args.runs / f"scores_{args.arm}_{seed}.parquet")
    frame = load_ieee_cis(args.zip, None, with_identity=True).frame
    for column in frame.columns:
        if frame[column].dtype == "object" or str(frame[column].dtype) == "str":
            frame[column] = frame[column].astype("category").cat.codes.astype("int32")

    numeric = [
        c
        for c in frame.select_dtypes(include=[np.number]).columns
        if c not in {"isFraud", "day", "TransactionDT", "TransactionID"}
    ]
    labels = frame["isFraud"].to_numpy()
    train_rows = scores[scores["block"] == "train"]["row"].to_numpy()

    low, high = band_edges(
        scores[scores["block"] == "band"]["score"].to_numpy(),
        cfg.decline_rate_budget,
        cfg.band.traffic_budget,
    )
    band_columns = select_band_features(
        frame, train_rows, labels, numeric, k=args.band_features, seed=seed
    )
    band_rows = rows_in_band(scores, "band", low, high)

    # The full-traffic scorer, on every feature, as the pipeline fits it.
    x_full, _ = prepare(frame, train_rows, numeric, None)
    full_model = fit_scorer(x_full, labels[train_rows], seed)

    # The in-band re-scorer, on the eight selected features.
    x_band, _ = prepare(frame, band_rows, band_columns, None)
    band_model = fit_scorer(x_band, labels[band_rows], seed)

    # The quantum kernel, priced per in-band row against its support set.  The screens rejected
    # it, so this is what it *would* have cost, and it is the figure the band argument turns on.
    circuit = build_feature_map("zz", args.band_features, entanglement="linear")
    support = x_band[: min(SUPPORT_ROWS, len(x_band))]

    components = (
        ("classical scorer", "all traffic", full_model, x_full),
        ("in-band re-scorer", "abstention band", band_model, x_band),
    )

    rng = np.random.default_rng(seed)
    work: list[tuple[str, str, str, int, object, np.ndarray]] = []
    for name, stage, model, source in components:
        for profile, device, threads in PROFILES:
            scorer = serve_as(model, device, threads)
            for batch in BATCH_SIZES:
                work.append(
                    (name, stage, profile, batch,
                     lambda p, b=scorer: b.inplace_predict(p), source)
                )
    # The simulator is a CPU statevector contraction with no GPU variant to compare against, and
    # it is priced one authorisation at a time because that is how the band would call it.
    work.append(
        ("quantum kernel (screened out)", "abstention band", "1-thread CPU", 1,
         lambda p: fidelity_gram(circuit, p, y=support), x_band)
    )

    rows: list[dict] = []
    print(
        f"latency on {args.arm}/{seed}: {len(work)} configurations, "
        f"{args.repetitions} repetitions each, {WARMUP} warm-up calls\n"
    )
    for name, stage, profile, batch, fn, source in work:
        payloads = [
            source[rng.choice(len(source), size=batch, replace=False)]
            for _ in range(args.repetitions)
        ]
        label = f"{name} @ {profile}, batch {batch}"
        slug = f"{name.split()[0]}_{profile.replace(' ', '-')}_{batch}"
        with ProgressReporter(
            label,
            total=args.repetitions,
            stream=sys.stdout,
            log_path=args.runs / f"latency_{slug}.jsonl",
            context={"component": name, "profile": profile, "batch_size": batch,
                     "n_features": source.shape[1]},
            every=max(1, args.repetitions // 20),
            label_width=44,
        ) as reporter:
            timings = time_calls(fn, payloads, reporter)
        rows.append(summarise(name, stage, profile, batch, timings, source.shape[1]))

    frame_out = pd.DataFrame(rows)
    frame_out["authorisation_budget_ms"] = AUTHORISATION_BUDGET_MS
    frame_out["network_response_ms"] = NETWORK_RESPONSE_MS
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "latency.csv"
    frame_out.to_csv(target, index=False)

    print(
        f"\n{'component':>30s} {'profile':>13s} {'batch':>6s} {'per call p50':>13s} {'p99':>9s} "
        f"{'per txn p50':>12s} {'p99':>9s}"
    )
    for row in frame_out.itertuples():
        print(
            f"{row.component:>30s} {row.profile:>13s} {row.batch_size:6d} "
            f"{row.per_call_p50_ms:11.3f}ms {row.per_call_p99_ms:7.3f}ms "
            f"{row.per_transaction_p50_ms:10.4f}ms {row.per_transaction_p99_ms:7.4f}ms"
        )
    residual = AUTHORISATION_BUDGET_MS - NETWORK_RESPONSE_MS
    print(
        f"\n  Authorisation budget {AUTHORISATION_BUDGET_MS:.0f} ms, of which about "
        f"{NETWORK_RESPONSE_MS:.0f} ms is network, leaving about {residual:.0f} ms."
    )
    print(f"  Wrote {display_path(target)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
