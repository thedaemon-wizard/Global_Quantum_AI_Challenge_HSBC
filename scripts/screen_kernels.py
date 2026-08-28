#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""E8 -- apply the a-priori screens to real band transactions, before spending compute.

Writes ``results/tables/screens.csv``: one row per candidate feature map, with both screen
axes, Huang's geometric difference, and the pass/reject decision.

The search is deliberately wide.  A narrow search that rejects everything invites the reply
that a better encoding was not tried, so the grid spans three encodings, four qubit counts,
two entanglement settings and five bandwidths, and every combination is reported whether it
passes or not.  The pre-registration commits to running the kernel arm only on maps that
pass, and to reporting a rejection as the result if none does.

Features are selected on ``D_band`` alone and scaled by a min-max fitted on ``D_band`` alone.
Fitting either on the calibration or test block would leak.

    .venv/bin/python scripts/screen_kernels.py
"""

from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from hsbcfraud.config import load_config
from hsbcfraud.data.ieee_cis import IEEE_CIS_ZIP, load_ieee_cis
from hsbcfraud.data.splits import temporal_blocks
from hsbcfraud.quantum.featuremaps import build_feature_map, scale_features
from hsbcfraud.quantum.kernel import fidelity_gram
from hsbcfraud.quantum.screens import screen_kernel

REPO = Path(__file__).resolve().parents[1]


def band_features(
    frame: pd.DataFrame, scores: pd.DataFrame, cfg, n_features: int, n_samples: int, seed: int
):
    """Stratified sample of band transactions, reduced to ``n_features`` and scaled.

    Stratified because the challenge statement requires it: subsampling for quantum execution
    "must be performed using stratified sampling to preserve the fraud/non-fraud ratio of the
    original dataset", and the total number of samples used must be stated.

    Feature selection is mutual information against the label, computed on the band block
    only.  Encoding hundreds of features into a circuit is not practical, which the challenge
    statement also says, so the reduction is required rather than a convenience -- and the
    selected columns are reported so the reduction is reproducible.
    """
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import MinMaxScaler

    band = scores[scores["block"] == "band"]
    rows = band["row"].to_numpy()
    y = band["y"].to_numpy()

    numeric = frame.iloc[rows].select_dtypes(include=[np.number])
    numeric = numeric.drop(columns=[c for c in ("isFraud", "day", "TransactionDT", "TransactionID") if c in numeric])
    filled = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)

    rng = np.random.default_rng(seed)
    # Mutual information on a subsample: it is O(n log n) per feature and the ranking is
    # stable well below the full block.
    probe = rng.choice(len(filled), size=min(20_000, len(filled)), replace=False)
    mi = mutual_info_classif(filled.iloc[probe], y[probe], random_state=seed)
    chosen = list(filled.columns[np.argsort(mi)[::-1][:n_features]])

    fraud_idx = np.flatnonzero(y == 1)
    legit_idx = np.flatnonzero(y == 0)
    n_fraud = max(1, int(round(n_samples * len(fraud_idx) / len(y))))
    take = np.concatenate(
        [
            rng.choice(fraud_idx, size=min(n_fraud, len(fraud_idx)), replace=False),
            rng.choice(legit_idx, size=n_samples - n_fraud, replace=False),
        ]
    )
    rng.shuffle(take)

    x = MinMaxScaler().fit_transform(filled[chosen].to_numpy()[take])
    return x, y[take], chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--zip", type=Path, default=REPO / "datasets" / IEEE_CIS_ZIP)
    parser.add_argument("--runs", type=Path, default=REPO / "results" / "runs")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "tables")
    parser.add_argument("--arm", default="temporal")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--n-screen",
        type=int,
        default=300,
        help="band transactions per Gram matrix; the screen is a spectral property and is "
        "stable well below the full band, which keeps the wide search affordable",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed = args.seed or cfg.split.seeds[0]
    scores = pd.read_parquet(args.runs / f"scores_{args.arm}_{seed}.parquet")
    loaded = load_ieee_cis(args.zip, None, with_identity=True)
    frame = loaded.frame
    temporal_blocks(frame["day"], cfg.split)  # asserts the split still forms

    rows: list[dict] = []
    print(
        f"Screening on {args.n_screen} stratified band transactions. "
        f"Bands: {cfg.quantum.feature_maps} x {cfg.quantum.qubits} qubits x "
        f"{cfg.quantum.entanglement} x {len(cfg.quantum.bandwidths)} bandwidths\n"
    )
    print(
        f"{'map':12s} {'q':>2s} {'ent':>6s} {'bw':>7s} {'eff-rank':>9s} {'top-eig':>8s} "
        f"{'offdiag':>8s} {'RBF corr':>9s} {'geom diff':>10s}  verdict"
    )

    from sklearn.metrics.pairwise import rbf_kernel

    for n_qubits in cfg.quantum.qubits:
        n_feat = n_qubits
        x, y, chosen = band_features(frame, scores, cfg, n_feat, args.n_screen, seed)
        classical = rbf_kernel(x, gamma=1.0 / n_feat)
        for name, ent, bw in itertools.product(
            cfg.quantum.feature_maps, cfg.quantum.entanglement, cfg.quantum.bandwidths
        ):
            if name == "z" and ent != "none":
                continue  # the Z map has no entangling layer to vary
            circuit = build_feature_map(name, n_feat, reps=cfg.quantum.reps, entanglement=ent)
            angles = scale_features(x, bw)
            start = time.perf_counter()
            gram = fidelity_gram(circuit, angles, backend="statevector")
            elapsed = time.perf_counter() - start
            result = screen_kernel(
                gram,
                angles,
                name=name,
                n_qubits=circuit.num_qubits,
                bandwidth=bw,
                entanglement=ent,
                classical_gram=classical,
                effective_rank_min=cfg.quantum.screen_effective_rank_min,
                effective_rank_max=cfg.quantum.screen_effective_rank_max,
                rbf_correlation_max=cfg.quantum.screen_rbf_correlation_max,
            )
            rows.append(
                {
                    **asdict(result),
                    "n_features": n_feat,
                    "gram_seconds": elapsed,
                    "selected_features": ";".join(chosen),
                }
            )
            print(
                f"{name:12s} {circuit.num_qubits:2d} {ent:>6s} {bw:7.4f} "
                f"{result.effective_rank:9.4f} {result.top_eigenvalue:8.4f} "
                f"{result.off_diagonal_mean:8.5f} {result.rbf_correlation:9.4f} "
                f"{result.geometric_difference:10.3f}  {result.summary()}"
            )

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "screens.csv", index=False)

    passed = table[table["passes_conditioning"] & table["passes_distinctness"]]
    print(f"\n{len(passed)} of {len(table)} candidates pass both screens.")
    if passed.empty:
        cond = int(table["passes_conditioning"].sum())
        dist = int(table["passes_distinctness"].sum())
        print(
            f"  {cond} pass conditioning alone, {dist} pass distinctness alone, none pass both.\n"
            "  Under the pre-registration this is the reported result: the quantum kernel arm\n"
            "  is not run, and the rejection answers the challenge's secondary objective on\n"
            "  the conditions under which quantum approaches behave differently."
        )
    else:
        for _, row in passed.iterrows():
            print(
                f"  {row['name']} {row['n_qubits']}q ent={row['entanglement']} "
                f"bw={row['bandwidth']:.4f}"
            )
    print(f"\nWrote {args.out / 'screens.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
