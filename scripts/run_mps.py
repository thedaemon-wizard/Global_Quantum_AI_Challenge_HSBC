#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""E10 -- the matrix-product-state arm, in the band and at full scale.

Writes ``results/tables/mps_band.csv`` (the in-band re-ranking comparison, which is the body
claim) and ``results/tables/mps_full.csv`` (a standalone classifier on all features, which is
the appendix claim).

Two experiments, because they answer different questions.

**In the band.**  The same band, the same features and the same evaluation as the quantum
kernel arm, so the three candidates -- MPS, quantum kernel, gradient boosting -- are directly
comparable.  The quantum kernel was rejected by its a-priori screens, so the honest question
is what a quantum-inspired model does in the place the kernel could not be used.

**At full scale.**  All features, no band, against the tuned gradient-boosted baseline. This
comparison appears to be absent from the 2025-2026 literature for tabular fraud data, and it
is only affordable because an MPS has no qubit ceiling: 439 sites is a long chain, not an
intractable state space.

Both sweep the bond dimension rather than tuning it, so the capacity dependence is measured.

    .venv/bin/python scripts/run_mps.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from hsbcfraud.config import load_config
from hsbcfraud.data.ieee_cis import IEEE_CIS_ZIP, load_ieee_cis
from hsbcfraud.quantum.mps import MPSConfig, fit_mps

REPO = Path(__file__).resolve().parents[1]


def prepare(frame: pd.DataFrame, rows: np.ndarray, columns: list[str], scaler: MinMaxScaler | None):
    """Numeric matrix scaled to [0, 1], which is the domain the local feature map needs.

    The scaler is fitted once on the training block and reused everywhere else.  Fitting it
    per block would leak the deployment distribution into the encoding.
    """
    numeric = frame.iloc[rows][columns]
    filled = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0).to_numpy(dtype=np.float32)
    if scaler is None:
        scaler = MinMaxScaler(clip=True).fit(filled)
    return scaler.transform(filled).astype(np.float32), scaler


def evaluate(y_true, scores) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--zip", type=Path, default=REPO / "datasets" / IEEE_CIS_ZIP)
    parser.add_argument("--runs", type=Path, default=REPO / "results" / "runs")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "tables")
    parser.add_argument("--arm", default="temporal")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bonds", type=int, nargs="*", default=[4, 8, 16, 32])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--band-features", type=int, default=8)
    parser.add_argument("--skip-full", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed = args.seed or cfg.split.seeds[0]
    scores = pd.read_parquet(args.runs / f"scores_{args.arm}_{seed}.parquet")
    loaded = load_ieee_cis(args.zip, None, with_identity=True)
    frame = loaded.frame
    for column in frame.columns:
        if frame[column].dtype == "object" or str(frame[column].dtype) == "str":
            frame[column] = frame[column].astype("category").cat.codes.astype("int32")

    all_numeric = [
        c
        for c in frame.select_dtypes(include=[np.number]).columns
        if c not in {"isFraud", "day", "TransactionDT", "TransactionID"}
    ]
    train_rows = scores[scores["block"] == "train"]["row"].to_numpy()
    y = frame["isFraud"].to_numpy()

    # ---------------------------------------------------------------- in-band comparison
    # The band is the region between the decline threshold and the approve threshold, fixed
    # on D_band exactly as scripts/run_conformal.py fixes it.
    band_block = scores[scores["block"] == "band"]
    s_band = band_block["score"].to_numpy()
    hi = float(np.quantile(s_band, 1.0 - cfg.decline_rate_budget))
    lo = float(np.quantile(s_band, 1.0 - cfg.decline_rate_budget - cfg.band.traffic_budget))

    from sklearn.feature_selection import mutual_info_classif

    rng = np.random.default_rng(seed)
    probe = rng.choice(len(train_rows), size=min(20_000, len(train_rows)), replace=False)
    probe_x, _ = prepare(frame, train_rows[probe], all_numeric, None)
    mi = mutual_info_classif(probe_x, y[train_rows[probe]], random_state=seed)
    band_columns = [all_numeric[i] for i in np.argsort(mi)[::-1][: args.band_features]]

    rows: list[dict] = []
    band_train = scores[(scores["block"] == "band")]
    band_train_rows = band_train[
        (band_train["score"] >= lo) & (band_train["score"] < hi)
    ]["row"].to_numpy()
    band_eval = scores[scores["block"] == "cal"]
    band_eval_rows = band_eval[
        (band_eval["score"] >= lo) & (band_eval["score"] < hi)
    ]["row"].to_numpy()

    x_bt, scaler = prepare(frame, band_train_rows, band_columns, None)
    x_be, _ = prepare(frame, band_eval_rows, band_columns, scaler)
    y_bt, y_be = y[band_train_rows], y[band_eval_rows]
    print(
        f"in-band: fit on {len(x_bt):,} band rows from D_band ({int(y_bt.sum())} fraud), "
        f"evaluate on {len(x_be):,} from D_cal ({int(y_be.sum())} fraud), "
        f"{args.band_features} features"
    )

    for chi in args.bonds:
        start = time.perf_counter()
        model, history = fit_mps(
            x_bt,
            y_bt,
            MPSConfig(bond_dimension=chi, epochs=args.epochs, seed=seed, device="cuda"),
        )
        elapsed = time.perf_counter() - start
        metrics = evaluate(y_be, model.predict_proba(x_be))
        rows.append(
            {
                "experiment": "band",
                "model": "mps",
                "bond_dimension": chi,
                "n_features": args.band_features,
                "n_train": len(x_bt),
                "n_eval": len(x_be),
                "seed": seed,
                "fit_seconds": elapsed,
                "final_loss": history[-1],
                **metrics,
            }
        )
        print(
            f"  chi={chi:3d}  {elapsed:6.1f}s  loss {history[0]:.4f} -> {history[-1]:.4f}  "
            f"AUC {metrics['roc_auc']:.4f}  AP {metrics['average_precision']:.4f}"
        )

    # The in-band GBDT control, on the identical features and rows.
    import xgboost as xgb

    control = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05, tree_method="hist",
        device="cuda", eval_metric="aucpr", random_state=seed, verbosity=0,
    )
    start = time.perf_counter()
    control.fit(x_bt, y_bt)
    elapsed = time.perf_counter() - start
    metrics = evaluate(y_be, control.predict_proba(x_be)[:, 1])
    rows.append(
        {
            "experiment": "band", "model": "xgboost", "bond_dimension": np.nan,
            "n_features": args.band_features, "n_train": len(x_bt), "n_eval": len(x_be),
            "seed": seed, "fit_seconds": elapsed, "final_loss": np.nan, **metrics,
        }
    )
    print(
        f"  xgboost   {elapsed:6.1f}s  AUC {metrics['roc_auc']:.4f}  "
        f"AP {metrics['average_precision']:.4f}   (same rows, same features)"
    )

    pd.DataFrame([r for r in rows if r["experiment"] == "band"]).to_csv(
        args.out / "mps_band.csv", index=False
    )

    # ------------------------------------------------------------------ full-scale arm
    if not args.skip_full:
        test_rows = scores[scores["block"] == "test"]["row"].to_numpy()
        x_tr, scaler_full = prepare(frame, train_rows, all_numeric, None)
        x_te, _ = prepare(frame, test_rows, all_numeric, scaler_full)
        print(
            f"\nfull scale: {len(x_tr):,} train rows x {len(all_numeric)} features, "
            f"evaluate on {len(x_te):,} test rows"
        )
        full_rows = []
        for chi in args.bonds:
            start = time.perf_counter()
            model, history = fit_mps(
                x_tr,
                y[train_rows],
                MPSConfig(bond_dimension=chi, epochs=args.epochs, seed=seed, device="cuda"),
            )
            elapsed = time.perf_counter() - start
            metrics = evaluate(y[test_rows], model.predict_proba(x_te))
            full_rows.append(
                {
                    "experiment": "full", "model": "mps", "bond_dimension": chi,
                    "n_features": len(all_numeric), "n_train": len(x_tr), "n_eval": len(x_te),
                    "seed": seed, "fit_seconds": elapsed, "final_loss": history[-1], **metrics,
                }
            )
            print(
                f"  chi={chi:3d}  {elapsed:6.1f}s  loss {history[0]:.4f} -> {history[-1]:.4f}  "
                f"AUC {metrics['roc_auc']:.4f}  AP {metrics['average_precision']:.4f}"
            )
        pd.DataFrame(full_rows).to_csv(args.out / "mps_full.csv", index=False)
        print(
            "\n  The tuned gradient-boosted baseline on the same split reaches AUC 0.8837-0.8884 "
            "and AP 0.5055-0.5114 (results/tables/baselines.csv)."
        )

    print(f"\nWrote {args.out / 'mps_band.csv'}" + ("" if args.skip_full else f" and {args.out / 'mps_full.csv'}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
