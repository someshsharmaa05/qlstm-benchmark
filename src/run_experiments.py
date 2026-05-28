"""
run_experiments.py
==================
Top-level driver. Reads seeds, sweeps all (model, asset, seed) triples,
writes per-seed predictions and a master raw_results.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .classical_models import build_classical, n_params
from .data_loader import ASSETS, load_all
from .metrics import (
    COST_BP, bootstrap_ci, directional_accuracy, long_short_trade,
    mae, mape, rmse, wilcoxon_signed_rank,
)
from .qlstm import QLSTM
from .train import TrainConfig, predict, train_one


def read_seeds(path: str = "seeds.txt") -> list[int]:
    return [int(x.strip()) for x in open(path) if x.strip()]


def model_fn_factory(name: str, seq_len: int = 30, in_features: int = 4,
                     n_vqc_blocks: int = 6, noise_p: float = 0.0):
    name = name.lower()
    if name == "qlstm":
        return lambda: QLSTM(in_features=in_features,
                              n_vqc_blocks=n_vqc_blocks,
                              noise_p=noise_p)
    return lambda: build_classical(name, seq_len=seq_len, in_features=in_features) \
        if name in {"patchtst", "itransformer", "nbeats"} \
        else build_classical(name, in_features=in_features)


def evaluate(model, split, device):
    yp = predict(model, split.X_test, device=device)
    yt = split.y_test
    return {
        "rmse": rmse(yt, yp),
        "mae":  mae(yt, yp),
        "mape": mape(yt, yp),
        "da":   directional_accuracy(yt, yp),
        "preds": yp,
    }


def run_main_benchmark(splits, seeds, models, device, cfg):
    rows = []
    for asset, sp in splits.items():
        for model_name in models:
            print(f"\n>>> {model_name:13s} {asset}")
            for seed in tqdm(seeds, leave=False):
                t0 = time.time()
                mfn = model_fn_factory(model_name)
                model = train_one(mfn, sp.X_train, sp.y_train,
                                  sp.X_val, sp.y_val, seed=seed, cfg=cfg)
                ev = evaluate(model, sp, device)
                rows.append({
                    "model": model_name, "asset": asset, "seed": seed,
                    "rmse": ev["rmse"], "mae": ev["mae"], "mape": ev["mape"],
                    "da": ev["da"], "params": n_params(model),
                    "train_sec": time.time() - t0,
                })
                # Save per-seed predictions for trading evaluation
                npz_path = Path("results/raw/preds") / f"{model_name}_{asset}_seed{seed}.npz"
                npz_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(npz_path, preds=ev["preds"], targets=sp.y_test,
                         prices=sp.test_prices)
    return pd.DataFrame(rows)


def run_walk_forward(splits, seeds, models, device, cfg, n_folds=5, win=60):
    """Expanding-window walk-forward on a subset of assets/models."""
    rows = []
    target_assets = ["NVDA", "AAPL", "BTC-USD", "ETH-USD"]
    for asset in target_assets:
        sp = splits[asset]
        full_X = np.concatenate([sp.X_train, sp.X_val, sp.X_test], axis=0)
        full_y = np.concatenate([sp.y_train, sp.y_val, sp.y_test], axis=0)
        # Anchor: 60% of full
        n = len(full_X)
        anchor = int(0.60 * n)
        for fold in range(n_folds):
            tr_end = anchor + fold * win
            te_end = tr_end + win
            if te_end > n: break
            X_tr_f = full_X[:tr_end]; y_tr_f = full_y[:tr_end]
            # carve a small inner validation set from the tail of training
            v_cut = max(int(0.85 * len(X_tr_f)), len(X_tr_f) - 60)
            X_tr_in, y_tr_in = X_tr_f[:v_cut], y_tr_f[:v_cut]
            X_va_in, y_va_in = X_tr_f[v_cut:], y_tr_f[v_cut:]
            X_te_f, y_te_f = full_X[tr_end:te_end], full_y[tr_end:te_end]

            for model_name in models:
                seed_rmses = []
                for seed in seeds:
                    mfn = model_fn_factory(model_name)
                    model = train_one(mfn, X_tr_in, y_tr_in,
                                       X_va_in, y_va_in, seed=seed, cfg=cfg)
                    yp = predict(model, X_te_f, device=device)
                    seed_rmses.append(rmse(y_te_f, yp))
                rows.append({"asset": asset, "model": model_name,
                             "fold": fold + 1,
                             "rmse_mean": float(np.mean(seed_rmses)),
                             "rmse_std": float(np.std(seed_rmses))})
    return pd.DataFrame(rows)


def run_noise_sweep(splits, seeds, device, cfg):
    rows = []
    for asset in ["BTC-USD", "NVDA"]:
        sp = splits[asset]
        for p in [0.000, 0.001, 0.005, 0.010]:
            seed_rmses = []
            for seed in seeds:
                mfn = lambda p=p: QLSTM(n_vqc_blocks=6, noise_p=p)
                m = train_one(mfn, sp.X_train, sp.y_train,
                              sp.X_val, sp.y_val, seed=seed, cfg=cfg)
                yp = predict(m, sp.X_test, device=device)
                seed_rmses.append(rmse(sp.y_test, yp))
            rows.append({"asset": asset, "noise_p": p,
                         "rmse_mean": float(np.mean(seed_rmses)),
                         "rmse_std":  float(np.std(seed_rmses))})
    return pd.DataFrame(rows)


def run_depth_ablation(splits, seeds, device, cfg):
    rows = []
    for asset in ["BTC-USD", "NVDA"]:
        sp = splits[asset]
        for n_blocks in [0, 2, 4, 6]:
            seed_rmses = []
            for seed in seeds:
                mfn = lambda n=n_blocks: QLSTM(n_vqc_blocks=n, noise_p=0.0)
                m = train_one(mfn, sp.X_train, sp.y_train,
                              sp.X_val, sp.y_val, seed=seed, cfg=cfg)
                yp = predict(m, sp.X_test, device=device)
                seed_rmses.append(rmse(sp.y_test, yp))
            rows.append({"asset": asset, "n_vqc_blocks": n_blocks,
                         "rmse_mean": float(np.mean(seed_rmses)),
                         "rmse_std":  float(np.std(seed_rmses))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="5 seeds, 3 assets, classical-only — for smoke testing")
    ap.add_argument("--classical-only", action="store_true",
                    help="skip QLSTM and all quantum sweeps")
    ap.add_argument("--seeds-file", default="seeds.txt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path("results/tables"); out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path("results/raw"); raw_dir.mkdir(parents=True, exist_ok=True)

    print("== loading data ==")
    splits = load_all()

    seeds = read_seeds(args.seeds_file)
    if args.quick: seeds = seeds[:5]
    print(f"using {len(seeds)} seeds: {seeds}")

    models = ["lstm", "bilstm", "patchtst", "nbeats", "itransformer"]
    if not args.classical_only: models.append("qlstm")
    if args.quick:
        models = [m for m in models if m in {"lstm", "patchtst", "qlstm"}]

    cfg = TrainConfig(device=args.device,
                      epochs=60 if args.quick else 100,
                      patience=10 if args.quick else 15)

    print(f"== main benchmark: {len(splits)} assets x {len(models)} models x {len(seeds)} seeds ==")
    main_df = run_main_benchmark(splits, seeds, models, args.device, cfg)
    main_df.to_csv(raw_dir / "main_results.csv", index=False)

    if not args.classical_only and not args.quick:
        print("== walk-forward ==")
        wf_df = run_walk_forward(splits, seeds, models, args.device, cfg)
        wf_df.to_csv(raw_dir / "walk_forward.csv", index=False)

        print("== noise sweep ==")
        noise_df = run_noise_sweep(splits, seeds, args.device, cfg)
        noise_df.to_csv(raw_dir / "noise.csv", index=False)

        print("== depth ablation ==")
        depth_df = run_depth_ablation(splits, seeds, args.device, cfg)
        depth_df.to_csv(raw_dir / "depth.csv", index=False)

    print("== done. now run `python -m src.aggregate` ==")


if __name__ == "__main__":
    main()
