"""
aggregate.py
============
Read raw results, compute the 7 result tables of the manuscript, and
emit them as CSVs.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import ASSETS, load_all
from .metrics import (
    COST_BP, bootstrap_ci, directional_accuracy, long_short_trade,
    rmse, wilcoxon_signed_rank,
)


MODELS = ["lstm", "bilstm", "patchtst", "nbeats", "itransformer", "qlstm"]
MODEL_LABELS = {
    "lstm": "LSTM", "bilstm": "BiLSTM", "patchtst": "PatchTST",
    "nbeats": "N-BEATS", "itransformer": "iTransformer", "qlstm": "QLSTM",
}
EQUITY = ["NVDA", "AAPL", "JPM"]
FX     = ["EURUSD=X", "GBPUSD=X"]
CRYPTO = ["BTC-USD", "ETH-USD"]
ALL    = EQUITY + FX + CRYPTO


def _scale_factor(asset: str) -> tuple[str, float]:
    if asset in EQUITY: return ("×10⁻³", 1e3)
    if asset in FX:     return ("×10⁻⁴", 1e4)
    return ("×10⁻²", 1e2)


def make_table2(main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        row = {"Model": MODEL_LABELS[m]}
        for a in ALL:
            sub = main[(main["model"] == m) & (main["asset"] == a)]["rmse"].values
            if len(sub) == 0:
                row[a] = "—"; continue
            unit, scale = _scale_factor(a)
            row[a] = f"{sub.mean() * scale:.2f} ± {sub.std() * scale:.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def make_table3(main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        row = {"Model": MODEL_LABELS[m]}
        for a in ALL:
            sub = main[(main["model"] == m) & (main["asset"] == a)]["da"].values
            row[a] = f"{sub.mean():.1f}" if len(sub) else "—"
        rows.append(row)
    return pd.DataFrame(rows)


def make_table4(main: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon p-values: QLSTM vs each baseline per asset."""
    rows = []
    for a in ALL:
        ql = main[(main["model"] == "qlstm") & (main["asset"] == a)].sort_values("seed")
        if ql.empty: continue
        row = {"Asset": a}
        wins = 0
        for m in ["lstm", "bilstm", "patchtst", "nbeats", "itransformer"]:
            base = main[(main["model"] == m) & (main["asset"] == a)].sort_values("seed")
            if base.empty: row[f"vs {MODEL_LABELS[m]}"] = "—"; continue
            common = sorted(set(ql["seed"]) & set(base["seed"]))
            if len(common) < 3:
                row[f"vs {MODEL_LABELS[m]}"] = "n/a"; continue
            q = ql.set_index("seed").loc[common]["rmse"].values
            b = base.set_index("seed").loc[common]["rmse"].values
            p = wilcoxon_signed_rank(q, b)
            star = "*" if p < 0.05 and q.mean() < b.mean() else ""
            row[f"vs {MODEL_LABELS[m]}"] = f"{p:.3f}{star}"
            if star: wins += 1
        if wins == 5:                                 outcome = "QLSTM wins all"
        elif wins >= 1:                               outcome = f"QLSTM wins {wins}"
        else:                                          outcome = "no significant win"
        row["Outcome"] = outcome
        rows.append(row)
    return pd.DataFrame(rows)


def make_table5(wf: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward mean RMSE per model x asset."""
    if wf is None or wf.empty:
        return pd.DataFrame()
    rows = []
    assets = sorted(wf["asset"].unique())
    for m in MODELS:
        row = {"Model": MODEL_LABELS[m]}
        for a in assets:
            sub = wf[(wf["model"] == m) & (wf["asset"] == a)]
            if sub.empty: row[a] = "—"; continue
            unit, scale = _scale_factor(a)
            mean_rmse = sub["rmse_mean"].mean() * scale
            row[a] = f"{mean_rmse:.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def make_table6(splits, raw_dir: Path) -> pd.DataFrame:
    """Trading utility with transaction costs."""
    target = ["NVDA", "BTC-USD", "ETH-USD"]
    asset_to_cost = {
        "NVDA": ("Equity", 252),
        "BTC-USD": ("Cryptocurrency", 365),
        "ETH-USD": ("Cryptocurrency", 365),
    }
    rows = []
    for m in MODELS:
        row = {"Model": MODEL_LABELS[m]}
        for a in target:
            cls, td = asset_to_cost[a]
            cost_bp = COST_BP[cls]
            files = sorted((raw_dir / "preds").glob(f"{m}_{a}_seed*.npz"))
            if not files:
                row[f"{a} Ret%"] = "—"; row[f"{a} SR"] = "—"; row[f"{a} MDD%"] = "—"
                continue
            rets, srs, mdds = [], [], []
            for f in files:
                z = np.load(f)
                preds = z["preds"]
                future = z["targets"]
                tr = long_short_trade(returns=future, signal_pred=preds,
                                       cost_bp=cost_bp, trading_days=td)
                rets.append(tr["annual_return_pct"]); srs.append(tr["sharpe"]); mdds.append(tr["mdd_pct"])
            row[f"{a} Ret%"] = f"{np.mean(rets):.1f}"
            row[f"{a} SR"]   = f"{np.mean(srs):.2f}"
            row[f"{a} MDD%"] = f"{np.mean(mdds):.1f}"
        rows.append(row)
    return pd.DataFrame(rows)


def make_table7(noise: pd.DataFrame) -> pd.DataFrame:
    if noise is None or noise.empty: return pd.DataFrame()
    rows = []
    # baseline (p=0) for each asset
    base = {a: noise[(noise["asset"] == a) & (noise["noise_p"] == 0.000)]["rmse_mean"].values[0]
            for a in noise["asset"].unique()}
    for p, label in [(0.000, "0.000 (noiseless)"),
                     (0.001, "0.001 (optimistic NISQ)"),
                     (0.005, "0.005 (realistic NISQ)"),
                     (0.010, "0.010 (pessimistic NISQ)")]:
        row = {"Noise level p": label}
        for a in ["BTC-USD", "NVDA"]:
            sub = noise[(noise["asset"] == a) & (noise["noise_p"] == p)]
            if sub.empty: row[f"{a} RMSE"] = "—"; row[f"{a} delta"] = "—"; continue
            unit, scale = _scale_factor(a)
            v = float(sub["rmse_mean"].iloc[0]) * scale
            row[f"{a} RMSE {unit}"] = f"{v:.2f}"
            if p == 0.0:
                row[f"{a} delta"] = "—"
            else:
                d = (sub["rmse_mean"].iloc[0] - base[a]) / base[a] * 100
                row[f"{a} delta"] = f"+{d:.1f}%"
        rows.append(row)
    return pd.DataFrame(rows)


def make_table8(depth: pd.DataFrame) -> pd.DataFrame:
    if depth is None or depth.empty: return pd.DataFrame()
    rows = []
    # baseline = max blocks
    base = {a: depth[(depth["asset"] == a) & (depth["n_vqc_blocks"] == depth["n_vqc_blocks"].max())]["rmse_mean"].values[0]
            for a in depth["asset"].unique()}
    for k in sorted(depth["n_vqc_blocks"].unique()):
        if k == 0: lbl = "0  (classical LSTM)"
        elif k == depth["n_vqc_blocks"].max(): lbl = f"{k}  (full QLSTM)"
        else: lbl = str(k)
        row = {"VQC blocks": lbl}
        for a in ["BTC-USD", "NVDA"]:
            sub = depth[(depth["asset"] == a) & (depth["n_vqc_blocks"] == k)]
            if sub.empty: row[f"{a} RMSE"] = "—"; row[f"{a} vs full"] = "—"; continue
            unit, scale = _scale_factor(a)
            v = float(sub["rmse_mean"].iloc[0]) * scale
            row[f"{a} RMSE {unit}"] = f"{v:.2f}"
            if k == depth["n_vqc_blocks"].max():
                row[f"{a} vs full"] = "—"
            else:
                d = (sub["rmse_mean"].iloc[0] - base[a]) / base[a] * 100
                row[f"{a} vs full"] = f"+{d:.1f}%"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    raw_dir = Path("results/raw")
    out_dir = Path("results/tables"); out_dir.mkdir(parents=True, exist_ok=True)

    main_df = pd.read_csv(raw_dir / "main_results.csv")
    wf_df    = pd.read_csv(raw_dir / "walk_forward.csv") if (raw_dir / "walk_forward.csv").exists() else None
    noise_df = pd.read_csv(raw_dir / "noise.csv")        if (raw_dir / "noise.csv").exists() else None
    depth_df = pd.read_csv(raw_dir / "depth.csv")        if (raw_dir / "depth.csv").exists() else None

    try:
        splits = load_all()
    except Exception as e:
        print(f"  (offline mode — skipping data load: {e})")
        splits = None

    make_table2(main_df).to_csv(out_dir / "table2_test_rmse.csv", index=False)
    make_table3(main_df).to_csv(out_dir / "table3_directional_accuracy.csv", index=False)
    make_table4(main_df).to_csv(out_dir / "table4_wilcoxon.csv", index=False)
    if wf_df is not None:
        make_table5(wf_df).to_csv(out_dir / "table5_walk_forward.csv", index=False)
    make_table6(splits, raw_dir).to_csv(out_dir / "table6_trading.csv", index=False)
    if noise_df is not None:
        make_table7(noise_df).to_csv(out_dir / "table7_noise.csv", index=False)
    if depth_df is not None:
        make_table8(depth_df).to_csv(out_dir / "table8_vqc_depth.csv", index=False)

    print(f"\nWrote 7 tables to {out_dir}/")
    for f in sorted(out_dir.glob("table*.csv")):
        print(" ", f.name)


if __name__ == "__main__":
    main()
