"""
smoke_test.py
=============
DO NOT use this for the published paper.

This script synthesises plausible raw_results.csv / walk_forward.csv /
noise.csv / depth.csv files so that the user can verify the
aggregation -> figure -> manuscript-fill pipeline end-to-end in under
60 seconds, BEFORE committing to the 30-CPU-hour real run.

It is a pipeline rehearsal. The numbers it produces are intentionally
random samples drawn around values consistent with published QLSTM
literature; THEY ARE NOT EXPERIMENTAL RESULTS. The file
`results/SMOKE_TEST_DATA.txt` is created to mark the directory as
rehearsal data so it cannot be mistaken for real output.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = [42, 137, 271, 389, 521, 683, 829, 967,
         1117, 1259, 1399, 1543, 1693, 1847, 2003]

ASSETS = ["NVDA", "AAPL", "JPM", "EURUSD=X", "GBPUSD=X", "BTC-USD", "ETH-USD"]
MODELS = ["lstm", "bilstm", "patchtst", "nbeats", "itransformer", "qlstm"]

# Hypothesised mean RMSE in unscaled (normalised log-return) space.
# These are placeholder numbers consistent with the published QLSTM
# literature ranges; they are NOT measurements.
BASE_RMSE = {
    "NVDA":     {"lstm": 0.0161, "bilstm": 0.0149, "patchtst": 0.0139, "nbeats": 0.0143, "itransformer": 0.0138, "qlstm": 0.0131},
    "AAPL":     {"lstm": 0.0116, "bilstm": 0.0110, "patchtst": 0.0101, "nbeats": 0.0104, "itransformer": 0.0102, "qlstm": 0.0106},
    "JPM":      {"lstm": 0.0098, "bilstm": 0.0095, "patchtst": 0.0089, "nbeats": 0.0091, "itransformer": 0.0089, "qlstm": 0.0093},
    "EURUSD=X": {"lstm": 0.00091, "bilstm": 0.00088, "patchtst": 0.00082, "nbeats": 0.00085, "itransformer": 0.00083, "qlstm": 0.00086},
    "GBPUSD=X": {"lstm": 0.00096, "bilstm": 0.00092, "patchtst": 0.00087, "nbeats": 0.00090, "itransformer": 0.00088, "qlstm": 0.00090},
    "BTC-USD":  {"lstm": 0.0391, "bilstm": 0.0378, "patchtst": 0.0358, "nbeats": 0.0371, "itransformer": 0.0362, "qlstm": 0.0336},
    "ETH-USD":  {"lstm": 0.0424, "bilstm": 0.0405, "patchtst": 0.0381, "nbeats": 0.0396, "itransformer": 0.0385, "qlstm": 0.0360},
}

# Directional accuracy mean
BASE_DA = {
    "NVDA":     {"lstm": 53.4, "bilstm": 54.2, "patchtst": 55.6, "nbeats": 54.7, "itransformer": 55.3, "qlstm": 56.4},
    "AAPL":     {"lstm": 52.1, "bilstm": 53.0, "patchtst": 54.8, "nbeats": 54.0, "itransformer": 54.4, "qlstm": 53.7},
    "JPM":      {"lstm": 51.6, "bilstm": 52.4, "patchtst": 53.9, "nbeats": 53.2, "itransformer": 53.6, "qlstm": 52.8},
    "EURUSD=X": {"lstm": 52.0, "bilstm": 52.8, "patchtst": 55.4, "nbeats": 54.2, "itransformer": 54.7, "qlstm": 53.6},
    "GBPUSD=X": {"lstm": 51.4, "bilstm": 52.3, "patchtst": 54.5, "nbeats": 53.7, "itransformer": 54.0, "qlstm": 53.2},
    "BTC-USD":  {"lstm": 53.5, "bilstm": 54.3, "patchtst": 55.9, "nbeats": 55.0, "itransformer": 55.4, "qlstm": 56.9},
    "ETH-USD":  {"lstm": 53.0, "bilstm": 53.8, "patchtst": 55.4, "nbeats": 54.4, "itransformer": 54.8, "qlstm": 56.2},
}

# Noise mapping for the depolarising sweep on QLSTM
NOISE_RMSE = {
    "BTC-USD": {0.000: 0.0336, 0.001: 0.0342, 0.005: 0.0362, 0.010: 0.0389},
    "NVDA":    {0.000: 0.01315, 0.001: 0.01345, 0.005: 0.01406, 0.010: 0.01486},
}
DEPTH_RMSE = {
    "BTC-USD": {0: 0.0391, 2: 0.0368, 4: 0.0351, 6: 0.0336},
    "NVDA":    {0: 0.01612, 2: 0.01482, 4: 0.01391, 6: 0.01315},
}


def _noise_per_seed(mean: float, rel_std: float = 0.12) -> np.ndarray:
    """Generate per-seed RMSE around `mean` with relative std `rel_std`."""
    rng = np.random.default_rng(seed=int(mean * 1e6))
    return mean * (1 + rel_std * rng.standard_normal(len(SEEDS)))


def main():
    print("== smoke_test: synthesising rehearsal data ==", file=sys.stderr)

    base = Path("results/raw"); base.mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    (Path("results") / "SMOKE_TEST_DATA.txt").write_text(
        "This directory contains rehearsal data from src.smoke_test, NOT real\n"
        "experimental results. Delete this directory and run `bash run_all.sh`\n"
        "for actual results.\n")

    # Main results
    rows = []
    for a in ASSETS:
        for m in MODELS:
            mean = BASE_RMSE[a][m]
            rmses = _noise_per_seed(mean, rel_std=0.12 if m != "qlstm" else 0.18)
            das_mean = BASE_DA[a][m]
            rng = np.random.default_rng(int(mean * 1e6) + 1)
            das = das_mean + 1.2 * rng.standard_normal(len(SEEDS))
            for s, r, d in zip(SEEDS, rmses, das):
                rows.append({"model": m, "asset": a, "seed": s,
                              "rmse": float(r), "mae": float(r * 0.8),
                              "mape": float(abs(d - 50)), "da": float(d),
                              "params": 295, "train_sec": 12.0})
                # also synthesise predictions file
                rng2 = np.random.default_rng(s + hash(m + a) % 1000)
                n_test = 180
                preds = rng2.standard_normal(n_test) * 0.5
                # Bias positive direction toward DA mean
                targets = rng2.standard_normal(n_test) * 0.5
                preds = 0.3 * targets + preds  # correlation gives target DA roughly
                prices = 100 + np.cumsum(targets * 2)
                npz_path = Path("results/raw/preds") / f"{m}_{a}_seed{s}.npz"
                npz_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(npz_path, preds=preds, targets=targets, prices=prices)
    pd.DataFrame(rows).to_csv(base / "main_results.csv", index=False)

    # Walk-forward
    wf_rows = []
    for a in ["NVDA", "AAPL", "BTC-USD", "ETH-USD"]:
        for m in MODELS:
            mean = BASE_RMSE[a][m] * 1.03
            wf_rows.append({"asset": a, "model": m, "fold": "mean",
                             "rmse_mean": float(mean), "rmse_std": float(mean * 0.05)})
    pd.DataFrame(wf_rows).to_csv(base / "walk_forward.csv", index=False)

    # Noise
    noise_rows = []
    for a, mp in NOISE_RMSE.items():
        for p, v in mp.items():
            noise_rows.append({"asset": a, "noise_p": p,
                                "rmse_mean": float(v), "rmse_std": float(v * 0.05)})
    pd.DataFrame(noise_rows).to_csv(base / "noise.csv", index=False)

    # Depth
    depth_rows = []
    for a, mp in DEPTH_RMSE.items():
        for k, v in mp.items():
            depth_rows.append({"asset": a, "n_vqc_blocks": k,
                                "rmse_mean": float(v), "rmse_std": float(v * 0.05)})
    pd.DataFrame(depth_rows).to_csv(base / "depth.csv", index=False)

    # Table 1 — placeholder data
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"Asset class": "Equity",           "Ticker": "NVDA",     "Source": "Yahoo Finance", "Period": "2020-01-01 / 2024-12-31", "Obs.": 1259, "Annualised sigma": "0.521"},
        {"Asset class": "Equity",           "Ticker": "AAPL",     "Source": "Yahoo Finance", "Period": "2020-01-01 / 2024-12-31", "Obs.": 1259, "Annualised sigma": "0.298"},
        {"Asset class": "Equity",           "Ticker": "JPM",      "Source": "Yahoo Finance", "Period": "2020-01-01 / 2024-12-31", "Obs.": 1259, "Annualised sigma": "0.264"},
        {"Asset class": "Foreign exchange", "Ticker": "EURUSD=X", "Source": "Yahoo Finance", "Period": "2020-01-01 / 2024-12-31", "Obs.": 1304, "Annualised sigma": "0.071"},
        {"Asset class": "Foreign exchange", "Ticker": "GBPUSD=X", "Source": "Yahoo Finance", "Period": "2020-01-01 / 2024-12-31", "Obs.": 1304, "Annualised sigma": "0.089"},
        {"Asset class": "Cryptocurrency",   "Ticker": "BTC-USD",  "Source": "Yahoo Finance", "Period": "2020-01-01 / 2024-12-31", "Obs.": 1827, "Annualised sigma": "0.674"},
        {"Asset class": "Cryptocurrency",   "Ticker": "ETH-USD",  "Source": "Yahoo Finance", "Period": "2020-01-01 / 2024-12-31", "Obs.": 1827, "Annualised sigma": "0.812"},
    ]).to_csv("results/tables/table1_dataset.csv", index=False)

    print("== smoke_test: done. Rehearsal CSVs in results/raw/", file=sys.stderr)


if __name__ == "__main__":
    main()
