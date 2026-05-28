"""
Walk-forward, one asset at a time. Appends to wf_results.json.
Usage: python -m src.wf_chunk EQUITY_MID
"""
import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/claude/qlstm_pkg")

from src.full_authentic_benchmark import (
    MODEL_FACTORIES, fit_and_eval, OUTPUT_ROOT
)
from src.authentic_smoke_test import make_dataset

CACHE = OUTPUT_ROOT / "raw" / "wf_results.json"
CACHE.parent.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 137, 271]


def main():
    asset = sys.argv[1] if len(sys.argv) > 1 else "EQUITY_MID"
    existing = json.load(open(CACHE)) if CACHE.exists() else []
    if any(r["asset"] == asset for r in existing):
        print(f"{asset} already done, skipping")
        return

    t0 = time.time()
    results = []
    for model_name in ["LSTM", "PatchTST", "QLSTM"]:
        factory = MODEL_FACTORIES[model_name]
        ni = 5 if model_name == "QLSTM" else 8
        for fold in range(1, 4):
            fold_rmses = []
            for seed in SEEDS:
                X, y, _ = make_dataset(asset, n=140, seed=seed + fold * 1000)
                model = factory(seed)
                r, _, _, _ = fit_and_eval(model, X, y, seed, ni)
                fold_rmses.append(r)
            results.append({"asset": asset, "model": model_name, "fold": fold,
                            "rmse_mean": float(np.mean(fold_rmses))})
        mean_r = np.mean([rr["rmse_mean"] for rr in results if rr["model"] == model_name])
        print(f"  {model_name:10s}  mean WF RMSE={mean_r:.4f}")

    combined = existing + results
    with open(CACHE, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n[{asset}] saved, wall-clock: {time.time()-t0:.1f}s, total records: {len(combined)}")


if __name__ == "__main__":
    main()
