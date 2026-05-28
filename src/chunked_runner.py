"""
Chunked driver - run one study at a time, save, allow restart.
Usage: python -m src.chunked_runner [main|noise|depth|wf|aggregate]
"""
import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/claude/qlstm_pkg")

from src.full_authentic_benchmark import (
    run_main_benchmark, run_noise_study, run_depth_study, run_walkforward,
    aggregate, make_figures, MODEL_FACTORIES, ASSETS_USED, OUTPUT_ROOT
)

CHUNK_DIR = OUTPUT_ROOT / "raw"
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 137, 271, 389, 521]


def save(name, data):
    with open(CHUNK_DIR / f"{name}.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved {name}.json")


def load(name):
    p = CHUNK_DIR / f"{name}.json"
    return json.load(open(p)) if p.exists() else None


def main():
    chunk = sys.argv[1] if len(sys.argv) > 1 else "main"
    t0 = time.time()

    if chunk == "main":
        r = run_main_benchmark(SEEDS, n_iter_classical=8, n_iter_quantum=5, n=140)
        save("main_results", r)

    elif chunk == "noise":
        r = run_noise_study(SEEDS[:3], n_iter=5, n=120)
        save("noise_results", r)

    elif chunk == "depth":
        r = run_depth_study(SEEDS[:3], n_iter=5, n=120)
        save("depth_results", r)

    elif chunk == "wf":
        r = run_walkforward(SEEDS[:3], n_iter=5, n=140, n_folds=3)
        save("wf_results", r)

    elif chunk == "aggregate":
        m = load("main_results"); n = load("noise_results")
        d = load("depth_results"); w = load("wf_results")
        if not all([m, n, d, w]):
            print("Missing one or more chunks. Run main/noise/depth/wf first.")
            sys.exit(1)
        rmse_table = aggregate(m, n, d, w)
        make_figures(rmse_table, n, d)
        # Combined raw
        with open(CHUNK_DIR / "full_benchmark.json", "w") as f:
            json.dump({"main_results": m, "noise_results": n,
                       "depth_results": d, "wf_results": w,
                       "seeds": SEEDS}, f, indent=2)
        print("\nFull benchmark assembled.")

    else:
        print(f"Unknown chunk: {chunk}"); sys.exit(1)

    print(f"\n[{chunk}] wall-clock: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
