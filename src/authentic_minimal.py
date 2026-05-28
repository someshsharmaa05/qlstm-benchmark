"""
Minimal authentic experiment — guaranteed to complete in this environment.
Single CPU, pure numpy, 2 assets x 2 models x 2 seeds, tiny problem.

Purpose: prove the QLSTM benchmark methodology produces sensible numbers
when executed in a real environment with real computation, even at
greatly reduced scale.
"""
import sys, time, json
sys.path.insert(0, "/home/claude/qlstm_pkg")
from pathlib import Path
import numpy as np
from scipy import stats

from src.authentic_smoke_test import (
    make_dataset, CompactQLSTM, CompactLSTM, spsa_train, rmse, da
)

OUT = Path("/home/claude/qlstm_pkg/results/raw/authentic_minimal.json")


def run(asset, model_kind, seed, n=120, n_iter=8):
    X, y, _ = make_dataset(asset, n=n, seed=seed)
    # CORRECT split: use sequence count, not feature count
    n_seq = len(X)
    n_train = int(0.65 * n_seq)
    n_val   = max(int(0.15 * n_seq), 4)
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_va, y_va = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_te, y_te = X[n_train + n_val:], y[n_train + n_val:]
    assert len(X_te) >= 4, f"test set too small: {len(X_te)}"
    if model_kind == "qlstm":
        m = CompactQLSTM(in_features=4, n_blocks=1, seed=seed)
    else:
        m = CompactLSTM(in_features=4, hidden=5, seed=seed)
    t0 = time.time()
    rng = np.random.default_rng(seed + 7)
    m, _ = spsa_train(m, X_tr, y_tr, X_va, y_va, n_iter=n_iter, rng=rng)
    yp = m.predict_batch(X_te)
    return dict(asset=asset, model=model_kind, seed=seed, n_params=m.n_params(),
                rmse=rmse(y_te, yp), da=da(y_te, yp),
                train_sec=time.time() - t0,
                y_test=y_te.tolist(), y_pred=yp.tolist())


def main():
    assets = ["EQUITY_HIGH", "CRYPTO"]
    models = ["lstm", "qlstm"]
    seeds = [42, 137]
    print(f"AUTHENTIC MINIMAL: {len(assets)} assets x {len(models)} models x {len(seeds)} seeds")
    print("Real numpy state-vector QLSTM simulation with real SPSA training.\n")
    all_r = []
    t0 = time.time()
    for a in assets:
        for mk in models:
            for s in seeds:
                r = run(a, mk, s)
                all_r.append(r)
                print(f"  {a:13s} {mk:6s} seed={s}: RMSE={r['rmse']:.4f}  "
                      f"DA={r['da']:5.1f}%  params={r['n_params']:4d}  "
                      f"t={r['train_sec']:.1f}s")
    total = time.time() - t0

    print(f"\nWall-clock: {total:.1f} sec")
    print("\n--- aggregated ---")
    summary = {}
    for a in assets:
        for mk in models:
            rs = [r for r in all_r if r["asset"] == a and r["model"] == mk]
            rmses = np.array([r["rmse"] for r in rs])
            das = np.array([r["da"] for r in rs])
            summary[f"{a}/{mk}"] = dict(
                rmse_mean=float(rmses.mean()), rmse_std=float(rmses.std()),
                da_mean=float(das.mean()), n_seeds=len(rs)
            )
            print(f"  {a:13s} {mk:6s}  RMSE={rmses.mean():.4f}±{rmses.std():.4f}  "
                  f"DA={das.mean():5.1f}%")

    # Wilcoxon (only 2 seeds — too few for real significance, but show the protocol)
    wp = {}
    for a in assets:
        q = sorted([r for r in all_r if r["asset"] == a and r["model"] == "qlstm"],
                   key=lambda r: r["seed"])
        l = sorted([r for r in all_r if r["asset"] == a and r["model"] == "lstm"],
                   key=lambda r: r["seed"])
        try:
            _, p = stats.wilcoxon([r["rmse"] for r in q], [r["rmse"] for r in l])
            wp[a] = float(p)
        except Exception as e:
            wp[a] = None
        print(f"  {a:13s} Wilcoxon QLSTM vs LSTM:  p={wp[a]} (warning: n=2 is too small for real inference)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "results": all_r, "summary": summary, "wilcoxon_p": wp,
        "wall_clock_sec": total,
        "caveats": (
            "Tiny-scale authentic numpy proof-of-concept. "
            "GARCH(1,1) synthetic data, NOT Yahoo Finance. "
            "2 seeds per cell - insufficient for real Wilcoxon inference. "
            "Demonstrates the methodology works; for paper-grade numbers run qlstm_pkg full."
        ),
    }, indent=2))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
