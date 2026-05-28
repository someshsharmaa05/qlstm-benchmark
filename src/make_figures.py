"""
make_figures.py
===============
Generate the four manuscript figures from real results.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG_DIR = Path("results/figures"); FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = Path("results/raw")

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 200,
})


# ----------------------------------------------------------------------------
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(8.5, 2.4))
    boxes = [
        (0.02, "Yahoo Finance\nDaily closes\n2020-2024"),
        (0.18, "Preprocessing\nwinsorise + log-ret\n+RSI+BBW+vol"),
        (0.36, "Sequence\n30 steps\n4 features"),
        (0.52, "Multi-seed\ntraining\n15 seeds × 6 models"),
        (0.72, "Statistical\naggregation\nWilcoxon + bootstrap"),
        (0.88, "Economic\nevaluation\nSharpe, MDD"),
    ]
    for i, (x, txt) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x, 0.30), 0.12, 0.45,
                                     boxstyle="round,pad=0.012",
                                     fc="#E8F0FB", ec="#1F4E79", lw=1.0,
                                     transform=ax.transAxes))
        ax.text(x + 0.06, 0.52, txt, transform=ax.transAxes,
                ha="center", va="center", fontsize=7.5)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(boxes[i + 1][0], 0.52), xytext=(x + 0.12, 0.52),
                         xycoords="axes fraction",
                         arrowprops=dict(arrowstyle="->", lw=1.0, color="#1F4E79"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.savefig(FIG_DIR / "fig1_pipeline.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig2_qlstm_cell():
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    # Inputs
    ax.text(0.04, 0.85, r"$x_t$", fontsize=12, transform=ax.transAxes)
    ax.text(0.04, 0.45, r"$h_{t-1}$", fontsize=12, transform=ax.transAxes)
    ax.text(0.04, 0.05, r"$c_{t-1}$", fontsize=12, transform=ax.transAxes)

    # Concat → Linear → tanh
    ax.add_patch(FancyBboxPatch((0.15, 0.40), 0.10, 0.18,
                                  boxstyle="round,pad=0.01",
                                  fc="#FFF2CC", ec="#806000", lw=1.0,
                                  transform=ax.transAxes))
    ax.text(0.20, 0.49, "concat\n+ linear", ha="center", va="center", fontsize=8,
              transform=ax.transAxes)

    # Four VQCs
    labels = [r"VQC$_f$ (forget)", r"VQC$_i$ (input)", r"VQC$_g$ (cand.)", r"VQC$_o$ (output)"]
    y_positions = [0.78, 0.58, 0.38, 0.18]
    for lbl, y in zip(labels, y_positions):
        ax.add_patch(FancyBboxPatch((0.33, y - 0.05), 0.18, 0.10,
                                      boxstyle="round,pad=0.008",
                                      fc="#D9E2F3", ec="#1F4E79", lw=1.0,
                                      transform=ax.transAxes))
        ax.text(0.42, y, lbl, ha="center", va="center", fontsize=8,
                  transform=ax.transAxes)

    ax.text(0.62, 0.78, r"$\sigma$", fontsize=12, ha="center", transform=ax.transAxes)
    ax.text(0.62, 0.58, r"$\sigma$", fontsize=12, ha="center", transform=ax.transAxes)
    ax.text(0.62, 0.38, r"$\tanh$", fontsize=11, ha="center", transform=ax.transAxes)
    ax.text(0.62, 0.18, r"$\sigma$", fontsize=12, ha="center", transform=ax.transAxes)

    ax.add_patch(FancyBboxPatch((0.72, 0.42), 0.10, 0.16,
                                  boxstyle="round,pad=0.008",
                                  fc="#E2EFDA", ec="#385723", lw=1.0,
                                  transform=ax.transAxes))
    ax.text(0.77, 0.50, r"$c_t$"+"\n"+r"$h_t$", ha="center", va="center", fontsize=10,
              transform=ax.transAxes)

    ax.text(0.92, 0.65, r"$h_t$", fontsize=12, transform=ax.transAxes)
    ax.text(0.92, 0.35, r"$c_t$", fontsize=12, transform=ax.transAxes)

    ax.text(0.5, 0.02, "Each VQC block: Hadamard + R$_y$ angle encoding · "
                       "R$_x$R$_y$R$_z$ ansatz · ring CNOT · Pauli-Z readout (4 qubits, 6 blocks)",
            ha="center", fontsize=7.5, style="italic", transform=ax.transAxes)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.savefig(FIG_DIR / "fig2_qlstm_cell.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig3_rmse_bar():
    """Bar chart of test RMSE per model, grouped by asset class."""
    df = pd.read_csv(RES_DIR / "main_results.csv")
    asset_classes = {
        "Equity":           ["NVDA", "AAPL", "JPM"],
        "Foreign exchange": ["EURUSD=X", "GBPUSD=X"],
        "Cryptocurrency":   ["BTC-USD", "ETH-USD"],
    }
    scales = {"Equity": 1e3, "Foreign exchange": 1e4, "Cryptocurrency": 1e2}
    units  = {"Equity": "×10⁻³", "Foreign exchange": "×10⁻⁴", "Cryptocurrency": "×10⁻²"}
    models_order = ["lstm", "bilstm", "patchtst", "nbeats", "itransformer", "qlstm"]
    labels = ["LSTM", "BiLSTM", "PatchTST", "N-BEATS", "iTransformer", "QLSTM"]
    colors = ["#7F7F7F", "#A6A6A6", "#5B9BD5", "#70AD47", "#FFC000", "#C00000"]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=False)
    for ax, (cls, assets) in zip(axes, asset_classes.items()):
        x = np.arange(len(assets))
        width = 0.13
        for i, m in enumerate(models_order):
            means, errs = [], []
            for a in assets:
                sub = df[(df["model"] == m) & (df["asset"] == a)]["rmse"]
                means.append(sub.mean() * scales[cls] if len(sub) else 0)
                errs.append(sub.std() * scales[cls] if len(sub) else 0)
            ax.bar(x + i * width - 2.5 * width, means, width,
                    yerr=errs, capsize=2, color=colors[i], label=labels[i],
                    edgecolor="black", linewidth=0.3)
        ax.set_xticks(x); ax.set_xticklabels(assets, rotation=0, fontsize=8)
        ax.set_title(cls)
        ax.set_ylabel(f"Test RMSE  {units[cls]}")
        ax.grid(axis="y", lw=0.4, alpha=0.5)
    axes[-1].legend(loc="upper right", ncol=2, fontsize=7, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_rmse_bar.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig4_noise_and_depth():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))

    # (a) Noise
    if (RES_DIR / "noise.csv").exists():
        noise = pd.read_csv(RES_DIR / "noise.csv")
        for a, color, marker in [("BTC-USD", "#C00000", "o"), ("NVDA", "#1F4E79", "s")]:
            sub = noise[noise["asset"] == a].sort_values("noise_p")
            scale = 1e2 if a == "BTC-USD" else 1e3
            unit  = "×10⁻²" if a == "BTC-USD" else "×10⁻³"
            axes[0].errorbar(sub["noise_p"].values, sub["rmse_mean"].values * scale,
                              yerr=sub["rmse_std"].values * scale,
                              marker=marker, color=color, label=f"{a} ({unit})",
                              lw=1.2, capsize=2.5)
        axes[0].set_xlabel("Depolarising noise $p$ per gate layer")
        axes[0].set_ylabel("Test RMSE")
        axes[0].set_title("(a) Noise robustness")
        axes[0].grid(lw=0.4, alpha=0.5); axes[0].legend(fontsize=8)

    # (b) Depth
    if (RES_DIR / "depth.csv").exists():
        depth = pd.read_csv(RES_DIR / "depth.csv")
        for a, color, marker in [("BTC-USD", "#C00000", "o"), ("NVDA", "#1F4E79", "s")]:
            sub = depth[depth["asset"] == a].sort_values("n_vqc_blocks")
            scale = 1e2 if a == "BTC-USD" else 1e3
            unit  = "×10⁻²" if a == "BTC-USD" else "×10⁻³"
            axes[1].errorbar(sub["n_vqc_blocks"].values, sub["rmse_mean"].values * scale,
                              yerr=sub["rmse_std"].values * scale,
                              marker=marker, color=color, label=f"{a} ({unit})",
                              lw=1.2, capsize=2.5)
        axes[1].set_xlabel("Number of VQC blocks")
        axes[1].set_ylabel("Test RMSE")
        axes[1].set_title("(b) VQC depth ablation")
        axes[1].grid(lw=0.4, alpha=0.5); axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_noise_and_depth.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    print("Building figures...")
    fig1_pipeline()
    fig2_qlstm_cell()
    if (RES_DIR / "main_results.csv").exists():
        fig3_rmse_bar()
    if (RES_DIR / "noise.csv").exists() or (RES_DIR / "depth.csv").exists():
        fig4_noise_and_depth()
    print(f"Figures written to {FIG_DIR}/")


if __name__ == "__main__":
    main()
