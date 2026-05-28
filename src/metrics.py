"""
metrics.py
==========
All quantitative metrics referenced in the paper: RMSE, MAE, MAPE,
directional accuracy, annualised Sharpe ratio, maximum drawdown,
Wilcoxon signed-rank, bootstrap CI.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float(np.mean(true_dir == pred_dir) * 100.0)


def long_short_trade(returns: np.ndarray, signal_pred: np.ndarray,
                      cost_bp: float, trading_days: int) -> dict[str, float]:
    """
    Long-short strategy: position +1 if signal_pred > 0 else -1.
    `returns` is the realised next-day return aligned with the prediction.
    `cost_bp` is the round-trip transaction cost in basis points.
    """
    positions = np.where(signal_pred > 0, 1.0, -1.0)
    # Position-change cost
    pos_changes = np.abs(np.diff(positions, prepend=0.0))
    costs = pos_changes * (cost_bp / 1e4)
    strat_ret = positions * returns - costs
    cum = np.cumprod(1.0 + strat_ret)
    cum_final = max(cum[-1], 1e-12)              # avoid NaN on negative cumulatives
    ann_return = (cum_final ** (trading_days / max(len(strat_ret), 1)) - 1.0) * 100.0
    sd = float(np.std(strat_ret, ddof=1))
    if sd <= 1e-12:
        sharpe = 0.0
    else:
        sharpe = float((np.mean(strat_ret) - 0.03 / trading_days) / sd * np.sqrt(trading_days))
    peak = np.maximum.accumulate(cum)
    mdd = float(np.max((peak - cum) / np.maximum(peak, 1e-12)) * 100.0)
    return {"annual_return_pct": float(ann_return), "sharpe": sharpe, "mdd_pct": mdd}


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value for paired samples a, b."""
    diff = a - b
    if np.allclose(diff, 0):
        return 1.0
    try:
        _, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(p)
    except ValueError:
        return 1.0


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000,
                 alpha: float = 0.05, rng: np.random.Generator | None = None
                 ) -> tuple[float, float]:
    rng = rng or np.random.default_rng(20251218)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = values[idx].mean(axis=1)
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return lo, hi


# Transaction cost defaults per the manuscript
COST_BP = {
    "Equity":           10.0,
    "Foreign exchange":  2.0,
    "Cryptocurrency":   25.0,
}
