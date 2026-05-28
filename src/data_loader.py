"""
data_loader.py
==============
Download daily closing prices for the 7 benchmark instruments from Yahoo
Finance and produce model-ready feature sequences.

Usage
-----
    python -m src.data_loader --start 2020-01-01 --end 2024-12-31 \
        --outdir results/raw/data
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------------------------------
#  Asset universe — paper Table 1
# ----------------------------------------------------------------------------
ASSETS: dict[str, dict] = {
    "NVDA":     {"class": "Equity",           "trading_days": 252},
    "AAPL":     {"class": "Equity",           "trading_days": 252},
    "JPM":      {"class": "Equity",           "trading_days": 252},
    "EURUSD=X": {"class": "Foreign exchange", "trading_days": 252},
    "GBPUSD=X": {"class": "Foreign exchange", "trading_days": 252},
    "BTC-USD":  {"class": "Cryptocurrency",   "trading_days": 365},
    "ETH-USD":  {"class": "Cryptocurrency",   "trading_days": 365},
}

SEQ_LEN = 30   # input sequence length used by every model
LOOKBACK_SCALER = 60


@dataclass
class AssetSplit:
    ticker: str
    asset_class: str
    trading_days: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_prices: np.ndarray         # un-normalised closes for trading strategy
    realised_vol: float


# ----------------------------------------------------------------------------
def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch one ticker; retry once on transient errors."""
    for attempt in range(2):
        try:
            df = yf.download(
                ticker, start=start, end=end,
                interval="1d", progress=False, auto_adjust=False,
            )
            if df.empty:
                raise RuntimeError(f"empty frame for {ticker}")
            return df
        except Exception as e:
            if attempt == 1:
                raise
            print(f"  retry {ticker}: {e}", file=sys.stderr)
    raise RuntimeError("unreachable")


def _winsorize(s: pd.Series, window: int = 20, n_sigma: float = 3.0) -> pd.Series:
    """Replace rolling-z-score outliers via linear interpolation."""
    roll_mu = s.rolling(window, min_periods=1).mean()
    roll_sd = s.rolling(window, min_periods=1).std().fillna(method="bfill")
    z = (s - roll_mu) / roll_sd.replace(0, np.nan)
    mask = z.abs() > n_sigma
    s = s.copy()
    s[mask] = np.nan
    return s.interpolate(limit_direction="both")


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _bollinger_width(close: pd.Series, period: int = 20, k: float = 2.0) -> pd.Series:
    ma = close.rolling(period, min_periods=1).mean()
    sd = close.rolling(period, min_periods=1).std().fillna(0)
    upper = ma + k * sd
    lower = ma - k * sd
    return (upper - lower) / ma.replace(0, np.nan)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = _winsorize(df["Close"].astype(float))
    vol   = df["Volume"].astype(float).fillna(0)

    out = pd.DataFrame(index=df.index)
    out["close"]   = close
    out["log_ret"] = np.log(close / close.shift(1))
    out["rsi14"]   = _rsi(close)
    out["bbw20"]   = _bollinger_width(close)
    out["log_vol"] = np.log1p(vol)
    out = out.dropna()
    return out


def _make_sequences(feat_arr: np.ndarray, target: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(feat_arr) - seq_len):
        X.append(feat_arr[i : i + seq_len])
        y.append(target[i + seq_len])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def build_split(df: pd.DataFrame, ticker: str) -> AssetSplit:
    feats = build_features(df)
    if len(feats) < SEQ_LEN + 30:
        raise RuntimeError(f"{ticker}: not enough usable rows ({len(feats)})")

    # Target = next-day log return
    target = feats["log_ret"].shift(-1).fillna(0).values
    feat_cols = ["log_ret", "rsi14", "bbw20", "log_vol"]
    arr = feats[feat_cols].values.astype(np.float32)

    n = len(arr)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)

    # Sliding-window MinMax: fit on rolling LOOKBACK_SCALER window
    scaler = MinMaxScaler()
    scaler.fit(arr[:n_train])
    arr_n = scaler.transform(arr)

    target_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    target_scaler.fit(target[:n_train].reshape(-1, 1))
    target_n = target_scaler.transform(target.reshape(-1, 1)).ravel()

    train_X, train_y = _make_sequences(
        arr_n[:n_train + SEQ_LEN], target_n[:n_train + SEQ_LEN], SEQ_LEN
    )
    val_X, val_y = _make_sequences(
        arr_n[n_train : n_train + n_val + SEQ_LEN],
        target_n[n_train : n_train + n_val + SEQ_LEN], SEQ_LEN
    )
    test_X, test_y = _make_sequences(
        arr_n[n_train + n_val :], target_n[n_train + n_val :], SEQ_LEN
    )
    test_prices = feats["close"].values[n_train + n_val + SEQ_LEN :]

    realised_vol = float(np.std(feats["log_ret"]) * np.sqrt(ASSETS[ticker]["trading_days"]))

    return AssetSplit(
        ticker=ticker,
        asset_class=ASSETS[ticker]["class"],
        trading_days=ASSETS[ticker]["trading_days"],
        X_train=train_X, y_train=train_y,
        X_val=val_X,     y_val=val_y,
        X_test=test_X,   y_test=test_y,
        test_prices=test_prices,
        realised_vol=realised_vol,
    )


def load_all(start: str = "2020-01-01", end: str = "2024-12-31",
             cache_dir: str | os.PathLike = "results/raw/data") -> dict[str, AssetSplit]:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for tk in ASSETS:
        raw_path = cache / f"{tk}_raw.csv"
        if raw_path.exists():
            df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
            print(f"  cached  {tk:10s} rows={len(df)}")
        else:
            df = fetch(tk, start, end)
            df.to_csv(raw_path)
            print(f"  fetched {tk:10s} rows={len(df)}")
        out[tk] = build_split(df, tk)
    return out


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end",   default="2024-12-31")
    ap.add_argument("--outdir", default="results/raw/data")
    args = ap.parse_args()

    splits = load_all(args.start, args.end, args.outdir)

    # Emit Table 1 from real data
    rows = []
    for tk, sp in splits.items():
        rows.append({
            "Asset class": sp.asset_class,
            "Ticker": tk,
            "Source": "Yahoo Finance",
            "Period": f"{args.start} / {args.end}",
            "Obs.": sp.X_train.shape[0] + sp.X_val.shape[0] + sp.X_test.shape[0] + SEQ_LEN,
            "Annualised sigma": f"{sp.realised_vol:.3f}",
        })
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "table1_dataset.csv", index=False)
    print(f"\nWrote {out_dir/'table1_dataset.csv'}")


if __name__ == "__main__":
    main()
