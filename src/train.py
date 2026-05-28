"""
train.py
========
Generic training loop used by every model: Adam, MSE loss, early stopping
on validation MSE.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    patience: int = 15
    device: str = "cpu"


def _seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one(model_fn: Callable[[], nn.Module],
              X_tr: np.ndarray, y_tr: np.ndarray,
              X_va: np.ndarray, y_va: np.ndarray,
              seed: int, cfg: TrainConfig) -> nn.Module:
    _seed_everything(seed)
    model = model_fn().to(cfg.device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()

    ds_tr = TensorDataset(
        torch.from_numpy(X_tr.astype(np.float32)),
        torch.from_numpy(y_tr.astype(np.float32)),
    )
    loader = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    X_va_t = torch.from_numpy(X_va.astype(np.float32)).to(cfg.device)
    y_va_t = torch.from_numpy(y_va.astype(np.float32)).to(cfg.device)

    best_val = math.inf
    best_state = None
    bad = 0
    for ep in range(cfg.epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(cfg.device); yb = yb.to(cfg.device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(X_va_t), y_va_t).item())
        if val < best_val - 1e-7:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    return model(X_t).detach().cpu().numpy()
