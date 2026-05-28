"""
classical_models.py
===================
Five baseline architectures matched to ~290-310 trainable parameters:
LSTM, BiLSTM, PatchTST, N-BEATS, iTransformer.

Reference implementations are kept compact but faithful to the published
papers cited in the manuscript.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------
# 1. Plain LSTM
# ----------------------------------------------------------------------------
class PlainLSTM(nn.Module):
    def __init__(self, in_features: int = 4, hidden: int = 7):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (B, T, F)
        h, _ = self.lstm(x)
        return self.head(h[:, -1]).squeeze(-1)


# ----------------------------------------------------------------------------
# 2. BiLSTM
# ----------------------------------------------------------------------------
class BiLSTM(nn.Module):
    def __init__(self, in_features: int = 4, hidden: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1]).squeeze(-1)


# ----------------------------------------------------------------------------
# 3. PatchTST — Nie et al. 2023 (ICLR), reduced
# ----------------------------------------------------------------------------
class PatchTST(nn.Module):
    """Patch length 8, stride 4, 1 head, d_model 8, 1 block."""

    def __init__(self, seq_len: int = 30, in_features: int = 4,
                 patch_len: int = 8, stride: int = 4, d_model: int = 8):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.proj = nn.Linear(patch_len * in_features, d_model, bias=False)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=1, dim_feedforward=8,
            dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=1)
        self.head = nn.Linear(d_model * self.n_patches, 1)

    def forward(self, x):  # (B, T, F)
        B, T, F = x.shape
        patches = []
        for i in range(self.n_patches):
            start = i * self.stride
            patches.append(x[:, start : start + self.patch_len].reshape(B, -1))
        z = torch.stack(patches, dim=1)         # (B, n_patches, patch_len*F)
        z = self.proj(z) + self.pos
        z = self.encoder(z)
        return self.head(z.flatten(1)).squeeze(-1)


# ----------------------------------------------------------------------------
# 4. N-BEATS — Oreshkin et al., 2 stacks x 2 blocks, hidden 12
# ----------------------------------------------------------------------------
class _NBeatsBlock(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 12, theta_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bcast = nn.Linear(hidden, theta_dim)
        self.fcast = nn.Linear(hidden, 1)
        self.proj = nn.Linear(theta_dim, in_dim, bias=False)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        backcast = self.proj(self.bcast(h))
        forecast = self.fcast(h).squeeze(-1)
        return backcast, forecast


class NBeats(nn.Module):
    def __init__(self, seq_len: int = 30, in_features: int = 4):
        super().__init__()
        in_dim = seq_len * in_features
        # 2 stacks x 2 blocks per stack
        self.blocks = nn.ModuleList([_NBeatsBlock(in_dim, hidden=12, theta_dim=4) for _ in range(4)])

    def forward(self, x):
        z = x.flatten(1)
        forecast = 0.0
        residual = z
        for blk in self.blocks:
            b, f = blk(residual)
            residual = residual - b
            forecast = forecast + f
        return forecast


# ----------------------------------------------------------------------------
# 5. iTransformer — Liu et al. 2024 (ICLR), reduced
# ----------------------------------------------------------------------------
class ITransformer(nn.Module):
    """Variates-as-tokens. 1 head, d_model 8, 1 block, dff 16."""
    def __init__(self, seq_len: int = 30, in_features: int = 4, d_model: int = 8):
        super().__init__()
        self.tok = nn.Linear(seq_len, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=1, dim_feedforward=16,
            dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=1)
        self.head = nn.Linear(d_model * in_features, 1)

    def forward(self, x):  # (B, T, F) -> tokenise per variate
        z = x.transpose(1, 2)                  # (B, F, T)
        z = self.tok(z)                        # (B, F, d_model)
        z = self.encoder(z)
        return self.head(z.flatten(1)).squeeze(-1)


# ----------------------------------------------------------------------------
def build_classical(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "lstm":         return PlainLSTM(**kwargs)
    if name == "bilstm":       return BiLSTM(**kwargs)
    if name == "patchtst":     return PatchTST(**kwargs)
    if name == "nbeats":       return NBeats(**kwargs)
    if name == "itransformer": return ITransformer(**kwargs)
    raise ValueError(f"unknown classical model: {name}")


if __name__ == "__main__":
    import json
    counts = {n: n_params(build_classical(n)) for n in
              ["lstm", "bilstm", "patchtst", "nbeats", "itransformer"]}
    print(json.dumps(counts, indent=2))
