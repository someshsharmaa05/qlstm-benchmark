"""
qlstm.py
========
Quantum Long Short-Term Memory cell.

Implementation follows Chen, Yoo, Kao (2020), "Quantum Long Short-Term
Memory", arXiv:2009.01783, with one important deviation: in addition to
selecting the number of VQC blocks (used by the depth-ablation experiment),
we expose a depolarising noise channel for the NISQ robustness study.

The cell replaces the four classical projection matrices inside the
standard LSTM gating equations with four parameterised quantum circuits
(VQCs). Each VQC has:

    1. an angle-encoding layer (Hadamard + R_y)
    2. a variational ansatz layer (R_x, R_y, R_z per qubit + ring CNOT)
    3. measurement of <Z_i> on each qubit

Implementation uses PennyLane's TorchLayer so PyTorch optimisers can train
quantum parameters jointly with classical ones via the parameter-shift rule.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


N_QUBITS = 4


def _vqc_factory(n_qubits: int, n_blocks: int, noise_p: float = 0.0):
    """Build a PennyLane QNode and PyTorch wrapper for a single VQC."""
    if noise_p <= 0.0:
        dev = qml.device("default.qubit", wires=n_qubits)
    else:
        dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # ---- encoding layer ----
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
        for w in range(n_qubits):
            qml.RY(2.0 * inputs[w], wires=w)
            if noise_p > 0.0:
                qml.DepolarizingChannel(noise_p, wires=w)

        # ---- variational blocks ----
        for b in range(n_blocks):
            for w in range(n_qubits):
                qml.RX(weights[b, w, 0], wires=w)
                qml.RY(weights[b, w, 1], wires=w)
                qml.RZ(weights[b, w, 2], wires=w)
                if noise_p > 0.0:
                    qml.DepolarizingChannel(noise_p, wires=w)
            for w in range(n_qubits):
                qml.CNOT(wires=[w, (w + 1) % n_qubits])

        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    weight_shapes = {"weights": (max(n_blocks, 1), n_qubits, 3)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class QLSTMCell(nn.Module):
    """Single Quantum LSTM cell. Hidden size is fixed at n_qubits."""

    def __init__(self, input_size: int, n_qubits: int = N_QUBITS,
                 n_vqc_blocks: int = 6, noise_p: float = 0.0):
        super().__init__()
        self.n_qubits = n_qubits
        self.input_size = input_size
        self.hidden_size = n_qubits

        # Classical pre-projection from concat([x_t, h_{t-1}]) onto n_qubits
        self.proj = nn.Linear(input_size + n_qubits, n_qubits, bias=False)

        if n_vqc_blocks == 0:
            # Pure-classical fallback (ablation): four small linear gates
            self.classical_fallback = True
            self.gate_f = nn.Linear(n_qubits, n_qubits)
            self.gate_i = nn.Linear(n_qubits, n_qubits)
            self.gate_g = nn.Linear(n_qubits, n_qubits)
            self.gate_o = nn.Linear(n_qubits, n_qubits)
        else:
            self.classical_fallback = False
            self.vqc_f = _vqc_factory(n_qubits, n_vqc_blocks, noise_p)
            self.vqc_i = _vqc_factory(n_qubits, n_vqc_blocks, noise_p)
            self.vqc_g = _vqc_factory(n_qubits, n_vqc_blocks, noise_p)
            self.vqc_o = _vqc_factory(n_qubits, n_vqc_blocks, noise_p)

    def forward(self, x_t, state):
        h_prev, c_prev = state
        v = torch.tanh(self.proj(torch.cat([x_t, h_prev], dim=-1)))

        if self.classical_fallback:
            f = torch.sigmoid(self.gate_f(v))
            i = torch.sigmoid(self.gate_i(v))
            g = torch.tanh(self.gate_g(v))
            o = torch.sigmoid(self.gate_o(v))
        else:
            # PennyLane TorchLayer requires per-sample loop in eager mode
            f_list, i_list, g_list, o_list = [], [], [], []
            for k in range(v.shape[0]):
                vk = v[k]
                f_list.append(self.vqc_f(vk))
                i_list.append(self.vqc_i(vk))
                g_list.append(self.vqc_g(vk))
                o_list.append(self.vqc_o(vk))
            f = torch.sigmoid(torch.stack(f_list))
            i = torch.sigmoid(torch.stack(i_list))
            g = torch.tanh(torch.stack(g_list))
            o = torch.sigmoid(torch.stack(o_list))

        c_new = f * c_prev + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class QLSTM(nn.Module):
    def __init__(self, in_features: int = 4, n_qubits: int = N_QUBITS,
                 n_vqc_blocks: int = 6, noise_p: float = 0.0):
        super().__init__()
        self.cell = QLSTMCell(in_features, n_qubits, n_vqc_blocks, noise_p)
        self.head = nn.Linear(n_qubits, 1)
        self.n_qubits = n_qubits

    def forward(self, x):  # (B, T, F)
        B, T, _ = x.shape
        h = x.new_zeros(B, self.n_qubits)
        c = x.new_zeros(B, self.n_qubits)
        for t in range(T):
            h, c = self.cell(x[:, t], (h, c))
        return self.head(h).squeeze(-1)


if __name__ == "__main__":
    # parameter count sanity check
    for k in [0, 2, 4, 6]:
        m = QLSTM(in_features=4, n_vqc_blocks=k)
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"QLSTM n_blocks={k:d}: {n} trainable params")
