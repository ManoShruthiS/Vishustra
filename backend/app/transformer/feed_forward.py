"""PHASE 4 — Position-wise Feed-Forward Network (paper §3.3, eq. 2).

    FFN(x) = max(0, x W1 + b1) W2 + b2

Applied identically to every position, with different parameters per
layer. The paper uses d_ff = 2048 for the base model (d_model = 512).
ReLU activation is evaluated in place so the pre-activation signal and
the gate are inspectable (plan.md §40).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .linear import Linear


@dataclass
class FFNTrace:
    x: Tensor
    pre_activation: Tensor   # x W1 + b1
    activated: Tensor        # max(0, ·)
    output: Tensor           # activated W2 + b2

    def steps(self) -> list[dict[str, Any]]:
        return [
            {"step": "expand", "label": "x W1 + b1  (d_model → d_ff)", "value": self.pre_activation},
            {"step": "relu", "label": "max(0, ·)", "value": self.activated},
            {"step": "contract", "label": "(·) W2 + b2  (d_ff → d_model)", "value": self.output},
        ]


class FeedForward(nn.Module):
    """Position-wise FFN with paper-configurable widths (§3.3, eq. 2)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> tuple[Tensor, FFNTrace]:
        pre = self.w1(x)
        activated = torch.relu(pre)
        dropped = self.dropout(activated)
        output = self.w2(dropped)
        trace = FFNTrace(x, pre, activated, output)
        return output, trace

    def parameters_count(self) -> int:
        return self.w1.parameters_count() + self.w2.parameters_count()