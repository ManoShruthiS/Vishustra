"""PHASE 1 — Linear layer: our implementation, not torch.nn.Linear.

The Transformer uses linear projections everywhere: Q, K, V projections
(paper §3.2.2), the output projection W^O, the position-wise FFN
(paper §3.3, eq. 2), and the pre-softmax linear (§3.4).

We implement it with explicit weight and bias tensors so the laboratory
can inspect and modify them directly (Transformer Surgery, plan.md §12).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from . import tensor_ops


class Linear(nn.Module):
    """``y = x W^T + b`` — the position-wise linear transformation."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        scale = (2.0 / in_features) ** 0.5
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=scale)
        self.bias: nn.Parameter | None = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: Tensor) -> Tensor:
        out = tensor_ops.matmul(x, self.weight.transpose(-1, -2))
        if self.bias is not None:
            out = out + self.bias
        return out

    def parameters_count(self) -> int:
        n = self.weight.numel()
        if self.bias is not None:
            n += self.bias.numel()
        return n