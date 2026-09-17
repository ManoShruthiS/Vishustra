"""PHASE 4 — Layer Normalization (paper §3.1).

    LayerNorm(x) = γ · (x − μ) / σ + β       μ, σ over the d_model axis

We implement it directly (not torch.nn.LayerNorm) so every statistic —
mean, variance, normalized signal, gain, bias — stays inspectable for
the "Show Me the Calculation" renderer (plan.md §40).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class LayerNormTrace:
    x: Tensor
    mean: Tensor
    variance: Tensor
    normalized: Tensor
    output: Tensor

    def steps(self) -> list[dict[str, Any]]:
        return [
            {"step": "mean", "label": "μ (over d_model)", "value": self.mean},
            {"step": "variance", "label": "σ² (over d_model)", "value": self.variance},
            {"step": "normalize", "label": "(x − μ) / √(σ² + ε)", "value": self.normalized},
            {"step": "scale_shift", "label": "γ · x̂ + β", "value": self.output},
        ]


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor) -> tuple[Tensor, LayerNormTrace]:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        output = normalized * self.gamma + self.beta
        trace = LayerNormTrace(x, mean, variance, normalized, output)
        return output, trace

    def parameters_count(self) -> int:
        return 2 * self.d_model