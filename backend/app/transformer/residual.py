"""PHASE 4 — Residual connection + Add & Norm (paper §3.1).

The paper employs a residual connection around each sub-layer, followed
by layer normalization:  y = LayerNorm(x + Dropout(SubLayer(x)))

Dropout is applied to the sub-layer output before it is added to the
residual (paper §5.4: "We apply dropout to the output of each sub-layer,
before it is added to the residual... throughout the network").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from .layer_norm import LayerNorm, LayerNormTrace


@dataclass
class AddNormTrace:
    residual: Tensor             # x (untouched identity path)
    sub_output: Tensor           # SubLayer(x)
    sub_dropped: Tensor          # Dropout(SubLayer(x))
    added: Tensor                # x + Dropout(SubLayer(x))
    normalized: Tensor           # LayerNorm(added)
    norm: LayerNormTrace

    def steps(self) -> list[dict[str, Any]]:
        return [
            {"step": "sub_layer", "label": "SubLayer(x)", "value": self.sub_output},
            {"step": "dropout", "label": "Dropout(SubLayer(x))", "value": self.sub_dropped},
            {"step": "add", "label": "x + SubLayer(x)", "value": self.added},
            {"step": "norm", "label": "LayerNorm", "value": self.normalized},
        ]


class AddNorm(nn.Module):
    """Add & Norm wrapper around a sub-layer output (attention, feed-forward).

    The sub-layer itself lives in its owning layer (so parameters are not
    double-registered); ``AddNorm`` applies dropout, the residual add, and
    layer normalization to the sub-layer output.
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model)

    def forward(self, x: Tensor, sub_output: Tensor) -> tuple[Tensor, AddNormTrace]:
        sub_dropped = self.dropout(sub_output)
        added = x + sub_dropped
        normalized, norm_trace = self.norm(added)
        trace = AddNormTrace(x, sub_output, sub_dropped, added, normalized, norm_trace)
        return normalized, trace