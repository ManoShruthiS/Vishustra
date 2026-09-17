"""PHASE 4 — Encoder layer (paper §3.1).

Each encoder layer = Multi-Head Attention + Add & Norm, then a
position-wise Feed-Forward Network + Add & Norm. The paper stacks N of
these on top of the (embedding + positional encoding) input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from .feed_forward import FeedForward, FFNTrace
from .multi_head_attention import MultiHeadAttention
from .residual import AddNorm, AddNormTrace


@dataclass
class EncoderLayerTrace:
    layer_index: int
    attention: Any           # MultiHeadTrace
    addnorm1: AddNormTrace
    ffn: FFNTrace
    addnorm2: AddNormTrace
    output: Tensor

    def steps(self) -> list[dict[str, Any]]:
        return [
            {"step": "mha", "label": f"Multi-Head Attention [layer {self.layer_index}]", "value": self.attention.output},
            {"step": "addnorm1", "label": "Add & Norm", "value": self.addnorm1.normalized},
            {"step": "ffn", "label": "Feed-Forward (eq. 2)", "value": self.ffn.output},
            {"step": "addnorm2", "label": "Add & Norm", "value": self.addnorm2.normalized},
            {"step": "output", "label": "encoder layer output", "value": self.output},
        ]


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        dropout: float,
        layer_index: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.layer_index = layer_index
        self.self_attn = MultiHeadAttention(d_model, num_heads, d_k, d_v)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, EncoderLayerTrace]:
        attn_out, attn_trace = self.self_attn(x, x, x, mask=mask, layer_index=self.layer_index)
        h1, an1 = self.addnorm1(x, attn_out)
        ffn_out, ffn_trace = self.ffn(h1)
        h2, an2 = self.addnorm2(h1, ffn_out)
        trace = EncoderLayerTrace(self.layer_index, attn_trace, an1, ffn_trace, an2, h2)
        return h2, trace