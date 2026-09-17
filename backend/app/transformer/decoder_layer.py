"""PHASE 4 — Decoder layer (paper §3.1).

Each decoder layer contains three sub-layers:
  1. masked self-attention (no attending to future positions, §3.2.3)
  2. encoder-decoder (cross) attention, querying the encoder output
  3. position-wise feed-forward network
each followed by Add & Norm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from .feed_forward import FeedForward, FFNTrace
from .multi_head_attention import MultiHeadAttention
from .residual import AddNorm, AddNormTrace


@dataclass
class DecoderLayerTrace:
    layer_index: int
    self_attention: Any      # MultiHeadTrace (masked)
    cross_attention: Any     # MultiHeadTrace (over encoder memory)
    addnorm1: AddNormTrace
    addnorm2: AddNormTrace
    ffn: FFNTrace
    addnorm3: AddNormTrace
    output: Tensor

    def steps(self) -> list[dict[str, Any]]:
        return [
            {"step": "self_attn", "label": f"masked self-attention [layer {self.layer_index}]", "value": self.self_attention.output},
            {"step": "addnorm1", "label": "Add & Norm", "value": self.addnorm1.normalized},
            {"step": "cross_attn", "label": "encoder-decoder attention", "value": self.cross_attention.output},
            {"step": "addnorm2", "label": "Add & Norm", "value": self.addnorm2.normalized},
            {"step": "ffn", "label": "Feed-Forward (eq. 2)", "value": self.ffn.output},
            {"step": "addnorm3", "label": "Add & Norm", "value": self.addnorm3.normalized},
            {"step": "output", "label": "decoder layer output", "value": self.output},
        ]


class DecoderLayer(nn.Module):
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
        self.cross_attn = MultiHeadAttention(d_model, num_heads, d_k, d_v)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.addnorm3 = AddNorm(d_model, dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> tuple[Tensor, DecoderLayerTrace]:
        self_out, self_trace = self.self_attn(x, x, x, mask=tgt_mask, layer_index=self.layer_index)
        h1, an1 = self.addnorm1(x, self_out)
        cross_out, cross_trace = self.cross_attn(h1, memory, memory, mask=src_mask, layer_index=self.layer_index)
        h2, an2 = self.addnorm2(h1, cross_out)
        ffn_out, ffn_trace = self.ffn(h2)
        h3, an3 = self.addnorm3(h2, ffn_out)
        trace = DecoderLayerTrace(self.layer_index, self_trace, cross_trace, an1, an2, ffn_trace, an3, h3)
        return h3, trace