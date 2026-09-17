"""PHASE 3 — Multi-Head Attention (paper §3.2.2, eq. 2).

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

Instead of performing a single attention function with d_model-dim
keys/values/queries, the paper performs several learned projections in
parallel — each head attends in its own subspace (d_k / d_v) — then
concatenates and projects once more with W^O.

Every head's Q/K/V projections and its attention trace are kept as
first-class objects so the laboratory can render LAB 05 (Multi-Head
Attention Studio) and, later, cut individual heads (Transformer Surgery,
plan.md §12).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .attention import AttentionTrace, ScaledDotProductAttention
from .linear import Linear


@dataclass
class MultiHeadTrace:
    """The full bookkeeping of a multi-head attention call."""

    d_model: int
    num_heads: int
    d_k: int
    d_v: int
    heads: list[AttentionTrace]
    concat: Tensor            # [batch, seq, h*d_v]
    output: Tensor            # [batch, seq, d_model]
    layer_index: int = -1

    def steps(self) -> list[dict[str, Any]]:
        """Ordered calculation for the 'show me' renderer (plan.md §40)."""
        steps: list[dict[str, Any]] = []
        for i, head in enumerate(self.heads):
            steps.append({"step": f"head_{i}", "label": f"HEAD {i}", "value": head.output})
            for inner in head.steps():
                steps.append(
                    {**inner, "label": f"h{i} · {inner['label']}"}
                )
        steps.append({"step": "concat", "label": "CONCAT", "value": self.concat})
        steps.append({"step": "project", "label": "W^O (linear)", "value": self.output})
        return steps


class MultiHeadAttention(nn.Module):
    """Parallel attention heads + concat + output projection (eq. 2)."""

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_k: int | None = None,
        d_v: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k if d_k is not None else d_model // num_heads
        self.d_v = d_v if d_v is not None else d_model // num_heads

        self.w_q = nn.ModuleList([Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.w_k = nn.ModuleList([Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.w_v = nn.ModuleList([Linear(d_model, self.d_v) for _ in range(num_heads)])
        self.w_out = Linear(num_heads * self.d_v, d_model)
        self.attention = ScaledDotProductAttention(d_k=self.d_k)

    def forward(
        self,
        query: Tensor,      # (batch, seq, d_model)
        key: Tensor,        # (batch, seq, d_model)
        value: Tensor,      # (batch, seq, d_model)
        mask: Tensor | None = None,
        layer_index: int = -1,
    ) -> tuple[Tensor, MultiHeadTrace]:
        head_outputs: list[Tensor] = []
        head_traces: list[AttentionTrace] = []
        for i in range(self.num_heads):
            q_h = self.w_q[i](query)
            k_h = self.w_k[i](key)
            v_h = self.w_v[i](value)
            out_h, trace_h = self.attention(
                q_h, k_h, v_h,
                mask=mask,
                layer_index=layer_index,
                head_index=i,
            )
            head_outputs.append(out_h)
            head_traces.append(trace_h)

        concat = torch.cat(head_outputs, dim=-1)                      # [b, seq, h*d_v]
        output = self.w_out(concat)                                    # [b, seq, d_model]
        trace = MultiHeadTrace(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_k=self.d_k,
            d_v=self.d_v,
            heads=head_traces,
            concat=concat,
            output=output,
            layer_index=layer_index,
        )
        return output, trace

    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def split_into_heads(x: Tensor, num_heads: int, d_head: int) -> Tensor:
    """Reshape [.., seq, num_heads*d_head] -> [.., seq, num_heads, d_head]."""
    return x.unflatten(-1, (num_heads, d_head))


def merge_heads(x: Tensor) -> Tensor:
    """Collapse [.., seq, num_heads, d_head] -> [.., seq, num_heads*d_head]."""
    return x.flatten(-2)