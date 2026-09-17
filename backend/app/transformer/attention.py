"""PHASE 2 — Scaled Dot-Product Attention (paper §3.2.1, eq. 1).

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

This is the first major milestone of plan.md (§62): implement Q/K/V,
QK^T, scaling, softmax, output — then visualize it.

The module exposes every intermediate via its `trace` so the laboratory
can render "Show me the calculation" (plan.md §40) step by step:

    Q -> K^T -> scores -> scaled -> softmax -> weights -> output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from . import tensor_ops
from .softmax import softmax


@dataclass
class AttentionTrace:
    """Every intermediate of a scaled dot-product attention call."""

    q: Tensor
    k: Tensor
    v: Tensor
    scores: Tensor          # Q K^T
    d_k: int
    scaler: float           # 1 / sqrt(d_k)
    scaled: Tensor          # scores / sqrt(d_k)
    mask: Tensor | None
    masked: Tensor          # scaled (+ mask applied)
    weights: Tensor         # softmax(masked)
    output: Tensor          # weights @ V
    head_index: int = -1
    layer_index: int = -1
    token_ids: list[int] = field(default_factory=list)

    def steps(self) -> list[dict[str, Any]]:
        """Ordered calculation steps for the 'show me' renderer (plan.md §40)."""
        return [
            {"step": "query", "label": "Q", "value": self.q},
            {"step": "key_transpose", "label": "K^T", "value": tensor_ops.transpose_last_two(self.k)},
            {"step": "scores", "label": "QK^T", "value": self.scores},
            {"step": "scale", "label": f"1/sqrt(d_k={self.d_k}) = {self.scaler:.4f}", "value": self.scaled},
            {"step": "mask", "label": "mask", "value": self.masked},
            {"step": "softmax", "label": "softmax", "value": self.weights},
            {"step": "output", "label": "weights @ V", "value": self.output},
        ]


class ScaledDotProductAttention(nn.Module):
    """The attention equation implemented directly (eq. 1, paper §3.2.1)."""

    def __init__(self, d_k: int = 64) -> None:
        super().__init__()
        self.d_k = d_k
        self.scaler = d_k ** -0.5

    def forward(
        self,
        query: Tensor,      # (..., seq, d_k)
        key: Tensor,        # (..., seq, d_k)
        value: Tensor,      # (..., seq, d_v)
        mask: Tensor | None = None,
        head_index: int = -1,
        layer_index: int = -1,
        token_ids: list[int] | None = None,
    ) -> tuple[Tensor, AttentionTrace]:
        scores = tensor_ops.qkt(query, key)
        scaled = tensor_ops.scale(scores, self.d_k)
        masked = scaled if mask is None else scaled + mask
        weights = softmax(masked)
        output = tensor_ops.matmul(weights, value)
        trace = AttentionTrace(
            q=query, k=key, v=value, scores=scores, d_k=self.d_k,
            scaler=self.scaler, scaled=scaled, mask=mask, masked=masked,
            weights=weights, output=output, head_index=head_index,
            layer_index=layer_index, token_ids=list(token_ids or []),
        )
        return output, trace


def create_causal_mask(seq_len: int) -> Tensor:
    """Upper-triangular inf mask: position i cannot attend to j > i (§3.2.3).

    Returns a tensor of the same shape conventions used in
    ``ScaledDotProductAttention``: add to scaled scores, where 0 admits
    attention and -inf forbids it.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
    return mask


def create_padding_mask(valid_lengths: Tensor, seq_len: int) -> Tensor:
    """Mask out positions beyond each row's valid length (−inf)."""
    indices = torch.arange(seq_len, device=valid_lengths.device)
    masked_bool = indices[None, :] >= valid_lengths[:, None]
    return torch.where(masked_bool, torch.full_like(indices[None, :], float("-inf"), dtype=torch.float32), torch.zeros(1, seq_len, dtype=torch.float32))