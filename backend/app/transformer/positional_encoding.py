"""PHASE 3 — Positional encoding (paper §3.5, eq. 3/4).

Because the Transformer contains no recurrence and no convolution, it must
inject information about the position of each token. The paper adds a
sinusoidal encoding to the input embeddings:

    PE_(pos, 2i)   = sin(pos / 10000 ** (2i / d_model))
    PE_(pos, 2i+1) = cos(pos / 10000 ** (2i / d_model))

LAB 03 (Positional Encoding Laboratory) compares this sinusoidal encoding
against a learned alternative, so both are implemented here.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def positional_encoding(max_len: int, d_model: int, base: float = 10000.0) -> Tensor:
    """Sinusoidal position-encoding table of shape (max_len, d_model), §3.5."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(max_len).unsqueeze(1).to(torch.float32)          # (max_len, 1)
    even = torch.arange(0, d_model, 2).to(torch.float32)                     # index 2i
    angle = position / (base ** (even / d_model))                            # (max_len, n_even)
    pe[:, 0::2] = torch.sin(angle)
    if d_model > 1:
        pe[:, 1::2] = torch.cos(angle[:, : pe[:, 1::2].shape[1]])
    return pe


class SinusoidalPositionalEncoding(nn.Module):
    """PE added to input embeddings (eq. 3/4). None of it is learned."""

    def __init__(self, max_len: int, d_model: int, base: float = 10000.0) -> None:
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.base = base
        self.register_buffer("pe", positional_encoding(max_len, d_model, base))

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[-2]
        assert seq_len <= self.max_len, "sequence longer than the PE table"
        return x + self.pe[:seq_len]

    def encoding_for(self, pos: int) -> Tensor:
        return self.pe[pos].clone()


class LearnedPositionalEncoding(nn.Module):
    """Learned per-position vectors (LAB 03 toggle: 'Sinusoidal | Learned')."""

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.pe = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[-2]
        assert seq_len <= self.max_len, "sequence longer than the learnable table"
        return x + self.pe[:seq_len]


def create_positional_encoding(
    kind: str = "sinusoidal",
    max_len: int = 512,
    d_model: int = 512,
    base: float = 10000.0,
) -> nn.Module:
    """Factory for ``kind in {"sinusoidal", "learned"}`` (LAB 03)."""
    if kind == "sinusoidal":
        return SinusoidalPositionalEncoding(max_len, d_model, base)
    if kind == "learned":
        return LearnedPositionalEncoding(max_len, d_model)
    raise ValueError(f"unknown positional encoding kind: {kind!r}")