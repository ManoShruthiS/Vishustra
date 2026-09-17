"""PHASE 4 — Token Embedding (paper §3.4).

Input/output tokens are converted to vectors of dimension d_model by a
learned embedding table. The paper multiplies the learned embeddings by
sqrt(d_model) (§3.4) and shares the embedding weights between the
encoder, the decoder, and the pre-softmax linear (also here, §3.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class EmbeddingTrace:
    token_ids: Tensor       # (batch, seq)
    vectors: Tensor         # (batch, seq, d_model) raw lookup
    scaled: Tensor          # vectors * sqrt(d_model)

    def steps(self) -> list[dict[str, Any]]:
        return [
            {"step": "lookup", "label": f"embedding[id] (table {self.vectors.shape[-1]}×{self.vectors.shape[-1]})", "value": self.vectors},
            {"step": "scale", "label": "· √d_model", "value": self.scaled},
        ]


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, multiply_sqrt: bool = True) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.multiply_sqrt = multiply_sqrt
        scale = (2.0 / d_model) ** 0.5
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.normal_(self.weight, mean=0.0, std=scale)

    def forward(self, token_ids: Tensor) -> tuple[Tensor, EmbeddingTrace]:
        vectors = self.weight.index_select(0, token_ids.reshape(-1)).reshape(*token_ids.shape, self.d_model)
        scaled = vectors * (self.d_model**0.5) if self.multiply_sqrt else vectors
        trace = EmbeddingTrace(token_ids, vectors, scaled)
        return scaled, trace

    def parameters_count(self) -> int:
        return self.weight.numel()