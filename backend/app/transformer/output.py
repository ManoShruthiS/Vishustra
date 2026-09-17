"""PHASE 4 — Output projection (paper §3.4).

The decoder output goes through a linear transformation to vocabulary
logits, then softmax to token probabilities. The paper optionally shares
the weight matrix of the pre-softmax linear with the embedding layers
(§3.4) — we support tying it to the decoder embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from .linear import Linear
from .softmax import log_softmax


@dataclass
class OutputTrace:
    x: Tensor
    logits: Tensor
    log_probs: Tensor

    def steps(self) -> list[dict[str, Any]]:
        return [
            {"step": "linear", "label": "pre-softmax linear (d_model -> vocab)", "value": self.logits},
            {"step": "softmax", "label": "log-softmax", "value": self.log_probs},
        ]


class OutputProjection(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, tied_weight: nn.Parameter | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linear = Linear(d_model, vocab_size)
        if tied_weight is not None:
            assert tuple(tied_weight.shape) == (vocab_size, d_model), "tied weight shape mismatch"
            self.linear.weight = tied_weight

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, OutputTrace]:
        logits = self.linear(x)
        log_probs = log_softmax(logits)
        trace = OutputTrace(x, logits, log_probs)
        return logits, log_probs, trace

    def parameters_count(self) -> int:
        return self.linear.parameters_count()