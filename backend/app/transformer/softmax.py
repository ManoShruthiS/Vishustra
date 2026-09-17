"""PHASE 1 — Softmax: our implementation, not torch.nn.

The softmax is the final nonlinearity of scaled dot-product attention
(paper eq. 1). We implement it directly so the laboratory can expose every
intermediate: raw logits, max-subtraction, exponentiation, normalization.

Numerical stability: we subtract the per-row max before exponentiating,
which is required for large scores (see paper §3.2.1 motivation).
"""

from __future__ import annotations

import torch
from torch import Tensor


def softmax(
    logits: Tensor,
    mask: Tensor | None = None,
    dim: int = -1,
) -> Tensor:
    """Softmax over ``dim`` with optional additive mask.

    ``mask`` is added to ``logits`` before normalization; use ``-inf`` for
    positions that must be excluded (paper §3.2.3 masks illegal connections
    by setting pre-softmax values to ``-inf``).
    """
    if mask is not None:
        logits = logits + mask
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    total = exp.sum(dim=dim, keepdim=True)
    return exp / total


def log_softmax(logits: Tensor, dim: int = -1) -> Tensor:
    """Log-softmax used inside attention metrics and experiments."""
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    logs = shifted - torch.log(torch.exp(shifted).sum(dim=dim, keepdim=True))
    return logs