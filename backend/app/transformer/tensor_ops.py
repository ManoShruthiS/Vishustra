"""PHASE 1 — Tensor operations laboratory.

Vectors, matrices, and the operations that attention is built from:
matmul, transpose chaining, and the per-dimension primitives the
visualizer needs to render "Show me the calculation" (plan.md §40).
"""

from __future__ import annotations

import torch
from torch import Tensor


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiply: ``a @ b``, the operation behind QK^T (paper eq. 1)."""
    return torch.matmul(a, b)


def transpose_last_two(x: Tensor) -> Tensor:
    """Transpose the last two dimensions (e.g. K -> K^T in eq. 1)."""
    return torch.transpose(x, -1, -2)


def scale(scores: Tensor, d_k: int) -> Tensor:
    """Scale scores by 1/sqrt(d_k) (paper §3.2.1 eq. 1)."""
    return scores / (d_k**0.5)


def qkt(q: Tensor, k: Tensor) -> Tensor:
    """Q K^T — the compatibility scores before scaling and softmax."""
    return matmul(q, transpose_last_two(k))