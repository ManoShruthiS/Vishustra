"""Transformer engine — our implementation from the paper's mathematics.

Phase order (plan.md §61-64):
  1. Foundation: tensor ops, Linear, Softmax.
  2. Attention: scaled dot-product (eq. 1).
  3. Multi-head attention.
  4. Blocks: embedding, positional encoding, encoder, decoder.
"""

from app.transformer.attention import (
    ScaledDotProductAttention,
    create_causal_mask,
    create_padding_mask,
)
from app.transformer.linear import Linear
from app.transformer.softmax import log_softmax, softmax
from app.transformer.tensor_ops import matmul, qkt, scale, transpose_last_two

__all__ = [
    "ScaledDotProductAttention",
    "create_causal_mask",
    "create_padding_mask",
    "Linear",
    "log_softmax",
    "matmul",
    "qkt",
    "scale",
    "softmax",
    "transpose_last_two",
]