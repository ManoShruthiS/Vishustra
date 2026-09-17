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
from app.transformer.multi_head_attention import (
    MultiHeadAttention,
    MultiHeadTrace,
    merge_heads,
    split_into_heads,
)
from app.transformer.positional_encoding import (
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
    create_positional_encoding,
    positional_encoding,
)
from app.transformer.softmax import log_softmax, softmax
from app.transformer.tensor_ops import matmul, qkt, scale, transpose_last_two

__all__ = [
    "LearnedPositionalEncoding",
    "Linear",
    "MultiHeadAttention",
    "MultiHeadTrace",
    "ScaledDotProductAttention",
    "SinusoidalPositionalEncoding",
    "create_causal_mask",
    "create_padding_mask",
    "create_positional_encoding",
    "log_softmax",
    "matmul",
    "merge_heads",
    "positional_encoding",
    "qkt",
    "scale",
    "softmax",
    "split_into_heads",
    "transpose_last_two",
]