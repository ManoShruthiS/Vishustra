"""Transformer engine — our implementation from the paper's mathematics.

Phase order (plan.md §61-64):
  1. Foundation: tensor ops, Linear, Softmax.
  2. Attention: scaled dot-product (eq. 1).
  3. Multi-head attention (eq. 2).
  4. Blocks: embedding, positional encoding, add+norm, feed-forward,
     encoder, decoder, output, and the full Transformer.
"""

from app.transformer.attention import (
    ScaledDotProductAttention,
    create_causal_mask,
    create_padding_mask,
)
from app.transformer.decoder import Decoder, DecoderTrace
from app.transformer.decoder_layer import DecoderLayer, DecoderLayerTrace
from app.transformer.embedding import TokenEmbedding
from app.transformer.encoder import Encoder, EncoderTrace
from app.transformer.encoder_layer import EncoderLayer, EncoderLayerTrace
from app.transformer.feed_forward import FeedForward, FFNTrace
from app.transformer.layer_norm import LayerNorm, LayerNormTrace
from app.transformer.linear import Linear
from app.transformer.multi_head_attention import (
    MultiHeadAttention,
    MultiHeadTrace,
    merge_heads,
    split_into_heads,
)
from app.transformer.output import OutputProjection, OutputTrace
from app.transformer.positional_encoding import (
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
    create_positional_encoding,
    positional_encoding,
)
from app.transformer.residual import AddNorm, AddNormTrace
from app.transformer.softmax import log_softmax, softmax
from app.transformer.tensor_ops import matmul, qkt, scale, transpose_last_two
from app.transformer.transformer import Transformer, TransformerTrace

__all__ = [
    "AddNorm",
    "AddNormTrace",
    "Decoder",
    "DecoderLayer",
    "DecoderLayerTrace",
    "DecoderTrace",
    "Encoder",
    "EncoderLayer",
    "EncoderLayerTrace",
    "EncoderTrace",
    "FFNTrace",
    "FeedForward",
    "LayerNorm",
    "LayerNormTrace",
    "LearnedPositionalEncoding",
    "Linear",
    "MultiHeadAttention",
    "MultiHeadTrace",
    "OutputProjection",
    "OutputTrace",
    "ScaledDotProductAttention",
    "SinusoidalPositionalEncoding",
    "TokenEmbedding",
    "Transformer",
    "TransformerTrace",
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