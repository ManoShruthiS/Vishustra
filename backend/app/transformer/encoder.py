"""PHASE 4 — Encoder (paper §3.1, Diagram).

Input embeddings are added to the positional encoding, then passed
through N identical encoder layers. Pure self-attention: every position
sees every position (mask only hides padding).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from app.core.config import PaperConfig
from app.transformer.embedding import TokenEmbedding
from app.transformer.encoder_layer import EncoderLayer, EncoderLayerTrace
from app.transformer.positional_encoding import create_positional_encoding


@dataclass
class EncoderTrace:
    token_ids: Tensor
    embedded: Any            # EmbeddingTrace
    positioned: Tensor       # embedded + PE
    layers: list[EncoderLayerTrace]
    output: Tensor

    def steps(self) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = [{"step": "embed", "label": "Token Embedding", "value": self.embedded.scaled}]
        steps.append({"step": "pe", "label": "+ Positional Encoding", "value": self.positioned})
        for layer in self.layers:
            steps.append({"step": f"layer_{layer.layer_index}", "label": f"EncoderLayer {layer.layer_index}", "value": layer.output})
        steps.append({"step": "output", "label": "encoder output", "value": self.output})
        return steps


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        num_layers: int,
        dropout: float,
        vocab_size: int,
        max_len: int = 512,
        pe_kind: str = "sinusoidal",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.pe = create_positional_encoding(pe_kind, max_len, d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_ff, num_heads, d_k, d_v, dropout, layer_index=i) for i in range(num_layers)]
        )

    @classmethod
    def from_config(cls, config: PaperConfig, vocab_size: int, max_len: int = 512, pe_kind: str = "sinusoidal") -> Encoder:
        return cls(
            config.d_model, config.d_ff, config.num_heads, config.d_k, config.d_v,
            config.num_encoder_layers, config.dropout, vocab_size, max_len, pe_kind,
        )

    def forward(self, token_ids: Tensor, mask: Tensor | None = None) -> tuple[Tensor, EncoderTrace]:
        x, emb_trace = self.embedding(token_ids)
        positioned = self.pe(x)
        layer_traces: list[EncoderLayerTrace] = []
        h = positioned
        for layer in self.layers:
            h, layer_trace = layer(h, mask=mask)
            layer_traces.append(layer_trace)
        trace = EncoderTrace(token_ids, emb_trace, positioned, layer_traces, h)
        return h, trace

    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters())