"""PHASE 4 — The Transformer (paper §3.1, Diagram).

Assembles encoder + decoder + output projection from a paper-locked
``PaperConfig`` (Table 3). Weight sharing (§3.4): the pre-softmax linear
shares with the decoder embedding, and — when the vocabularies match —
the encoder, decoder, and output share one embedding table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from app.core.config import PaperConfig
from app.transformer.decoder import Decoder, DecoderTrace
from app.transformer.encoder import Encoder, EncoderTrace
from app.transformer.output import OutputProjection, OutputTrace


@dataclass
class TransformerTrace:
    encoder: EncoderTrace
    decoder: DecoderTrace
    output: OutputTrace
    logits: Tensor
    log_probs: Tensor

    def steps(self) -> list[dict[str, Any]]:
        return self.encoder.steps() + self.decoder.steps() + self.output.steps()


class Transformer(nn.Module):
    def __init__(
        self,
        config: PaperConfig,
        src_vocab_size: int,
        tgt_vocab_size: int,
        max_len: int = 512,
        pe_kind: str = "sinusoidal",
        tie_encoder_decoder_embeddings: bool = True,
        tie_decoder_output: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        if tie_encoder_decoder_embeddings and src_vocab_size != tgt_vocab_size:
            raise ValueError("encoder/decoder embedding sharing requires equal vocab sizes")
        self.encoder = Encoder.from_config(config, src_vocab_size, max_len, pe_kind)
        self.decoder = Decoder.from_config(config, tgt_vocab_size, max_len, pe_kind)
        if tie_encoder_decoder_embeddings:
            self.decoder.embedding.weight = self.encoder.embedding.weight
        tied_weight = self.decoder.embedding.weight if tie_decoder_output else None
        self.output = OutputProjection(config.d_model, tgt_vocab_size, tied_weight=tied_weight)

    @classmethod
    def from_config(cls, config: PaperConfig, **kwargs: Any) -> Transformer:
        return cls(config, **kwargs)

    def forward(
        self,
        src_ids: Tensor,
        tgt_ids: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, TransformerTrace]:
        memory, enc_trace = self.encoder(src_ids, src_mask)
        dec_out, dec_trace = self.decoder(tgt_ids, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        logits, log_probs, out_trace = self.output(dec_out)
        trace = TransformerTrace(enc_trace, dec_trace, out_trace, logits, log_probs)
        return logits, log_probs, trace

    def parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters())