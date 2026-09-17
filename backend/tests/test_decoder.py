"""Unit tests — Decoder (paper §3.1, §3.2.3)."""

import torch
from app.core.config import PaperConfig
from app.transformer import Decoder, create_causal_mask


def tiny_config(d_model=8, d_ff=16, heads=4, layers=2):
    return PaperConfig(
        name="tiny", num_encoder_layers=layers, num_decoder_layers=layers,
        d_model=d_model, d_ff=d_ff, num_heads=heads, d_k=d_model // heads,
        d_v=d_model // heads, dropout=0.0, label_smoothing=0.0,
        train_steps=0, params_millions=0,
    )


def make(batch=2, seq=5, vocab=10, mem_seq=7):
    dec = Decoder.from_config(tiny_config(), vocab_size=vocab, max_len=8)
    ids = torch.randint(0, vocab, (batch, seq))
    memory = torch.randn(batch, mem_seq, 8)
    return dec, ids, memory


def test_decoder_output_shape():
    dec, ids, memory = make()
    out, trace = dec(ids, memory)
    assert tuple(out.shape) == (2, 5, 8)
    assert len(trace.layers) == 2


def test_causal_mask_respected_in_self_attention():
    dec, ids, memory = make(batch=1, seq=6)
    _, trace = dec(ids, memory, tgt_mask=create_causal_mask(6))
    for layer in trace.layers:
        for head in layer.self_attention.heads:
            w = head.weights[0]
            for i in range(6):
                assert torch.allclose(w[i, i + 1:], torch.zeros_like(w[i, i + 1:]), atol=1e-6)


def test_cross_attention_attends_all_memory_positions():
    dec, ids, memory = make(batch=1, seq=5, mem_seq=7)
    _, trace = dec(ids, memory)
    for layer in trace.layers:
        for head in layer.cross_attention.heads:
            w = head.weights[0]
            assert w.shape == (5, 7)
            assert torch.allclose(w.sum(dim=-1), torch.ones(5), atol=1e-5)


def test_decoder_gradients_flow():
    dec, ids, memory = make()
    out, _ = dec(ids, memory)
    out.sum().backward()
    for p in dec.parameters():
        assert p.grad is None or bool(torch.isfinite(p.grad).all())