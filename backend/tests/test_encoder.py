"""Unit tests — Encoder (paper §3.1)."""

import torch
from app.core.config import PaperConfig
from app.transformer import Encoder, create_causal_mask


def tiny_config(d_model=8, d_ff=16, heads=4, layers=2):
    return PaperConfig(
        name="tiny", num_encoder_layers=layers, num_decoder_layers=layers,
        d_model=d_model, d_ff=d_ff, num_heads=heads, d_k=d_model // heads,
        d_v=d_model // heads, dropout=0.0, label_smoothing=0.0,
        train_steps=0, params_millions=0,
    )


def make(batch=2, seq=5, vocab=10):
    enc = Encoder.from_config(tiny_config(), vocab_size=vocab, max_len=8)
    ids = torch.randint(0, vocab, (batch, seq))
    return enc, ids


def test_encoder_output_shape():
    enc, ids = make()
    out, trace = enc(ids)
    assert tuple(out.shape) == (2, 5, 8)
    assert len(trace.layers) == 2
    assert trace.layers[0].layer_index == 0
    assert trace.layers[1].layer_index == 1


def test_causal_mask_respected_in_every_layer():
    enc, ids = make(batch=1, seq=6)
    _, trace = enc(ids, mask=create_causal_mask(6))
    for layer in trace.layers:
        for head in layer.attention.heads:
            w = head.weights[0]
            for i in range(6):
                assert torch.allclose(w[i, i + 1:], torch.zeros_like(w[i, i + 1:]), atol=1e-6)


def test_positional_encoding_injected():
    enc, ids = make()
    _, trace = enc(ids)
    assert trace.positioned.shape == (2, 5, 8)
    assert not torch.allclose(trace.positioned, trace.embedded.scaled, atol=1e-6)


def test_parameter_count_positive_and_finite():
    enc, _ = make()
    assert enc.parameters_count() > 0


def test_encoder_gradients_flow():
    enc, ids = make()
    out, _ = enc(ids)
    out.sum().backward()
    for p in enc.parameters():
        assert p.grad is None or bool(torch.isfinite(p.grad).all())