"""Unit tests — the full Transformer (paper §3.1, §3.4)."""

import pytest
import torch
from app.core.config import PaperConfig
from app.transformer import Transformer, create_causal_mask


def tiny_config(d_model=8, d_ff=16, heads=4, layers=1):
    return PaperConfig(
        name="tiny", num_encoder_layers=layers, num_decoder_layers=layers,
        d_model=d_model, d_ff=d_ff, num_heads=heads, d_k=d_model // heads,
        d_v=d_model // heads, dropout=0.0, label_smoothing=0.0,
        train_steps=0, params_millions=0,
    )


def make(config=None, src_vocab=10, tgt_vocab=10, **kw):
    config = config or tiny_config()
    model = Transformer(config=config, src_vocab_size=src_vocab, tgt_vocab_size=tgt_vocab, max_len=8, **kw)
    src = torch.randint(0, src_vocab, (2, 5))
    tgt = torch.randint(0, tgt_vocab, (2, 5))
    return model, src, tgt


def test_forward_shapes():
    model, src, tgt = make()
    logits, log_probs, trace = model(src, tgt)
    assert tuple(logits.shape) == (2, 5, 10)
    assert tuple(log_probs.shape) == (2, 5, 10)
    assert trace.encoder is not None and trace.decoder is not None and trace.output is not None


def test_log_probs_are_normalised():
    model, src, tgt = make()
    _, log_probs, _ = model(src, tgt)
    assert torch.allclose(torch.exp(log_probs).sum(-1), torch.ones(2, 5), atol=1e-5)


def test_decoder_output_is_tied_to_embedding():
    model, _, _ = make()
    tied = model.output.linear.weight
    assert tied is model.decoder.embedding.weight


def test_same_vocab_ties_all_three_tables():
    model, _, _ = make()
    assert model.encoder.embedding.weight is model.decoder.embedding.weight
    assert model.output.linear.weight is model.decoder.embedding.weight


def test_tying_requires_equal_vocab():
    with pytest.raises(ValueError):
        Transformer(config=tiny_config(), src_vocab_size=10, tgt_vocab_size=12, tie_encoder_decoder_embeddings=True)


def test_causal_mask_blocks_future_in_decoder_self_attention():
    model, src, tgt = make()
    _, _, trace = model(src, tgt, tgt_mask=create_causal_mask(5))
    for layer in trace.decoder.layers:
        for head in layer.self_attention.heads:
            w = head.weights[0]
            for i in range(5):
                assert torch.allclose(w[i, i + 1:], torch.zeros_like(w[i, i + 1:]), atol=1e-6)


def test_parameter_count():
    model, _, _ = make(tiny_config(layers=1), src_vocab=10, tgt_vocab=10)
    assert model.parameters_count() == sum(p.numel() for p in model.parameters())
    assert model.parameters_count() > 0


def test_trace_steps_cover_full_pipeline():
    model, src, tgt = make()
    _, _, trace = model(src, tgt)
    labels = [s["label"] for s in trace.steps()]
    assert "Token Embedding" in labels
    assert "+ Positional Encoding" in labels
    assert "pre-softmax linear (d_model -> vocab)" in labels
    assert "log-softmax" in labels


def test_backprop_flows_through_full_model():
    model, src, tgt = make(config=tiny_config(d_model=6, d_ff=12, heads=3, layers=2),
                           src_vocab=7, tgt_vocab=7)
    logits, _, _ = model(src, tgt)
    logits.sum().backward()
    grads = [p.grad for p in model.parameters()]
    assert any(g is not None for g in grads)
    for g in grads:
        if g is not None:
            assert bool(torch.isfinite(g).all())