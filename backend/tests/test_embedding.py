"""Unit tests — Token Embedding (paper §3.4)."""

import torch
from app.transformer import TokenEmbedding


def test_lookup_shape():
    emb = TokenEmbedding(vocab_size=10, d_model=8)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out, trace = emb(ids)
    assert tuple(out.shape) == (2, 3, 8)
    assert tuple(trace.vectors.shape) == (2, 3, 8)


def test_lookup_returns_correct_rows():
    emb = TokenEmbedding(vocab_size=4, d_model=2)
    with torch.no_grad():
        emb.weight.copy_(torch.tensor([[0.0, 0.0], [1.0, 11.0], [2.0, 22.0], [3.0, 33.0]]))
    ids = torch.tensor([[2, 1, 3]])
    out, _ = emb(ids)
    assert torch.allclose(out[0, 0], torch.tensor([2.0, 22.0]) * (2**0.5))
    assert torch.allclose(out[0, 1], torch.tensor([1.0, 11.0]) * (2**0.5))


def test_sqrt_d_model_scaling():
    emb = TokenEmbedding(vocab_size=6, d_model=9)
    ids = torch.tensor([[0, 1]])
    out, trace = emb(ids)
    assert torch.allclose(out, trace.vectors * (9**0.5), atol=1e-6)


def test_no_scale_option():
    emb = TokenEmbedding(vocab_size=6, d_model=9, multiply_sqrt=False)
    ids = torch.tensor([[0, 1]])
    out, trace = emb(ids)
    assert torch.allclose(out, trace.vectors, atol=1e-6)


def test_gradients_flow_to_table():
    emb = TokenEmbedding(5, 4)
    ids = torch.tensor([[0, 2]])
    out, _ = emb(ids)
    out.sum().backward()
    assert emb.weight.grad is not None
    assert bool(torch.isfinite(emb.weight.grad).all())