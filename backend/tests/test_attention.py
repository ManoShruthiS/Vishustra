"""Unit tests — scaled dot-product attention (paper eq. 1)."""

import torch
from torch.autograd import gradcheck

from app.transformer import (
    ScaledDotProductAttention,
    create_causal_mask,
    create_padding_mask,
)


def test_output_shape():
    d_k, d_v, seq = 8, 8, 5
    attn = ScaledDotProductAttention(d_k=d_k)
    q = torch.randn(1, seq, d_k)
    k = torch.randn(1, seq, d_k)
    v = torch.randn(1, seq, d_v)
    out, trace = attn(q, k, v)
    assert tuple(out.shape) == (1, seq, d_v)
    assert tuple(trace.weights.shape) == (1, seq, seq)


def test_weights_rows_sum_to_one():
    attn = ScaledDotProductAttention(d_k=4)
    q = torch.randn(1, 6, 4)
    k = torch.randn(1, 6, 4)
    v = torch.randn(1, 6, 4)
    _, trace = attn(q, k, v)
    row_sums = trace.weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(1, 6), atol=1e-6)


def test_scaling_matches_equation():
    """Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V, computed manually."""
    d_k, seq = 5, 4
    attn = ScaledDotProductAttention(d_k=d_k)
    gen = torch.Generator().manual_seed(0)
    q = torch.randn(1, seq, d_k, generator=gen)
    k = torch.randn(1, seq, d_k, generator=gen)
    v = torch.randn(1, seq, d_k, generator=gen)

    _, trace = attn(q, k, v)
    manual = torch.matmul(q, k.transpose(-1, -2)) / (d_k**0.5)
    manual_w = torch.softmax(manual, dim=-1)
    assert torch.allclose(trace.scores, torch.matmul(q, k.transpose(-1, -2)), atol=1e-6)
    assert torch.allclose(trace.scaled, manual, atol=1e-6)
    assert torch.allclose(trace.weights, manual_w, atol=1e-6)
    assert torch.allclose(trace.output, torch.matmul(manual_w, v), atol=1e-6)


def test_attention_concentrates_on_matching_key():
    """If one key exactly equals the query, the row concentrates on it."""
    d_k, seq = 8, 4
    attn = ScaledDotProductAttention(d_k=d_k)
    target = torch.randn(d_k)
    q = target.unsqueeze(0).unsqueeze(0)
    k = torch.stack([torch.randn(d_k) for _ in range(seq)])
    k[1] = target
    k = k.unsqueeze(0)
    v = torch.randn(1, seq, d_k)
    _, trace = attn(q, k, v)
    weights = trace.weights[0, 0]
    assert int(torch.argmax(weights)) == 1
    assert weights[1] > 2 * weights[weights != weights[1]].max()


def test_causal_mask_blocks_future():
    """Position i must not attend to j > i (§3.2.3)."""
    seq = 5
    attn = ScaledDotProductAttention(d_k=4)
    q = torch.randn(1, seq, 4)
    k = torch.randn(1, seq, 4)
    v = torch.randn(1, seq, 4)
    _, trace = attn(q, k, v, mask=create_causal_mask(seq))
    w = trace.weights[0]
    for i in range(seq):
        assert torch.allclose(w[i, i + 1:], torch.zeros(w[i, i + 1:].shape), atol=1e-6)


def test_padding_mask_blocks_padding():
    seq = 4
    attn = ScaledDotProductAttention(d_k=4)
    q = torch.randn(1, seq, 4)
    k = torch.randn(1, seq, 4)
    v = torch.randn(1, seq, 4)
    lengths = torch.tensor([2])
    _, trace = attn(q, k, v, mask=create_padding_mask(lengths, seq))
    w = trace.weights[0]
    assert torch.allclose(w[:, 2:], torch.zeros(w[:, 2:].shape), atol=1e-6)
    assert torch.allclose(w[:, :2].sum(dim=-1), torch.ones(seq), atol=1e-6)


def test_trace_reports_every_calculation_step():
    attn = ScaledDotProductAttention(d_k=4)
    q = torch.randn(1, 3, 4)
    k = torch.randn(1, 3, 4)
    v = torch.randn(1, 3, 4)
    _, trace = attn(q, k, v, layer_index=2, head_index=3, token_ids=[0, 1, 2])
    labels = [s["label"] for s in trace.steps()]
    assert labels == ["Q", "K^T", "QK^T", "1/sqrt(d_k=4) = 0.5000", "mask", "softmax", "weights @ V"]
    assert trace.layer_index == 2
    assert trace.head_index == 3
    assert trace.token_ids == [0, 1, 2]


def test_gradcheck_attentions_output():
    d_k, seq = 3, 4
    attn = ScaledDotProductAttention(d_k=d_k).double()
    gen = torch.Generator().manual_seed(1)
    q = torch.randn(1, seq, d_k, dtype=torch.float64, requires_grad=True, generator=gen)
    k = torch.randn(1, seq, d_k, dtype=torch.float64, requires_grad=True, generator=gen)
    v = torch.randn(1, seq, d_k, dtype=torch.float64, requires_grad=True, generator=gen)

    def fn(qi, ki, vi):
        out, _ = attn(qi, ki, vi)
        return out

    assert gradcheck(fn, (q, k, v), eps=1e-6, atol=1e-4)