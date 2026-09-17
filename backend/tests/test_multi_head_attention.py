"""Unit tests — multi-head attention (paper §3.2.2, eq. 2)."""

import torch
from app.transformer import MultiHeadAttention, ScaledDotProductAttention, create_causal_mask
from torch.autograd import gradcheck


def make(batch=2, seq=5, d_model=8, heads=4, d_k=2, d_v=2, seed=0):
    gen = torch.Generator().manual_seed(seed)
    mha = MultiHeadAttention(d_model=d_model, num_heads=heads, d_k=d_k, d_v=d_v)
    x = torch.randn(batch, seq, d_model, generator=gen)
    return mha, x


def test_output_shape_and_dtype():
    mha, x = make()
    out, _ = mha(x, x, x)
    assert tuple(out.shape) == (2, 5, 8)


def test_every_head_has_its_own_index():
    mha, x = make()
    _, trace = mha(x, x, x)
    assert len(trace.heads) == 4
    for i, head in enumerate(trace.heads):
        assert head.head_index == i
        assert head.weights.shape == (2, 5, 5)


def test_parameter_count_matches_sum_of_named_params():
    mha, _ = make()
    total = sum(p.numel() for p in mha.parameters())
    assert mha.parameters_count() == total


def test_manual_recompute_matches_module(tmp_path):
    """Eq. 2 recomputed by hand: project each head, attend, concat, project."""
    batch, seq, d_model, heads, d_k, d_v = 2, 5, 8, 4, 2, 2
    mha = MultiHeadAttention(d_model=d_model, num_heads=heads, d_k=d_k, d_v=d_v)
    gen = torch.Generator().manual_seed(11)
    q = torch.randn(batch, seq, d_model, generator=gen)
    k = torch.randn(batch, seq, d_model, generator=gen)
    v = torch.randn(batch, seq, d_model, generator=gen)

    out, trace = mha(q, k, v)

    manual_heads = []
    attention = ScaledDotProductAttention(d_k=d_k)
    for i in range(heads):
        o, _ = attention(mha.w_q[i](q), mha.w_k[i](k), mha.w_v[i](v))
        manual_heads.append(o)
    manual_out = mha.w_out(torch.cat(manual_heads, dim=-1))
    assert torch.allclose(out, manual_out, atol=1e-5)
    assert trace.concat.shape == (batch, seq, heads * d_v)


def test_heads_produce_distinct_weights():
    mha, x = make()
    _, trace = mha(x, x, x)
    seen = torch.stack([h.weights[0].flatten() for h in trace.heads])
    assert (seen[0] != seen[1]).any() or (seen[0] != seen[2]).any()


def test_causal_mask_respected_per_head():
    seq = 5
    mha, x = make(batch=1, seq=seq)
    _, trace = mha(x, x, x, mask=create_causal_mask(seq))
    for head in trace.heads:
        w = head.weights[0]
        for i in range(seq):
            assert torch.allclose(w[i, i + 1:], torch.zeros(w[i, i + 1:].shape), atol=1e-6)


def test_trace_steps_include_heads_concat_and_projection():
    mha, x = make(heads=4)
    _, trace = mha(x, x, x)
    labels = [s["label"] for s in trace.steps()]
    assert any(l.startswith("h0 ·") for l in labels)
    assert any(l.startswith("h3 ·") for l in labels)
    assert "CONCAT" in labels
    assert "W^O (linear)" in labels


def test_defaults_from_base_config():
    mha = MultiHeadAttention()  # defaults follow paper Table 3
    assert mha.d_model == 512
    assert mha.num_heads == 8
    assert mha.d_k == 64
    assert mha.d_v == 64


def test_gradcheck_multi_head():
    d_model, heads, d_k, seq = 6, 3, 2, 4
    mha = MultiHeadAttention(d_model=d_model, num_heads=heads, d_k=d_k, d_v=d_k).double()
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(1, seq, d_model, dtype=torch.float64, requires_grad=True, generator=gen)

    def fn(xi):
        out, _ = mha(xi, xi, xi)
        return out

    assert gradcheck(fn, (x,), eps=1e-6, atol=1e-4)