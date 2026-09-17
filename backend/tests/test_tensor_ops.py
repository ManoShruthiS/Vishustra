"""Unit tests — tensor operations laboratory."""

import torch
from app.transformer import qkt, scale, transpose_last_two


def test_transpose_last_two():
    x = torch.arange(24.0).reshape(2, 3, 4)
    t = transpose_last_two(x)
    assert tuple(t.shape) == (2, 4, 3)
    assert torch.equal(t[0], x[0].transpose(0, 1))


def test_qkt_is_q_dot_k_transpose():
    gen = torch.Generator().manual_seed(3)
    q = torch.randn(1, 5, 8, generator=gen)
    k = torch.randn(1, 5, 8, generator=gen)
    manual = torch.matmul(q, torch.transpose(k, 1, 2))
    assert torch.allclose(qkt(q, k), manual, atol=1e-6)


def test_scale_is_inverse_sqrt_dk():
    d_k = 16
    x = torch.randn(2, 3)
    scaled = scale(x, d_k)
    assert torch.allclose(scaled, x / (d_k**0.5), atol=1e-6)