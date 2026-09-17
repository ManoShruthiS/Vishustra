"""Unit tests — Layer Normalization (paper §3.1)."""

import torch
from app.transformer import LayerNorm
from torch.autograd import gradcheck


def test_output_shape():
    ln = LayerNorm(8)
    x = torch.randn(3, 5, 8)
    out, trace = ln(x)
    assert tuple(out.shape) == (3, 5, 8)
    assert tuple(trace.mean.shape) == (3, 5, 1)


def test_normalized_signal_has_zero_mean_and_unit_var():
    ln = LayerNorm(8)
    x = torch.randn(3, 5, 8)
    _, trace = ln(x)
    assert torch.allclose(trace.normalized.mean(-1), torch.zeros(3, 5), atol=1e-5)
    assert torch.allclose(trace.normalized.var(-1, unbiased=False), torch.ones(3, 5), atol=1e-5)


def test_gamma_beta_are_applied():
    ln = LayerNorm(8)
    with torch.no_grad():
        ln.gamma.fill_(2.0)
        ln.beta.fill_(3.0)
    x = torch.randn(2, 4, 8)
    _, trace = ln(x)
    assert torch.allclose(trace.output, trace.normalized * 2.0 + 3.0, atol=1e-6)


def test_parameters_count():
    assert LayerNorm(512).parameters_count() == 1024


def test_gradcheck():
    ln = LayerNorm(4).double()
    x = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)

    def fn(xi):
        out, _ = ln(xi)
        return out

    assert gradcheck(fn, (x,), eps=1e-6, atol=1e-4)