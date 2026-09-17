"""Unit tests — Position-wise Feed-Forward (paper §3.3, eq. 2)."""

import torch
from app.transformer import FeedForward
from torch.autograd import gradcheck


def test_output_shape():
    ff = FeedForward(d_model=8, d_ff=16)
    x = torch.randn(2, 5, 8)
    out, trace = ff(x)
    assert tuple(out.shape) == (2, 5, 8)
    assert tuple(trace.pre_activation.shape) == (2, 5, 16)


def test_parameter_count():
    ff = FeedForward(d_model=8, d_ff=16)
    expected = 8 * 16 + 16 + 16 * 8 + 8
    assert ff.parameters_count() == expected


def test_relu_applied_after_first_linear():
    ff = FeedForward(d_model=4, d_ff=8)
    x = torch.randn(1, 3, 4)
    _, trace = ff(x)
    assert torch.allclose(trace.activated, torch.relu(trace.pre_activation), atol=1e-6)
    assert bool((trace.activated >= 0.0).all())


def test_ffn_matches_manual_equation2():
    """FFN(x) = max(0, xW1+b1)W2+b2 computed by hand matches the module."""
    ff = FeedForward(d_model=4, d_ff=8, dropout=0.0)
    x = torch.randn(1, 5, 4)
    out, trace = ff(x)
    with torch.no_grad():
        pre = x @ ff.w1.weight.t() + ff.w1.bias
        manual = torch.relu(pre) @ ff.w2.weight.t() + ff.w2.bias
    assert torch.allclose(trace.pre_activation, pre, atol=1e-6)
    assert torch.allclose(out, manual, atol=1e-5)


def test_ffn_is_positionwise_identical_parameters():
    """The same W1/W2 act on every position independently (paper §3.3)."""
    ff = FeedForward(d_model=4, d_ff=8, dropout=0.0)
    x = torch.randn(1, 5, 4)
    row = torch.randn(4)
    x[0, 2] = row
    x[0, 4] = row
    out, _ = ff(x)
    assert torch.allclose(out[0, 2], out[0, 4], atol=1e-6)


def test_gradcheck():
    ff = FeedForward(d_model=4, d_ff=8, dropout=0.0)
    ff.eval()
    ff.double()
    x = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)

    def fn(xi):
        out, _ = ff(xi)
        return out

    assert gradcheck(fn, (x,), eps=1e-6, atol=1e-4)