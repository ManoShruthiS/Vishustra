"""Unit tests — softmax (our implementation)."""

import torch
from app.transformer import log_softmax, softmax
from torch import Tensor


def _rows(a: Tensor) -> Tensor:
    return a.sum(dim=-1)


def test_rows_sum_to_one():
    x = torch.randn(4, 5)
    out = softmax(x)
    assert torch.allclose(_rows(out), torch.ones(4), atol=1e-6)


def test_values_in_unit_range():
    x = torch.randn(3, 6)
    out = softmax(x)
    assert bool((out >= 0).all())
    assert bool((out <= 1).all())


def test_argument_reordering():
    x = torch.tensor([[2.0, 1.0, 0.0]])
    out = softmax(x)
    assert float(out[0, 0]) > float(out[0, 1]) > float(out[0, 2])


def test_stability_with_large_logits():
    x = torch.tensor([[1000.0, 1000.0]])
    out = softmax(x)
    assert bool(torch.isfinite(out).all())
    assert torch.allclose(_rows(out), torch.ones(1), atol=1e-6)
    assert torch.allclose(out[0, 0], out[0, 1], atol=1e-6)


def test_mask_excludes_positions():
    x = torch.tensor([[0.0, 0.0, 0.0]])
    mask = torch.tensor([[0.0, 0.0, float("-inf")]])
    out = softmax(x, mask)
    assert float(out[0, 2]) == 0.0
    assert torch.allclose(_rows(out), torch.ones(1), atol=1e-6)
    assert torch.allclose(out[0, 0], out[0, 1], atol=1e-6)


def test_log_softmax_consistent_with_softmax():
    x = torch.randn(2, 4)
    logs = log_softmax(x)
    assert torch.allclose(torch.exp(logs), softmax(x), atol=1e-6)


def test_gradient_exists():
    x = torch.randn(2, 3, requires_grad=True)
    out = softmax(x).sum()
    out.backward()
    assert x.grad is not None
    assert bool(torch.isfinite(x.grad).all())