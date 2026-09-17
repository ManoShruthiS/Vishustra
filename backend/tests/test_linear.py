"""Unit tests — linear layer (our implementation)."""

import torch

from app.transformer import Linear


def test_forward_shapes():
    layer = Linear(in_features=4, out_features=6)
    x = torch.randn(3, 4)
    out = layer(x)
    assert tuple(out.shape) == (3, 6)


def test_batched_forward():
    layer = Linear(5, 3)
    x = torch.randn(2, 8, 5)
    out = layer(x)
    assert tuple(out.shape) == (2, 8, 3)


def test_parameters_count():
    layer = Linear(4, 3, bias=True)
    assert layer.parameters_count() == 4 * 3 + 3
    no_bias = Linear(4, 3, bias=False)
    assert no_bias.parameters_count() == 4 * 3


def test_output_is_affine_of_input():
    layer = Linear(2, 1)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2.0, -1.0]]))
        layer.bias.copy_(torch.tensor([0.5]))
    x = torch.tensor([[3.0, 4.0]])
    out = layer(x)
    assert torch.allclose(out, torch.tensor([[2 * 3 - 4 + 0.5]]))


def test_parameters_are_trainable():
    layer = Linear(2, 2)
    x = torch.randn(1, 2)
    loss = layer(x).sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert bool(torch.isfinite(layer.weight.grad).all())


def test_gradcheck():
    from torch.autograd import gradcheck

    layer = Linear(3, 2)
    x = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
    w = layer.weight.detach().to(torch.float64).clone().requires_grad_(True)
    b = layer.bias.detach().to(torch.float64).clone().requires_grad_(True)

    def fn(xi, wi, bi):
        return torch.matmul(xi, wi.transpose(-1, -2)) + bi

    assert gradcheck(fn, (x, w, b), eps=1e-6, atol=1e-4)