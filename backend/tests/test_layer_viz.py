"""PHASE 6 — Layer-by-layer attention strip renderer tests."""

import torch
from app.visualization.layer_visualization import plot_layer_strip
from torch.nn.functional import softmax


def test_layer_strip_renders_file(tmp_path):
    layers = [softmax(torch.randn(4, 4), dim=-1) for _ in range(3)]
    out = plot_layer_strip(
        layers, ["a", "b", "c", "d"],
        layer_labels=["enc 0", "enc 1", "enc 2"],
        out_path=tmp_path / "strip.png",
    )
    assert out.stat().st_size > 0


def test_layer_strip_requires_matching_tokens(tmp_path):
    layers = [softmax(torch.randn(4, 4), dim=-1)]
    try:
        plot_layer_strip(layers, ["a", "b"], out_path=tmp_path / "bad.png")
    except AssertionError:
        pass
    else:
        raise AssertionError("expected token-count mismatch to raise")