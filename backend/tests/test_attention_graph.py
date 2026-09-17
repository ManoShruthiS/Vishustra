"""PHASE 6 — Attention graph renderer tests (plan.md §66, View B)."""

import torch
from app.visualization.attention_graph import plot_attention_graph
from torch.nn.functional import softmax


def test_graph_renders_bipartite_file(tmp_path):
    weights = softmax(torch.randn(3, 3), dim=-1)
    out = plot_attention_graph(
        ["a", "b", "c"], ["a", "b", "c"], weights,
        out_path=tmp_path / "graph.png",
    )
    assert out == tmp_path / "graph.png"
    assert out.stat().st_size > 0


def test_graph_accepts_rectangular_query_key(tmp_path):
    weights = softmax(torch.randn(4, 2), dim=-1)
    out = plot_attention_graph(
        ["q0", "q1", "q2", "q3"], ["k0", "k1"], weights,
        top_k=2, out_path=tmp_path / "rect.png",
    )
    assert out.stat().st_size > 0