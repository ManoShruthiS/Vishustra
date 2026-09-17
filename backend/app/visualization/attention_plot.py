"""Attention matrix heatmap.

Renders the attention weight matrix from a scaled dot-product attention
trace (plan.md §8 'View A — Matrix'). Monochrome scientific styling:
black / white / grayscale, minimal, data-dense.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import Tensor


def plot_attention_weights(
    weights: Tensor,
    tokens: list[str],
    layer_index: int = -1,
    head_index: int = -1,
    title: str | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Save (or show) an attention weight heatmap.

    ``weights`` must have the shape (seq, seq) where row i = the attention
    distribution of token i over tokens j.
    """
    seq = weights.shape[-1]
    assert seq == len(tokens), "token count must match attention sequence length"

    fig, ax = plt.subplots(figsize=(max(5.0, 0.9 * seq), max(4.5, 0.9 * seq)))
    ax.imshow(weights.detach().cpu().numpy(), cmap="Greys", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(seq))
    ax.set_yticks(range(seq))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)
    ax.set_xlabel("Keys (attended to)")
    ax.set_ylabel("Queries (attending token)")

    heading = title or "Attention weights — softmax(QKᵀ/√d_k)  (paper eq. 1)"
    if layer_index >= 0:
        heading += f"  ·  layer {layer_index}"
    if head_index >= 0:
        heading += f"  ·  head {head_index}"
    ax.set_title(heading)

    fig.tight_layout()
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(out_path) if out_path is not None else Path("<inline>")


def plot_head_grid(
    rows: list[Tensor],
    tokens: list[str],
    layer_index: int = -1,
    title: str | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Multi-Head Attention Studio grid (LAB 05) — one heatmap per head.

    ``rows`` holds one (seq, seq) weight matrix per head, in head order.
    """
    seq = rows[0].shape[-1]
    assert seq == len(tokens), "token count must match attention sequence length"
    n_heads = len(rows)
    cols = 2
    fig_height = max(4.0, 4.0 * ((n_heads + cols - 1) // cols))
    fig, axes = plt.subplots((n_heads + cols - 1) // cols, cols, figsize=(10, fig_height))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= n_heads:
            ax.set_visible(False)
            continue
        ax.imshow(rows[i].detach().cpu().numpy(), cmap="Greys", vmin=0.0, vmax=1.0)
        ax.set_title(f"head {i}", fontsize=10)
        ax.set_xticks(range(seq))
        ax.set_yticks(range(seq))
        ax.set_xticklabels(tokens, rotation=90, fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)

    fig.suptitle(
        title or "Multi-head attention — each head attends in its own projection subspace",
        fontsize=11,
    )
    fig.tight_layout()
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(out_path) if out_path is not None else Path("<inline>")