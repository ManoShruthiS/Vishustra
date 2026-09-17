"""PHASE 6 — Layer-by-layer attention heatmap strip.

Stacks the attention weight matrix from each encoder/decoder layer
vertically so the viewer can compare how attention patterns evolve
through the stack (plan.md §8 'View C — Layer strip').
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import Tensor


def plot_layer_strip(
    weights_list: list[Tensor],
    tokens: list[str],
    *,
    layer_labels: list[str] | None = None,
    title: str | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Render a vertical strip of attention heatmaps — one row per layer.

    ``weights_list``: list of (seq, seq) attention matrices (detached or
    not; they're moved to CPU internally).  Usually the *first head* or
    *mean across heads* is passed so each layer gets one representative
    heatmap.
    """
    n = len(weights_list)
    seq = weights_list[0].shape[-1]
    assert seq == len(tokens), "token count must match attention sequence length"
    labels = layer_labels or [f"layer {i}" for i in range(n)]

    fig_height = max(3.5, 1.8 * n)
    fig, axes = plt.subplots(n, 1, figsize=(max(5.5, 0.9 * seq), fig_height), squeeze=False)
    for i, (w, lab) in enumerate(zip(weights_list, labels)):
        ax = axes[i, 0]
        ax.imshow(w.detach().cpu().numpy(), cmap="Greys", vmin=0.0, vmax=1.0)
        ax.set_ylabel(lab, fontsize=9, rotation=0, labelpad=50, va="center")
        ax.set_yticks([])
        if i == n - 1:
            ax.set_xticks(range(seq))
            ax.set_xticklabels(tokens, rotation=90, fontsize=8)
        else:
            ax.set_xticks([])
    fig.suptitle(title or "Layer attention strip", fontsize=11, y=1.01)
    fig.tight_layout()
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path) if out_path is not None else Path("<inline>")