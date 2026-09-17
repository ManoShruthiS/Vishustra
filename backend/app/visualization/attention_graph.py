"""PHASE 6 — Attention graph (plan.md §8 'View B — Graph').

Bipartite token-connection diagram: query tokens on the left column,
key tokens on the right; edge width and darkness encode attention
weight.  Only the top-k strongest connections per query are drawn so
the graph stays readable for sequences up to ~30 tokens.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyArrowPatch
from torch import Tensor


def plot_attention_graph(
    query_tokens: list[str],
    key_tokens: list[str],
    weights: Tensor,
    *,
    top_k: int = 3,
    min_weight: float = 0.02,
    title: str | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Render a bipartite directed attention graph.

    ``weights``: (q_len, k_len) tensor of non-negative attention
    probabilities — row *i* is the distribution of query *i* over keys.
    """
    weights = weights.detach().cpu()
    q_len, k_len = weights.shape
    fig, ax = plt.subplots(figsize=(max(5, 0.8 * max(q_len, k_len)), max(4, 1.2 * max(q_len, k_len))))
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.5, max(q_len, k_len) - 0.5)
    ax.invert_yaxis()
    ax.axis("off")

    q_y = torch.arange(q_len).float()
    k_y = torch.arange(k_len).float()

    for x, toks, ys, side in [
        (0.0, query_tokens, q_y, "query"),
        (1.0, key_tokens, k_y, "key"),
    ]:
        for y, tok in zip(ys, toks):
            ax.plot(x, y, "o", ms=10, color="#444", zorder=5)
            ax.text(
                x + (0.06 if side == "query" else -0.06),
                y,
                tok,
                va="center",
                ha="right" if side == "query" else "left",
                fontsize=9,
                family="monospace",
                zorder=5,
            )

    for qi in range(q_len):
        scores, indices = torch.topk(weights[qi], min(top_k, k_len))
        for val, ki in zip(scores, indices):
            if val < min_weight:
                continue
            alpha = float(max(0.15, val))
            lw = float(max(0.3, 3.0 * val))
            arrow = FancyArrowPatch(
                (0.0, float(q_y[qi])),
                (1.0, float(k_y[ki])),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=lw,
                color="#333",
                alpha=alpha,
                connectionstyle="arc3,rad=0.12",
                shrinkA=6,
                shrinkB=6,
            )
            ax.add_patch(arrow)

    ax.set_title(
        title or "Attention graph — strongest connections per query",
        fontsize=10,
        pad=8,
    )
    fig.tight_layout()
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(out_path) if out_path is not None else Path("<inline>")