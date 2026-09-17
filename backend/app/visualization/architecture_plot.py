"""Phase 4 visualization — Transformer architecture diagram (LAB 06 BUILD).

Draws the block flow from the paper's Diagram (§3.1) that LAB 06 renders
after the user presses BUILD:

    EMBEDDING → POSITIONAL ENCODING → MULTI-HEAD ATTENTION → ADD+NORM
    → FEED FORWARD → ADD+NORM → ...  (repeated per encoder/decoder layer)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def plot_transformer_architecture(
    num_encoder_layers: int,
    num_decoder_layers: int,
    out_path: str | Path | None = None,
) -> Path:
    """Render the encoder/decoder block stack as a schematic."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4 + max(num_encoder_layers, num_decoder_layers) * 1.15))
    enc_ax, dec_ax = axes

    block_y = lambda n: 1.0 + n * 1.0
    layers = [
        ("EMBEDDING + POSITIONAL ENCODING", 0.0),
    ] + [
        ("MULTI-HEAD ATTENTION\n                 ↓\n            ADD + NORM\n                 ↓\n        FEED FORWARD\n                 ↓\n            ADD + NORM", 1.0 + i * 1.0)
        for i in range(num_encoder_layers)
    ]

    def draw(ax, blocks, title):
        ax.set_xlim(0, 3)
        ax.set_ylim(0, max(block_y(len(blocks) - 1), 1.5) + 0.9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=11)
        for i, (label, base) in enumerate(blocks):
            top = base + 0.55
            box = FancyBboxPatch((0.35, base), 2.3, 0.5 - 0.05, boxstyle="round,pad=0.02",
                                 fc="#e8e8e8", ec="black", lw=1.0)
            ax.add_patch(box)
            ax.text(1.5, top, label, ha="center", va="top", fontsize=7.2, linespacing=1.3)
            if i < len(blocks) - 1:
                arrow = FancyArrowPatch((1.5, base), (1.5, base + 0.5), arrowstyle="-|>",
                                        mutation_scale=10, lw=1.0)
                ax.add_patch(arrow)

    enc_blocks = layers[:]
    dec_blocks = [
        ("EMBEDDING + POSITIONAL ENCODING", 0.0),
    ] + [
        ("MASKED SELF-ATTENTION\n         ↓\n    ADD + NORM\n         ↓\n  CROSS-ATTENTION\n         ↓\n    ADD + NORM\n         ↓\n   FEED FORWARD\n         ↓\n    ADD + NORM", 1.0 + i * 1.0)
        for i in range(num_decoder_layers)
    ]
    draw(enc_ax, enc_blocks, "ENCODER")
    draw(dec_ax, dec_blocks, "DECODER")
    fig.suptitle("Transformer block flow (Attention Is All You Need, Fig. 1)", fontsize=12)
    fig.tight_layout()

    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(out_path) if out_path is not None else Path("<inline>")