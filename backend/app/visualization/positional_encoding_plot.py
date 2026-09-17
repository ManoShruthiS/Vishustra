"""PHASE 6 — Positional encoding plotter.

Waveform view of the sinusoidal table (§3.5, eq. 3): for the first few
dimensions, plots how PE varies with position — sin on even indices,
cos on odd — as a warm invitation to "why even dims are sin and odd
dims are cos" (LAB 03 reference, used by the observatory).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.transformer import positional_encoding


def plot_positional_encoding(
    max_len: int,
    d_model: int,
    base: float = 10000.0,
    num_waves: int = 8,
    *,
    title: str | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Waveform plots of the first ``num_waves`` PE dimensions."""
    pe = positional_encoding(max_len, d_model, base)
    num_waves = min(num_waves, d_model)
    positions = torch.arange(max_len)

    fig, ax = plt.subplots(figsize=(9, 4))
    for d in range(num_waves):
        kind = "sin" if d % 2 == 0 else "cos"
        ax.plot(positions, pe[:, d].numpy(), label=f"dim {d} ({kind})", linewidth=1.6)
    ax.set_title(title or f"Sinusoidal positional encoding  (§3.5, eq. 3)  d_model={d_model}")
    ax.set_xlabel("position")
    ax.set_ylabel("PE value")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(out_path) if out_path is not None else Path("<inline>")