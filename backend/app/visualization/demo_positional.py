"""PHASE 3 demo — Positional Encoding Laboratory (LAB 03).

Run:  python -m app.visualization.demo_positional

Builds the sinusoidal table from paper §3.5 (eq. 3/4), prints sample
vectors / wavelength structure, and saves a waveform + heatmap figure to
reports/positional_encoding.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from app.core.logging import configure_logging, get_logger
from app.transformer import positional_encoding

logger = get_logger("demo.positional")


def main() -> int:
    configure_logging()
    max_len, d_model = 20, 128
    pe = positional_encoding(max_len, d_model)

    logger.info("Sinusoidal positional encoding (paper §3.5, eq. 3/4)")
    logger.info("  PE(pos,2i)   = sin(pos / 10000^(2i/d_model))")
    logger.info("  PE(pos,2i+1) = cos(pos / 10000^(2i/d_model))")
    logger.info("  table shape: %s", tuple(pe.shape))
    logger.info("  wavelength dims: 2pi .. 10000*2pi over %d dims", d_model)

    for pos in (0, 1, 2, 3, 10):
        logger.info("  pos %2d  pe[%d,:6] = %s", pos, pos, torch.round(pe[pos, :6] * 1e2) / 1e2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for pos in (0, 1, 2, 3, 10, 19):
        ax1.plot(pe[pos].numpy(), label=f"pos {pos}")
    ax1.set_title("PE waveform over dimensions (§3.5)")
    ax1.set_xlabel("dimension i")
    ax1.set_ylabel("encoding value")
    ax1.legend(fontsize=8)

    ax2.imshow(pe.numpy(), cmap="Greys", aspect="auto")
    ax2.set_title("Positional encoding heatmap")
    ax2.set_xlabel("dimension i")
    ax2.set_ylabel("position pos")

    fig.tight_layout()
    out_dir = Path("reports")
    png = out_dir / "positional_encoding.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    logger.info(f"Figure saved -> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())