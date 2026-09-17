"""PHASE 4 demo — the Transformer assembled and executed.

Run:  python -m app.visualization.demo_transformer

Builds a tiny seq2seq Transformer (2 encoder + 2 decoder layers) from a
paper-style config, runs a dummy padded sequence through it, prints the
full block-by-block calculation trace (embed → PE → layers → output),
and saves the LAB 06 architecture diagram to reports/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from app.core.config import PaperConfig
from app.core.logging import configure_logging, get_logger
from app.transformer import Transformer
from app.visualization.architecture_plot import plot_transformer_architecture

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = get_logger("demo.transformer")

TINY_CONFIG = PaperConfig(
    name="tiny-demo",
    num_encoder_layers=2,
    num_decoder_layers=2,
    d_model=16,
    d_ff=32,
    num_heads=4,
    d_k=4,
    d_v=4,
    dropout=0.0,
    label_smoothing=0.0,
    train_steps=0,
    params_millions=0,
    notes=("Synthetic Phase 4 milestone config; not a paper Table 3 row.",),
)

VOCAB = list("abc")
ID_PAD = 0


def main() -> int:
    configure_logging()
    vocab_size = len(VOCAB) + 1
    transform = Transformer(
        config=TINY_CONFIG,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        max_len=8,
        pe_kind="sinusoidal",
    )
    logger.info("Transformer assembled from %s config", TINY_CONFIG.name)
    logger.info("  %d parameters  (%d encoder + %d decoder layers)",
                transform.parameters_count(),
                TINY_CONFIG.num_encoder_layers,
                TINY_CONFIG.num_decoder_layers)

    gen = torch.Generator().manual_seed(4)
    src = torch.randint(1, vocab_size, (1, 6), generator=gen)
    tgt = torch.randint(1, vocab_size, (1, 6), generator=gen)

    with torch.no_grad():
        logits, log_probs, trace = transform(src, tgt)

    logger.info("Forward trace:")
    for step in trace.steps():
        logger.info("  step '%s': shape=%s", step["label"], tuple(step["value"].shape))
    logger.info("logits shape: %s   log-probs shape: %s",
                tuple(logits.shape), tuple(log_probs.shape))
    logger.info("  (d_model=%d, d_ff=%d, heads=%d, d_k=d_v=%d)",
                TINY_CONFIG.d_model, TINY_CONFIG.d_ff,
                TINY_CONFIG.num_heads, TINY_CONFIG.d_k)

    out_dir = Path("reports")
    png = out_dir / "transformer_blocks.png"
    plot_transformer_architecture(
        TINY_CONFIG.num_encoder_layers,
        TINY_CONFIG.num_decoder_layers,
        out_path=png,
    )
    logger.info(f"Architecture diagram saved -> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())