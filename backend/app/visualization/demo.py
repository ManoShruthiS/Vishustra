"""PHASE 2 demo — the paper's attention equation, made visible.

Run:  python -m app.visualization.demo

"Sentence: The scientist studied the paper because it contained new results."
Tokenize character-wise, build random Q/K/V, compute equation 1, print the
full calculation, and save an attention heatmap under reports/.
"""

from __future__ import annotations

from pathlib import Path

import torch

from app.transformer import ScaledDotProductAttention
from app.visualization.attention_plot import plot_attention_weights
from app.core.logging import configure_logging, get_logger

logger = get_logger("demo.attention")

SENTENCE = "the scientist studied the paper because it contained new results"


def tokenize(sentence: str) -> list[str]:
    cleaned = sentence.lower().replace(".", "")
    return cleaned.split(" ")


def build_qkv(tokens: list[str], d_k: int = 8, seed: int = 7) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random Q/K/V (batch=1, seq, d_k) from a fixed seed for reproducibility."""
    gen = torch.Generator().manual_seed(seed)
    n = len(tokens)
    q = torch.randn(1, n, d_k, generator=gen)
    k = torch.randn(1, n, d_k, generator=gen)
    v = torch.randn(1, n, d_k, generator=gen)
    return q, k, v


def main() -> int:
    configure_logging()
    tokens = tokenize(SENTENCE)
    d_k = 8
    q, k, v = build_qkv(tokens, d_k=d_k)

    attention = ScaledDotProductAttention(d_k=d_k)
    output, trace = attention(q, k, v, token_ids=list(range(len(tokens))))

    logger.info("Scaled dot-product attention  Attention(Q,K,V) = softmax(QK^T/v(d_k))V")
    widths = max(len(t) for t in tokens)
    for t in tokens:
        logger.info(f"  token: {t:<{widths}}")
    for step in trace.steps():
        shape = tuple(step["value"].shape)
        logger.info(f"  step '{step['label']}': shape={shape}")

    logger.info("Attention weights (token -> keys):")
    w = trace.weights[0]
    header = "        " + "  ".join(f"{t:>{max(widths, 3)}}" for t in tokens)
    logger.info(header)
    for i, row in enumerate(w):
        fmt = "  ".join(f"{x:>{max(widths, 3)}.2f}" for x in row)
        logger.info(f"{tokens[i]:<{7}}{fmt}")

    out_dir = Path("reports")
    png = out_dir / "attention_first_milestone.png"
    plot_attention_weights(w, tokens, out_path=png)
    logger.info(f"Heatmap saved -> {png}")

    logger.info("FINAL attention output shape: %s", tuple(output.shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())