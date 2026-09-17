"""PHASE 3 demo — Multi-Head Attention Studio (LAB 05).

Run:  python -m app.visualization.demo_multi_head

Builds a small multi-head attention (d_model=32, h=4, d_k=d_v=8), runs it on
the same sentence as the Phase 2 demo, prints the per-head weights and the
full calculation trace, and saves a head grid to reports/.
"""

from __future__ import annotations

from pathlib import Path

from app.core.logging import configure_logging, get_logger
from app.transformer import MultiHeadAttention
from app.visualization.attention_plot import plot_head_grid
from app.visualization.demo import SENTENCE, build_qkv, tokenize

logger = get_logger("demo.multi_head")


def main() -> int:
    configure_logging()
    tokens = tokenize(SENTENCE)
    d_model, num_heads, d_k = 32, 4, 8

    attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, d_k=d_k, d_v=d_k)
    q, k, v = build_qkv(tokens, d_k=d_model)
    output, trace = attn(q, k, v)

    logger.info("Multi-head attention  MultiHead(Q,K,V) = Concat(head_1..head_h) W^O  (eq. 2)")
    logger.info("  d_model=%d  heads=%d  d_k=d_v=%d  parameters=%d",
                d_model, num_heads, d_k, attn.parameters_count())

    for step in trace.steps():
        logger.info("  step '%s': shape=%s", step["label"], tuple(step["value"].shape))

    widths = max(len(t) for t in tokens)
    for i, head in enumerate(trace.heads):
        logger.info(f"HEAD {i} attention weights (token -> keys):")
        w = head.weights[0]
        header = "        " + "  ".join(f"{t:>{max(widths, 3)}}" for t in tokens)
        logger.info(header)
        for r, row in enumerate(w):
            fmt = "  ".join(f"{x:>{max(widths, 3)}.2f}" for x in row)
            logger.info(f"{tokens[r]:<{7}}{fmt}")

    logger.info("FINAL multi-head output shape: %s", tuple(output.shape))

    out_dir = Path("reports")
    png = out_dir / "multi_head_studio.png"
    plot_head_grid(
        [h.weights[0] for h in trace.heads],
        tokens,
        out_path=png,
    )
    logger.info(f"Head grid saved -> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())