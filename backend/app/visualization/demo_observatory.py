"""PHASE 6 — Demo: The Attention Observatory (plan.md §8, LAB 08).

Renders the full Phase 6 visualization family from one trained tiny
Transformer on the copy task into ``reports/observatory/``:

    attention_matrix.png          View A — single head heatmap
    head_grid_encoder_l{i}.png    Multi-Head Attention Studio (LAB 05)
    head_grid_cross_l{i}.png      decoder cross-attention head grid
    layer_strip_encoder.png       encoder self-attention, layer by layer
    layer_strip_cross.png         decoder cross-attention, layer by layer
    layer_strip_decoder_self.png  masked decoder self-attention strip
    attention_graph.png           View B — bipartite token connections
    positional_encoding.png       sine/cosine waves (paper §3.5, eq. 3)
    training_curve.png            Phase 5 training graph

Reuses the Phase 5 checkpoint (``models/copy_demo/epoch_*.pt``) when it
exists; otherwise trains a short run itself so the demo stays fast and
reproducible.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from app.core.config import PaperConfig
from app.datasets import CharacterTokenizer, CopyTask, make_loaders, seed_all, train_val_split
from app.training import TrainConfig, Trainer, greedy_decode, load_checkpoint, save_checkpoint
from app.training.evaluator import src_mask_for, tgt_mask_for
from app.transformer import Transformer
from app.visualization.attention_graph import plot_attention_graph
from app.visualization.attention_plot import plot_attention_weights, plot_head_grid
from app.visualization.layer_visualization import plot_layer_strip
from app.visualization.positional_encoding_plot import plot_positional_encoding
from app.visualization.training_plot import plot_training_curve

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("vishustra")

ALPHABET = "abcdefghijkl"
SEED = 42


def _tokens(tokenizer: CharacterTokenizer, ids: list[int]) -> list[str]:
    special = {
        tokenizer.pad_id: "<PAD>",
        tokenizer.sos_id: "<SOS>",
        tokenizer.eos_id: "<EOS>",
    }
    return [special.get(int(i), t) for i, t in zip(ids, tokenizer.decode(ids))]


def observe(
    model: Transformer,
    tokenizer: CharacterTokenizer,
    src_ids: list[int],
    tgt_ids: list[int],
    out_dir: str | Path,
    *,
    history: list[dict[str, float]] | None = None,
) -> list[Path]:
    """Render every Phase 6 visualization for one teacher-forced example."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    src_t = torch.tensor([src_ids], dtype=torch.long)
    tgt_in = torch.tensor([[tokenizer.sos_id, *tgt_ids]], dtype=torch.long)
    batch = {
        "src_len": torch.tensor([len(src_ids)]),
        "tgt_len": torch.tensor([tgt_in.shape[1]]),
    }
    src_mask = src_mask_for(batch, len(src_ids))
    tgt_mask = tgt_mask_for(batch, tgt_in.shape[1])

    model.eval()
    with torch.no_grad():
        _, _, trace = model(src_t, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask)

    src_tokens = _tokens(tokenizer, src_ids)
    tgt_tokens = _tokens(tokenizer, tgt_ids)
    dec_tokens = [tokenizer.tokens[tokenizer.sos_id]] + tgt_tokens
    enc_layers = trace.encoder.layers
    dec_layers = trace.decoder.layers

    m = artifacts.append

    w = enc_layers[0].attention.heads[0].weights[0]
    m(plot_attention_weights(
        w, src_tokens, layer_index=0, head_index=0,
        title="Encoder self-attention — softmax(QKᵀ/√d_k)  (eq. 1)",
        out_path=out_dir / "attention_matrix.png",
    ))

    for i, layer in enumerate(enc_layers):
        rows = [h.weights[0] for h in layer.attention.heads]
        m(plot_head_grid(
            rows, src_tokens, layer_index=i,
            title=f"Encoder layer {i} — multi-head attention (eq. 2)",
            out_path=out_dir / f"head_grid_encoder_l{i}.png",
        ))

    for i, layer in enumerate(dec_layers):
        rows = [h.weights[0] for h in layer.cross_attention.heads]
        m(plot_head_grid(
            rows, src_tokens, layer_index=i,
            title=f"Decoder layer {i} — cross-attention over the encoder (eq. 2)",
            out_path=out_dir / f"head_grid_cross_l{i}.png",
        ))

    m(plot_layer_strip(
        [lay.attention.heads[0].weights[0] for lay in enc_layers], src_tokens,
        layer_labels=[f"enc layer {i}" for i in range(len(enc_layers))],
        title="Encoder self-attention — layer strip (head 0)",
        out_path=out_dir / "layer_strip_encoder.png",
    ))
    m(plot_layer_strip(
        [lay.cross_attention.heads[0].weights[0] for lay in dec_layers], src_tokens,
        layer_labels=[f"dec cross {i}" for i in range(len(dec_layers))],
        title="Decoder cross-attention — layer strip (head 0)",
        out_path=out_dir / "layer_strip_cross.png",
    ))
    m(plot_layer_strip(
        [lay.self_attention.heads[0].weights[0] for lay in dec_layers], dec_tokens,
        layer_labels=[f"dec self {i}" for i in range(len(dec_layers))],
        title="Masked decoder self-attention — layer strip (head 0)",
        out_path=out_dir / "layer_strip_decoder_self.png",
    ))

    m(plot_attention_graph(
        src_tokens, src_tokens, enc_layers[0].attention.heads[0].weights[0],
        title="Encoder layer 0, head 0 — attention graph (View B)",
        out_path=out_dir / "attention_graph.png",
    ))
    m(plot_attention_graph(
        dec_tokens, src_tokens, dec_layers[0].cross_attention.heads[0].weights[0],
        title="Decoder layer 0, head 0 — cross-attention graph",
        out_path=out_dir / "attention_graph_cross.png",
    ))

    m(plot_positional_encoding(
        32, model.config.d_model, num_waves=6,
        out_path=out_dir / "positional_encoding.png",
    ))

    if history:
        m(plot_training_curve(
            history,
            out_path=out_dir / "training_curve.png",
            title="Transformer training curve — copy task",
        ))
    return artifacts


def run() -> None:
    seed_all(SEED)
    tokenizer = CharacterTokenizer(chars=ALPHABET)
    vocab = tokenizer.vocab_size

    config = PaperConfig(
        name="tiny-copy",
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_model=16,
        d_ff=32,
        num_heads=4,
        d_k=4,
        d_v=4,
        dropout=0.0,
        label_smoothing=0.1,
        train_steps=0,
        params_millions=0,
        notes=("Tiny model reused by the Phase 6 Attention Observatory demo.",),
    )
    model = Transformer(config, src_vocab_size=vocab, tgt_vocab_size=vocab, max_len=8)

    checkpoint_dir = Path("models") / "copy_demo"
    ckpts = sorted(checkpoint_dir.glob("epoch_*.pt"))
    history: list[dict[str, float]] = []
    if ckpts:
        state = load_checkpoint(ckpts[-1], model)
        history = state.get("history", [])
        logger.info("reused checkpoint %s (epoch %s)", ckpts[-1].name, state.get("epoch"))
    else:
        train_ds = CopyTask(vocab_size=vocab, n_examples=1500, min_len=2, max_len=6, seed=SEED, content_start=3)
        val_ds = CopyTask(vocab_size=vocab, n_examples=200, min_len=2, max_len=6, seed=SEED + 1, content_start=3)
        train_view, _ = train_val_split(train_ds, val_fraction=0.05, seed=SEED)
        train_loader, val_loader = make_loaders(train_view, val_ds, tokenizer, batch_size=32, max_len=8)
        trainer = Trainer(
            model,
            TrainConfig(epochs=12, batch_size=32, init_lr=0.5, warmup_steps=100, max_len=8,
                        checkpoint_dir="models", run_name="copy_demo", seed=SEED),
            vocab_size=vocab,
        )
        history = trainer.fit(train_loader, val_loader)
        save_checkpoint(
            checkpoint_dir / "epoch_012.pt", model, None, None, 12, history,
            vocab=tokenizer.tokens,
        )
        logger.info("trained short run (12 epochs) and saved checkpoint")

    val_ds = CopyTask(vocab_size=vocab, n_examples=200, min_len=2, max_len=6, seed=SEED + 1, content_start=3)
    src, tgt = val_ds.tensor_pairs()
    idx = 0
    src_ids = src[idx][src[idx] != 0].tolist()
    tgt_ids = tgt[idx][tgt[idx] != 0].tolist()
    pred = greedy_decode(model, torch.tensor(src_ids), tokenizer, max_steps=12)
    logger.info(
        "observed example: src=%s  gold=%s  pred=%s  copied=%s",
        "".join(tokenizer.decode(src_ids)),
        "".join(tokenizer.decode(tgt_ids)),
        "".join(tokenizer.decode(pred)),
        pred == tgt_ids,
    )

    artifacts = observe(model, tokenizer, src_ids, tgt_ids, Path("reports") / "observatory", history=history)
    logger.info("Attention Observatory -> %d artifacts", len(artifacts))
    for p in artifacts:
        logger.info("  %s  (%d bytes)", p, p.stat().st_size)


if __name__ == "__main__":
    run()