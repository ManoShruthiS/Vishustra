"""PHASE 5 — Demo: training a toy seq2seq copy task (plan.md §65, LAB 07).

Trains the Phase 4 Transformer (tiny) on a character "copy" task, logs
per-epoch metrics, saves checkpoint snapshots to models/ and a training
curve to reports/training_curve.png. Finally, runs greedy decoding on a
few held-out examples to show the model has actually learned to copy.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from app.core.config import PaperConfig
from app.datasets import CharacterTokenizer, CopyTask, make_loaders, seed_all, train_val_split
from app.training import TrainConfig, Trainer, greedy_decode
from app.transformer import Transformer
from app.visualization.training_plot import plot_training_curve

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("vishustra")

ALPHABET = "abcdefghijkl"
SEED = 42


def run() -> None:
    seed_all(SEED)

    tokenizer = CharacterTokenizer(chars=ALPHABET)
    vocab = tokenizer.vocab_size
    logger.info("vocab size: %d (%s ...)", vocab, tokenizer.tokens)

    train_ds = CopyTask(
        vocab_size=vocab, n_examples=1500, min_len=2, max_len=6, seed=SEED, content_start=3
    )
    val_ds = CopyTask(
        vocab_size=vocab, n_examples=200, min_len=2, max_len=6, seed=SEED + 1, content_start=3
    )
    train_view, _ = train_val_split(train_ds, val_fraction=0.05, seed=SEED)
    train_loader, val_loader = make_loaders(train_view, val_ds, tokenizer, batch_size=32, max_len=8)

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
        notes=("Tiny model for the Phase 5 copy-task demo.",),
    )
    model = Transformer(config, src_vocab_size=vocab, tgt_vocab_size=vocab, max_len=8)

    trainer = Trainer(
        model,
        TrainConfig(
            epochs=25,
            batch_size=32,
            init_lr=0.5,
            warmup_steps=100,
            max_len=8,
            checkpoint_dir="models",
            run_name="copy_demo",
            seed=SEED,
        ),
        vocab_size=vocab,
    )

    history = trainer.fit(train_loader, val_loader)
    for row in history:
        logger.info(
            "  e=%d train_loss=%.4f train_acc=%.3f val_loss=%.4f val_acc=%.3f val_ppl=%.2f",
            int(row["epoch"]),
            row["train_loss"],
            row["train_acc"],
            row["val_loss"],
            row["val_acc"],
            row["val_ppl"],
        )

    artifact = plot_training_curve(
        history,
        Path(__file__).resolve().parents[2] / "reports" / "training_curve.png",
        title=f"Transformer training curve — copy task ({config.name})",
    )
    logger.info("training curve -> %s", artifact)

    logger.info("greedy decode on held-out examples:")
    src, tgt = val_ds.tensor_pairs()
    for i in range(5):
        ids = src[i][src[i] != 0].tolist()
        gold = tgt[i][tgt[i] != 0].tolist()
        out = greedy_decode(model, torch.tensor(ids), tokenizer, max_steps=12)
        logger.info(
            "  src=%s  gold=%s  pred=%s  ok=%s",
            "".join(tokenizer.decode(ids)),
            "".join(tokenizer.decode(gold)),
            "".join(tokenizer.decode(out)),
            out == gold,
        )


if __name__ == "__main__":
    run()