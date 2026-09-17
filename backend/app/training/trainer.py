"""PHASE 5 — Trainer (plan.md §65 'Training loop').

Teacher-forced seq2seq training loop built from the paper's pieces:

    optimizer  Adam(beta1=0.9, beta2=0.98, eps=1e-9)   §5.3
    scheduler  Noam warmup, eq. 3                      §5.3
    loss       label smoothing eps_ls                   §5.4

Every epoch records train loss, train accuracy, validation metrics and
the current learning rate for the reports in Phase 6.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from app.core.config import PAPER_TRAINING
from app.training.checkpoint import save_best, save_checkpoint
from app.training.evaluator import evaluate, src_mask_for, tgt_mask_for
from app.training.loss import LabelSmoothingLoss
from app.training.optimizer import paper_adam
from app.training.scheduler import NoamScheduler

logger = logging.getLogger("vishustra.trainer")


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 32
    init_lr: float = 0.25
    warmup_steps: int = field(default_factory=lambda: PAPER_TRAINING.warmup_steps)
    max_len: int | None = None
    grad_clip: float | None = None
    checkpoint_dir: str = "models"
    run_name: str = "run"
    save_every: int | None = None  # None = save last + best only
    seed: int = 0


class Trainer:
    def __init__(self, model: nn.Module, config: TrainConfig, vocab_size: int, pad_id: int = 0) -> None:
        self.model = model
        self.config = config
        self.pad_id = pad_id
        self.optimizer = paper_adam(model, config.init_lr)
        self.scheduler = NoamScheduler(self.optimizer, model.config.d_model, config.warmup_steps)
        self.loss_fn = LabelSmoothingLoss(
            vocab_size, smoothing=model.config.label_smoothing, ignore_index=pad_id
        )

    def _forward(self, batch: dict) -> tuple[Tensor, Tensor]:
        tgt_len = batch["tgt_in"].shape[1]
        src_len = batch["src"].shape[1]
        logits, _, _ = self.model(
            batch["src"],
            batch["tgt_in"],
            src_mask=src_mask_for(batch, src_len),
            tgt_mask=tgt_mask_for(batch, tgt_len),
        )
        targets = batch["tgt_out"]
        loss = self.loss_fn(logits, targets)
        return logits, loss

    def train_epoch(self, loader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        for batch in loader:
            self.optimizer.zero_grad(set_to_none=True)
            logits, loss = self._forward(batch)
            loss.backward()
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            targets = batch["tgt_out"]
            valid = targets != self.pad_id
            n_valid = int(valid.sum())
            total_loss += float(loss.item()) * n_valid
            total_tokens += n_valid
            total_correct += int(((logits.argmax(-1) == targets) & valid).sum())
        return {
            "loss": total_loss / max(total_tokens, 1),
            "acc": total_correct / max(total_tokens, 1),
        }

    def fit(self, train_loader, val_loader) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        out_dir = Path(self.config.checkpoint_dir) / self.config.run_name
        best_val = float("inf")
        for epoch in range(1, self.config.epochs + 1):
            train_stats = self.train_epoch(train_loader)
            val_stats = evaluate(self.model, val_loader, self.loss_fn, ignore_index=self.pad_id)
            lr = self.scheduler.get_last_lr()[0]
            row = {
                "epoch": float(epoch),
                "train_loss": train_stats["loss"],
                "train_acc": train_stats["acc"],
                "val_loss": val_stats["loss"],
                "val_acc": val_stats["acc"],
                "val_ppl": val_stats["ppl"],
                "lr": lr,
            }
            history.append(row)
            logger.info(
                "epoch %d train_loss=%.4f val_loss=%.4f val_acc=%.3f val_ppl=%.2f lr=%.5f",
                epoch,
                train_stats["loss"],
                val_stats["loss"],
                val_stats["acc"],
                val_stats["ppl"],
                lr,
            )
            if self.config.save_every is None or epoch % self.config.save_every == 0:
                save_checkpoint(
                    out_dir / f"epoch_{epoch:03d}.pt",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    history,
                )
            if val_stats["loss"] < best_val:
                best_val = val_stats["loss"]
                save_best(out_dir / "best.pt", epoch, history)
        return history