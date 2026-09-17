"""Training package — optimizer, scheduler, loss, trainer, eval, checkpoints (plan.md §35)."""

from app.training.checkpoint import load_checkpoint, save_best, save_checkpoint
from app.training.evaluator import evaluate, src_mask_for, tgt_mask_for
from app.training.greedy import greedy_decode
from app.training.loss import LabelSmoothingLoss
from app.training.optimizer import paper_adam
from app.training.scheduler import NoamScheduler, noam_lr
from app.training.trainer import TrainConfig, Trainer

__all__ = [
    "LabelSmoothingLoss",
    "NoamScheduler",
    "TrainConfig",
    "Trainer",
    "evaluate",
    "greedy_decode",
    "load_checkpoint",
    "noam_lr",
    "paper_adam",
    "save_best",
    "save_checkpoint",
    "src_mask_for",
    "tgt_mask_for",
]