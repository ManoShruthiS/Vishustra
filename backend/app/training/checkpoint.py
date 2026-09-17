"""PHASE 5 — Checkpointing (plan.md §35 training/checkpoint.py).

Saves and restores the full training run-state: model, optimizer,
scheduler, configuration, tokenizer vocabulary and metric history.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Any,
    scheduler: Any,
    epoch: int,
    history: list[dict[str, float]],
    vocab: list[str] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "config": model.config,
        "epoch": epoch,
        "history": history,
        "vocab": vocab,
    }
    torch.save(state, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Any = None,
    scheduler: Any = None,
) -> dict[str, Any]:
    state = torch.load(path, weights_only=False)
    model.load_state_dict(state["model_state"])
    if optimizer is not None and state.get("optimizer_state") is not None:
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and state.get("scheduler_state") is not None:
        scheduler.load_state_dict(state["scheduler_state"])
    return state


def save_best(path: str | Path, epoch: int, history: list[dict[str, float]]) -> Path:
    """Copy convenience for the best checkout — writes alongside the run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    best = min(history, key=lambda h: h["val_loss"]) if history else {}
    torch.save({"epoch": epoch, "best": best}, path)
    return path