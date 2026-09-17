"""PHASE 5 — Training-curve plotter.

Renders loss and accuracy over epochs from a trainer `history`, saving
to `reports/` for the laboratory (plan.md §42, LAB 07).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curve(history: list[dict], out_path: str | Path, title: str = "Training Curve") -> Path:
    """Loss + accuracy curves; saving the figure to `out_path`."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, train_loss, marker="o", label="train")
    ax1.plot(epochs, val_loss, marker="s", label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("epoch")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_acc, marker="o", label="train")
    ax2.plot(epochs, val_acc, marker="s", label="val")
    ax2.set_title("Token accuracy")
    ax2.set_xlabel("epoch")
    ax2.set_ylim(0.0, 1.05)
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path