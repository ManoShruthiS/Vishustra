"""PHASE 5 — Data loader helper (plan.md §35 datasets/loader.py).

Wraps the toy tasks in PyTorch DataLoaders with the Vishustra collate
function so seq2seq batches carry src/tgt/masks ready to train on.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from app.datasets.preprocessing import collate_batch
from app.datasets.tokenizer import CharacterTokenizer


def make_loaders(
    train_dataset,
    val_dataset,
    tokenizer: CharacterTokenizer,
    batch_size: int = 32,
    max_len: int | None = None,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    collate = lambda batch: collate_batch(batch, tokenizer, max_len=max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader


def seed_all(seed: int) -> None:
    """Reproducible demo/training runs (plan.md §31 reproducibility)."""
    torch.manual_seed(seed)