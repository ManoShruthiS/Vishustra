"""PHASE 5 — Train/validation splitter (plan.md §65, §55).

Splits a dataset into train and validation views deterministically.
"""

from __future__ import annotations

import torch


class SplitView:
    """A read-only view over a fraction of a dataset."""

    def __init__(self, dataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]


def train_val_split(dataset, val_fraction: float = 0.1, seed: int = 0) -> tuple[SplitView, SplitView]:
    """Deterministic train/validation split over a dataset's indices."""
    assert 0.0 < val_fraction < 1.0
    gen = torch.Generator().manual_seed(seed)
    n = len(dataset)
    indices = torch.randperm(n, generator=gen).tolist()
    n_val = int(n * val_fraction)
    return SplitView(dataset, indices[n_val:]), SplitView(dataset, indices[:n_val])