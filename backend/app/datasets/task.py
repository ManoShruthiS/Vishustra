"""PHASE 5 — Toy seq2seq task: a learnable copy dataset (plan.md §65).

A small, CPU-fast sequence-to-sequence task used to exercise the full
encoder + decoder + cross-attention pipeline. Every example is a return
(label-hidden) copy — or, optionally, reversal — of a short random token
sequence over a fixed alphabet.
"""

from __future__ import annotations

import torch
from torch import Tensor


class CopyTask:
    """Deterministic source/target pairs of integer token ids.

    ``(src, tgt)`` both hold python lists of ids WITHOUT special tokens;
    the collate/preprocessing step adds ``SOS``/``EOS``.
    """

    def __init__(
        self,
        vocab_size: int,
        n_examples: int,
        min_len: int = 2,
        max_len: int = 6,
        seed: int = 0,
        reverse: bool = False,
        content_start: int = 0,
    ) -> None:
        self.vocab_size = vocab_size
        self.n_examples = n_examples
        self.min_len = min_len
        self.max_len = max_len
        self.reverse = reverse
        self.content_start = content_start
        gen = torch.Generator().manual_seed(seed)
        self.pairs: list[tuple[list[int], list[int]]] = []
        for _ in range(n_examples):
            length = int(torch.randint(min_len, max_len + 1, (1,), generator=gen))
            src = torch.randint(content_start, vocab_size, (length,), generator=gen).tolist()
            tgt = list(reversed(src)) if reverse else list(src)
            self.pairs.append((src, tgt))

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.pairs[idx]

    def tensor_pairs(self) -> tuple[Tensor, Tensor]:
        """Stacked (src, tgt) tensors for tests — padded to the max length."""
        max_len = max(len(s) for s, _ in self.pairs)
        src = torch.zeros(len(self), max_len, dtype=torch.long)
        tgt = torch.zeros(len(self), max_len, dtype=torch.long)
        for i, (s, t) in enumerate(self.pairs):
            src[i, : len(s)] = torch.tensor(s)
            tgt[i, : len(t)] = torch.tensor(t)
        return src, tgt