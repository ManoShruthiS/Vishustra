"""PHASE 5 — Preprocessing / batching (plan.md §65 'Batching').

Turns variable-length examples into fixed batches for teacher forcing:

    src      -> padded source ids,            src_mask (padding over source)
    tgt_in   -> [SOS] + tgt ids               (decoder input, shifted right)
    tgt_out  -> tgt ids + [EOS]               (targets to predict)
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from app.datasets.tokenizer import CharacterTokenizer


def _pad(rows: list[list[int]], length: int, pad_id: int) -> Tensor:
    out = torch.full((len(rows), length), pad_id, dtype=torch.long)
    for i, row in enumerate(rows):
        out[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    return out


def collate_batch(
    batch: list[tuple[list[int], list[int]]],
    tokenizer: CharacterTokenizer,
    max_len: int | None = None,
) -> dict[str, Any]:
    """Collate function for teacher-forced seq2seq batches."""
    src_rows, tgt_rows = zip(*batch)
    src_rows = list(src_rows)
    tgt_rows = list(tgt_rows)

    max_src = max(max(len(r) for r in src_rows), 1)
    max_tgt = max(max(len(r) + 1 for r in tgt_rows), 1)  # SOS prefix / EOS suffix
    if max_len is not None:
        max_src = min(max_src, max_len)
        max_tgt = min(max_tgt, max_len)

    src = _pad(src_rows, max_src, tokenizer.pad_id)
    tgt_input = _pad([[tokenizer.sos_id, *r][:max_tgt] for r in tgt_rows], max_tgt, tokenizer.pad_id)
    tgt_output = _pad([[*r, tokenizer.eos_id][:max_tgt] for r in tgt_rows], max_tgt, tokenizer.pad_id)

    src_lengths = torch.tensor([min(len(r), max_src) for r in src_rows])
    tgt_lengths = torch.tensor([min(len(r) + 1, max_tgt) for r in tgt_rows], dtype=torch.long)

    return {
        "src": src,
        "tgt_in": tgt_input,
        "tgt_out": tgt_output,
        "src_len": src_lengths,
        "tgt_len": tgt_lengths,
    }