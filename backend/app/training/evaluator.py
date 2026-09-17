"""PHASE 5 — Evaluation (plan.md §65 'Evaluation').

Computes held-out loss, token accuracy and perplexity over validation
data without updating parameters. Masks are built here (rather than in
the collate) so the full batch/head dimensions broadcast correctly.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from app.transformer import create_causal_mask, create_padding_mask


def tgt_mask_for(batch: dict, tgt_len: int) -> Tensor:
    """Causal + padding mask over decoder inputs: (batch, tgt_len, tgt_len).

    Per-head scores live in the multi-head implementation as 3D
    (batch, q_len, k_len), so masks must broadcast to that rank.
    """
    causal = create_causal_mask(tgt_len)
    padding = create_padding_mask(batch["tgt_len"], tgt_len)
    return causal[None, :, :] + padding[:, None, :]


def src_mask_for(batch: dict, src_len: int) -> Tensor:
    """Padding mask over encoder inputs: (batch, 1, src_len)."""
    return create_padding_mask(batch["src_len"], src_len)[:, None, :]


def evaluate(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    ignore_index: int = -100,
) -> dict[str, float]:
    """Average loss, token accuracy and perplexity over a validation loader."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    with torch.no_grad():
        for batch in loader:
            tgt_len = batch["tgt_in"].shape[1]
            src_len = batch["src"].shape[1]
            logits, _, _ = model(
                batch["src"],
                batch["tgt_in"],
                src_mask=src_mask_for(batch, src_len),
                tgt_mask=tgt_mask_for(batch, tgt_len),
            )
            targets = batch["tgt_out"]
            valid = targets != ignore_index
            loss = loss_fn(logits, targets)
            n_valid = int(valid.sum())
            total_loss += float(loss.item()) * n_valid
            total_tokens += n_valid
            total_correct += int(((logits.argmax(-1) == targets) & valid).sum())
    mean_loss = total_loss / max(total_tokens, 1)
    acc = total_correct / max(total_tokens, 1)
    return {"loss": mean_loss, "acc": acc, "ppl": math.exp(min(mean_loss, 20.0))}