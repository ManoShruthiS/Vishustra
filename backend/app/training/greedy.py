"""PHASE 5 — Greedy autoregressive decoding.

Teacher forcing trains the decoder; at inference it must generate its own
next token. `greedy_decode` runs the decoder loop step by step, passing
the previously-decoded tokens in for the next prediction (paper §3.1).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from app.datasets.tokenizer import CharacterTokenizer
from app.transformer import create_causal_mask, create_padding_mask


@torch.no_grad()
def greedy_decode(
    model: nn.Module,
    src_ids: Tensor,
    tokenizer: CharacterTokenizer,
    max_steps: int = 16,
) -> list[int]:
    """Autoregressive greedy decoding for a single (unbatched) source."""
    model.eval()
    pe_max = getattr(model.decoder.pe, "max_len", max_steps)
    max_steps = min(max_steps, pe_max)
    src_len = src_ids.shape[-1]
    src_mask = create_padding_mask(torch.tensor([src_len]), src_len)[:, None, :]

    decoded: list[int] = []
    prompt = torch.full((1, 1), tokenizer.sos_id, dtype=torch.long)
    for _ in range(max_steps):
        tgt_len = prompt.shape[1]
        tgt_mask = create_causal_mask(tgt_len)[None, :, :]
        logits, _, _ = model(src_ids[None, :], prompt, src_mask=src_mask, tgt_mask=tgt_mask)
        next_id = int(logits[:, -1, :].argmax(-1).item())
        if next_id == tokenizer.eos_id:
            break
        decoded.append(next_id)
        prompt = torch.cat([prompt, torch.full((1, 1), next_id, dtype=torch.long)], dim=1)
    return decoded