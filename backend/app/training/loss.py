"""PHASE 5 — Label-smoothing cross-entropy (paper §5.4).

Instead of one-hot targets, the model is asked to assign probability:

    (1 - eps) * one_hot(y) + eps / V

This improves generalization and makes the model less over-confident.
All arithmetic is done on log-probabilities using our `log_softmax`
(plan.md §81 traceability), and padded positions are ignored.
"""

from __future__ import annotations

from torch import Tensor, nn

from app.transformer import log_softmax


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = -100) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Batch-average smoothing loss over non-ignored target positions.

        ``logits``: (batch, tgt_len, vocab); ``targets``: (batch, tgt_len).
        """
        log_probs = log_softmax(logits, dim=-1)
        valid = targets != self.ignore_index
        safe = targets.clamp(min=0)
        nll_gt = -log_probs.gather(dim=-1, index=safe.unsqueeze(-1)).squeeze(-1)
        nll_avg = -log_probs.mean(dim=-1)

        # Exact closed form of the smoothed cross-entropy (paper §5.4):
        #   loss = (1-eps) * nll_gt + eps * nll_avg
        losses = (1.0 - self.smoothing) * nll_gt + self.smoothing * nll_avg

        if valid.any():
            return losses[valid].mean()
        return losses.sum() * 0.0