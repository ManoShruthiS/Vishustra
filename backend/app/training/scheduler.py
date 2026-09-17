"""PHASE 5 — Noam learning-rate schedule (paper §5.3, eq. 3).

    lr = d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)

Linear warmup for the first `warmup` steps, then inverse-sqrt decay.
"""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def noam_lr(d_model: float, step: int, warmup_steps: int) -> float:
    """Equation 3 of the paper, a pure function — easily unit-tested."""
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)


class NoamScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, d_model: int, warmup_steps: int = 4000) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._noam_step = 0
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        self._noam_step += 1
        lr = noam_lr(self.d_model, self._noam_step, self.warmup_steps)
        return [lr for _ in self.base_lrs]

    @property
    def step_count(self) -> int:
        return self._noam_step