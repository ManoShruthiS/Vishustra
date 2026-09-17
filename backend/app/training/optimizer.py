"""PHASE 5 — Paper-configured Adam optimizer (paper §5.3).

The Transformer paper uses Adam with beta1=0.9, beta2=0.98 and
epsilon=1e-9 (section 5.3). `paper_adam` builds a torch optimizer with
exactly those hyperparameters.
"""

from __future__ import annotations

from torch import nn
from torch.optim import Adam

from app.core.config import PAPER_TRAINING

PAPER_BETAS = (PAPER_TRAINING.beta1, PAPER_TRAINING.beta2)
PAPER_EPS = PAPER_TRAINING.epsilon


def paper_adam(model: nn.Module, lr: float) -> Adam:
    """Adam with the paper's exact hyperparameters (§5.3: beta1, beta2, eps).

    ``foreach=True`` vectorizes the per-parameter updates; on CPU this is
    several times faster than torch's default single-tensor loop.
    """
    return Adam(model.parameters(), lr=lr, betas=PAPER_BETAS, eps=PAPER_EPS, foreach=True)