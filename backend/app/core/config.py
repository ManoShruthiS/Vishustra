"""Paper-locked configuration.

These constants are lifted verbatim from *Attention Is All You Need*
(Vaswani et al., 2017) — Table 3 (model variations), §5.3 (optimizer),
§5.4 (regularization). They are the scientific anchors of VISHUSTRA:
every experiment compares against the base configuration the paper reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PaperConfig:
    """A Transformer configuration as defined in the paper (Table 3)."""

    name: str
    num_encoder_layers: int        # N
    num_decoder_layers: int        # N
    d_model: int
    d_ff: int
    num_heads: int                 # h
    d_k: int
    d_v: int
    dropout: float                 # P_drop
    label_smoothing: float         # eps_ls
    train_steps: int
    params_millions: int
    notes: tuple[str, ...] = field(default_factory=tuple)


BASE_CONFIG = PaperConfig(
    name="base",
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_model=512,
    d_ff=2048,
    num_heads=8,
    d_k=64,
    d_v=64,
    dropout=0.1,
    label_smoothing=0.1,
    train_steps=100_000,
    params_millions=65,
    notes=(
        "Table 3 row 1: train-perplexity 4.92, dev BLEU 25.8 (newstest2013).",
        "WMT14 EN-DE: 27.3 BLEU; EN-FR: 38.1 BLEU (Table 2).",
    ),
)

BIG_CONFIG = PaperConfig(
    name="big",
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_model=1024,
    d_ff=4096,
    num_heads=16,
    d_k=64,
    d_v=64,
    dropout=0.3,
    label_smoothing=0.1,
    train_steps=300_000,
    params_millions=213,
    notes=(
        "Table 3 bottom line: train-perplexity 4.33, dev BLEU 26.4.",
        "WMT14 EN-DE: 28.4 BLEU; EN-FR: 41.8 BLEU (Table 2).",
    ),
)


@dataclass(frozen=True)
class PaperTraining:
    """Training regime constants from §5.3 and §5.4."""

    optimizer: str = "Adam"
    beta1: float = 0.9
    beta2: float = 0.98
    epsilon: float = 1e-9
    warmup_steps: int = 4000
    label_smoothing: float = 0.1


PAPER_TRAINING = PaperTraining()


def paper_configs() -> list[PaperConfig]:
    return [BASE_CONFIG, BIG_CONFIG]