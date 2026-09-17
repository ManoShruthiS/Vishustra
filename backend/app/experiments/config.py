"""PHASE 7 — Experiment configuration (plan.md §14 LAB 10, §67 'Configuration').

An :class:`ExperimentConfig` is the complete, serializable description of
one laboratory run: dataset, architecture and training hyperparameters.
``generate_grid`` builds the cartesian sweep from plan.md §14 — the user
defines value lists (heads, layers, d_model) and VISHUSTRA creates one
experiment per combination.
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import asdict, dataclass, field
from typing import Any

from app.core.config import PaperConfig

DEFAULT_DATASET = {
    "alphabet": "abcdefghij",
    "n_examples": 600,
    "val_n_examples": 100,
    "min_len": 2,
    "max_len": 6,
    "content_start": 3,
}

DEFAULT_ARCHITECTURE = {
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "d_model": 16,
    "d_ff": None,  # resolved to 2 * d_model unless overridden
    "num_heads": 4,
    "d_k": None,  # resolved to d_model // num_heads unless overridden
    "d_v": None,  # resolved to d_model // num_heads unless overridden
    "dropout": 0.0,
    "label_smoothing": 0.1,
    "max_len": 8,
}

DEFAULT_TRAINING = {
    "epochs": 6,
    "batch_size": 32,
    "init_lr": 0.5,
    "warmup_steps": 100,
}


@dataclass(frozen=True)
class ExperimentConfig:
    """One experiment's full specification (plan.md §15 experiment record)."""

    name: str
    hypothesis: str = ""
    notes: str = ""
    seed: int = 42
    dataset: dict = field(default_factory=lambda: dict(DEFAULT_DATASET))
    architecture: dict = field(default_factory=lambda: dict(DEFAULT_ARCHITECTURE))
    training: dict = field(default_factory=lambda: dict(DEFAULT_TRAINING))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ExperimentConfig:
        return cls(**data)

    def _architecture(self) -> dict:
        return {**DEFAULT_ARCHITECTURE, **self.architecture}

    def model_name(self, sep: str = "x") -> str:
        a = self._architecture()
        dims = f"H{a['num_heads']}"
        dims += f"{sep}L{a['num_encoder_layers']}"
        dims += f"{sep}d{a['d_model']}"
        return dims

    def summary(self) -> str:
        return f"{self.name} ({self.model_name()})"

    def as_paper_config(self) -> PaperConfig:
        """Resolve the architecture dict into a paper-locked PaperConfig."""
        a = self._architecture()
        d_model = int(a["d_model"])
        d_ff = int(a.get("d_ff") or 2 * d_model)
        return PaperConfig(
            name=self.name,
            num_encoder_layers=int(a["num_encoder_layers"]),
            num_decoder_layers=int(a["num_decoder_layers"]),
            d_model=d_model,
            d_ff=d_ff,
            num_heads=int(a["num_heads"]),
            d_k=int(a.get("d_k") or d_model // int(a["num_heads"])),
            d_v=int(a.get("d_v") or d_model // int(a["num_heads"])),
            dropout=float(a.get("dropout") or 0.0),
            label_smoothing=float(a.get("label_smoothing") or 0.0),
            train_steps=0,
            params_millions=0,
            notes=(self.hypothesis, self.notes),
        )


def generate_grid(
    base: ExperimentConfig,
    overrides: dict[str, dict[str, list[Any]]],
) -> list[ExperimentConfig]:
    """Cartesian sweep over per-plugin value lists (plan.md §14).

    ``overrides`` maps section → parameter → list of values, e.g.::

        {"architecture": {"num_heads": [1, 2, 4],
                          "num_encoder_layers": [1, 2]},
         "training":     {"epochs": [6, 10]}}

    Every combination becomes one ``ExperimentConfig`` with the same
    name and a zero-based variant tag in the notes.
    """
    axes: list[tuple[str, str, list[Any]]] = []
    for section, params in overrides.items():
        if section not in ("dataset", "architecture", "training"):
            raise ValueError(f"unknown config section: {section!r}")
        for key, values in params.items():
            axes.append((section, key, list(values)))

    grid: list[ExperimentConfig] = []
    if not axes:
        return [base]
    values_lists = [values for _, _, values in axes]
    for combo in itertools.product(*values_lists):
        cfg_dict = base.to_dict()
        for (section, key, _), value in zip(axes, combo):
            cfg_dict[section][key] = value
        cfg = ExperimentConfig.from_dict(cfg_dict)
        grid.append(cfg)
    for i, cfg in enumerate(grid):
        grid[i] = dataclasses.replace(cfg, notes=base.notes + f" [grid variant {i + 1}]")
    return grid