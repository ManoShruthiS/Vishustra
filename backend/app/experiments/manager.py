"""PHASE 7 — Experiment manager (plan.md §67 'Creation', §15 RE-RUN).

High-level orchestration over the registry + runner: create single
experiments or grid sweeps (plan.md §14), run them, re-run ("same
configuration → new experiment", §15) and compare results.
"""

from __future__ import annotations

from pathlib import Path

from app.experiments.comparator import Comparison, compare
from app.experiments.config import ExperimentConfig, generate_grid
from app.experiments.registry import ExperimentRegistry
from app.experiments.runner import ExperimentResult, run_experiment


class ExperimentManager:
    def __init__(self, root: str | Path | None = None) -> None:
        root = Path(root) if root is not None else Path(__file__).resolve().parents[2] / "experiments"
        self.registry = ExperimentRegistry(root)

    # -- creation ---------------------------------------------------------
    def create(self, name: str, hypothesis: str = "", **sections) -> str:
        """Create a single experiment from defaults plus partial overrides."""
        config = ExperimentConfig(name=name, hypothesis=hypothesis)
        config = _merge(config, sections)
        return self.registry.create(config)

    def create_grid(self, name: str, overrides: dict, hypothesis: str = "") -> list[str]:
        """Create one experiment per cartesian combination (plan.md §14)."""
        base = ExperimentConfig(name=name, hypothesis=hypothesis)
        grid = generate_grid(base, overrides)
        return [self.registry.create(cfg) for cfg in grid]

    def rerun(self, exp_id: str) -> str:
        """RE-RUN: same configuration produces a brand-new experiment (§15)."""
        config = self.registry.get(exp_id).config_obj
        new_id = self.registry.create(config)
        return new_id

    # -- execution ----------------------------------------------------------
    def run(self, exp_id: str, save_model: bool = True) -> ExperimentResult:
        return run_experiment(self.registry, exp_id, save_model=save_model)

    def run_all(self, ids: list[str] | None = None) -> list[ExperimentResult]:
        target = ids or [r.exp_id for r in self.registry.where_status("created")]
        return [self.run(exp_id) for exp_id in target]

    # -- reading ------------------------------------------------------------
    def get(self, exp_id: str):
        return self.registry.get(exp_id)

    def list(self):
        return self.registry.all()

    def compare(self, ids: list[str] | None = None) -> Comparison:
        return compare(self.registry, ids)


def _merge(config: ExperimentConfig, sections: dict) -> ExperimentConfig:
    """Apply dotted ``architecture.num_heads``-style overrides onto config."""
    data = config.to_dict()
    for key, value in sections.items():
        if key in ("dataset", "architecture", "training"):
            data[key].update(value)
        else:
            data[key] = value
    return ExperimentConfig.from_dict(data)