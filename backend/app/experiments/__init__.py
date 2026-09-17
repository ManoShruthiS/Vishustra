"""PHASE 7 — Experiment engine (plan.md §14 LAB 10, §67).

Experiment creation, configuration, execution, logging, metrics, storage
and comparison.  Experiments live under ``backend/experiments/`` as
``EXP-XXXX`` directories with a shared ``index.json`` registry.
"""

from app.experiments.comparator import Comparison, ComparisonRow, compare, plot_comparison, to_csv
from app.experiments.config import (
    DEFAULT_ARCHITECTURE,
    DEFAULT_DATASET,
    DEFAULT_TRAINING,
    ExperimentConfig,
    generate_grid,
)
from app.experiments.manager import ExperimentManager
from app.experiments.registry import ExperimentRecord, ExperimentRegistry
from app.experiments.runner import ExperimentResult, run_experiment

__all__ = [
    "DEFAULT_ARCHITECTURE",
    "DEFAULT_DATASET",
    "DEFAULT_TRAINING",
    "Comparison",
    "ComparisonRow",
    "ExperimentConfig",
    "ExperimentManager",
    "ExperimentRecord",
    "ExperimentRegistry",
    "ExperimentResult",
    "compare",
    "generate_grid",
    "plot_comparison",
    "run_experiment",
    "to_csv",
]