"""PHASE 7 — Experiment comparison (plan.md §67 'Comparison', §45 'Collect').

Flattens completed experiment results into a table, exports it as CSV and
renders monochrome bar charts (by validation accuracy or another metric)
so a sweep like plan.md §14 can be read at a glance.
"""

from __future__ import annotations

import csv
import dataclasses
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.experiments.registry import ExperimentRegistry

METRICS = ("val_acc", "val_loss", "val_ppl", "train_time_s", "params_millions",
           "attention_entropy", "attention_concentration")


@dataclass(frozen=True)
class ComparisonRow:
    exp_id: str
    name: str
    model_name: str
    status: str
    val_loss: float | None
    val_acc: float | None
    val_ppl: float | None
    train_time_s: float | None
    params_millions: float | None
    attention_entropy: float | None
    attention_concentration: float | None
    epochs: int | None

    @classmethod
    def from_record(cls, record) -> ComparisonRow:
        cfg = record.config_obj
        r = record.result or {}
        return cls(
            exp_id=record.exp_id,
            name=record.name,
            model_name=cfg.model_name(),
            status=record.status,
            val_loss=r.get("val_loss"),
            val_acc=r.get("val_acc"),
            val_ppl=r.get("val_ppl"),
            train_time_s=r.get("train_time_s"),
            params_millions=r.get("params_millions"),
            attention_entropy=r.get("attention_entropy"),
            attention_concentration=r.get("attention_concentration"),
            epochs=int(r["epochs"]) if r.get("epochs") is not None else None,
        )


@dataclass(frozen=True)
class Comparison:
    rows: list[ComparisonRow]

    def completed(self) -> list[ComparisonRow]:
        return [r for r in self.rows if r.status == "completed"]

    def by_metric(self, metric: str) -> list[ComparisonRow]:
        return sorted(
            self.completed(),
            key=lambda r: (getattr(r, metric) is None, getattr(r, metric)),
            reverse=True,
        )


def compare(registry: ExperimentRegistry, ids: list[str] | None = None) -> Comparison:
    records = registry.all() if ids is None else [registry.get(i) for i in ids]
    return Comparison([ComparisonRow.from_record(r) for r in records])


def to_csv(comparison: Comparison, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [f.name for f in dataclasses.fields(ComparisonRow)]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in comparison.rows:
            writer.writerow(asdict(row))
    return out_path


def plot_comparison(
    comparison: Comparison,
    metric: str = "val_acc",
    *,
    title: str | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Horizontal monochrome bar chart of one metric across experiments."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric: {metric!r}")
    rows = comparison.by_metric(metric)
    labels = [f"{r.exp_id} - {r.model_name}" for r in rows]
    values = [getattr(r, metric) for r in rows]

    fig, ax = plt.subplots(figsize=(7, max(2.6, 0.45 * len(labels))))
    colors = ["#333333" if i % 2 == 0 else "#666666" for i in range(len(labels))]
    ax.barh(labels, values, color=colors)
    ax.set_xlabel(metric)
    ax.set_title(title or f"Experiment sweep — {metric}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = Path(out_path) if out_path is not None else Path("<inline>")
    if out_path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return path