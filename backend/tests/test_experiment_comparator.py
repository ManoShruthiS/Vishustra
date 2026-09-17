"""PHASE 7 — Experiment comparator tests (plan.md §67 Comparison, §45)."""

import csv

from app.experiments import ExperimentManager
from app.experiments.comparator import plot_comparison, to_csv


def _completed(manager: ExperimentManager) -> list[str]:
    ids = [
        manager.create(f"compare-{i}", seed=i,
                       architecture={"num_encoder_layers": 1, "d_model": 8, "num_heads": h,
                                     "d_k": 4, "d_v": 4, "label_smoothing": 0.0, "max_len": 6},
                       training={"epochs": 1, "batch_size": 8, "warmup_steps": 3},
                       dataset={"alphabet": "abc", "n_examples": 24, "val_n_examples": 6, "min_len": 2, "max_len": 3})
        for i, h in enumerate([1, 2])
    ]
    manager.run_all(ids)
    return ids


def test_compare_builds_table(tmp_path):
    manager = ExperimentManager(tmp_path)
    ids = _completed(manager)
    comparison = manager.compare(ids)
    assert len(comparison.rows) == 2
    for row in comparison.completed():
        assert row.status == "completed"
        assert row.val_acc is not None
        assert row.epochs == 1
        assert row.exp_id in ids


def test_plot_comparison_renders_chart(tmp_path):
    manager = ExperimentManager(tmp_path)
    ids = _completed(manager)
    path = plot_comparison(manager.compare(ids), metric="val_acc", out_path=tmp_path / "chart.png")
    assert path.exists() and path.stat().st_size > 0


def test_plot_comparison_rejects_unknown_metric(tmp_path):
    manager = ExperimentManager(tmp_path)
    try:
        plot_comparison(manager.compare(), metric="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_to_csv_exports_rows(tmp_path):
    manager = ExperimentManager(tmp_path)
    ids = _completed(manager)
    path = to_csv(manager.compare(ids), tmp_path / "sweep.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["exp_id"] in ids
    assert rows[0]["status"] == "completed"
    assert rows[0]["val_acc"]