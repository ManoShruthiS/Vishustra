"""PHASE 7 — Demo: LAB 10, the Experiment Engine (plan.md §14).

Builds the plan.md §14 sweep end to end: define value lists (heads,
layers), let VISHUSTRA generate one experiment per combination, execute
every run with logging/metrics, store reproducibility packages, compare
the results in a chart + CSV, and demonstrate RE-RUN (§15: same
configuration -> brand-new experiment).

Everything lives under ``backend/experiments/demo_sweep``; artifacts go
to ``reports/experiments``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.experiments import ExperimentManager, plot_comparison, to_csv

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("vishustra.experiments")

ROOT = Path(__file__).resolve().parents[2] / "experiments" / "demo_sweep"
REPORTS = Path(__file__).resolve().parents[2] / "reports" / "experiments"


def run() -> None:
    manager = ExperimentManager(ROOT)

    logger.info("generating sweep: num_heads x num_layers (plan 14)")
    grid_ids = manager.create_grid(
        "heads-sweep",
        {
            "architecture": {"num_heads": [1, 2, 4], "num_encoder_layers": [1, 2]},
        },
        hypothesis="Different numbers of attention heads may produce different attention patterns and performance.",
    )
    for exp_id in grid_ids:
        record = manager.get(exp_id)
        logger.info("  created  %s  %s", exp_id, record.config_obj.summary())

    logger.info("executing sweep...")
    manager.run_all()

    comparison = manager.compare()
    header = f"{'EXP':<10} {'head/layer':<12} {'val_acc':>8} {'val_ppl':>8} {'params':>9} {'train_s':>8} {'entropy':>8}"
    logger.info(header)
    for row in comparison.completed():
        logger.info(
            "%-10s %-12s %8.3f %8.2f %9d %8.2f %8.3f",
            row.exp_id, row.model_name, row.val_acc or 0, row.val_ppl or 0,
            int(row.params_millions * 1e6) if row.params_millions else 0,
            row.train_time_s or 0, row.attention_entropy or 0,
        )

    chart = plot_comparison(
        comparison, metric="val_acc",
        title="Head × layer sweep — validation accuracy (LAB 10)",
        out_path=REPORTS / "heads_sweep_val_acc.png",
    )
    csv_path = to_csv(comparison, REPORTS / "heads_sweep.csv")
    logger.info("chart  -> %s", chart)
    logger.info("csv    -> %s", csv_path)

    first = grid_ids[0]
    logger.info("RE-RUN demo (plan 15): re-running %s with the same configuration", first)
    rerun_id = manager.rerun(first)
    manager.run(rerun_id)
    logger.info("reproduced as new experiment %s (same config, new run)", rerun_id)
    for row in manager.compare([rerun_id]).completed():
        logger.info("  %s  val_acc=%.3f  val_ppl=%.2f", rerun_id, row.val_acc or 0, row.val_ppl or 0)

    logger.info("sweep done: %d experiments under %s", len(list(ROOT.iterdir())), ROOT)

    best = max((row for row in comparison.completed()), key=lambda r: r.val_acc or 0)
    logger.info("best configuration: %s (%s, val_acc=%.3f)", best.exp_id, best.model_name, best.val_acc)


if __name__ == "__main__":
    run()