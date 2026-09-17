"""PHASE 7 — Experiment runner tests (plan.md §67 Execution/Metrics, §31 reproduce)."""

from app.experiments import ExperimentManager, ExperimentRegistry
from app.experiments.config import ExperimentConfig
from app.experiments.runner import run_experiment


def _tiny_cfg() -> ExperimentConfig:
    return ExperimentConfig(
        name="runner-test",
        seed=1,
        dataset={"alphabet": "abcd", "n_examples": 40, "val_n_examples": 10, "min_len": 2, "max_len": 4, "content_start": 3},
        architecture={"num_encoder_layers": 1, "num_decoder_layers": 1, "d_model": 8, "num_heads": 2,
                      "d_k": 4, "d_v": 4, "dropout": 0.0, "label_smoothing": 0.0, "max_len": 6},
        training={"epochs": 2, "batch_size": 8, "init_lr": 0.5, "warmup_steps": 5},
    )


def test_run_experiment_produces_reproducibility_package(tmp_path):
    registry = ExperimentRegistry(tmp_path)
    exp_id = registry.create(_tiny_cfg())
    result = run_experiment(registry, exp_id)

    assert registry.get(exp_id).status == "completed"
    assert result.metrics["val_acc"] >= 0.0
    assert result.metrics["train_time_s"] > 0.0
    assert result.metrics["params"] > 0
    assert 0.0 <= result.metrics["attention_entropy"]
    assert 0.0 <= result.metrics["attention_concentration"] <= 1.0

    exp_dir = tmp_path / exp_id
    package = exp_dir / "reproducibility.json"
    metrics = exp_dir / "metrics.json"
    assert package.exists() and metrics.exists()
    assert (exp_dir / "run.log").exists()
    assert (exp_dir / "history.json").exists()

    import json

    repro = json.loads(package.read_text())
    assert repro["experiment_id"] == exp_id
    assert repro["random_seed"] == _tiny_cfg().seed
    assert len(repro["dataset_version"]) == 12
    assert "python" in repro["software_environment"]
    assert repro["metrics"]["val_acc"] == result.metrics["val_acc"]

    m = json.loads(metrics.read_text())
    assert m["epochs"] == 2.0
    assert m["val_ppl"] > 1.0


def test_run_all_and_rerun(tmp_path):
    manager = ExperimentManager(tmp_path)
    first = manager.create("native", seed=2, architecture={"num_encoder_layers": 1, "d_model": 8, "num_heads": 2, "d_k": 4, "d_v": 4, "label_smoothing": 0.0, "max_len": 6}, training={"epochs": 1, "batch_size": 8, "warmup_steps": 3}, dataset={"alphabet": "abc", "n_examples": 24, "val_n_examples": 6, "min_len": 2, "max_len": 3})
    results = manager.run_all()  # runs the one created experiment
    assert len(results) == 1
    assert results[0].exp_id == first

    rerun_id = manager.rerun(first)
    assert rerun_id != first
    assert manager.get(rerun_id).config["seed"] == manager.get(first).config["seed"]


def test_rerun_same_config(tmp_path):
    registry = ExperimentRegistry(tmp_path)
    exp_id = registry.create(_tiny_cfg())
    run_experiment(registry, exp_id)
    second = registry.create(registry.get(exp_id).config_obj)
    run_experiment(registry, second)
    a = registry.get(exp_id).result
    b = registry.get(second).result
    assert a["params"] == b["params"]  # same architecture → same parameter count