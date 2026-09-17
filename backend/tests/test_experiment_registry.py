"""PHASE 7 — Experiment registry tests (plan.md §14 ids, §15 tracking, §16 notes)."""

from app.experiments.config import ExperimentConfig
from app.experiments.registry import ExperimentRegistry


def test_create_assigns_sequential_ids(tmp_path):
    registry = ExperimentRegistry(tmp_path)
    cfg = ExperimentConfig(name="a")
    ids = [registry.create(cfg) for _ in range(3)]
    assert ids == ["EXP-0001", "EXP-0002", "EXP-0003"]


def test_index_persists_across_instances(tmp_path):
    registry = ExperimentRegistry(tmp_path)
    registry.create(ExperimentConfig(name="persisted", seed=7))
    reloaded = ExperimentRegistry(tmp_path)
    assert len(reloaded) == 1
    record = reloaded.get("EXP-0001")
    assert record.name == "persisted"
    assert record.config["seed"] == 7
    assert record.status == "created"


def test_update_and_result(tmp_path):
    registry = ExperimentRegistry(tmp_path)
    exp_id = registry.create(ExperimentConfig(name="x"))
    registry.update(exp_id, status="running")
    registry.update(exp_id, status="completed", result={"val_acc": 0.9})
    record = registry.get(exp_id)
    assert record.status == "completed"
    assert record.result["val_acc"] == 0.9


def test_notes_append(tmp_path):
    registry = ExperimentRegistry(tmp_path)
    exp_id = registry.create(ExperimentConfig(name="x"))
    registry.add_note(exp_id, "hypothesis: more heads helps")
    record = registry.get(exp_id)
    assert record.notes == ["hypothesis: more heads helps"]


def test_same_config_new_experiment(tmp_path):
    # §15: reproducing = same configuration, new experiment
    registry = ExperimentRegistry(tmp_path)
    config = ExperimentConfig(name="repeat", seed=1)
    first = registry.create(config)
    second = registry.create(ExperimentConfig(name="repeat", seed=1))
    assert first != second
    assert registry.get(first).config == registry.get(second).config


def test_where_status(tmp_path):
    registry = ExperimentRegistry(tmp_path)
    a = registry.create(ExperimentConfig(name="a"))
    registry.update(a, status="completed")
    b = registry.create(ExperimentConfig(name="b"))
    assert [r.exp_id for r in registry.where_status("created")] == [b]
    assert [r.exp_id for r in registry.where_status("completed")] == [a]


def test_get_unknown_raises(tmp_path):
    registry = ExperimentRegistry(tmp_path)
    try:
        registry.get("EXP-9999")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")