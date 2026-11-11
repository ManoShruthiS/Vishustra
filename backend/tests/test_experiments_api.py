"""Tests for the experiment workspace API endpoints."""

import pytest
from app.api.experiments import (
    ExperimentCreateRequest,
    create_experiment,
    get_experiment,
    list_experiments,
    rerun_experiment,
    run_experiment,
)
from app.experiments import ExperimentManager
from fastapi import HTTPException
from pydantic import ValidationError


def test_create_and_list_experiments_persists_the_requested_configuration(tmp_path):
    manager = ExperimentManager(tmp_path)
    request = ExperimentCreateRequest(
        name="head sweep",
        hypothesis="More heads may change attention concentration.",
        architecture={"num_heads": 2, "d_model": 16},
        training={"epochs": 3},
    )

    created = create_experiment(request, manager)
    records = list_experiments(manager)

    assert created.exp_id == "EXP-0001"
    assert created.status == "created"
    assert created.config["architecture"]["num_heads"] == 2
    assert created.config["training"]["epochs"] == 3
    assert [record.exp_id for record in records] == [created.exp_id]


def test_rerun_creates_a_new_record_with_the_same_configuration(tmp_path):
    manager = ExperimentManager(tmp_path)
    created = create_experiment(ExperimentCreateRequest(name="position study", seed=19), manager)

    rerun = rerun_experiment(created.exp_id, manager)

    assert rerun.exp_id == "EXP-0002"
    assert rerun.config == created.config


def test_unknown_experiment_returns_a_clear_not_found_response(tmp_path):
    with pytest.raises(HTTPException) as error:
        get_experiment("EXP-9999", ExperimentManager(tmp_path))

    assert error.value.status_code == 404
    assert error.value.detail == "unknown experiment: EXP-9999"


def test_run_experiment_returns_the_completed_record_without_hiding_metrics(tmp_path, monkeypatch):
    manager = ExperimentManager(tmp_path)
    created = create_experiment(ExperimentCreateRequest(name="run this"), manager)

    def complete(exp_id: str):
        manager.registry.update(exp_id, status="completed", result={"val_acc": 0.875})

    monkeypatch.setattr(manager, "run", complete)
    completed = run_experiment(created.exp_id, manager)

    assert completed.status == "completed"
    assert completed.result == {"val_acc": 0.875}


def test_run_experiment_returns_conflict_when_the_same_record_is_already_running(
    tmp_path, monkeypatch
):
    manager = ExperimentManager(tmp_path)
    created = create_experiment(ExperimentCreateRequest(name="busy"), manager)

    def already_running(_: str):
        raise RuntimeError("EXP-0001 is already running")

    monkeypatch.setattr(manager, "run", already_running)
    with pytest.raises(HTTPException) as error:
        run_experiment(created.exp_id, manager)

    assert error.value.status_code == 409
    assert error.value.detail == "EXP-0001 is already running"


def test_experiment_name_cannot_be_blank():
    with pytest.raises(ValidationError, match="name must not be blank"):
        ExperimentCreateRequest(name="   ")
