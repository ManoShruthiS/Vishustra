"""Experiment registry endpoints for the VISHUSTRA research workspace."""

from __future__ import annotations

from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

from app.experiments import ExperimentManager, ExperimentRecord

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


class ExperimentCreateRequest(BaseModel):
    """A user-defined experiment, optionally overriding the lab defaults."""

    name: str = Field(min_length=1, max_length=120)
    hypothesis: str = Field(default="", max_length=2_000)
    notes: str = Field(default="", max_length=5_000)
    seed: int = Field(default=42, ge=0, le=2**63 - 1)
    dataset: dict[str, Any] = Field(default_factory=dict)
    architecture: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def name_must_have_text(cls, name: str) -> str:
        if not name.strip():
            raise ValueError("name must not be blank")
        return name


class ExperimentRecordResponse(BaseModel):
    """The serializable state of a tracked laboratory experiment."""

    exp_id: str
    name: str
    status: str
    path: str
    created_at: str
    updated_at: str
    config: dict[str, Any]
    notes: list[str]
    result: dict[str, Any] | None


def _to_response(record: ExperimentRecord) -> ExperimentRecordResponse:
    return ExperimentRecordResponse.model_validate(record.to_dict())


def get_experiment_manager(request: Request) -> ExperimentManager:
    """Get the file-backed experiment manager configured on this application."""
    return cast(ExperimentManager, request.app.state.experiment_manager)


Manager = Annotated[ExperimentManager, Depends(get_experiment_manager)]


@router.get("", response_model=list[ExperimentRecordResponse])
def list_experiments(manager: Manager) -> list[ExperimentRecordResponse]:
    """List all known experiments, newest identifier last."""
    return [_to_response(record) for record in manager.list()]


@router.post("", response_model=ExperimentRecordResponse, status_code=status.HTTP_201_CREATED)
def create_experiment(
    request: ExperimentCreateRequest, manager: Manager
) -> ExperimentRecordResponse:
    """Register a reproducible experiment without starting its training run."""
    exp_id = manager.create(
        name=request.name,
        hypothesis=request.hypothesis,
        notes=request.notes,
        seed=request.seed,
        dataset=request.dataset,
        architecture=request.architecture,
        training=request.training,
    )
    return _to_response(manager.get(exp_id))


@router.get("/{exp_id}", response_model=ExperimentRecordResponse)
def get_experiment(exp_id: str, manager: Manager) -> ExperimentRecordResponse:
    """Return one experiment record, including its reproducible configuration."""
    try:
        return _to_response(manager.get(exp_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown experiment: {exp_id}") from exc


@router.post("/{exp_id}/run", response_model=ExperimentRecordResponse)
def run_experiment(exp_id: str, manager: Manager) -> ExperimentRecordResponse:
    """Train one registered experiment and return its completed record.

    Training is intentionally synchronous in this first API version: the caller
    receives the recorded metrics only after the run has either completed or
    failed. A future job queue can preserve this contract while moving the work
    out of the request lifecycle.
    """
    try:
        manager.run(exp_id)
        return _to_response(manager.get(exp_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown experiment: {exp_id}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/{exp_id}/rerun", response_model=ExperimentRecordResponse, status_code=status.HTTP_201_CREATED
)
def rerun_experiment(exp_id: str, manager: Manager) -> ExperimentRecordResponse:
    """Create a fresh record that retains an experiment's exact configuration."""
    try:
        new_id = manager.rerun(exp_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown experiment: {exp_id}") from exc
    return _to_response(manager.get(new_id))
