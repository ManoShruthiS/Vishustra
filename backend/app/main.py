"""VISHUSTRA's FastAPI application entry point."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from app.api.experiments import router as experiments_router
from app.api.labs import router as labs_router
from app.experiments import ExperimentManager


class HealthResponse(BaseModel):
    """A small, stable response for uptime checks and local smoke tests."""

    status: str
    service: str
    version: str


def create_app(experiment_root: str | Path | None = None) -> FastAPI:
    """Create the API application without starting a server."""
    app = FastAPI(
        title="VISHUSTRA API",
        version="0.2.0",
        description=(
            "An interactive Transformer research laboratory. "
            "Explore paper configurations, attention calculations, and "
            "reproducible experiment records."
        ),
    )
    app.state.experiment_manager = ExperimentManager(experiment_root)
    app.include_router(labs_router)
    app.include_router(experiments_router)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health_check() -> HealthResponse:
        return HealthResponse(status="ok", service="vishustra", version=app.version)

    return app


app = create_app()


def main() -> None:
    """Run the local development server for the ``vishustra`` command."""
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
