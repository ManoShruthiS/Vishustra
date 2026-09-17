"""PHASE 7 — Experiment registry (plan.md §15 tracking, §31 reproducibility).

Every experiment gets a unique record: ``EXP-0001``, ``EXP-0002``, …
(plan.md §14). The registry persists an index file (``index.json``) of
all records so experiments survive restarts, can be re-run ("RE-RUN with
the same configuration creates a new experiment", plan.md §15) and can
carry research notes (plan.md §16).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.experiments.config import ExperimentConfig

STATUSES = ("created", "running", "completed", "failed")


@dataclass
class ExperimentRecord:
    """The persistent record of one experiment (§15)."""

    exp_id: str
    name: str
    status: str
    path: str
    created_at: str
    updated_at: str
    config: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    result: dict | None = None

    @property
    def config_obj(self) -> ExperimentConfig:
        return ExperimentConfig.from_dict(self.config)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ExperimentRecord:
        return cls(**data)


class ExperimentRegistry:
    """File-backed store of experiment records under one root directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.index_path = self.root / "index.json"
        self._records: dict[str, ExperimentRecord] = {}
        self._sequence = 0
        self._load()

    # -- persistence ---------------------------------------------------
    def _load(self) -> None:
        if not self.index_path.exists():
            return
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        self._sequence = int(data.get("sequence", 0))
        for rec in data.get("records", {}).values():
            record = ExperimentRecord.from_dict(rec)
            self._records[record.exp_id] = record

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "sequence": self._sequence,
            "records": {rid: rec.to_dict() for rid, rec in sorted(self._records.items())},
        }
        self.index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # -- ids -------------------------------------------------------------
    @staticmethod
    def format_id(number: int) -> str:
        return f"EXP-{number:04d}"

    def next_id(self) -> str:
        self._sequence += 1
        return self.format_id(self._sequence)

    # -- records ----------------------------------------------------------
    def create(self, config: ExperimentConfig) -> str:
        """Register a new (not yet run) experiment and return its id."""
        exp_id = self.next_id()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path = str(self.root / exp_id)
        record = ExperimentRecord(
            exp_id=exp_id,
            name=config.name,
            status="created",
            path=path,
            created_at=now,
            updated_at=now,
            config=config.to_dict(),
        )
        self._records[exp_id] = record
        self._save()
        return exp_id

    def update(self, exp_id: str, **fields) -> ExperimentRecord:
        """Update mutable fields of a record (status, result, notes …)."""
        record = self.get(exp_id)
        for key, value in fields.items():
            if not hasattr(record, key):
                raise KeyError(f"unknown record field: {key!r}")
            setattr(record, key, value)
        record.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._save()
        return record

    def get(self, exp_id: str) -> ExperimentRecord:
        if exp_id not in self._records:
            raise KeyError(f"unknown experiment: {exp_id}")
        return self._records[exp_id]

    def add_note(self, exp_id: str, note: str) -> ExperimentRecord:
        record = self.get(exp_id)
        record.notes.append(note)
        record.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._save()
        return record

    def all(self) -> list[ExperimentRecord]:
        return [self._records[rid] for rid in sorted(self._records)]

    def where_status(self, status: str) -> list[ExperimentRecord]:
        return [r for r in self.all() if r.status == status]

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, exp_id: str) -> bool:
        return exp_id in self._records