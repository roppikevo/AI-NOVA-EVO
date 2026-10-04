from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExperimentRecord:
    """Persistent experiment record used by NOVA-EVO."""

    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    generation: int = 0
    candidate_id: str = ""
    parent_id: str | None = None

    state: str = "CREATED"

    hypothesis: str = ""
    objective: str = ""

    architecture: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)

    seeds: list[int] = field(default_factory=list)

    metrics: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)

    validation: dict[str, Any] = field(default_factory=dict)
    failure: dict[str, Any] = field(default_factory=dict)
    repair: dict[str, Any] = field(default_factory=dict)

    mentor: dict[str, Any] = field(default_factory=dict)

    artifacts: list[str] = field(default_factory=list)

    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    checksum: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["checksum"] = self.compute_checksum(data)
        return data

    @staticmethod
    def compute_checksum(data: dict[str, Any]) -> str:
        payload = dict(data)
        payload.pop("checksum", None)

        canonical = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        return hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()

    def finalize(self) -> None:
        self.checksum = self.compute_checksum(asdict(self))

    def verify_integrity(self) -> bool:
        data = asdict(self)
        stored = data.pop("checksum", "")
        calculated = self.compute_checksum(data)

        return stored == calculated

    def start(self) -> None:
        if self.state != "CREATED":
            raise ValueError(
                f"Cannot start experiment in state: {self.state}"
            )

        self.state = "RUNNING"
        self.started_at = time.time()
        self.finalize()

    def complete(self, metrics: dict[str, Any] | None = None) -> None:
        if self.state != "RUNNING":
            raise ValueError(
                f"Cannot complete experiment in state: {self.state}"
            )

        if metrics:
            self.metrics.update(metrics)

        self.state = "COMPLETED"
        self.finished_at = time.time()
        self.finalize()

    def fail(
        self,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        if self.state not in {"CREATED", "RUNNING", "REPAIRING"}:
            raise ValueError(
                f"Cannot fail experiment in state: {self.state}"
            )

        self.state = "FAILED"
        self.failure = {
            "reason": reason,
            "details": details or {},
            "timestamp": time.time(),
        }
        self.finished_at = time.time()
        self.finalize()

    def begin_repair(self) -> None:
        if self.state != "FAILED":
            raise ValueError(
                f"Cannot repair experiment in state: {self.state}"
            )

        self.state = "REPAIRING"
        self.repair.setdefault("attempts", 0)
        self.repair["attempts"] += 1
        self.finalize()

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            indent=2,
            ensure_ascii=False,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentRecord":
        allowed = {
            field_name
            for field_name in cls.__dataclass_fields__
        }

        filtered = {
            key: value
            for key, value in data.items()
            if key in allowed
        }

        record = cls(**filtered)

        if not record.verify_integrity():
            raise ValueError(
                f"Experiment integrity check failed: "
                f"{record.experiment_id}"
            )

        return record

    @classmethod
    def from_json(cls, payload: str) -> "ExperimentRecord":
        return cls.from_dict(json.loads(payload))


def self_test() -> None:
    record = ExperimentRecord(
        generation=4,
        candidate_id="GEN4-SELFTEST-001",
        parent_id="GEN3-007",
        hypothesis="Test experiment lifecycle",
        objective="Verify experiment record integrity",
        architecture={
            "d_model": 384,
            "d_state": 384,
            "layers": 6,
            "kernel": 5,
        },
        dataset={
            "name": "selftest",
            "split": "validation",
        },
        training={
            "steps": 10,
        },
        seeds=[1, 2, 3],
    )

    record.finalize()

    assert record.state == "CREATED"
    assert record.verify_integrity()

    serialized = record.to_json()
    restored = ExperimentRecord.from_json(serialized)

    assert restored.experiment_id == record.experiment_id
    assert restored.candidate_id == "GEN4-SELFTEST-001"
    assert restored.state == "CREATED"
    assert restored.verify_integrity()

    restored.start()

    assert restored.state == "RUNNING"
    assert restored.started_at is not None
    assert restored.verify_integrity()

    restored.complete(
        {
            "validation_loss": 8.5,
            "params": 9856512,
        }
    )

    assert restored.state == "COMPLETED"
    assert restored.finished_at is not None
    assert restored.metrics["validation_loss"] == 8.5
    assert restored.verify_integrity()

    failed = ExperimentRecord(
        generation=4,
        candidate_id="GEN4-SELFTEST-FAIL",
    )

    failed.finalize()
    failed.start()
    failed.fail(
        "synthetic failure",
        {"error": "self-test"},
    )

    assert failed.state == "FAILED"
    assert failed.failure["reason"] == "synthetic failure"
    assert failed.verify_integrity()

    failed.begin_repair()

    assert failed.state == "REPAIRING"
    assert failed.repair["attempts"] == 1
    assert failed.verify_integrity()

    print("EXPERIMENT RECORD SELFTEST: PASSED")


if __name__ == "__main__":
    self_test()
