from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evo.engine.experiment_record import ExperimentRecord

class ExperimentStore:
    """Persistent experiment storage for NOVA-EVO."""

    def __init__(self, root: str | Path = "evo/experiments") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, experiment_id: str) -> Path:
        return self.root / f"{experiment_id}.json"

    def save(self, record: ExperimentRecord) -> Path:
        if not record.verify_integrity():
            record.finalize()

        path = self._path(record.experiment_id)

        if path.exists():
            raise FileExistsError(
                f"Experiment already exists: {path}"
            )

        path.write_text(
            record.to_json() + "\n",
            encoding="utf-8",
        )

        return path

    def update(self, record: ExperimentRecord) -> Path:
        if not record.verify_integrity():
            record.finalize()

        path = self._path(record.experiment_id)

        if not path.exists():
            raise FileNotFoundError(path)

        path.write_text(
            record.to_json() + "\n",
            encoding="utf-8",
        )

        return path

    def load(self, experiment_id: str) -> ExperimentRecord:
        path = self._path(experiment_id)

        if not path.exists():
            raise FileNotFoundError(path)

        payload = path.read_text(encoding="utf-8")
        return ExperimentRecord.from_json(payload)

    def exists(self, experiment_id: str) -> bool:
        return self._path(experiment_id).exists()

    def list_ids(self) -> list[str]:
        return sorted(
            path.stem
            for path in self.root.glob("*.json")
            if path.is_file()
        )

    def list_records(
        self,
        state: str | None = None,
        generation: int | None = None,
    ) -> list[ExperimentRecord]:

        records: list[ExperimentRecord] = []

        for path in sorted(self.root.glob("*.json")):
            if not path.is_file():
                continue

            try:
                record = ExperimentRecord.from_json(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError, json.JSONDecodeError):
                continue

            if state is not None and record.state != state:
                continue

            if generation is not None:
                if record.generation != generation:
                    continue

            records.append(record)

        records.sort(
            key=lambda record: record.created_at,
            reverse=True,
        )

        return records

    def recover_incomplete(self) -> list[ExperimentRecord]:
        """Return experiments that were not completed before shutdown."""

        incomplete_states = {
            "CREATED",
            "RUNNING",
            "REPAIRING",
        }

        return [
            record
            for record in self.list_records()
            if record.state in incomplete_states
        ]

    def count(self) -> int:
        return sum(
            1
            for path in self.root.glob("*.json")
            if path.is_file()
        )

    def summary(self) -> dict[str, Any]:
        records = self.list_records()

        states: dict[str, int] = {}

        for record in records:
            states[record.state] = states.get(record.state, 0) + 1

        return {
            "root": str(self.root),
            "total": len(records),
            "states": states,
            "incomplete": len(self.recover_incomplete()),
        }


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = ExperimentStore(tmp)

        record = ExperimentRecord(
            generation=4,
            candidate_id="GEN4-STORE-SELFTEST",
            parent_id="GEN3-007",
            hypothesis="Verify persistent experiment storage",
            objective="Test save/load/recovery lifecycle",
        )

        record.finalize()

        path = store.save(record)

        assert path.exists()
        assert store.exists(record.experiment_id)
        assert store.count() == 1

        loaded = store.load(record.experiment_id)

        assert loaded.experiment_id == record.experiment_id
        assert loaded.candidate_id == "GEN4-STORE-SELFTEST"
        assert loaded.verify_integrity()

        loaded.start()
        store.update(loaded)

        recovered = store.recover_incomplete()

        assert len(recovered) == 1
        assert recovered[0].state == "RUNNING"

        loaded.complete(
            {
                "validation_loss": 8.5,
                "params": 9856512,
            }
        )

        store.update(loaded)

        assert len(store.recover_incomplete()) == 0

        completed = store.list_records(
            state="COMPLETED",
            generation=4,
        )

        assert len(completed) == 1
        assert completed[0].metrics["validation_loss"] == 8.5

        summary = store.summary()

        assert summary["total"] == 1
        assert summary["states"]["COMPLETED"] == 1
        assert summary["incomplete"] == 0

    print("EXPERIMENT STORE SELFTEST: PASSED")


if __name__ == "__main__":
    self_test()
