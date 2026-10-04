from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any


class KnowledgeStore:
    """Persistent JSON knowledge store for NOVA-EVO."""

    VALID_TYPES = {
        "success",
        "failure",
        "repair",
        "architecture",
        "experiment",
        "hypothesis",
        "mentor",
        "lineage",
    }

    VALID_STATUS = {
        "UNTESTED",
        "EXPERIMENTAL",
        "VALIDATED",
        "REPEATED",
        "ROBUST",
    }

    def __init__(self, root: str | Path = "evo/knowledge") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _directory_for(self, knowledge_type: str) -> Path:
        if knowledge_type not in self.VALID_TYPES:
            raise ValueError(f"Invalid knowledge type: {knowledge_type}")

        directory = self.root / ("successes" if knowledge_type == "success" else f"{knowledge_type}s")

        if knowledge_type == "hypothesis":
            directory = self.root / "hypotheses"
        elif knowledge_type == "mentor":
            directory = self.root / "mentor"
        elif knowledge_type == "lineage":
            directory = self.root / "lineage"

        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def _canonical_json(data: dict[str, Any]) -> str:
        return json.dumps(
            data,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def _checksum(cls, data: dict[str, Any]) -> str:
        canonical = cls._canonical_json(data)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def create(
        self,
        knowledge_type: str,
        title: str,
        content: dict[str, Any],
        status: str = "EXPERIMENTAL",
        source: str = "NOVA-EVO",
        confidence: float = 0.0,
        parent_id: str | None = None,
    ) -> dict[str, Any]:

        if knowledge_type not in self.VALID_TYPES:
            raise ValueError(f"Invalid knowledge type: {knowledge_type}")

        if status not in self.VALID_STATUS:
            raise ValueError(f"Invalid knowledge status: {status}")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

        record: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "type": knowledge_type,
            "title": title,
            "status": status,
            "confidence": confidence,
            "source": source,
            "created_at": time.time(),
            "parent_id": parent_id,
            "content": content,
        }

        record["checksum"] = self._checksum(record)
        return record

    def save(self, record: dict[str, Any]) -> Path:
        required = {
            "id",
            "type",
            "title",
            "status",
            "confidence",
            "source",
            "created_at",
            "content",
            "checksum",
        }

        missing = required - record.keys()

        if missing:
            raise ValueError(f"Missing knowledge fields: {sorted(missing)}")

        directory = self._directory_for(record["type"])
        path = directory / f'{record["id"]}.json'

        if path.exists():
            raise FileExistsError(f"Knowledge record already exists: {path}")

        path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        return path

    def load(self, knowledge_id: str, knowledge_type: str) -> dict[str, Any]:
        directory = self._directory_for(knowledge_type)
        path = directory / f"{knowledge_id}.json"

        if not path.exists():
            raise FileNotFoundError(path)

        record = json.loads(path.read_text(encoding="utf-8"))

        checksum = record.pop("checksum", None)
        calculated = self._checksum(record)
        record["checksum"] = checksum

        if checksum != calculated:
            raise ValueError(f"Knowledge integrity check failed: {path}")

        return record

    def search(
        self,
        knowledge_type: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:

        if knowledge_type is not None:
            directories = [self._directory_for(knowledge_type)]
        else:
            directories = [
                path
                for path in self.root.iterdir()
                if path.is_dir()
            ]

        results: list[dict[str, Any]] = []

        for directory in directories:
            for path in sorted(directory.glob("*.json")):
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue

                if status is not None and record.get("status") != status:
                    continue

                results.append(record)

        results.sort(key=lambda item: item.get("created_at", 0), reverse=True)
        return results

    def count(self) -> int:
        return sum(
            1
            for path in self.root.rglob("*.json")
            if path.is_file()
        )


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = KnowledgeStore(tmp)

        record = store.create(
            knowledge_type="success",
            title="Self-test knowledge",
            content={"result": "ok"},
            status="VALIDATED",
            confidence=0.9,
        )

        path = store.save(record)

        assert path.exists()
        assert store.count() == 1

        loaded = store.load(record["id"], "success")

        assert loaded["title"] == "Self-test knowledge"
        assert loaded["content"]["result"] == "ok"
        assert loaded["status"] == "VALIDATED"

        results = store.search(
            knowledge_type="success",
            status="VALIDATED",
        )

        assert len(results) == 1
        assert results[0]["id"] == record["id"]

    print("KNOWLEDGE STORE SELFTEST: PASSED")


if __name__ == "__main__":
    self_test()
