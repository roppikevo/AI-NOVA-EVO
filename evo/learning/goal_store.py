from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any


class LearningGoalStore:
    """Persistent storage for NOVA-EVO learning goals."""

    def __init__(self, root: str | Path = "evo/learning/goals") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _slug(self, goal: str) -> str:
        slug = goal.lower().strip()
        slug = re.sub(r"[^a-z0-9áäčďéíĺľňóôŕšťúýž]+", "_", slug)
        slug = slug.strip("_")

        return slug or "learning_goal"

    def _path(self, goal: str) -> Path:
        return self.root / f"{self._slug(goal)}.json"

    def create(self, goal: str) -> dict[str, Any]:
        now = time.time()

        record = {
            "goal": goal,
            "status": "NEW",
            "current_phase": "UNDERSTAND",
            "completed_tasks": [],
            "failed_tasks": [],
            "knowledge": [],
            "next_task": None,
            "created": now,
            "updated": now,
        }

        self.save(record)
        return record

    def save(self, record: dict[str, Any]) -> Path:
        goal = str(record["goal"])
        path = self._path(goal)

        record["updated"] = time.time()

        temporary = path.with_suffix(".tmp")

        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                record,
                handle,
                ensure_ascii=False,
                indent=2,
            )

        temporary.replace(path)

        return path

    def load(self, goal: str) -> dict[str, Any] | None:
        path = self._path(goal)

        if not path.exists():
            return None

        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def exists(self, goal: str) -> bool:
        return self._path(goal).exists()

    def delete(self, goal: str) -> bool:
        path = self._path(goal)

        if not path.exists():
            return False

        path.unlink()
        return True

    def list_goals(self) -> list[dict[str, Any]]:
        goals = []

        for path in sorted(self.root.glob("*.json")):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    goals.append(json.load(handle))
            except (OSError, json.JSONDecodeError):
                continue

        return goals
