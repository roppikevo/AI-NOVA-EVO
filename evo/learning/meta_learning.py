from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from evo.engine.knowledge_store import KnowledgeStore


@dataclass
class CoreRecommendation:
    """Recommendation produced from accumulated learning experience."""

    task_class: str
    selected_core: str
    selected_config: dict[str, Any]
    confidence: float
    attempts: int
    successes: int
    success_rate: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_class": self.task_class,
            "selected_core": self.selected_core,
            "selected_config": self.selected_config,
            "confidence": self.confidence,
            "attempts": self.attempts,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "reason": self.reason,
        }


class MetaLearningEngine:
    """
    Learn which core and strategy work best for different task classes.

    This component does not modify the model itself.
    It learns from measured experience and produces recommendations
    that a later self-adaptation layer can apply.
    """

    def __init__(
        self,
        knowledge: KnowledgeStore | None = None,
        min_attempts_for_preference: int = 2,
    ) -> None:
        self.knowledge = knowledge or KnowledgeStore()
        self.min_attempts_for_preference = min_attempts_for_preference

    @staticmethod
    def classify_task(
        goal: str,
        task: str,
    ) -> str:
        text = f"{goal} {task}".lower()

        if any(
            term in text
            for term in (
                "vlastn",
                "ownership",
                "borrow",
                "lifetime",
                "rust",
                "cargo",
                "rustc",
            )
        ):
            return "rust_programming"

        if any(
            term in text
            for term in (
                "python",
                "pytest",
                "pip",
                "programovanie",
                "programming",
                "kód",
                "code",
            )
        ):
            return "python_programming"

        if any(
            term in text
            for term in (
                "linux",
                "bash",
                "shell",
                "systemd",
                "ubuntu",
            )
        ):
            return "linux"

        if any(
            term in text
            for term in (
                "slovensk",
                "gramatik",
                "pravopis",
                "jazyk",
                "slovenčin",
            )
        ):
            return "slovak_language"

        if any(
            term in text
            for term in (
                "repair",
                "opr",
                "debug",
                "chybu",
                "error",
                "self-repair",
            )
        ):
            return "self_repair"

        if any(
            term in text
            for term in (
                "architekt",
                "jadro",
                "model",
                "d_model",
                "d_state",
                "memory",
                "pamäť",
            )
        ):
            return "model_architecture"

        return "general_learning"

    @staticmethod
    def _core_name(config: dict[str, Any]) -> str:
        state = config.get(
            "state_architecture",
            "equal",
        )
        fusion = config.get(
            "fusion_architecture",
            "standard",
        )
        local = config.get(
            "local_context",
            "single_depthwise",
        )

        if (
            state == "equal"
            and fusion == "standard"
            and local == "single_depthwise"
        ):
            return "standard"

        if state == "expanded":
            return "expanded_state"

        if state == "compressed":
            return "compressed_state"

        if fusion == "dual_gate":
            return "dual_gate"

        if fusion == "separate_state_gate":
            return "separate_state_gate"

        if fusion == "separate_conv_gate":
            return "separate_conv_gate"

        if local == "multi_scale":
            return "multi_scale"

        return "unknown"

    def record_experience(
        self,
        *,
        goal: str,
        task: str,
        core_config: dict[str, Any],
        success: bool,
        metrics: dict[str, Any] | None = None,
        strategy: str = "default",
        source: str = "learning",
    ) -> dict[str, Any]:
        """
        Store one measured learning experience.

        The record is deliberately explicit so later self-adaptation
        can use it without guessing what happened.
        """

        task_class = self.classify_task(
            goal,
            task,
        )

        core = self._core_name(
            core_config
        )

        metrics = metrics or {}

        record = self.knowledge.create(
            knowledge_type="experiment",
            title=(
                f"Meta-learning experience: "
                f"{task_class} / {core}"
            ),
            content={
                "goal": goal,
                "task": task,
                "task_class": task_class,
                "core": core,
                "core_config": core_config,
                "strategy": strategy,
                "success": bool(success),
                "metrics": metrics,
                "source": source,
                "timestamp": time.time(),
            },
            status=(
                "VALIDATED"
                if success
                else "EXPERIMENTAL"
            ),
            source="meta_learning",
            confidence=(
                1.0
                if success
                else 0.0
            ),
        )

        path = self.knowledge.save(record)

        record["path"] = str(path)

        return record

    def _experiences(
        self,
        task_class: str,
    ) -> list[dict[str, Any]]:
        experiences = []

        for record in self.knowledge.search(
            knowledge_type="experiment"
        ):
            if record.get("source") != "meta_learning":
                continue

            content = record.get(
                "content",
                {},
            )

            if content.get("task_class") != task_class:
                continue

            experiences.append(record)

        return experiences

    @staticmethod
    def _candidate_score(
        experiences: list[dict[str, Any]],
        core_name: str,
    ) -> tuple[int, int, float]:
        attempts = 0
        successes = 0

        for record in experiences:
            content = record.get(
                "content",
                {},
            )

            if content.get("core") != core_name:
                continue

            attempts += 1

            if bool(
                content.get(
                    "success",
                    False,
                )
            ):
                successes += 1

        # Laplace smoothing prevents one lucky result
        # from dominating the decision.
        smoothed_rate = (
            (successes + 1.0)
            / (attempts + 2.0)
        )

        return (
            attempts,
            successes,
            smoothed_rate,
        )

    def recommend_core(
        self,
        *,
        goal: str,
        task: str,
        candidates: list[dict[str, Any]],
    ) -> CoreRecommendation:
        if not candidates:
            raise ValueError(
                "At least one core candidate is required."
            )

        task_class = self.classify_task(
            goal,
            task,
        )

        experiences = self._experiences(
            task_class
        )

        stats: list[dict[str, Any]] = []

        for candidate in candidates:
            config = candidate.get(
                "config",
                candidate,
            )

            core_name = self._core_name(
                config
            )

            attempts, successes, rate = (
                self._candidate_score(
                    experiences,
                    core_name,
                )
            )

            stats.append(
                {
                    "core": core_name,
                    "config": config,
                    "attempts": attempts,
                    "successes": successes,
                    "rate": rate,
                }
            )

        eligible = [
            item
            for item in stats
            if item["attempts"]
            >= self.min_attempts_for_preference
        ]

        if not eligible:
            selected = stats[0]

            return CoreRecommendation(
                task_class=task_class,
                selected_core=selected["core"],
                selected_config=selected["config"],
                confidence=0.0,
                attempts=selected["attempts"],
                successes=selected["successes"],
                success_rate=selected["rate"],
                reason=(
                    "Insufficient repeated evidence. "
                    "Using baseline candidate."
                ),
            )

        selected = max(
            eligible,
            key=lambda item: (
                item["rate"],
                item["successes"],
                -item["attempts"],
            ),
        )

        confidence = min(
            1.0,
            selected["attempts"]
            / 10.0,
        )

        return CoreRecommendation(
            task_class=task_class,
            selected_core=selected["core"],
            selected_config=selected["config"],
            confidence=confidence,
            attempts=selected["attempts"],
            successes=selected["successes"],
            success_rate=selected["rate"],
            reason=(
                "Selected from repeated measured "
                "experience for this task class."
            ),
        )

    def summary(self) -> dict[str, Any]:
        grouped: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(
                lambda: {
                    "attempts": 0,
                    "successes": 0,
                }
            )
        )

        for record in self.knowledge.search(
            knowledge_type="experiment"
        ):
            if record.get("source") != "meta_learning":
                continue

            content = record.get(
                "content",
                {},
            )

            task_class = content.get(
                "task_class",
                "unknown",
            )

            core = content.get(
                "core",
                "unknown",
            )

            grouped[task_class][core]["attempts"] += 1

            if content.get(
                "success",
                False,
            ):
                grouped[task_class][core]["successes"] += 1

        result: dict[str, Any] = {}

        for task_class, cores in grouped.items():
            result[task_class] = {}

            for core, values in cores.items():
                attempts = values["attempts"]
                successes = values["successes"]

                result[task_class][core] = {
                    "attempts": attempts,
                    "successes": successes,
                    "success_rate": (
                        successes / attempts
                        if attempts
                        else 0.0
                    ),
                }

        return result


def self_test() -> None:
    import tempfile

    from evo.engine.knowledge_store import KnowledgeStore

    with tempfile.TemporaryDirectory() as tmp:
        knowledge = KnowledgeStore(tmp)

        engine = MetaLearningEngine(
            knowledge=knowledge
        )

        standard = {
            "vocab_size": 16384,
            "d_model": 384,
            "d_state": 384,
            "num_layers": 6,
            "conv_kernel": 5,
            "forget_bias": 1.125,
            "learnable_initial_state": False,
            "state_architecture": "equal",
            "fusion_architecture": "standard",
            "local_context": "single_depthwise",
        }

        expanded = dict(standard)

        expanded.update(
            {
                "d_state": 512,
                "state_architecture": "expanded",
            }
        )

        for _ in range(3):
            engine.record_experience(
                goal="Rust",
                task="Rust ownership test",
                core_config=standard,
                success=True,
                metrics={
                    "score": 0.90,
                },
            )

        for _ in range(3):
            engine.record_experience(
                goal="Rust",
                task="Rust ownership test",
                core_config=expanded,
                success=False,
                metrics={
                    "score": 0.40,
                },
            )

        recommendation = engine.recommend_core(
            goal="Rust",
            task="Rust ownership test",
            candidates=[
                {
                    "candidate": "standard",
                    "config": standard,
                },
                {
                    "candidate": "expanded_state",
                    "config": expanded,
                },
            ],
        )

        assert recommendation.task_class == (
            "rust_programming"
        )

        assert recommendation.selected_core == (
            "standard"
        )

        assert recommendation.attempts == 3
        assert recommendation.successes == 3

        summary = engine.summary()

        assert (
            summary["rust_programming"]["standard"]["attempts"]
            == 3
        )

        print(
            "META LEARNING SELFTEST: PASSED"
        )


if __name__ == "__main__":
    self_test()
