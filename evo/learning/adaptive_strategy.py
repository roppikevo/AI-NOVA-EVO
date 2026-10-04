from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evo.engine.knowledge_store import KnowledgeStore
from evo.learning.meta_learning import MetaLearningEngine


@dataclass
class TrainingStrategy:
    """One candidate training configuration."""

    name: str
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }


class AdaptiveTrainingStrategy:
    """
    Choose training hyperparameters from accumulated experience.

    The strategy is conservative:
    with insufficient evidence it uses the default configuration.
    Repeated measured results are required before preferring another
    configuration.
    """

    def __init__(
        self,
        knowledge: KnowledgeStore | None = None,
        min_attempts: int = 2,
    ) -> None:
        self.knowledge = knowledge or KnowledgeStore()
        self.meta = MetaLearningEngine(
            knowledge=self.knowledge,
            min_attempts_for_preference=min_attempts,
        )
        self.min_attempts = min_attempts

    @staticmethod
    def task_class(
        goal: str,
        task: str,
    ) -> str:
        return MetaLearningEngine.classify_task(
            goal,
            task,
        )

    @staticmethod
    def default_strategy() -> TrainingStrategy:
        return TrainingStrategy(
            name="default",
            steps=100,
            batch_size=8,
            learning_rate=3e-4,
            weight_decay=0.01,
        )

    @staticmethod
    def candidate_strategies() -> list[TrainingStrategy]:
        return [
            TrainingStrategy(
                name="default",
                steps=100,
                batch_size=8,
                learning_rate=3e-4,
                weight_decay=0.01,
            ),
            TrainingStrategy(
                name="small_lr",
                steps=100,
                batch_size=8,
                learning_rate=1e-4,
                weight_decay=0.01,
            ),
            TrainingStrategy(
                name="large_batch",
                steps=100,
                batch_size=16,
                learning_rate=3e-4,
                weight_decay=0.01,
            ),
            TrainingStrategy(
                name="long_training",
                steps=200,
                batch_size=8,
                learning_rate=3e-4,
                weight_decay=0.01,
            ),
            TrainingStrategy(
                name="low_decay",
                steps=100,
                batch_size=8,
                learning_rate=3e-4,
                weight_decay=0.001,
            ),
        ]

    def record_experience(
        self,
        *,
        goal: str,
        task: str,
        strategy: TrainingStrategy,
        success: bool,
        validation_loss: float | None = None,
        perplexity: float | None = None,
        core_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        task_class = self.task_class(
            goal,
            task,
        )

        content = {
            "goal": goal,
            "task": task,
            "task_class": task_class,
            "strategy": strategy.to_dict(),
            "success": bool(success),
            "metrics": {
                "validation_loss": validation_loss,
                "perplexity": perplexity,
            },
            "core_config": core_config or {},
        }

        record = self.knowledge.create(
            knowledge_type="experiment",
            title=(
                f"Training strategy: "
                f"{task_class} / {strategy.name}"
            ),
            content=content,
            status=(
                "VALIDATED"
                if success
                else "EXPERIMENTAL"
            ),
            source="adaptive_training",
            confidence=(
                1.0
                if success
                else 0.0
            ),
        )

        path = self.knowledge.save(record)
        record["path"] = str(path)

        return record

    def _history(
        self,
        goal: str,
        task: str,
    ) -> list[dict[str, Any]]:
        task_class = self.task_class(
            goal,
            task,
        )

        history = []

        for record in self.knowledge.search(
            knowledge_type="experiment"
        ):
            if record.get("source") != "adaptive_training":
                continue

            content = record.get(
                "content",
                {},
            )

            if content.get("task_class") != task_class:
                continue

            history.append(content)

        return history

    def _score(
        self,
        history: list[dict[str, Any]],
        strategy_name: str,
    ) -> dict[str, Any]:
        attempts = 0
        successes = 0
        losses: list[float] = []

        for item in history:
            strategy = item.get(
                "strategy",
                {},
            )

            if strategy.get("name") != strategy_name:
                continue

            attempts += 1

            if item.get("success", False):
                successes += 1

            metrics = item.get(
                "metrics",
                {},
            )

            loss = metrics.get(
                "validation_loss"
            )

            if isinstance(loss, (int, float)):
                losses.append(
                    float(loss)
                )

        mean_loss = (
            sum(losses) / len(losses)
            if losses
            else None
        )

        success_rate = (
            successes / attempts
            if attempts
            else 0.0
        )

        return {
            "name": strategy_name,
            "attempts": attempts,
            "successes": successes,
            "success_rate": success_rate,
            "mean_validation_loss": mean_loss,
        }

    def select(
        self,
        *,
        goal: str,
        task: str,
        candidates: list[TrainingStrategy] | None = None,
    ) -> TrainingStrategy:
        candidates = (
            candidates
            if candidates is not None
            else self.candidate_strategies()
        )

        if not candidates:
            raise ValueError(
                "At least one training strategy is required."
            )

        history = self._history(
            goal,
            task,
        )

        scored = [
            self._score(
                history,
                strategy.name,
            )
            for strategy in candidates
        ]

        eligible = [
            item
            for item in scored
            if item["attempts"] >= self.min_attempts
        ]

        if not eligible:
            return candidates[0]

        eligible.sort(
            key=lambda item: (
                item["mean_validation_loss"]
                if item["mean_validation_loss"] is not None
                else float("inf"),
                -item["success_rate"],
                -item["successes"],
            )
        )

        selected_name = eligible[0]["name"]

        for strategy in candidates:
            if strategy.name == selected_name:
                return strategy

        return candidates[0]

    def summary(
        self,
        goal: str,
        task: str,
    ) -> list[dict[str, Any]]:
        history = self._history(
            goal,
            task,
        )

        return [
            self._score(
                history,
                strategy.name,
            )
            for strategy in self.candidate_strategies()
        ]


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        knowledge = KnowledgeStore(tmp)

        engine = AdaptiveTrainingStrategy(
            knowledge=knowledge,
            min_attempts=2,
        )

        candidates = engine.candidate_strategies()

        best = next(
            item
            for item in candidates
            if item.name == "small_lr"
        )

        default = next(
            item
            for item in candidates
            if item.name == "default"
        )

        for _ in range(3):
            engine.record_experience(
                goal="Python",
                task="Python syntax",
                strategy=best,
                success=True,
                validation_loss=1.0,
            )

        for _ in range(3):
            engine.record_experience(
                goal="Python",
                task="Python syntax",
                strategy=default,
                success=True,
                validation_loss=2.0,
            )

        selected = engine.select(
            goal="Python",
            task="Python syntax",
            candidates=candidates,
        )

        assert selected.name == "small_lr"

        summary = engine.summary(
            "Python",
            "Python syntax",
        )

        assert len(summary) == 5

        print(
            "ADAPTIVE TRAINING STRATEGY SELFTEST: PASSED"
        )


if __name__ == "__main__":
    self_test()
