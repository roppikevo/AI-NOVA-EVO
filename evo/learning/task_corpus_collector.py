from __future__ import annotations

from typing import Any

from evo.engine.knowledge_store import KnowledgeStore
from evo.learning.learning_engine import LearningEngine
from evo.learning.task_dataset import TaskLearningDataset


class TaskCorpusCollector:
    """
    Autonomously collect validated learning experiences.

    The collector never trains the model. It only creates executable
    learning tasks through LearningEngine and keeps looping until the
    requested number of unique validated examples is reached.
    """

    def __init__(
        self,
        engine: LearningEngine | None = None,
        knowledge: KnowledgeStore | None = None,
    ) -> None:
        self.engine = engine or LearningEngine()
        self.knowledge = knowledge or KnowledgeStore()
        self.dataset = TaskLearningDataset(
            knowledge=self.knowledge
        )

    def unique_validated_examples(self) -> int:
        records = self.knowledge.search(
            knowledge_type="success"
        )

        identities: set[str] = set()

        for record in records:
            if record.get("source") != "learning_executor":
                continue

            if record.get("status") != "VALIDATED":
                continue

            content = record.get("content") or {}

            if not content.get("source_code"):
                continue

            identities.add(
                self.dataset._identity(record)
            )

        return len(identities)

    def collect(
        self,
        *,
        goal_prefix: str,
        target_examples: int = 32,
        max_batches: int = 20,
        tasks_per_batch: int = 5,
    ) -> dict[str, Any]:
        if not goal_prefix.strip():
            raise ValueError(
                "goal_prefix cannot be empty"
            )

        if target_examples < 1:
            raise ValueError(
                "target_examples must be positive"
            )

        if tasks_per_batch < 1:
            raise ValueError(
                "tasks_per_batch must be positive"
            )

        start_count = (
            self.unique_validated_examples()
        )

        created = 0
        successful = 0
        failed = 0
        duplicate_count = 0
        batch_results: list[dict[str, Any]] = []

        for batch_index in range(
            1,
            max_batches + 1,
        ):
            current = self.unique_validated_examples()

            if current >= target_examples:
                break

            goal = (
                f"{goal_prefix} "
                f"Autonomous Batch {batch_index}"
            )

            learner = self.engine.create_goal(
                goal
            )

            learner = self.engine.plan(
                learner
            )

            batch = {
                "batch": batch_index,
                "goal": goal,
                "status": learner.status,
                "tasks": [],
            }

            if learner.status != "PLANNED":
                batch["error"] = learner.knowledge.get(
                    "error",
                    "Planning failed."
                )
                batch_results.append(batch)
                continue

            max_tasks = min(
                tasks_per_batch,
                len(learner.plan),
            )

            for _ in range(max_tasks):
                before = self.unique_validated_examples()

                result = self.engine.execute_next(
                    goal
                )

                created += 1

                execution = result.knowledge.get(
                    "last_execution",
                    {}
                )

                success = bool(
                    execution.get("success")
                )

                if success:
                    after = self.unique_validated_examples()

                    if after > before:
                        successful += 1
                    else:
                        duplicate_count += 1

                    batch["tasks"].append(
                        {
                            "task": result.knowledge.get(
                                "last_task"
                            ),
                            "status": result.status,
                            "success": True,
                            "unique_example_added": (
                                after > before
                            ),
                        }
                    )

                else:
                    failed += 1

                    batch["tasks"].append(
                        {
                            "task": result.knowledge.get(
                                "last_task"
                            ),
                            "status": result.status,
                            "success": False,
                            "error": result.knowledge.get(
                                "last_error"
                            ),
                        }
                    )

                    break

                if (
                    self.unique_validated_examples()
                    >= target_examples
                ):
                    break

            batch_results.append(batch)

        final_count = (
            self.unique_validated_examples()
        )

        return {
            "goal_prefix": goal_prefix,
            "target_examples": target_examples,
            "start_unique_examples": start_count,
            "final_unique_examples": final_count,
            "new_unique_examples": (
                max(
                    0,
                    final_count - start_count,
                )
            ),
            "attempted_tasks": created,
            "successful_tasks": successful,
            "failed_tasks": failed,
            "duplicate_successes": duplicate_count,
            "target_reached": (
                final_count >= target_examples
            ),
            "batches_used": len(batch_results),
            "batches": batch_results,
        }


def self_test() -> None:
    collector = TaskCorpusCollector()

    count = collector.unique_validated_examples()

    assert isinstance(count, int)
    assert count >= 0

    print(
        "TASK CORPUS COLLECTOR SELFTEST: PASSED"
    )
    print(
        "CURRENT_UNIQUE_EXAMPLES:",
        count,
    )


if __name__ == "__main__":
    self_test()
