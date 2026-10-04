from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from evo.engine.agent_controller import AgentController
from evo.engine.architecture_factory import build_model
from evo.engine.mutation_engine import architecture_signature
from nova.data import TokenSequenceDataset
from nova.training import TrainConfig, train


ROOT = Path(__file__).resolve().parents[2]


class GenerationManager:
    """
    Autonomous generation manager.

    GEN3 remains the protected parent until a GEN4 candidate
    passes the existing robust Evaluator gate.

    The manager:
      1. reads the existing GEN4 candidate pool,
      2. skips invalid/unsupported candidates,
      3. reuses already completed robust experiments,
      4. evaluates candidates sequentially,
      5. continues after REJECT,
      6. promotes only after the Evaluator returns PROMOTE,
      7. writes an autonomous state file so execution can resume.
    """

    def __init__(
        self,
        root: str | Path = ROOT,
    ) -> None:
        self.root = Path(root).resolve()

        self.candidate_dir = (
            self.root / "evo" / "gen4" / "candidates"
        )

        self.result_dir = (
            self.root / "evo" / "gen4" / "results"
        )

        self.state_file = (
            self.result_dir
            / "autonomous_generation_state.json"
        )

        self.evo_state_file = (
            self.root / "evo" / "engine" / "evo_state.json"
        )

        self.controller = AgentController(
            project_root=self.root,
            generation=4,
            dry_run=False,
        )

    def load_manager_state(self) -> dict[str, Any]:
        if not self.state_file.exists():
            return {
                "version": 1,
                "generation": 4,
                "parent": "GEN3-007",
                "next_index": 1,
                "attempts": 0,
                "processed": {},
                "status": "READY",
                "updated_at": time.time(),
            }

        return json.loads(
            self.state_file.read_text(
                encoding="utf-8"
            )
        )

    def save_manager_state(
        self,
        state: dict[str, Any],
    ) -> None:
        self.result_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        state["updated_at"] = time.time()

        self.state_file.write_text(
            json.dumps(
                state,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load_candidates(self) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []

        for path in sorted(
            self.candidate_dir.glob("GEN4-*.json")
        ):
            try:
                data = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )
            except (
                OSError,
                json.JSONDecodeError,
            ):
                continue

            candidate_id = str(
                data.get("candidate", "")
            )

            if not candidate_id.startswith("GEN4-"):
                continue

            if not isinstance(
                data.get("config"),
                dict,
            ):
                continue

            candidates.append(data)

        def candidate_number(
            item: dict[str, Any],
        ) -> int:
            try:
                return int(
                    str(
                        item["candidate"]
                    ).split("-", 1)[1]
                )
            except (
                KeyError,
                ValueError,
                TypeError,
            ):
                return 999999

        candidates.sort(
            key=candidate_number
        )

        return candidates

    def historical_robust(
        self,
    ) -> dict[str, dict[str, Any]]:
        """
        Reconstruct completed robust evaluations from
        individual ExperimentStore records.
        """

        experiment_dir = (
            self.root / "evo" / "experiments"
        )

        grouped: dict[
            str,
            dict[int, dict[str, Any]],
        ] = {}

        for path in experiment_dir.glob("*.json"):
            try:
                record = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )
            except (
                OSError,
                json.JSONDecodeError,
            ):
                continue

            candidate_id = str(
                record.get(
                    "candidate_id",
                    "",
                )
            )

            if not candidate_id.startswith(
                "GEN4-"
            ):
                continue

            if record.get("state") != "COMPLETED":
                continue

            training = record.get(
                "training",
                {},
            )

            if training.get("stage") != "robust":
                continue

            seeds = record.get(
                "seeds",
                [],
            )

            if not seeds:
                continue

            try:
                seed = int(seeds[0])
            except (
                TypeError,
                ValueError,
            ):
                continue

            metrics = record.get(
                "metrics",
                {},
            )

            loss = metrics.get(
                "validation_loss"
            )

            if loss is None:
                continue

            grouped.setdefault(
                candidate_id,
                {},
            )[seed] = {
                "seed": seed,
                "validation_loss": float(loss),
                "experiment_id": record.get(
                    "experiment_id"
                ),
            }

        result: dict[
            str,
            dict[str, Any],
        ] = {}

        for candidate_id, seed_map in grouped.items():
            seeds = sorted(seed_map)
            losses = [
                seed_map[seed]["validation_loss"]
                for seed in seeds
            ]

            if len(losses) < 3:
                continue

            mean_loss = statistics.mean(losses)
            std_loss = statistics.pstdev(losses)

            result[candidate_id] = {
                "candidate_id": candidate_id,
                "parent_id": "GEN3-007",
                "state": "COMPLETED",
                "training": {
                    "stage": "robust",
                    "seeds": seeds,
                },
                "metrics": {
                    "validation_loss_mean": mean_loss,
                    "validation_loss_std": std_loss,
                    "validation_loss_min": min(losses),
                    "validation_loss_max": max(losses),
                },
                "seed_results": [
                    seed_map[seed]
                    for seed in seeds
                ],
            }

        return result

    def _result_path(
        self,
        candidate_id: str,
    ) -> Path:
        return (
            self.result_dir
            / f"{candidate_id}.json"
        )

    def save_candidate_result(
        self,
        result: dict[str, Any],
    ) -> None:
        self.result_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        candidate_id = str(
            result["candidate"]
        )

        path = self._result_path(
            candidate_id
        )

        path.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _parent_config(
        self,
    ) -> dict[str, Any]:
        state = json.loads(
            self.evo_state_file.read_text(
                encoding="utf-8"
            )
        )

        config = dict(
            state["primary_parent_config"]
        )

        config.setdefault(
            "state_architecture",
            "equal",
        )
        config.setdefault(
            "fusion_architecture",
            "standard",
        )
        config.setdefault(
            "local_context",
            "single_depthwise",
        )

        return config

    def _save_promoted_checkpoint(
        self,
        candidate: dict[str, Any],
        robust_result: dict[str, Any],
    ) -> str:
        """
        Materialize the exact checkpoint produced by the
        robust experiment that passed the promotion gate.

        No additional training is performed here.
        """

        candidate_id = candidate["candidate"]

        seed_results = robust_result.get(
            "seed_results",
            [],
        )

        if not seed_results:
            raise RuntimeError(
                f"No robust seed results available for {candidate_id}."
            )

        selected = None

        for item in seed_results:
            metrics = item.get("metrics", {})

            if int(
                metrics.get("seed", item.get("seed", -1))
            ) == 1001:
                selected = item
                break

        if selected is None:
            selected = seed_results[0]

        metrics = selected.get(
            "metrics",
            {},
        )

        source_checkpoint = metrics.get(
            "checkpoint"
        )

        if not source_checkpoint:
            raise RuntimeError(
                "Promotion blocked: robust experiment "
                f"for {candidate_id} has no checkpoint."
            )

        source = Path(source_checkpoint)

        if not source.is_absolute():
            source = self.root / source

        if not source.exists():
            raise FileNotFoundError(
                f"Robust checkpoint missing: {source}"
            )

        checkpoint_dir = (
            self.root
            / "evo"
            / "final"
            / "checkpoints"
            / candidate_id
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        seed = int(
            selected.get(
                "seed",
                metrics.get("seed", 1001),
            )
        )

        destination = (
            checkpoint_dir
            / f"seed_{seed}.pt"
        )

        import shutil

        shutil.copy2(
            source,
            destination,
        )

        return str(destination)

    def _promote(
        self,
        candidate: dict[str, Any],
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        candidate_id = candidate["candidate"]

        robust_result = evaluation.get(
            "robust"
        ) or {}

        checkpoint = self._save_promoted_checkpoint(
            candidate,
            robust_result,
        )

        state = json.loads(
            self.evo_state_file.read_text(
                encoding="utf-8"
            )
        )

        previous_state = dict(state)

        backup = (
            self.evo_state_file.with_suffix(
                ".pre-gen4-promotion.json"
            )
        )

        backup.write_text(
            json.dumps(
                previous_state,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        candidate_config = dict(
            candidate["config"]
        )

        state["parent_generation"] = "GEN4"
        state["current_generation"] = 5
        state["primary_parent"] = candidate_id
        state["primary_parent_config"] = candidate_config

        best_known = dict(
            state.get(
                "best_known",
                {},
            )
        )

        robust_metrics = (
            evaluation.get(
                "evaluation"
            )
            or {}
        )

        candidate_loss = robust_metrics.get(
            "candidate_loss"
        )

        best_known.update(
            {
                "candidate": candidate_id,
                "source_generation": "GEN4",
                "validation_loss": (
                    float(candidate_loss)
                    if candidate_loss is not None
                    else best_known.get(
                        "validation_loss"
                    )
                ),
                "parameters": (
                    candidate.get(
                        "implementation_validation",
                        {},
                    ).get(
                        "parameters"
                    )
                    or best_known.get(
                        "parameters"
                    )
                ),
                "checkpoint": checkpoint,
            }
        )

        state["best_known"] = best_known

        state.setdefault(
            "generation_history",
            [],
        )

        state["generation_history"].append(
            {
                "generation": 4,
                "parent": "GEN3-007",
                "promoted": candidate_id,
                "evaluation": evaluation,
                "checkpoint": checkpoint,
                "promoted_at": time.time(),
            }
        )

        state.setdefault(
            "lineage",
            [],
        )

        state["lineage"].append(
            {
                "generation": 4,
                "parent": "GEN3-007",
                "candidate": candidate_id,
                "checkpoint": checkpoint,
            }
        )

        self.evo_state_file.write_text(
            json.dumps(
                state,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return {
            "status": "PROMOTED",
            "candidate": candidate_id,
            "checkpoint": checkpoint,
            "evo_state": str(
                self.evo_state_file
            ),
        }

    def run(
        self,
        *,
        max_attempts: int = 5,
        screening_steps: int = 100,
        robust_steps: int = 1000,
        seeds: list[int] | None = None,
    ) -> dict[str, Any]:
        if seeds is None:
            seeds = [1001, 2002, 3003]

        if len(seeds) < 3:
            raise ValueError(
                "GEN4 robust evaluation requires at least 3 seeds."
            )

        candidates = self.load_candidates()

        state = self.load_manager_state()

        historical = self.historical_robust()

        processed = dict(
            state.get(
                "processed",
                {},
            )
        )

        attempts = int(
            state.get(
                "attempts",
                0,
            )
        )

        print("=" * 72)
        print("NOVA-EVO AUTONOMOUS GEN4 MANAGER")
        print("=" * 72)
        print("Parent       :", "GEN3-007")
        print("Candidates   :", len(candidates))
        print("Max attempts:", max_attempts)
        print("Robust seeds :", seeds)
        print()

        for candidate in candidates:
            if attempts >= max_attempts:
                break

            candidate_id = candidate["candidate"]

            if candidate_id in processed:
                continue

            technical = candidate.get(
                "technical_validation",
                {},
            )

            if not technical.get(
                "valid",
                False,
            ):
                processed[candidate_id] = {
                    "status": "SKIPPED",
                    "reason": "Technical validation failed.",
                }
                continue

            if (
                architecture_signature(
                    candidate["config"]
                )
                == architecture_signature(
                    self._parent_config()
                )
            ):
                processed[candidate_id] = {
                    "status": "SKIPPED",
                    "reason": "Candidate architecture equals parent.",
                }
                continue

            implementation = (
                self.controller.evolution
                .implementation_screen(
                    [candidate]
                )
            )

            if not implementation[
                "supported"
            ]:
                reason = (
                    candidate.get(
                        "implementation_validation",
                        {},
                    )
                )

                processed[candidate_id] = {
                    "status": "SKIPPED",
                    "reason": reason,
                }

                continue

            attempts += 1

            print("=" * 72)
            print(
                f"ATTEMPT {attempts}/{max_attempts}: "
                f"{candidate_id}"
            )
            print(
                "Mutation     :",
                candidate.get(
                    "mutation"
                ),
            )
            print("=" * 72)

            candidate["implementation_validation"] = (
                implementation["supported"][0]
                .get(
                    "implementation_validation",
                    {},
                )
            )

            if candidate_id in historical:
                robust = historical[
                    candidate_id
                ]

                evaluation = (
                    self.controller.evaluator.evaluate(
                        robust
                    )
                )

                evaluation_dict = (
                    evaluation.to_dict()
                )

                result = {
                    "candidate": candidate_id,
                    "source": "historical_robust",
                    "robust": robust,
                    "evaluation": evaluation_dict,
                    "status": evaluation.action,
                }

            else:
                evaluation = (
                    self.controller.evaluate_candidate(
                        candidate=candidate,
                        screening_steps=screening_steps,
                        robust_steps=robust_steps,
                        robust_seeds=seeds,
                        problem=(
                            "Autonomously improve "
                            "NOVA-EVO GEN4 over GEN3-007."
                        ),
                    )
                )

                result = evaluation

            self.save_candidate_result(
                result
            )

            processed[candidate_id] = {
                "status": result.get(
                    "status"
                ),
                "result_file": str(
                    self._result_path(
                        candidate_id
                    )
                ),
                "timestamp": time.time(),
            }

            state["processed"] = processed
            state["attempts"] = attempts
            state["next_index"] = attempts + 1
            state["last_candidate"] = candidate_id
            state["last_status"] = result.get(
                "status"
            )
            self.save_manager_state(
                state
            )

            status = result.get(
                "status"
            )

            print()
            print(
                "RESULT     :",
                status,
            )

            if status == "PROMOTE":
                promoted = self._promote(
                    candidate,
                    result,
                )

                state["status"] = "PROMOTED"
                state["promoted"] = promoted
                self.save_manager_state(
                    state
                )

                print(
                    "GEN4 RESULT:",
                    "PROMOTED",
                )
                print(
                    "Candidate   :",
                    candidate_id,
                )
                print(
                    "Checkpoint  :",
                    promoted["checkpoint"],
                )

                return {
                    "status": "PROMOTED",
                    "candidate": candidate_id,
                    "attempts": attempts,
                    "processed": processed,
                    "promotion": promoted,
                }

            print(
                "Continuing to next candidate..."
            )

        if attempts >= max_attempts:
            final_status = "ATTEMPT_LIMIT"
        else:
            final_status = "EXHAUSTED"

        state["status"] = final_status
        state["processed"] = processed
        state["attempts"] = attempts
        self.save_manager_state(
            state
        )

        print()
        print("=" * 72)
        print("GEN4 SEARCH FINISHED")
        print("=" * 72)
        print("STATUS  :", final_status)
        print("ATTEMPTS:", attempts)
        print("PARENT  :", "GEN3-007")
        print("=" * 72)

        return {
            "status": final_status,
            "parent": "GEN3-007",
            "attempts": attempts,
            "processed": processed,
        }


def self_test() -> None:
    manager = object.__new__(
        GenerationManager
    )

    manager.result_dir = Path(
        "/tmp/nova-evo-generation-manager-test"
    )

    assert (
        manager._result_path(
            "GEN4-001"
        ).name
        == "GEN4-001.json"
    )

    assert (
        manager._result_path(
            "GEN4-048"
        ).name
        == "GEN4-048.json"
    )

    print(
        "GENERATION MANAGER SELFTEST: PASSED"
    )


if __name__ == "__main__":
    self_test()
