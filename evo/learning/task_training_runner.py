from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from evo.engine.architecture_factory import build_model
from evo.engine.knowledge_store import KnowledgeStore
from evo.learning.task_dataset import TaskLearningDataset
from nova.data import TokenSequenceDataset
from nova.training import TrainConfig, evaluate, train


ROOT = Path(__file__).resolve().parents[2]


class TaskTrainingRunner:
    """
    Incremental task-learning runner.

    Training source:
        validated LearningExecutor experiences.

    Selection gate:
        validation loss only.

    Holdout:
        test split is measured for reporting but never used
        to decide checkpoint promotion.

    Evolution:
        completely separate from TrainingRunner / Gen4.
    """

    def __init__(
        self,
        root: str | Path = ROOT,
    ) -> None:
        self.root = Path(root)

        self.knowledge = KnowledgeStore()

        self.dataset = TaskLearningDataset(
            knowledge=self.knowledge,
            seq_len=128,
        )

        self.dataset_dir = (
            self.root
            / "data"
            / "task_learning_autonomous"
        )

        self.state_file = (
            self.root
            / "evo"
            / "learning"
            / "task_learning_state.json"
        )

        self.checkpoint_dir = (
            self.root
            / "evo"
            / "learning"
            / "checkpoints"
            / "task_autonomous"
        )

        self.bootstrap_checkpoint = (
            self.root
            / "evo"
            / "learning"
            / "checkpoints"
            / "task_v1_standard_step100_padaware.pt"
        )

    def load_state(self) -> dict[str, Any]:
        if not self.state_file.exists():
            return {}

        try:
            return json.loads(
                self.state_file.read_text(
                    encoding="utf-8"
                )
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid task-learning state: {exc}"
            )

    def save_state(
        self,
        state: dict[str, Any],
    ) -> None:
        self.state_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.state_file.write_text(
            json.dumps(
                state,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def build_dataset(
        self,
        *,
        min_examples: int = 32,
    ) -> dict[str, Any]:
        return self.dataset.build(
            self.dataset_dir,
            min_examples=min_examples,
        )

    def _parent_config(self) -> dict[str, Any]:
        state_file = (
            self.root
            / "evo"
            / "engine"
            / "evo_state.json"
        )

        data = json.loads(
            state_file.read_text(
                encoding="utf-8"
            )
        )

        config = dict(
            data.get(
                "primary_parent_config",
                {},
            )
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

    def _resolve_active_checkpoint(
        self,
        manifest: dict[str, Any],
    ) -> tuple[Path | None, dict[str, Any]]:
        state = self.load_state()

        active = state.get(
            "active_checkpoint"
        )

        if active:
            path = Path(active)

            if not path.is_absolute():
                path = self.root / path

            if not path.exists():
                raise FileNotFoundError(
                    f"Active task checkpoint missing: {path}"
                )

            return path, state

        if self.bootstrap_checkpoint.exists():
            previous_manifest = (
                self.root
                / "data"
                / "task_learning_v1"
                / "manifest.json"
            )

            trained_examples = 0

            if previous_manifest.exists():
                try:
                    previous = json.loads(
                        previous_manifest.read_text(
                            encoding="utf-8"
                        )
                    )
                    trained_examples = int(
                        previous.get(
                            "validated_examples",
                            0,
                        )
                    )
                except (
                    OSError,
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                ):
                    trained_examples = 0

            state = {
                "version": 1,
                "active_checkpoint": str(
                    self.bootstrap_checkpoint
                ),
                "last_trained_examples": trained_examples,
                "last_status": "BOOTSTRAPPED",
                "updated_at": time.time(),
            }

            self.save_state(state)

            return (
                self.bootstrap_checkpoint,
                state,
            )

        return None, state

    def _load_model(
        self,
        checkpoint: Path | None,
    ) -> tuple[torch.nn.Module, dict[str, Any]]:
        checkpoint_data: dict[str, Any] = {}

        config = self._parent_config()

        if checkpoint is not None:
            checkpoint_data = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
            )

            saved_config = checkpoint_data.get(
                "config"
            )

            if isinstance(saved_config, dict):
                config = dict(saved_config)

        model = build_model(config)

        if checkpoint is not None:
            model.load_state_dict(
                checkpoint_data["model_state_dict"]
            )

        return model, config

    @staticmethod
    def _loader(
        dataset: TokenSequenceDataset,
        *,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def inspect(
        self,
        *,
        min_examples: int = 32,
    ) -> dict[str, Any]:
        manifest = self.build_dataset(
            min_examples=min_examples,
        )

        checkpoint, state = (
            self._resolve_active_checkpoint(
                manifest
            )
        )

        return {
            "status": "READY"
            if checkpoint is not None
            else "WAITING_FOR_BOOTSTRAP",
            "manifest": manifest,
            "active_checkpoint": (
                str(checkpoint)
                if checkpoint
                else None
            ),
            "last_trained_examples": int(
                state.get(
                    "last_trained_examples",
                    0,
                )
            ),
        }

    def maybe_train(
        self,
        *,
        min_examples: int = 32,
        min_new_examples: int = 4,
        steps: int = 100,
        seed: int = 1001,
        batch_size: int = 8,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
    ) -> dict[str, Any]:
        manifest = self.build_dataset(
            min_examples=min_examples,
        )

        current_examples = int(
            manifest["validated_examples"]
        )

        checkpoint, state = (
            self._resolve_active_checkpoint(
                manifest
            )
        )

        if checkpoint is None:
            return {
                "status": "WAITING_FOR_BOOTSTRAP",
                "validated_examples": current_examples,
            }

        last_trained = int(
            state.get(
                "last_trained_examples",
                0,
            )
        )

        if (
            current_examples < min_examples
            or (
                current_examples - last_trained
                < min_new_examples
            )
        ):
            return {
                "status": "SKIPPED",
                "reason": "Not enough new validated examples.",
                "validated_examples": current_examples,
                "last_trained_examples": last_trained,
                "new_examples": (
                    current_examples - last_trained
                ),
                "active_checkpoint": str(
                    checkpoint
                ),
            }

        train_path = (
            self.dataset_dir / "train.txt"
        )
        val_path = (
            self.dataset_dir / "val.txt"
        )
        test_path = (
            self.dataset_dir / "test.txt"
        )

        train_dataset = TokenSequenceDataset(
            train_path,
            seq_len=128,
        )
        val_dataset = TokenSequenceDataset(
            val_path,
            seq_len=128,
        )
        test_dataset = TokenSequenceDataset(
            test_path,
            seq_len=128,
        )

        val_loader = self._loader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        test_loader = self._loader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        model, config = self._load_model(
            checkpoint
        )

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        model = model.to(device)

        before_val = evaluate(
            model,
            val_loader,
            device,
            pad_id=0,
        )

        before_test = evaluate(
            model,
            test_loader,
            device,
            pad_id=0,
        )

        train_config = TrainConfig(
            seed=seed,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_steps=steps,
            eval_every=steps,
            log_every=steps,
            device="cuda",
            pad_id=0,
        )

        started = time.perf_counter()

        model, history = train(
            model,
            train_dataset,
            val_dataset,
            train_config,
        )

        duration = (
            time.perf_counter()
            - started
        )

        after_val = evaluate(
            model,
            val_loader,
            device,
            pad_id=0,
        )

        after_test = evaluate(
            model,
            test_loader,
            device,
            pad_id=0,
        )

        promoted = (
            after_val["loss"]
            < before_val["loss"]
        )

        result: dict[str, Any] = {
            "status": (
                "PROMOTED"
                if promoted
                else "REJECTED"
            ),
            "validated_examples": current_examples,
            "new_examples": (
                current_examples - last_trained
            ),
            "parent_checkpoint": str(
                checkpoint
            ),
            "metrics": {
                "validation_before": before_val,
                "validation_after": after_val,
                "test_before": before_test,
                "test_after": after_test,
                "duration_seconds": duration,
                "steps": steps,
                "pad_id": 0,
            },
            "history": history,
        }

        if promoted:
            self.checkpoint_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            checkpoint_path = (
                self.checkpoint_dir
                / (
                    f"task_auto_"
                    f"{current_examples}_"
                    f"{int(time.time())}.pt"
                )
            )

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "config": config,
                    "training": {
                        "seed": seed,
                        "batch_size": batch_size,
                        "learning_rate": (
                            learning_rate
                        ),
                        "weight_decay": (
                            weight_decay
                        ),
                        "max_steps": steps,
                        "pad_id": 0,
                    },
                    "dataset": manifest,
                    "parent_checkpoint": str(
                        checkpoint
                    ),
                    "benchmark": result["metrics"],
                },
                checkpoint_path,
            )

            state = {
                "version": 1,
                "active_checkpoint": str(
                    checkpoint_path
                ),
                "last_trained_examples": (
                    current_examples
                ),
                "last_status": "PROMOTED",
                "last_validation_loss": (
                    after_val["loss"]
                ),
                "updated_at": time.time(),
            }

            self.save_state(state)

            result["checkpoint"] = str(
                checkpoint_path
            )

        else:
            state["last_status"] = "REJECTED"
            state["updated_at"] = time.time()
            self.save_state(state)

            result["checkpoint"] = None

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result


def self_test() -> None:
    runner = TaskTrainingRunner()

    assert runner.dataset_dir.name == (
        "task_learning_autonomous"
    )

    assert runner.dataset.seq_len == 128

    print(
        "TASK TRAINING RUNNER SELFTEST: PASSED"
    )


if __name__ == "__main__":
    self_test()
