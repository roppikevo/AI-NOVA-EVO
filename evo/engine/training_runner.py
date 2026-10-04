from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

import torch

from evo.engine.architecture_factory import build_model
from evo.engine.experiment_record import ExperimentRecord
from evo.engine.experiment_store import ExperimentStore
from evo.engine.resource_manager import ResourceManager
from nova.calibration import measure_calibration
from nova.efficiency import measure_cpu
from nova.data import TokenSequenceDataset
from nova.training import TrainConfig, train


ROOT = Path(__file__).resolve().parents[2]


class TrainingRunner:
    """
    Executes one controlled NOVA-EVO training experiment.

    Selection data:
        dedicated generation train/validation data only.

    Final holdout:
        never loaded by this runner.
    """

    def __init__(
        self,
        root: str | Path = ROOT,
    ) -> None:
        self.root = Path(root)

        self.experiments = ExperimentStore(
            root=self.root / "evo" / "experiments"
        )

        self.resources = ResourceManager(
            project_root=self.root
        )

        self.state_file = (
            self.root
            / "evo"
            / "engine"
            / "evo_state.json"
        )

    def load_state(self) -> dict[str, Any]:
        import json

        return json.loads(
            self.state_file.read_text(
                encoding="utf-8"
            )
        )

    def dataset_root(self) -> Path:
        state = self.load_state()

        paths = state["data_policy"]["training_data"]

        if not paths:
            raise ValueError(
                "No training dataset configured."
            )

        dataset = self.root / paths[0]

        if not dataset.exists():
            raise FileNotFoundError(
                f"Dataset not found: {dataset}"
            )

        return dataset

    def run_screening(
        self,
        candidate: dict[str, Any],
        steps: int = 100,
        seed: int = 1001,
        batch_size: int = 8,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        stage: str = "screening",
    ) -> dict[str, Any]:
        candidate_id = candidate["candidate"]
        generation = int(
            candidate["generation"]
        )
        parent_id = candidate.get("parent")

        config = candidate["config"]

        dataset_root = self.dataset_root()

        train_path = dataset_root / "train.txt"
        val_path = dataset_root / "val.txt"

        if not train_path.exists():
            raise FileNotFoundError(train_path)

        if not val_path.exists():
            raise FileNotFoundError(val_path)

        record = ExperimentRecord(
            generation=generation,
            candidate_id=candidate_id,
            parent_id=parent_id,
            hypothesis=(
                "Evaluate candidate architecture "
                "on dedicated GEN4 training data."
            ),
            objective=(
                "Minimize validation loss while "
                "respecting resource limits."
            ),
            architecture=config,
            dataset={
                "root": str(dataset_root),
                "train": str(train_path),
                "validation": str(val_path),
                "test_loaded": False,
                "selection_data": True,
            },
            training={
                "stage": stage,
                "steps": steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
            },
            seeds=[seed],
        )

        record.finalize()
        self.experiments.save(record)

        try:
            record.start()
            self.experiments.update(record)

            resource_before = (
                self.resources.snapshot()
            )

            train_dataset = TokenSequenceDataset(
                train_path,
                seq_len=128,
            )

            val_dataset = TokenSequenceDataset(
                val_path,
                seq_len=128,
            )

            from nova.gpu_guard import ensure_vram

            ensure_vram()
            model = build_model(config)

            parameter_count = sum(
                parameter.numel()
                for parameter in model.parameters()
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
            )

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

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

            if not history:
                raise RuntimeError(
                    "Training returned no evaluation history."
                )

            final_metrics = history[-1]

            resource_after = (
                self.resources.snapshot()
            )

            peak_vram_mb = (
                round(torch.cuda.max_memory_allocated() / 1e6, 1)
                if torch.cuda.is_available()
                else 0.0
            )

            try:
                efficiency = measure_cpu(
                    model,
                    vocab_size=int(config["vocab_size"]),
                )
            except Exception as exc:  # never fail training on a benchmark
                efficiency = {"error": f"{type(exc).__name__}: {exc}"}

            efficiency["peak_vram_mb"] = peak_vram_mb

            try:
                calibration = measure_calibration(
                    model,
                    val_dataset,
                    next(model.parameters()).device,
                )
            except Exception as exc:
                calibration = {"error": f"{type(exc).__name__}: {exc}"}

            metrics = {
                "efficiency": efficiency,
                "calibration": calibration,
                "validation_loss": float(
                    final_metrics["loss"]
                ),
                "perplexity": float(
                    final_metrics["perplexity"]
                ),
                "train_loss": float(
                    final_metrics["train_loss"]
                ),
                "steps": steps,
                "parameters": parameter_count,
                "duration_seconds": duration,
            }

            # Robust runs must preserve the exact trained weights
            # that produced the measured result. Screening runs do
            # not save checkpoints to avoid unnecessary disk usage.
            if stage == "robust":
                checkpoint_dir = (
                    self.root
                    / "evo"
                    / "gen4"
                    / "results"
                    / "checkpoints"
                    / candidate_id
                )
                checkpoint_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                checkpoint_path = (
                    checkpoint_dir
                    / f"seed_{seed}.pt"
                )

                cpu_state = {
                    key: value.detach().cpu()
                    for key, value
                    in model.state_dict().items()
                }

                torch.save(
                    {
                        "candidate": candidate_id,
                        "parent": parent_id,
                        "generation": generation,
                        "seed": seed,
                        "steps": steps,
                        "params": parameter_count,
                        "val_loss": float(
                            final_metrics["loss"]
                        ),
                        "config": config,
                        "model_state_dict": cpu_state,
                    },
                    checkpoint_path,
                )

                metrics["checkpoint"] = str(
                    checkpoint_path
                )

            resources = {
                "before": resource_before.to_dict(),
                "after": resource_after.to_dict(),
            }

            record.resources = resources

            record.complete(metrics)

            self.experiments.update(record)

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                "status": "COMPLETED",
                "experiment_id": record.experiment_id,
                "candidate": candidate_id,
                "metrics": metrics,
                "resources": resources,
            }

        except Exception as exc:
            record.fail(
                "training_failure",
                {
                    "error": str(exc),
                    "candidate": candidate_id,
                },
            )

            self.experiments.update(record)

            return {
                "status": "FAILED",
                "experiment_id": record.experiment_id,
                "candidate": candidate_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
    def run_robust(
        self,
        candidate: dict[str, Any],
        steps: int = 1000,
        seeds: list[int] | None = None,
        batch_size: int = 8,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
    ) -> dict[str, Any]:

        if seeds is None:
            seeds = [1001, 2002, 3003]

        results: list[dict[str, Any]] = []

        for seed in seeds:
            result = self.run_screening(
                candidate=candidate,
                steps=steps,
                seed=seed,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                stage="robust",
            )

            if result.get("status") != "COMPLETED":
                return {
                    "status": "FAILED",
                    "candidate": candidate["candidate"],
                    "stage": "robust",
                    "seed_results": results,
                    "failed_seed": seed,
                    "error": result.get("error"),
                    "traceback": result.get("traceback"),
                }

            results.append(result)

        losses = [
            float(
                result["metrics"]["validation_loss"]
            )
            for result in results
        ]

        mean_loss = sum(losses) / len(losses)

        variance = sum(
            (loss - mean_loss) ** 2
            for loss in losses
        ) / len(losses)

        std_loss = variance ** 0.5

        parameters = int(
            results[0]["metrics"]["parameters"]
        )

        durations = [
            float(
                result["metrics"]["duration_seconds"]
            )
            for result in results
        ]

        mean_duration = (
            sum(durations) / len(durations)
        )

        efficiency = aggregate_efficiency(
            [r["metrics"].get("efficiency", {}) for r in results]
        )
        calibration = aggregate_calibration(
            [r["metrics"].get("calibration", {}) for r in results]
        )

        return {
            "status": "COMPLETED",
            "candidate": candidate["candidate"],
            "candidate_id": candidate["candidate"],
            "parent_id": candidate.get("parent"),
            "state": "COMPLETED",
            "stage": "robust",
            "seeds": seeds,
            "seed_results": results,
            "training": {
                "stage": "robust",
                "steps": steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "seeds": seeds,
            },
            "metrics": {
                "validation_loss_mean": mean_loss,
                "validation_loss_std": std_loss,
                "validation_loss_min": min(losses),
                "validation_loss_max": max(losses),
                "parameters": parameters,
                "duration_seconds_mean": mean_duration,
                "efficiency": efficiency,
                "calibration": calibration,
            },
            "dataset": dataset_label(self.root, self.dataset_root()),
        }

def dataset_label(root: Path, dataset: Path) -> str:
    try:
        return str(dataset.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(dataset)


def aggregate_calibration(items: list[dict[str, Any]]) -> dict[str, Any]:
    items = [i for i in items if i and "ece" in i]
    if not items:
        return {}
    return {
        key: round(sum(i[key] for i in items) / len(items), 4)
        for key in ("ece", "accuracy", "mean_confidence")
    }


def aggregate_efficiency(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean of speeds, max of memory, first value of static sizes."""
    items = [i for i in items if i and "error" not in i]
    if not items:
        return {}

    def values(key):
        return [i[key] for i in items if isinstance(i.get(key), (int, float))]

    out: dict[str, Any] = {}
    for key in ("parameters", "param_mb", "state_kb", "cpu_threads"):
        v = values(key)
        if v:
            out[key] = v[0]
    for key in ("cpu_tokens_per_sec", "cpu_gen_tokens_per_sec"):
        v = values(key)
        if v:
            out[key] = round(sum(v) / len(v), 1)
    v = values("peak_vram_mb")
    if v:
        out["peak_vram_mb"] = max(v)
    return out


def self_test() -> None:
    runner = TrainingRunner()

    assert runner.state_file.exists()

    dataset = runner.dataset_root()

    assert (dataset / "train.txt").exists()
    assert (dataset / "val.txt").exists()

    assert not (dataset / "test.txt").name == ""

    print("TRAINING RUNNER SELFTEST: PASSED")


if __name__ == "__main__":
    self_test()

