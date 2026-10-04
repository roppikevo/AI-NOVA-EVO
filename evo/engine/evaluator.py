from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class EvaluationDecision:
    action: str
    candidate_id: str
    parent_id: str | None
    candidate_loss: float | None
    parent_loss: float | None
    delta: float | None
    reason: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "candidate_id": self.candidate_id,
            "parent_id": self.parent_id,
            "candidate_loss": self.candidate_loss,
            "parent_loss": self.parent_loss,
            "delta": self.delta,
            "reason": self.reason,
            "confidence": self.confidence,
        }


class Evaluator:

    def __init__(
        self,
        project_root: str | Path = ".",
    ) -> None:
        self.project_root = Path(
            project_root
        ).resolve()
    """
    Objective evaluator for NOVA-EVO experiments.

    The evaluator does not train models and does not modify code.
    It only converts experimental evidence into a bounded decision.
    """

    def load_parent_baseline(
        self,
        project_root: str | Path,
        parent_id: str,
    ) -> dict[str, Any] | None:
        """
        Load the immutable baseline for the specified parent.

        GEN3 uses its locked benchmark.
        Promoted CORE generations use their official promotion
        record, which contains the exact robust evidence that
        authorized promotion.
        """

        root = Path(project_root).resolve()

        # ---------------------------------------------------------
        # GEN3 — immutable locked baseline
        # ---------------------------------------------------------

        baseline_file = (
            root
            / "evo"
            / "gen3"
            / "LOCKED"
            / "robust_results.json"
        )

        if baseline_file.exists():
            payload = json.loads(
                baseline_file.read_text(
                    encoding="utf-8"
                )
            )

            summary = payload.get(
                "summary",
                {},
            )

            baseline = summary.get(
                parent_id
            )

            if baseline is not None:
                return {
                    "candidate_id": parent_id,
                    "validation_loss": float(
                        baseline["val_loss_mean"]
                    ),
                    "validation_std": float(
                        baseline["val_loss_std"]
                    ),
                    "parameters": int(
                        baseline["params"]
                    ),
                    "seeds": list(
                        baseline.get(
                            "seeds",
                            [],
                        )
                    ),
                    "source": str(
                        baseline_file
                    ),
                    "immutable": True,
                }

        # ---------------------------------------------------------
        # PROMOTED CORE — official promotion record
        # ---------------------------------------------------------

        official_file = (
            root
            / "evo"
            / "core_evolution"
            / "results"
            / f"{parent_id}.official.json"
        )

        if official_file.exists():
            official = json.loads(
                official_file.read_text(
                    encoding="utf-8"
                )
            )

            if official.get(
                "active_core"
            ) != parent_id:
                return None

            evaluation = official.get(
                "evaluation",
                {},
            )

            robust = official.get(
                "robust",
                {},
            )

            metrics = robust.get(
                "metrics",
                {},
            )

            if (
                evaluation.get("action")
                != "PROMOTE"
            ):
                return None

            loss = metrics.get(
                "validation_loss_mean"
            )

            if not isinstance(
                loss,
                (int, float),
            ):
                return None

            return {
                "candidate_id": parent_id,
                "validation_loss": float(
                    loss
                ),
                "validation_std": float(
                    metrics.get(
                        "validation_loss_std",
                        0.0,
                    )
                ),
                "parameters": int(
                    metrics.get(
                        "parameters",
                        0,
                    )
                ),
                "seeds": list(
                    robust.get(
                        "seeds",
                        [],
                    )
                ),
                "source": str(
                    official_file
                ),
                "immutable": True,
            }

        # ---------------------------------------------------------
        # Current best-known fallback
        # ---------------------------------------------------------

        state_file = (
            root
            / "evo"
            / "engine"
            / "evo_state.json"
        )

        if state_file.exists():
            state = json.loads(
                state_file.read_text(
                    encoding="utf-8"
                )
            )

            best = state.get(
                "best_known",
                {},
            )

            if best.get(
                "candidate"
            ) == parent_id:

                loss = best.get(
                    "validation_loss"
                )

                if isinstance(
                    loss,
                    (int, float),
                ):
                    return {
                        "candidate_id": parent_id,
                        "validation_loss": float(
                            loss
                        ),
                        "validation_std": 0.0,
                        "parameters": int(
                            best.get(
                                "parameters",
                                0,
                            )
                        ),
                        "seeds": [],
                        "source": str(
                            state_file
                        ),
                        "immutable": False,
                    }

        return None

    def evaluate(
        self,
        experiment: dict[str, Any],
        parent_metrics: dict[str, Any] | None = None,
    ) -> EvaluationDecision:

        candidate_id = str(
            experiment.get("candidate_id", "")
        )

        parent_id = experiment.get("parent_id")

        if parent_metrics is None and parent_id:
           parent_metrics = self.load_parent_baseline(
           self.project_root,
           parent_id,
        )
        state = experiment.get("state")

        if state != "COMPLETED":
            return EvaluationDecision(
                action="RETRY",
                candidate_id=candidate_id,
                parent_id=parent_id,
                candidate_loss=None,
                parent_loss=None,
                delta=None,
                reason="Experiment did not complete successfully.",
                confidence=0.95,
            )

        metrics = experiment.get("metrics", {})

        training = experiment.get("training", {})
        stage = training.get("stage", "")

        if stage == "screening":
            return EvaluationDecision(
                action="RETRY",
                candidate_id=candidate_id,
                parent_id=parent_id,
                candidate_loss=None,
                parent_loss=None,
                delta=None,
                reason=(
                    "Screening experiment cannot promote a candidate; "
                    "robust evaluation is required."
                ),
                confidence=0.99,
            )

        if stage != "robust":
            return EvaluationDecision(
                action="RETRY",
                candidate_id=candidate_id,
                parent_id=parent_id,
                candidate_loss=None,
                parent_loss=None,
                delta=None,
                reason=(
                    "Only robust experiments can be evaluated "
                    "for promotion."
                ),
                confidence=0.99,
            )

        seeds = training.get("seeds", [])

        if len(seeds) < 3:
            return EvaluationDecision(
                action="RETRY",
                candidate_id=candidate_id,
                parent_id=parent_id,
                candidate_loss=None,
                parent_loss=None,
                delta=None,
                reason=(
                    "Robust evaluation requires at least "
                    "three seeds."
                ),
                confidence=0.99,
            )
        candidate_loss = metrics.get(
            "validation_loss_mean"
        )
        if candidate_loss is None:
            return EvaluationDecision(
                action="RETRY",
                candidate_id=candidate_id,
                parent_id=parent_id,
                candidate_loss=None,
                parent_loss=None,
                delta=None,
                reason="Completed experiment has no validation loss.",
                confidence=0.95,
            )

        candidate_loss = float(candidate_loss)

        if not parent_metrics:
            return EvaluationDecision(
                action="REJECT",
                candidate_id=candidate_id,
                parent_id=parent_id,
                candidate_loss=candidate_loss,
                parent_loss=None,
                delta=None,
                reason="Parent baseline is missing.",
                confidence=0.9,
            )

        parent_loss = parent_metrics.get(
            "validation_loss"
        )

        if parent_loss is None:
            return EvaluationDecision(
                action="REJECT",
                candidate_id=candidate_id,
                parent_id=parent_id,
                candidate_loss=candidate_loss,
                parent_loss=None,
                delta=None,
                reason="Parent baseline has no validation loss.",
                confidence=0.9,
            )

        parent_loss = float(parent_loss)
        delta = candidate_loss - parent_loss

        if candidate_loss < parent_loss:
            return EvaluationDecision(
                action="PROMOTE",
                candidate_id=candidate_id,
                parent_id=parent_id,
                candidate_loss=candidate_loss,
                parent_loss=parent_loss,
                delta=delta,
                reason=(
                    "Candidate validation loss is lower "
                    "than the parent baseline."
                ),
                confidence=0.8,
            )

        return EvaluationDecision(
            action="REJECT",
            candidate_id=candidate_id,
            parent_id=parent_id,
            candidate_loss=candidate_loss,
            parent_loss=parent_loss,
            delta=delta,
            reason=(
                "Candidate validation loss is not lower "
                "than the parent baseline."
            ),
            confidence=0.8,
        )


def self_test() -> None:
    evaluator = Evaluator()

    experiment = {
        "candidate_id": "GEN4-TEST",
        "parent_id": "GEN3-007",
        "state": "COMPLETED",
        "training": {
            "stage": "robust",
            "steps": 1000,
            "batch_size": 8,
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "seeds": [1001, 2002, 3003],
        },
        "metrics": {
            "validation_loss_mean": 8.5,
        },
    }

    decision = evaluator.evaluate(
        experiment,
        {"validation_loss": 8.6},
    )

    print("DECISION:", decision.to_dict())
    assert decision.action == "PROMOTE" 
    assert decision.candidate_id == "GEN4-TEST"
    assert decision.parent_id == "GEN3-007"
    assert decision.candidate_loss == 8.5
    assert decision.parent_loss == 8.6
    assert abs(decision.delta - (-0.1)) < 1e-9

    print("EVALUATOR SELF-TEST: PASSED")


if __name__ == "__main__":
    self_test()
