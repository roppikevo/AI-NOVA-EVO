from __future__ import annotations

from evo.engine.training_runner import TrainingRunner
import math
import json
import time
import uuid
from evo.engine.evaluator import Evaluator
from evo.engine.evolution_engine import EvolutionEngine
from evo.learning.meta_learning import MetaLearningEngine
from evo.learning.adaptive_strategy import AdaptiveTrainingStrategy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from .knowledge_store import KnowledgeStore
    from .experiment_record import ExperimentRecord
    from .experiment_store import ExperimentStore
    from .resource_manager import ResourceManager
    from .validation_engine import ValidationEngine
    from .authority_manager import (
        AuthorityError,
        AuthorityManager,
    )
    from .oxcoder_mentor import (
        MentorRequest,
        OxCoderMentor,
    )
except ImportError:
    from knowledge_store import KnowledgeStore
    from experiment_record import ExperimentRecord
    from experiment_store import ExperimentStore
    from resource_manager import ResourceManager
    from validation_engine import ValidationEngine
    from authority_manager import (
        AuthorityError,
        AuthorityManager,
    )
    from oxcoder_mentor import (
        MentorRequest,
        OxCoderMentor,
    )


@dataclass
class AgentDecision:
    """Decision produced by the NOVA-EVO controller."""

    action: str
    reason: str
    confidence: float = 0.0
    requires_experiment: bool = True
    mentor_used: bool = False
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "confidence": self.confidence,
            "requires_experiment": (
                self.requires_experiment
            ),
            "mentor_used": self.mentor_used,
            "metadata": self.metadata,
        }


@dataclass
class AgentState:
    """Persistent high-level controller state."""

    run_id: str
    generation: int
    candidate_id: str | None = None
    phase: str = "IDLE"
    last_action: str | None = None
    cycle_count: int = 0
    repair_attempts: int = 0
    mentor_requests: int = 0
    experiments_started: int = 0
    experiments_completed: int = 0
    experiments_failed: int = 0
    started_at: float = field(
        default_factory=time.time
    )
    updated_at: float = field(
        default_factory=time.time
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "generation": self.generation,
            "candidate_id": self.candidate_id,
            "phase": self.phase,
            "last_action": self.last_action,
            "cycle_count": self.cycle_count,
            "repair_attempts": self.repair_attempts,
            "mentor_requests": self.mentor_requests,
            "experiments_started": (
                self.experiments_started
            ),
            "experiments_completed": (
                self.experiments_completed
            ),
            "experiments_failed": (
                self.experiments_failed
            ),
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }


class AgentController:
    """
    Autonomous NOVA-EVO controller.

    This layer coordinates the subsystems but does not bypass
    their safety boundaries.

    The initial implementation supports dry-run orchestration.
    Real candidate mutation, training and promotion are enabled
    only after explicit integration and validation.
    """

    VALID_PHASES = {
        "IDLE",
        "OBSERVE",
        "UNDERSTAND",
        "MEMORY",
        "PLAN",
        "IMPLEMENT",
        "VALIDATE",
        "EXPERIMENT",
        "MEASURE",
        "LEARN",
        "DECIDE",
        "STOPPED",
    }

    VALID_ACTIONS = {
        "PROMOTE",
        "REJECT",
        "REPAIR",
        "RETRY",
        "MUTATE",
        "INVESTIGATE",
        "ASK_MENTOR",
        "STOP",
    }

    def __init__(
        self,
        project_root: str | Path = ".",
        generation: int = 4,
        dry_run: bool = True,
    ) -> None:
        self.project_root = Path(
            project_root
        ).resolve()

        self.generation = generation
        self.dry_run = dry_run

        self.evo_root = (
            self.project_root / "evo"
        )

        self.engine_root = (
            self.evo_root / "engine"
        )

        self.state_dir = (
            self.engine_root / "state"
        )

        self.state_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.state_file = (
            self.state_dir / "agent_state.json"
        )

        self.knowledge = KnowledgeStore(
            root=(
                self.evo_root / "knowledge"
            )
        )

        self.experiments = ExperimentStore(
            root=(
                self.evo_root / "experiments"
            )
        )

        self.resources = ResourceManager(
            project_root=self.project_root
        )

        self.training_runner = TrainingRunner(
            root=self.project_root
        )

        self.evolution = EvolutionEngine(
            root=self.project_root
        )

        self.evaluator = Evaluator(
            project_root=self.project_root
        )

        self.validation = ValidationEngine(
            project_root=self.project_root
        )

        self.mentor = OxCoderMentor()

        self.meta_learning = MetaLearningEngine(
            knowledge=self.knowledge
        )

        self.adaptive_strategy = AdaptiveTrainingStrategy(
            knowledge=self.knowledge
        )

        self.authority = AuthorityManager(
            actor_id="NOVA-EVO"
        )

        self.creator_authority = AuthorityManager(
            actor_id=self.authority.creator_id()
        )

        self._creator_identity: str | None = None

        self.state = AgentState(
            run_id=str(uuid.uuid4()),
            generation=generation,
        )

    def authenticate_creator(
        self,
        username: str,
        password: str,
    ) -> bool:
        """Authenticate Creator without exposing or storing the password."""

        try:
            identity = self.creator_authority.authenticate_and_identify(
                username,
                password,
            )
        except AuthorityError:
            self._creator_identity = None
            return False

        if identity != self.creator_authority.creator_id():
            self._creator_identity = None
            return False

        self._creator_identity = identity
        return True

    def creator_authenticated(self) -> bool:
        return (
            self._creator_identity == self.creator_authority.creator_id()
        )

    def creator_command(
        self,
        username: str,
        password: str,
        command: str,
        *,
        problem: str = "",
        context: str = "",
        candidate_id: str | None = None,
        candidate_count: int = 3,
        screening_steps: int = 100,
        robust_steps: int = 1000,
        robust_seeds: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Execute one authenticated Creator command.

        The password is used only for authentication and is never stored,
        returned, or passed to the model.
        """

        if not self.authenticate_creator(
            username,
            password,
        ):
            return {
                "success": False,
                "error": "Creator authentication failed.",
            }

        command = command.strip().lower()

        try:
            if command in {
                "status",
                "inspect",
            }:
                self.creator_authority.assert_runtime_action(
                    "inspect"
                )

                return {
                    "success": True,
                    "identity": self._creator_identity,
                    "authority": "CREATOR",
                    "state": self.state.to_dict(),
                }

            if command in {
                "stop",
                "emergency_stop",
            }:
                self.creator_authority.assert_runtime_action(
                    "emergency_stop"
                )

                self.stop()

                return {
                    "success": True,
                    "identity": self._creator_identity,
                    "action": "STOP",
                    "state": self.state.to_dict(),
                }

            if command in {
                "run",
                "start",
                "set_goal",
            }:
                if not problem.strip():
                    return {
                        "success": False,
                        "identity": self._creator_identity,
                        "error": (
                            "Creator command requires "
                            "a non-empty problem/goal."
                        ),
                    }

                self.creator_authority.assert_runtime_action(
                    "set_goal"
                )

                self.creator_authority.assert_runtime_action(
                    "override_autonomy"
                )

                result = self.run_cycle(
                    problem=problem,
                    candidate_id=candidate_id,
                    context=context,
                    candidate_count=candidate_count,
                    screening_steps=screening_steps,
                    robust_steps=robust_steps,
                    robust_seeds=robust_seeds,
                )

                result["creator_command"] = True
                result["creator_identity"] = (
                    self._creator_identity
                )

                return result

            return {
                "success": False,
                "identity": self._creator_identity,
                "error": (
                    f"Unknown Creator command: {command}"
                ),
            }

        except AuthorityError as exc:
            return {
                "success": False,
                "identity": self._creator_identity,
                "error": str(exc),
            }

    def set_phase(
        self,
        phase: str,
    ) -> None:

        if phase not in self.VALID_PHASES:
            raise ValueError(
                f"Invalid agent phase: {phase}"
            )

        self.state.phase = phase
        self.state.updated_at = time.time()

    def observe(self) -> dict[str, Any]:
        """Observe current system state."""

        self.set_phase("OBSERVE")

        snapshot = self.resources.snapshot()
        limits = self.resources.check_limits(
            snapshot
        )

        return {
            "phase": self.state.phase,
            "resources": snapshot.to_dict(),
            "resource_limits": limits,
            "generation": self.generation,
            "dry_run": self.dry_run,
        }

    def understand(
        self,
        problem: str,
        context: str = "",
    ) -> dict[str, Any]:
        """Create a structured understanding of a problem."""

        self.set_phase("UNDERSTAND")

        return {
            "problem": problem,
            "context": context,
            "generation": self.generation,
            "constraints": [
                "Protected generations remain immutable",
                "Final holdout remains unavailable",
                "Candidates require technical validation",
                "Experimental evidence overrides mentor opinion",
            ],
        }

    def search_memory(
        self,
        query: str,
    ) -> list[dict[str, Any]]:
        """Search persistent NOVA-EVO knowledge."""

        self.set_phase("MEMORY")

        # KnowledgeStore currently provides structured search.
        # Query filtering is performed locally until semantic
        # retrieval is added to the memory subsystem.
        records = self.knowledge.search()

        if not query:
            return records

        query_terms = {
            term.lower()
            for term in query.split()
            if term.strip()
        }

        if not query_terms:
            return records

        matched: list[dict[str, Any]] = []

        for record in records:
            text = json.dumps(
                record,
                ensure_ascii=False,
            ).lower()

            if any(
                term in text
                for term in query_terms
            ):
                matched.append(record)

        return matched

    def plan(
        self,
        problem: str,
        memory: list[dict[str, Any]],
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a bounded execution plan."""

        self.set_phase("PLAN")

        return {
            "objective": problem,
            "steps": [
                "inspect_memory",
                "check_resources",
                "generate_or_select_candidate",
                "validate_candidate",
                "run_experiment",
                "measure_results",
                "store_learning",
                "make_decision",
            ],
            "memory_matches": len(memory),
            "resource_allowed": observation[
                "resource_limits"
            ]["allowed"],
            "dry_run": self.dry_run,
        }

    def ask_mentor(
        self,
        problem: str,
        context: str = "",
        candidate_id: str | None = None,
        previous_failures: list[str] | None = None,
        architecture: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        self.state.mentor_requests += 1

        request = MentorRequest(
            problem=problem,
            context=context,
            candidate_id=candidate_id,
            generation=self.generation,
            current_architecture=(
                architecture or {}
            ),
            previous_failures=(
                previous_failures or []
            ),
            constraints=[
                "Protected generations are immutable",
                "Final holdout is forbidden",
                "Every proposal requires experiment verification",
                "Mentor cannot promote candidates",
            ],
        )

        if self.dry_run:
            proposal = self.mentor.parse_proposal(
                """
PROPOSAL:
Perform a controlled architecture experiment.

HYPOTHESIS:
The candidate architecture may improve validation
performance without violating the NOVA state-space core.

RATIONALE:
The hypothesis must be tested against the current
primary parent using identical evaluation conditions.

CHANGES:
- Modify only the candidate architecture.
- Preserve the parent as a reference.

RISKS:
- Increased parameter count.
- Possible slower training.
- Possible regression.

EXPERIMENTS:
- Run technical validation.
- Run controlled training.
- Compare validation loss, parameters and speed.
""".strip()
            )
        else:
            proposal = self.mentor.ask(
                request
            )

        return {
            "proposal": proposal.to_dict(),
            "hypothesis": (
                self.mentor.proposal_to_hypothesis(
                    proposal
                )
            ),
        }

    def validate_candidate(
        self,
        candidate_id: str,
        files: list[str | Path],
    ) -> dict[str, Any]:

        self.set_phase("VALIDATE")

        result = self.validation.full_validation(
            candidate_id,
            files,
        )

        return result.to_dict()
    def select_candidate(
        self,
        count: int = 3,
        problem: str = "",
    ) -> dict[str, Any] | None:
        """
        Generate and technically screen candidates.

        Returns the first implementation-supported candidate,
        or None if no candidate is executable.
        """

        self.set_phase("PLAN")

        candidates = self.evolution.generate_generation(
            count=count
        )

        technical = self.evolution.technical_screen(
            candidates
        )

        supported = self.evolution.implementation_screen(
            technical["valid"]
        )

        if not supported["supported"]:
            return None

        candidates = supported["supported"]

        recommendation = self.meta_learning.recommend_core(
            goal=problem or "general learning",
            task=problem or "model optimization",
            candidates=candidates,
        )

        selected_core = recommendation.selected_core

        candidate = next(
            (
                item
                for item in candidates
                if self.meta_learning._core_name(
                    item.get("config", {})
                ) == selected_core
            ),
            candidates[0],
        )

        candidate["meta_learning_recommendation"] = (
            recommendation.to_dict()
        )

        self.state.candidate_id = candidate[
            "candidate"
        ]

        return candidate

    def run_experiment(
        self,
        candidate: dict[str, Any],
        steps: int = 100,
    ) -> dict[str, Any]:

        self.set_phase("EXPERIMENT")

        result = self.training_runner.run_screening(
            candidate=candidate,
            steps=steps,
            seed=1001,
            batch_size=8,
            learning_rate=3e-4,
            weight_decay=0.01,
        )

        self.set_phase("MEASURE")

        return result
    def evaluate_candidate(
        self,
        candidate: dict[str, Any],
        screening_steps: int = 100,
        robust_steps: int = 1000,
        robust_seeds: list[int] | None = None,
        problem: str = "",
    ) -> dict[str, Any]:
        """
        Run real screening and robust evaluation.

        Screening is used only as a filter.
        Promotion requires a robust three-seed evaluation.
        """

        self.set_phase("EXPERIMENT")

        if robust_seeds is None:
            robust_seeds = [1001, 2002, 3003]

        strategy = self.adaptive_strategy.select(
            goal=problem or "model optimization",
            task=problem or "model optimization",
        )

        screening = self.training_runner.run_screening(
            candidate=candidate,
            steps=min(
                screening_steps,
                strategy.steps,
            ),
            seed=1001,
            batch_size=strategy.batch_size,
            learning_rate=strategy.learning_rate,
            weight_decay=strategy.weight_decay,
            stage="screening",
        )

        screening_metrics = screening.get(
            "metrics",
            {},
        )

        screening_loss = screening_metrics.get(
            "validation_loss"
        )

        if (
            screening.get("status") != "COMPLETED"
            or screening_loss is None
            or not math.isfinite(
                float(screening_loss)
            )
        ):
            self.set_phase("DECIDE")

            return {
                "candidate": candidate["candidate"],
                "screening": screening,
                "robust": None,
                "evaluation": None,
                "status": "SCREENING_FAILED",
            }

        self.set_phase("EXPERIMENT")

        robust = self.training_runner.run_robust(
            candidate=candidate,
            steps=max(
                robust_steps,
                strategy.steps,
            ),
            seeds=robust_seeds,
            batch_size=strategy.batch_size,
            learning_rate=strategy.learning_rate,
            weight_decay=strategy.weight_decay,
        )

        self.set_phase("MEASURE")

        evaluation = self.evaluator.evaluate(
            robust
        )

        self.set_phase("DECIDE")

        return {
            "candidate": candidate["candidate"],
            "screening": screening,
            "robust": robust,
            "evaluation": evaluation.to_dict(),
            "training_strategy": strategy.to_dict(),
            "status": evaluation.action,
        }
    def decide(
        self,
        validation: dict[str, Any],
        experiment_success: bool | None = None,
        repair_possible: bool = False,
    ) -> AgentDecision:
        """Make a bounded controller decision."""

        self.set_phase("DECIDE")

        if not validation.get("valid", False):
            if (
                repair_possible
                and self.state.repair_attempts < 5
            ):
                return AgentDecision(
                    action="REPAIR",
                    reason=(
                        "Technical validation failed "
                        "and repair budget remains."
                    ),
                    confidence=0.9,
                )

            return AgentDecision(
                action="REJECT",
                reason=(
                    "Candidate failed technical validation."
                ),
                confidence=0.95,
                requires_experiment=False,
            )

        if experiment_success is None:
            return AgentDecision(
                action="INVESTIGATE",
                reason=(
                    "Candidate is technically valid but "
                    "experimental evidence is missing."
                ),
                confidence=0.9,
            )

        if experiment_success:
            return AgentDecision(
                action="PROMOTE",
                reason=(
                    "Candidate passed validation and "
                    "experimental evaluation."
                ),
                confidence=0.85,
            )

        if repair_possible:
            return AgentDecision(
                action="REPAIR",
                reason=(
                    "Candidate is valid but experiment "
                    "failed; controlled repair is possible."
                ),
                confidence=0.8,
            )

        return AgentDecision(
            action="REJECT",
            reason=(
                "Candidate is valid but experimental "
                "evaluation did not support it."
            ),
            confidence=0.85,
        )

    def learn(
        self,
        decision: AgentDecision,
        candidate_id: str,
        validation: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist controller learning into the NOVA-EVO knowledge store."""

        self.set_phase("LEARN")

        timestamp = time.time()

        record = {
            "candidate_id": candidate_id,
            "generation": self.generation,
            "decision": decision.to_dict(),
            "validation": validation,
            "dry_run": self.dry_run,
            "timestamp": timestamp,
        }


        knowledge_record = self.knowledge.create(
            knowledge_type="experiment",
            title=(
                f"NOVA-EVO generation {self.generation} "
                f"candidate {candidate_id}"
            ),
            content=record,
            status=(
                "VALIDATED"
                if decision.action == "PROMOTE"
                else "EXPERIMENTAL"
            ),
            source="agent_controller",
            confidence=decision.confidence,
        )

        knowledge_path = self.knowledge.save(
            knowledge_record
        )

        record["knowledge_id"] = knowledge_record["id"]
        record["knowledge_path"] = str(
            knowledge_path
        )

        return record
        return record

    def run_cycle(
        self,
        problem: str,
        candidate_id: str | None = None,
        context: str = "",
        candidate_count: int = 3,
        screening_steps: int = 100,
        robust_steps: int = 1000,
        robust_seeds: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Execute one autonomous NOVA-EVO cycle.

        Dry-run mode keeps the original bounded simulation.
        Real mode executes candidate generation, screening,
        robust evaluation and evaluator-based decision making.
        """

        self.state.cycle_count += 1

        observation = self.observe()

        understanding = self.understand(
            problem,
            context,
        )

        memory = self.search_memory(
            problem
        )

        plan = self.plan(
            problem,
            memory,
            observation,
        )

        if candidate_id is None:
            candidate_id = (
                self.state.candidate_id
                or "AUTO"
            )

        mentor = self.ask_mentor(
            problem=problem,
            context=context,
            candidate_id=candidate_id,
        )

        if self.dry_run:
            validation = {
                "candidate_id": candidate_id,
                "valid": True,
                "syntax_ok": True,
                "imports_ok": True,
                "build_ok": True,
                "forward_ok": True,
                "backward_ok": True,
                "gradient_ok": True,
                "parameter_count": None,
                "errors": [],
                "warnings": [
                    "DRY-RUN: no real candidate was executed"
                ],
                "checks": {
                    "dry_run": True
                },
            }

            decision = self.decide(
                validation=validation,
                experiment_success=None,
                repair_possible=True,
            )

            learning = self.learn(
                decision=decision,
                candidate_id=candidate_id,
                validation=validation,
            )

            self.state.last_action = (
                decision.action
            )

            self.save_state()

            return {
                "run_id": self.state.run_id,
                "phase": self.state.phase,
                "mode": "dry_run",
                "observation": observation,
                "understanding": understanding,
                "memory_matches": len(memory),
                "plan": plan,
                "mentor": mentor,
                "validation": validation,
                "decision": decision.to_dict(),
                "learning": learning,
                "state": self.state.to_dict(),
            }

        self.set_phase("PLAN")

        candidate = self.select_candidate(
            count=candidate_count,
            problem=problem,
        )

        if candidate is None:
            decision = AgentDecision(
                action="REJECT",
                reason=(
                    "No implementation-supported "
                    "candidate was generated."
                ),
                confidence=0.95,
            )

            learning = self.learn(
                decision=decision,
                candidate_id=(
                    self.state.candidate_id
                    or candidate_id
                ),
                validation={
                    "valid": False,
                    "errors": [
                        "No supported candidate"
                    ],
                },
            )

            self.state.last_action = (
                decision.action
            )

            self.save_state()

            return {
                "run_id": self.state.run_id,
                "phase": self.state.phase,
                "mode": "autonomous",
                "observation": observation,
                "understanding": understanding,
                "memory_matches": len(memory),
                "plan": plan,
                "mentor": mentor,
                "candidate": None,
                "decision": decision.to_dict(),
                "learning": learning,
                "state": self.state.to_dict(),
            }

        self.state.candidate_id = (
            candidate["candidate"]
        )

        evaluation = self.evaluate_candidate(
            candidate=candidate,
            screening_steps=screening_steps,
            robust_steps=robust_steps,
            robust_seeds=robust_seeds,
            problem=problem,
        )

        self.state.last_action = (
            evaluation["status"]
        )

        validation = candidate.get(
            "technical_validation",
            {},
        )

        learning_decision = AgentDecision(
            action=evaluation["status"],
            reason=(
                evaluation["evaluation"]
                or {}
            ).get(
                "reason",
                "Evaluation completed.",
            ),
            confidence=(
                evaluation["evaluation"]
                or {}
            ).get(
                "confidence",
                0.0,
            ),
            requires_experiment=False,
            metadata={
                "candidate": candidate["candidate"],
                "parent": candidate.get("parent"),
                "evaluation": evaluation[
                    "evaluation"
                ],
            },
        )

        meta_experience = self.meta_learning.record_experience(
            goal=problem,
            task=problem,
            core_config=candidate["config"],
            success=(learning_decision.action == "PROMOTE"),
            metrics=(
                evaluation.get("evaluation")
                or {}
            ),
            strategy=candidate.get(
                "mutation",
                "evolution",
            ),
            source="agent_controller",
        )

        learning = self.learn(
            decision=learning_decision,
            candidate_id=candidate["candidate"],
            validation=validation,
        )

        self.save_state()

        return {
            "run_id": self.state.run_id,
            "phase": self.state.phase,
            "mode": "autonomous",
            "observation": observation,
            "understanding": understanding,
            "memory_matches": len(memory),
            "plan": plan,
            "mentor": mentor,
            "candidate": candidate,
            "evaluation": evaluation,
            "meta_learning": meta_experience,
            "decision": learning_decision.to_dict(),
            "learning": learning,
            "state": self.state.to_dict(),
        }

    def save_state(self) -> None:
        self.state.updated_at = time.time()

        self.state_file.write_text(
            json.dumps(
                self.state.to_dict(),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load_state(self) -> dict[str, Any] | None:
        if not self.state_file.exists():
            return None

        return json.loads(
            self.state_file.read_text(
                encoding="utf-8"
            )
        )

    def stop(self) -> None:
        self.set_phase("STOPPED")
        self.state.last_action = "STOP"
        self.save_state()


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        (root / "evo/knowledge").mkdir(
            parents=True,
            exist_ok=True,
        )

        (root / "evo/experiments").mkdir(
            parents=True,
            exist_ok=True,
        )

        (root / "evo/engine").mkdir(
            parents=True,
            exist_ok=True,
        )

        policy_path = (
            root
            / "evo/engine/agent_policy.json"
        )

        policy_path.write_text(
            json.dumps(
                {
                    "limits": {
                        "max_candidate_runtime_seconds": 300,
                        "max_cpu_threads": 16,
                    }
                }
            ),
            encoding="utf-8",
        )

        controller = AgentController(
            project_root=root,
            generation=4,
            dry_run=True,
        )

        result = controller.run_cycle(
            problem=(
                "Determine whether a new NOVA "
                "architecture should be investigated."
            ),
            candidate_id="SELFTEST-001",
            context="Controller dry-run",
        )

        assert result["run_id"]
        assert result["decision"]["action"] == (
            "INVESTIGATE"
        )

        assert (
            result["mentor"]["hypothesis"]["status"]
            == "UNTESTED"
        )

        assert result["validation"]["valid"]
        assert result["validation"]["checks"][
            "dry_run"
        ]

        state = controller.load_state()

        assert state is not None
        assert state["cycle_count"] == 1
        assert state["candidate_id"] == (
            "SELFTEST-001"
        )

    print(
        "AGENT CONTROLLER SELFTEST: PASSED"
    )


if __name__ == "__main__":
    self_test()
