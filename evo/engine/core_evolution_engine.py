from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any

import torch

from evo.engine.authority_manager import AuthorityManager
from evo.engine.sandbox import create_workspace, write_file
from evo.engine.validation_engine import ValidationEngine
from nova.efficiency import efficiency_check


ROOT = Path("/opt/ai/work/nova-evo").resolve()

STATE_FILE = ROOT / "evo" / "engine" / "evo_state.json"
CORE_STATE_FILE = ROOT / "evo" / "core_evolution" / "autonomous_state.json"
CANDIDATE_DIR = ROOT / "evo" / "core_evolution" / "candidates"
RESULT_DIR = ROOT / "evo" / "core_evolution" / "results"
PASSED_DIR = ROOT / "evo" / "core_evolution" / "passed"

BASELINE_CORE = ROOT / "nova" / "blocks_scan.py"

TEACHER_URL = "http://127.0.0.1:8081/v1/chat/completions"
TEACHER_MODEL = "OxCoder-9B"

ROBUST_SEEDS = [1001, 2002, 3003]

# How many times an infrastructure-failed candidate is re-trained
# before recovery gives up on it.
MAX_RECOVERY_ATTEMPTS = 3

# First core-evolution phase only permits replacement of the recurrent block.
# model_scan.py remains unchanged until a later integration phase.
ALLOWED_CORE_TARGETS = {
    "nova/blocks_scan.py",
}

FORBIDDEN_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "pathlib",
    "shutil",
    "urllib",
    "requests",
    "httpx",
    "ftplib",
    "pickle",
    "marshal",
    "ctypes",
}

FORBIDDEN_CALLS = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "system",
    "popen",
}

FORBIDDEN_NAMES = {
    "__builtins__",
    "__globals__",
    "__loader__",
    "__spec__",
}


class CoreEvolutionError(RuntimeError):
    pass


REQUIRED_CONFIG_KEYS = {
    "vocab_size",
    "d_model",
    "d_state",
    "num_layers",
    "conv_kernel",
}


INFRA_FRAMES = (
    "evo/engine/training_runner.py",
    "evo/engine/architecture_factory.py",
    "train_candidate.py",
)


def is_infrastructure_failure(stderr: str) -> bool:
    """
    True when a training traceback ends in project infrastructure
    rather than in the generated core (nova/blocks_scan.py).

    Such failures say nothing about candidate quality and must not
    be recorded as REJECT or used as a learning lesson.
    """
    frames = re.findall(r'File "([^"]+)"', stderr or "")
    if not frames:
        return False
    if any(f.endswith("blocks_scan.py") for f in frames):
        return False
    last = frames[-1].replace("\\", "/")
    return any(last.endswith(tail) for tail in INFRA_FRAMES)


def _generation_from_id(value: Any) -> int | None:
    match = re.match(r"^GEN(\d+)", str(value or ""))
    return int(match.group(1)) if match else None


def resolve_training_metadata(
    candidate: dict[str, Any],
    project_state: dict[str, Any],
    result_dir: Path,
) -> dict[str, Any]:
    """
    Guarantee the fields TrainingRunner.run_screening() requires:
    candidate, generation, parent, config.

    Core candidates only replace nova/blocks_scan.py, so they are trained
    with the architecture config of their parent (the active core).

    Raises CoreEvolutionError when metadata cannot be resolved, so an
    infrastructure problem is never recorded as a bad candidate.
    """
    candidate.setdefault("candidate", candidate.get("candidate_id"))
    candidate.setdefault("parent", project_state.get("primary_parent"))

    if candidate.get("generation") is None:
        candidate["generation"] = (
            _generation_from_id(candidate.get("candidate_id"))
            or project_state.get("current_generation")
        )

    config = candidate.get("config")

    if not isinstance(config, dict):
        config = None
        parent = candidate.get("parent")

        for name in (f"{parent}.official.json", f"{parent}.json"):
            path = result_dir / name
            if parent and path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                found = (data.get("candidate") or {}).get("config")
                if isinstance(found, dict):
                    config = dict(found)
                    break

        if config is None and isinstance(
            project_state.get("primary_parent_config"), dict
        ):
            config = dict(project_state["primary_parent_config"])

    missing = []
    if not candidate.get("candidate"):
        missing.append("candidate")
    if candidate.get("generation") is None:
        missing.append("generation")
    if not isinstance(config, dict):
        missing.append("config")
    else:
        lacking = REQUIRED_CONFIG_KEYS - set(config)
        if lacking:
            missing.append(f"config keys {sorted(lacking)}")

    if missing:
        raise CoreEvolutionError(
            "Cannot build training metadata for "
            f"{candidate.get('candidate_id')}: missing {missing}"
        )

    candidate["generation"] = int(candidate["generation"])
    candidate["config"] = config
    return candidate


class CoreEvolutionEngine:
    """
    Autonomous NOVA core-evolution layer.

    Flow:

        observe
          -> diagnose
          -> hypothesize
          -> teacher proposal
          -> source generation
          -> sandbox
          -> static validation
          -> runtime smoke test
          -> screening / robust training
          -> result
          -> lesson
          -> next candidate

    Important:
        - GEN3 remains immutable.
        - Generated code never writes directly into nova/.
        - Teacher is advisory only.
        - Promotion of a core is deliberately separated from generation.
    """

    def __init__(
        self,
        project_root: str | Path = ROOT,
        actor_id: str = "NOVA-EVO",
    ) -> None:
        self.root = Path(project_root).resolve()

        self.authority = AuthorityManager(actor_id=actor_id)
        self.validation = ValidationEngine(
            project_root=self.root,
        )

        CANDIDATE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )
        RESULT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )
        PASSED_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.state = self.load_core_state()

        self.mentors = [TEACHER_MODEL]
        self.teacher_model = TEACHER_MODEL
        self.teacher_timeout_factor = 1.0
        self._registry = None
        try:
            from evo.learning.teacher_registry import TeacherRegistry

            self._registry = TeacherRegistry.load()
            self.mentors = self._registry.mentors() or [TEACHER_MODEL]
        except Exception:
            pass
        self.use_mentor(self.mentors[0])

    # ------------------------------------------------------------------
    # STATE
    # ------------------------------------------------------------------

    def load_project_state(self) -> dict[str, Any]:
        if not STATE_FILE.exists():
            raise CoreEvolutionError(
                f"Missing EVO state: {STATE_FILE}"
            )

        return json.loads(
            STATE_FILE.read_text(encoding="utf-8")
        )

    def load_core_state(self) -> dict[str, Any]:
        if not CORE_STATE_FILE.exists():
            state = {
                "engine": "core_evolution",
                "version": "0.1.0",
                "mode": "autonomous_core_evolution",
                "attempt": 0,
                "candidates": [],
                "lessons": [],
                "passed_candidates": [],
                "active_core": None,
            }
            CORE_STATE_FILE.write_text(
                json.dumps(
                    state,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return state

        return json.loads(
            CORE_STATE_FILE.read_text(
                encoding="utf-8"
            )
        )

    def save_core_state(self) -> None:
        CORE_STATE_FILE.write_text(
            json.dumps(
                self.state,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # AUTHORITY
    # ------------------------------------------------------------------

    def assert_allowed(self) -> None:
        self.authority.assert_runtime_action(
            "self_evolution"
        )
        self.authority.assert_runtime_action(
            "self_modification"
        )

    def assert_target_allowed(
        self,
        relative_path: str,
    ) -> None:
        if relative_path not in ALLOWED_CORE_TARGETS:
            raise CoreEvolutionError(
                f"Core target not allowed: {relative_path}"
            )

        target = (
            self.root / relative_path
        ).resolve()

        if not self.authority.may_modify_path(target):
            # We don't modify this target directly yet, but it must
            # still satisfy the authority policy before it becomes
            # an eventual promotion target.
            raise CoreEvolutionError(
                f"Authority denied core target: {target}"
            )

    # ------------------------------------------------------------------
    # OBSERVATION / DIAGNOSIS
    # ------------------------------------------------------------------

    def previous_results(
        self,
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        """
        Build canonical evolutionary memory.

        Multiple physical files may describe the same experiment.
        Only the highest-quality record for each candidate is exposed
        to the autonomous reasoning loop.

        Priority:
            official promotion
            CORE result
            sandbox robust result
            classic generation result
        """

        roots = [
            (
                self.root / "evo" / "core_evolution" / "results",
                "core_evolution",
                100,
            ),
            (
                self.root / "evo" / "engine" / "sandbox",
                "sandbox_experiment",
                70,
            ),
            (
                self.root / "evo" / "gen4" / "results",
                "generation",
                60,
            ),
        ]

        best_by_candidate: dict[str, dict[str, Any]] = {}

        for root, source_type, priority in roots:

            if not root.exists():
                continue

            for path in root.glob("**/*.json"):

                name = path.name

                # Skip manifests and administrative state files.
                if name in {
                    "generation_manifest.json",
                    "autonomous_generation_state.json",
                    "autonomous_state.json",
                }:
                    continue

                try:
                    data = json.loads(
                        path.read_text(
                            encoding="utf-8"
                        )
                    )
                except Exception:
                    continue

                if not isinstance(
                    data,
                    dict,
                ):
                    continue

                candidate = (
                    data.get("candidate")
                    or data.get("candidate_id")
                )

                candidate_metadata = {}

                if isinstance(
                    candidate,
                    dict,
                ):
                    candidate_metadata = candidate
                    candidate = (
                        candidate.get("candidate")
                        or candidate.get("candidate_id")
                    )

                if not candidate:
                    continue

                robust = data.get(
                    "robust",
                    {},
                )

                if not isinstance(
                    robust,
                    dict,
                ):
                    robust = {}

                metrics = data.get(
                    "metrics",
                    {},
                )

                if not isinstance(
                    metrics,
                    dict,
                ):
                    metrics = {}

                robust_metrics = robust.get(
                    "metrics",
                    {},
                )

                if not isinstance(
                    robust_metrics,
                    dict,
                ):
                    robust_metrics = {}

                merged_metrics = dict(
                    robust_metrics
                )

                merged_metrics.update(
                    {
                        key: value
                        for key, value in metrics.items()
                        if value is not None
                    }
                )

                training = data.get(
                    "training",
                    {},
                )

                if not isinstance(
                    training,
                    dict,
                ):
                    training = {}

                if not training:
                    training = robust.get(
                        "training",
                        {},
                    )

                if not isinstance(
                    training,
                    dict,
                ):
                    training = {}

                evaluation = data.get(
                    "evaluation"
                )

                if evaluation is None:
                    evaluation = robust.get(
                        "evaluation"
                    )

                status = data.get(
                    "status"
                )

                stage = (
                    data.get("stage")
                    or robust.get("stage")
                    or training.get("stage")
                )

                # Promotion records are stronger evidence than
                # the underlying sandbox record.
                if name.endswith(
                    ".official.json"
                ):
                    record_priority = priority + 30
                elif source_type == "core_evolution":
                    record_priority = priority + 10
                else:
                    record_priority = priority

                teacher_data = (
                    data.get("teacher")
                    or candidate_metadata.get("teacher")
                    or {}
                )

                if not isinstance(
                    teacher_data,
                    dict,
                ):
                    teacher_data = {}

                plan = (
                    teacher_data.get("plan")
                    or ""
                )

                hypothesis = ""

                if plan:
                    for section in (
                        "HYPOTHESIS:",
                        "MECHANISM:",
                        "RATIONALE:",
                    ):
                        if section in plan:
                            fragment = plan.split(
                                section,
                                1,
                            )[1]

                            next_sections = [
                                x
                                for x in (
                                    "MECHANISM:",
                                    "RATIONALE:",
                                    "RISKS:",
                                    "IMPLEMENTATION_PLAN:",
                                )
                                if x in fragment
                            ]

                            if next_sections:
                                fragment = fragment.split(
                                    next_sections[0],
                                    1,
                                )[0]

                            if section == "HYPOTHESIS:":
                                hypothesis = fragment.strip()
                                break

                row = {
                    "file": str(path),
                    "source": source_type,
                    "priority": record_priority,
                    "candidate": str(candidate),
                    "status": status,
                    "stage": stage,
                    "evaluation": evaluation,
                    "metrics": {
                        "validation_loss_mean":
                            merged_metrics.get(
                                "validation_loss_mean"
                            ),
                        "validation_loss_std":
                            merged_metrics.get(
                                "validation_loss_std"
                            ),
                        "validation_loss_min":
                            merged_metrics.get(
                                "validation_loss_min"
                            ),
                        "validation_loss_max":
                            merged_metrics.get(
                                "validation_loss_max"
                            ),
                        "parameters":
                            merged_metrics.get(
                                "parameters"
                            ),
                        "steps":
                            merged_metrics.get(
                                "steps"
                            ),
                    },
                    "hypothesis": hypothesis,
                    "parent": (
                        data.get("parent_id")
                        or data.get("parent")
                        or candidate_metadata.get("parent")
                    ),
                }

                existing = best_by_candidate.get(
                    str(candidate)
                )

                if (
                    existing is None
                    or record_priority > existing["priority"]
                ):
                    best_by_candidate[
                        str(candidate)
                    ] = row

        rows = list(
            best_by_candidate.values()
        )

        rows.sort(
            key=lambda row: (
                Path(
                    row["file"]
                ).stat().st_mtime
                if Path(
                    row["file"]
                ).exists()
                else 0
            )
        )

        return rows[-limit:]

    def diagnose(self) -> dict[str, Any]:
        project = self.load_project_state()

        # The live source is always the currently active core.
        source = BASELINE_CORE.read_text(
            encoding="utf-8"
        )

        previous = self.previous_results()

        rejected = []

        for item in previous:
            evaluation = item.get(
                "evaluation"
            )

            action = None

            if isinstance(
                evaluation,
                dict,
            ):
                action = (
                    evaluation.get("action")
                    or evaluation.get("decision")
                )
            elif isinstance(
                evaluation,
                str,
            ):
                action = evaluation

            if (
                action == "REJECT"
                or item.get("status") == "REJECT"
            ):
                rejected.append(item)

        losses: list[float] = []

        for row in rejected:
            metrics = row.get("metrics") or {}

            value = metrics.get(
                "validation_loss_mean"
            )

            if isinstance(
                value,
                (int, float),
            ):
                losses.append(float(value))

        best_known = project.get(
            "best_known",
            {},
        )

        active_core_state = {}

        if CORE_STATE_FILE.exists():
            try:
                active_core_state = json.loads(
                    CORE_STATE_FILE.read_text(
                        encoding="utf-8"
                    )
                )
            except Exception:
                active_core_state = {}

        active_core = (
            active_core_state.get(
                "active_core"
            )
            or best_known.get(
                "candidate"
            )
            or project.get(
                "primary_parent"
            )
        )

        active_source = (
            active_core_state.get(
                "active_source"
            )
        )

        if active_source:
            active_source_path = Path(
                active_source
            ).resolve()

            if active_source_path.exists():
                source = active_source_path.read_text(
                    encoding="utf-8"
                )

        source_hash = hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest()

        baseline = best_known.get(
            "validation_loss"
        )

        diagnosis_text = (
            f"The current active parent is {active_core}. "
            f"Its robust validation baseline is "
            f"{baseline}. "
            f"Previous core experiments must be used as evidence "
            f"for the next hypothesis. "
            f"The next candidate must improve the active core's "
            f"algorithmic mechanism rather than merely changing "
            f"standard hyperparameters. "
            f"The active core must preserve causality, "
            f"recurrent state semantics, finite forward/backward "
            f"behaviour and the public NovaScanBlock interface."
        )

        return {
            "current_generation": project.get(
                "current_generation"
            ),
            "parent_generation": project.get(
                "parent_generation"
            ),
            "primary_parent": project.get(
                "primary_parent"
            ),
            "active_core": active_core,
            "baseline_validation_loss": baseline,
            "previous_rejected_count": len(
                rejected
            ),
            "previous_rejected_losses": losses,
            "previous_results": previous,
            "baseline_core": str(BASELINE_CORE),
            "baseline_core_sha256": source_hash,
            "baseline_core_bytes": len(
                source.encode("utf-8")
            ),
            "diagnosis": diagnosis_text,
        }

    # ------------------------------------------------------------------
    # TEACHER
    # ------------------------------------------------------------------

    def teacher_prompt(
        self,
        diagnosis: dict[str, Any],
        previous_lessons: list[dict[str, Any]],
    ) -> str:
        source = BASELINE_CORE.read_text(
            encoding="utf-8"
        )

        lessons_text = json.dumps(
            previous_lessons[-8:],
            ensure_ascii=False,
            indent=2,
        )

        return f"""
You are the advisory architecture researcher for NOVA-EVO.

Your job is to invent ONE concrete improvement to the NOVA recurrent
core. You are NOT allowed to decide promotion and you are NOT allowed
to modify files yourself.

TARGET FILE:
nova/blocks_scan.py

IMMUTABLE PUBLIC API:

class NovaScanBlock(nn.Module):
    __init__(
        self,
        d_model,
        d_state,
        conv_kernel=5,
        forget_bias=1.5,
    )

    forward(self, x, state=None)
        -> y, final_state

MANDATORY INVARIANTS:
1. Causal behaviour: token t must not depend on future tokens.
2. State must remain recurrent and differentiable.
3. Forward output must remain finite.
4. Backward must produce finite gradients.
5. Existing constructor and forward signature must remain compatible.
6. Do not use filesystem, subprocesses, networking or reflection.
7. Use only PyTorch operations.
8. Keep the implementation practical on an RTX 4060 8GB.
9. The experiment must be meaningfully different from simple
   d_model/layer/kernel/forget_bias mutation.
10. Prefer an algorithmic improvement in state update, state gating,
    memory compression, retention, recurrence, local/global fusion,
    normalization, or another core mechanism.

CURRENT DIAGNOSIS:
{json.dumps(diagnosis, ensure_ascii=False, indent=2)}

PREVIOUS LESSONS:
{lessons_text}

EVOLUTION EXPERIMENT EVIDENCE:
{json.dumps(
    diagnosis.get("previous_results", []),
    ensure_ascii=False,
    indent=2,
)}

IMPORTANT EVOLUTION RULE:
Do not blindly repeat a mechanism merely because it succeeded once.
Treat previous successes and failures as evidence.
The next mechanism should be complementary, independently testable,
and justified by the active parent's actual behaviour.

CURRENT BASELINE SOURCE:
----- BEGIN BASELINE SOURCE -----
{source[:30000]}
----- END BASELINE SOURCE -----

Return EXACTLY these sections:

HYPOTHESIS:
one precise hypothesis

RATIONALE:
why the mechanism may improve learning or efficiency

RISKS:
main technical failure modes

EXPERIMENTS:
what should be measured

CORE_SOURCE_BEGIN
<complete replacement Python source for nova/blocks_scan.py>
CORE_SOURCE_END

The generated source MUST define NovaScanBlock.
Do not use markdown fences around the source.
"""

    def _teacher_request(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        timeout: int = 180,
    ) -> dict[str, Any]:

        timeout = int(timeout * self.teacher_timeout_factor)

        payload = {
            "model": self.teacher_model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": 0.1,
            "stream": False,
            "max_tokens": max_tokens,
            "reasoning_format": "none",
            "reasoning_effort": "none",
        }

        request = urllib.request.Request(
            TEACHER_URL,
            data=json.dumps(
                payload,
                ensure_ascii=False,
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            raw = response.read().decode(
                "utf-8"
            )

        data = json.loads(raw)

        choice = (
            data.get("choices", [{}])[0]
        )

        message = choice.get(
            "message",
            {},
        )

        content = message.get(
            "content",
            "",
        )

        reasoning = message.get(
            "reasoning_content",
            "",
        )

        if not content and reasoning:
            content = reasoning

        if not content:
            raise CoreEvolutionError(
                "Teacher returned empty response"
            )

        # Defensive removal of reasoning blocks.
        # CORE evolution only consumes explicit output.
        if "<think>" in content:
            while "<think>" in content and "</think>" in content:
                before, rest = content.split(
                    "<think>",
                    1,
                )
                _, after = rest.split(
                    "</think>",
                    1,
                )
                content = (
                    before + after
                ).strip()

        return {
            "content": content.strip(),
            "raw": raw,
            "usage": data.get(
                "usage",
                {},
            ),
            "finish_reason": choice.get(
                "finish_reason"
            ),
        }

    def _load_learning_memory(
        self,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """Load durable autonomous experience from disk."""

        path = (
            self.root
            / "evo"
            / "core_evolution"
            / "learning"
            / "learning_memory.json"
        )

        if not path.exists():
            return []

        try:
            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            return []

        lessons = data.get(
            "lessons",
            [],
        )

        if not isinstance(
            lessons,
            list,
        ):
            return []

        compact = []

        for lesson in lessons[-limit:]:

            if not isinstance(
                lesson,
                dict,
            ):
                continue

            compact.append(
                {
                    "candidate":
                        lesson.get("candidate"),

                    "generation":
                        lesson.get("generation"),

                    "parent":
                        lesson.get("parent"),

                    "decision":
                        lesson.get("decision"),

                    "stage":
                        lesson.get("stage"),

                    "validation_loss":
                        lesson.get(
                            "validation_loss"
                        ),

                    "parent_loss":
                        lesson.get(
                            "parent_loss"
                        ),

                    "delta":
                        lesson.get("delta"),

                    "mechanism":
                        str(
                            lesson.get(
                                "mechanism"
                            )
                            or ""
                        )[:800],

                    "lesson":
                        str(
                            lesson.get(
                                "lesson"
                            )
                            or ""
                        )[:1000],

                    "next_direction":
                        str(
                            lesson.get(
                                "next_direction"
                            )
                            or ""
                        )[:700],
                }
            )

        return compact

    def ask_teacher(
        self,
        diagnosis: dict[str, Any],
    ) -> dict[str, Any]:

        baseline = BASELINE_CORE.read_text(
            encoding="utf-8"
        )

        # --------------------------------------------------------------
        # PHASE 1 — ARCHITECTURE HYPOTHESIS
        # --------------------------------------------------------------

        plan_system = """
You are the advisory architecture researcher for NOVA-EVO.

You do NOT control NOVA-EVO.
You do NOT promote candidates.
You do NOT modify files.
You must not invent benchmark results.

Your task is to design ONE genuinely algorithmic improvement
to the recurrent state mechanism.

Return ONLY these sections:

HYPOTHESIS:
one precise testable hypothesis

MECHANISM:
exactly what should change in the recurrent state algorithm

RATIONALE:
why this may improve learning, memory or efficiency

RISKS:
specific failure modes

IMPLEMENTATION_PLAN:
5 to 10 precise implementation steps

Do not output <think>.
Do not output markdown fences.
""".strip()

        # Keep teacher context compact and evidence-focused.
        # The previous implementation serialized full nested experiment
        # records and could exceed the OxCoder context budget.

        compact_results = []

        for item in diagnosis.get(
            "previous_results",
            [],
        )[-10:]:

            if not isinstance(
                item,
                dict,
            ):
                continue

            metrics = item.get(
                "metrics",
                {},
            )

            if not isinstance(
                metrics,
                dict,
            ):
                metrics = {}

            evaluation = item.get(
                "evaluation",
                {},
            )

            if not isinstance(
                evaluation,
                dict,
            ):
                evaluation = {}

            compact_results.append(
                {
                    "candidate":
                        item.get(
                            "candidate"
                        ),

                    "source":
                        item.get(
                            "source"
                        ),

                    "stage":
                        item.get(
                            "stage"
                        ),

                    "status":
                        item.get(
                            "status"
                        ),

                    "decision":
                        (
                            evaluation.get(
                                "decision"
                            )
                            or evaluation.get(
                                "action"
                            )
                        ),

                    "validation_loss":
                        metrics.get(
                            "validation_loss_mean"
                        ),

                    "parent_loss":
                        evaluation.get(
                            "parent_loss"
                        ),

                    "delta":
                        evaluation.get(
                            "delta"
                        ),

                    "hypothesis":
                        str(
                            item.get(
                                "hypothesis"
                            )
                            or ""
                        )[:600],

                    "parent":
                        item.get(
                            "parent"
                        ),
                }
            )

        compact_lessons = []

        for item in diagnosis.get(
            "lessons",
            [],
        )[-8:]:

            if not isinstance(
                item,
                dict,
            ):
                continue

            compact_lessons.append(
                {
                    "candidate":
                        item.get(
                            "candidate"
                        ),

                    "decision":
                        item.get(
                            "decision"
                        ),

                    "stage":
                        item.get(
                            "stage"
                        ),

                    "validation_loss":
                        item.get(
                            "validation_loss"
                        ),

                    "mechanism":
                        str(
                            item.get(
                                "mechanism"
                            )
                            or ""
                        )[:500],

                    "lesson":
                        str(
                            item.get(
                                "lesson"
                            )
                            or ""
                        )[:600],

                    "next_direction":
                        str(
                            item.get(
                                "next_direction"
                            )
                            or ""
                        )[:500],
                }
            )

        teacher_diagnosis = {
            "current_generation":
                diagnosis.get(
                    "current_generation"
                ),

            "parent_generation":
                diagnosis.get(
                    "parent_generation"
                ),

            "primary_parent":
                diagnosis.get(
                    "primary_parent"
                ),

            "active_core":
                diagnosis.get(
                    "active_core"
                ),

            "baseline_validation_loss":
                diagnosis.get(
                    "baseline_validation_loss"
                ),

            "previous_rejected_count":
                diagnosis.get(
                    "previous_rejected_count"
                ),

            "previous_results":
                compact_results,

            "learning_memory":
                compact_lessons,

            "repair_feedback":
                diagnosis.get(
                    "repair_feedback"
                ),
        }

        persistent_learning_memory = (
            self._load_learning_memory(
                limit=8
            )
        )

        diagnosis = dict(
            diagnosis
        )

        diagnosis[
            "persistent_learning_memory"
        ] = persistent_learning_memory

        plan_user = f"""
CURRENT DIAGNOSIS:

{json.dumps(
    teacher_diagnosis,
    ensure_ascii=False,
    indent=2,
)}

CURRENT CORE:

{baseline[:12000]}

MANDATORY INVARIANTS:

- causal sequence behaviour
- recurrent differentiable state
- finite forward values
- finite backward gradients
- constructor compatibility
- forward signature compatibility
- PyTorch implementation
- practical on RTX 4060 8GB
- no filesystem access
- no networking
- no subprocesses
- no eval/exec/open/import tricks

The change must NOT be merely:
- d_model
- d_state
- num_layers
- conv_kernel
- forget_bias

CRITICAL AUTONOMOUS LEARNING RULES:

- Persistent learning memory is experimental history, not decoration.
- Do not repeat a failed mechanism blindly.
- Distinguish algorithmic failure from infrastructure failure.
- A candidate with SMOKE=PASS and TRAINING_FAILED due to metadata,
  runner or infrastructure is a recoverable experiment, not an
  algorithmic rejection.
- Preserve every functioning mechanism from the active parent.
- Every new learnable parameter must be defined as nn.Parameter and
  participate in the executable recurrence.
- Prefer one focused algorithmic change at a time.
- Do not silently remove inherited mechanisms.
- The implementation must match the plan exactly.
- Do not invent benchmark results.

Before proposing a new mechanism, inspect the persistent history and
determine whether an unfinished experiment should be recovered first.

We want a new computational mechanism in the core.
""".strip()

        plan_result = self._teacher_request(
            plan_system,
            plan_user,
            max_tokens=1400,
            timeout=180,
        )

        plan = plan_result["content"]

        # --------------------------------------------------------------
        # PHASE 2 — CORE SOURCE GENERATION
        # --------------------------------------------------------------

        source_system = """
You are the implementation engineer for NOVA-EVO.

Generate ONE complete replacement implementation for:

nova/blocks_scan.py

You must implement:

class NovaScanBlock(nn.Module):

with compatible constructor:

__init__(
    self,
    d_model,
    d_state,
    conv_kernel=5,
    forget_bias=1.5,
)

and compatible:

forward(self, x, state=None)

returning:

y, final_state

MANDATORY:

- PyTorch only
- causal computation
- recurrent differentiable state
- finite outputs
- finite gradients
- preserve residual/output dimensions
- compatible with existing nova.model_scan.NovaModel
- no filesystem access
- no networking
- no subprocesses
- no eval
- no exec
- no dynamic imports
- no reflection tricks
- do not modify other files
- do not write explanatory text outside the markers

The implementation must contain a genuinely changed
algorithmic state mechanism, not merely changed hyperparameters.

Output EXACTLY:

CORE_SOURCE_BEGIN
<complete Python source>
CORE_SOURCE_END

Do not output <think>.
Do not use markdown fences.
""".strip()

        required_parameters = ", ".join(
            sorted(set(re.findall(
                r"self\.(\w+)\s*(?::[^=\n]+)?=\s*nn\.Parameter", baseline
            )))
        ) or "(none)"

        source_user = f"""
ARCHITECTURE PLAN:

{plan}

BASELINE IMPLEMENTATION:

{baseline}

REQUIRED PARENT PARAMETERS (hard contract):
{required_parameters}
Every one of these must stay a `self.<name> = nn.Parameter(...)` with the
same name and must be used in forward(). Removing or renaming one is
only allowed if the ARCHITECTURE PLAN says explicitly "replace <name>".

Implement the proposed mechanism now.

Preserve every required public interface.
The resulting file must be self-contained except for
PyTorch imports.
""".strip()

        source_result = self._teacher_request(
            source_system,
            source_user,
            max_tokens=6000,
            timeout=360,
        )

        source_response = source_result["content"]

        combined = (
            "ARCHITECTURE_PLAN_BEGIN\n"
            + plan
            + "\nARCHITECTURE_PLAN_END\n\n"
            + source_response
        )

        return {
            "model": self.teacher_model,
            "plan": plan,
            "response": combined,
            "source_raw": source_response,
            "plan_usage": plan_result.get(
                "usage",
                {},
            ),
            "source_usage": source_result.get(
                "usage",
                {},
            ),
            "plan_finish_reason": plan_result.get(
                "finish_reason"
            ),
            "source_finish_reason": source_result.get(
                "finish_reason"
            ),
        }

    # ------------------------------------------------------------------
    # SOURCE EXTRACTION / SAFETY
    # ------------------------------------------------------------------

    @staticmethod
    def extract_source(
        text: str,
    ) -> str:

        if (
            "CORE_SOURCE_BEGIN" in text
            and "CORE_SOURCE_END" in text
        ):
            source = text.split(
                "CORE_SOURCE_BEGIN",
                1,
            )[1].split(
                "CORE_SOURCE_END",
                1,
            )[0]
            return source.strip()

        if "```python" in text:
            source = text.split(
                "```python",
                1,
            )[1].split(
                "```",
                1,
            )[0]
            return source.strip()

        raise CoreEvolutionError(
            "Teacher response did not contain a core source"
        )

    def source_contract_check(
        self,
        source: str,
        plan: str,
    ) -> tuple[bool, list[str]]:
        """
        Validate evolutionary continuity between active parent and child.

        The child may evolve the parent, but must not silently delete
        functioning learnable mechanisms. New mechanisms described by the
        teacher must become executable code before runtime/training.
        """

        import ast
        import re

        errors: list[str] = []

        try:
            parent_source = BASELINE_CORE.read_text(
                encoding="utf-8"
            )
        except Exception as exc:
            return (
                False,
                [
                    "Unable to read active parent core: "
                    f"{type(exc).__name__}: {exc}"
                ],
            )

        def parameter_names(text: str) -> set[str]:
            tree = ast.parse(text)
            names: set[str] = set()

            for node in ast.walk(tree):

                if isinstance(node, ast.Assign):

                    value = node.value

                    if not (
                        isinstance(value, ast.Call)
                        and isinstance(
                            value.func,
                            ast.Attribute,
                        )
                        and value.func.attr == "Parameter"
                        and isinstance(
                            value.func.value,
                            ast.Name,
                        )
                        and value.func.value.id == "nn"
                    ):
                        continue

                    for target in node.targets:

                        if (
                            isinstance(
                                target,
                                ast.Attribute,
                            )
                            and isinstance(
                                target.value,
                                ast.Name,
                            )
                            and target.value.id == "self"
                        ):
                            names.add(
                                target.attr
                            )

                elif isinstance(node, ast.AnnAssign):

                    value = node.value

                    if (
                        value is not None
                        and isinstance(
                            value,
                            ast.Call,
                        )
                        and isinstance(
                            value.func,
                            ast.Attribute,
                        )
                        and value.func.attr == "Parameter"
                        and isinstance(
                            value.func.value,
                            ast.Name,
                        )
                        and value.func.value.id == "nn"
                        and isinstance(
                            node.target,
                            ast.Attribute,
                        )
                        and isinstance(
                            node.target.value,
                            ast.Name,
                        )
                        and node.target.value.id == "self"
                    ):
                        names.add(
                            node.target.attr
                        )

            return names

        parent_parameters = parameter_names(
            parent_source
        )
        child_parameters = parameter_names(
            source
        )

        plan_lower = plan.lower()

        # --------------------------------------------------------------
        # Parent parameters may not disappear silently.
        # --------------------------------------------------------------

        for name in sorted(
            parent_parameters
        ):

            explicit_replace = any(
                phrase in plan_lower
                for phrase in (
                    f"replace {name}",
                    f"remove {name}",
                    f"without {name}",
                    f"eliminate {name}",
                    f"replace the {name}",
                )
            )

            if explicit_replace:
                continue

            if name not in child_parameters:

                errors.append(
                    f"Parent parameter '{name}' was silently removed "
                    "from the child core."
                )
                continue

            occurrences = len(
                re.findall(
                    rf"\bself\.{re.escape(name)}\b",
                    source,
                )
            )

            if occurrences < 2:

                errors.append(
                    f"Parent parameter '{name}' exists but is not "
                    "used by the child computation."
                )

        # --------------------------------------------------------------
        # Teacher-declared NEW parameters.
        # --------------------------------------------------------------

        planned_parameters: set[str] = set()

        for match in re.finditer(
            r"parameter[^`\n]{0,50}`([A-Za-z_]\w*)`",
            plan,
            flags=re.IGNORECASE,
        ):
            planned_parameters.add(
                match.group(1)
            )

        for match in re.finditer(
            r"(?:add|introduce|define|create)[^\n]{0,80}"
            r"parameter\s+(?:named\s+)?"
            r"([A-Za-z_]\w*)",
            plan,
            flags=re.IGNORECASE,
        ):
            planned_parameters.add(
                match.group(1)
            )

        planned_parameters.difference_update(
            parent_parameters
        )
        # class / method names picked up from prose are not parameters
        planned_parameters = {
            n for n in planned_parameters
            if not n.startswith("_") and not n[0].isupper()
            and n not in {"self", "forward", "init", "named", "called", "that", "which", "the"}
        }

        for name in sorted(
            planned_parameters
        ):

            if name not in child_parameters:

                errors.append(
                    f"Teacher planned new parameter '{name}' "
                    "but child source does not define it as "
                    "nn.Parameter."
                )
                continue

            occurrences = len(
                re.findall(
                    rf"\bself\.{re.escape(name)}\b",
                    source,
                )
            )

            if occurrences < 2:

                errors.append(
                    f"Teacher planned new parameter '{name}' "
                    "but it is declared without participating "
                    "in the executable core."
                )

        # --------------------------------------------------------------
        # Public NovaScanBlock interface.
        # --------------------------------------------------------------

        try:
            tree = ast.parse(source)

            block = None

            for node in tree.body:
                if (
                    isinstance(
                        node,
                        ast.ClassDef,
                    )
                    and node.name == "NovaScanBlock"
                ):
                    block = node
                    break

            if block is None:

                errors.append(
                    "NovaScanBlock class is missing."
                )

            else:

                methods = {
                    node.name: node
                    for node in block.body
                    if isinstance(
                        node,
                        (ast.FunctionDef, ast.AsyncFunctionDef),
                    )
                }

                init_node = methods.get(
                    "__init__"
                )

                forward_node = methods.get(
                    "forward"
                )

                if init_node is None:

                    errors.append(
                        "NovaScanBlock.__init__ is missing."
                    )

                else:

                    args = [
                        arg.arg
                        for arg in init_node.args.args
                    ]

                    for required in (
                        "self",
                        "d_model",
                        "d_state",
                        "conv_kernel",
                        "forget_bias",
                    ):

                        if required not in args:

                            errors.append(
                                "NovaScanBlock.__init__ lost "
                                f"required argument '{required}'."
                            )

                if forward_node is None:

                    errors.append(
                        "NovaScanBlock.forward is missing."
                    )

                else:

                    args = [
                        arg.arg
                        for arg in forward_node.args.args
                    ]

                    if "self" not in args:
                        errors.append(
                            "NovaScanBlock.forward lost 'self'."
                        )

                    if "x" not in args:
                        errors.append(
                            "NovaScanBlock.forward lost 'x'."
                        )

        except SyntaxError as exc:

            errors.append(
                "Child source could not be parsed during "
                f"contract validation: {exc}"
            )

        return (
            not errors,
            sorted(
                set(errors)
            ),
        )

    @staticmethod
    def source_safety_check(
        source: str,
    ) -> tuple[bool, list[str]]:

        errors: list[str] = []

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return (
                False,
                [
                    f"SyntaxError: {exc.msg} "
                    f"line={exc.lineno}"
                ],
            )

        found_class = False
        found_forward = False
        found_init = False

        for node in ast.walk(tree):

            if isinstance(node, ast.ClassDef):
                if node.name == "NovaScanBlock":
                    found_class = True
                    method_names = {
                        x.name
                        for x in node.body
                        if isinstance(
                            x,
                            ast.FunctionDef,
                        )
                    }
                    found_init |= "__init__" in method_names
                    found_forward |= "forward" in method_names

            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(
                        ".",
                        1,
                    )[0]

                    if root != "torch":
                        errors.append(
                            f"Forbidden import: {alias.name}"
                        )

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""

                if (
                    node.level != 0
                    or not (
                        module == "torch"
                        or module.startswith("torch.")
                    )
                ):
                    errors.append(
                        f"Forbidden import-from: {module}"
                    )

            elif isinstance(node, ast.Call):
                if isinstance(
                    node.func,
                    ast.Name,
                ):
                    if node.func.id in FORBIDDEN_CALLS:
                        errors.append(
                            f"Forbidden call: {node.func.id}"
                        )

                elif isinstance(
                    node.func,
                    ast.Attribute,
                ):
                    if node.func.attr in {
                        "system",
                        "popen",
                        "run",
                        "check_output",
                    }:
                        errors.append(
                            f"Forbidden attribute call: "
                            f"{node.func.attr}"
                        )

            elif isinstance(node, ast.Name):
                if node.id in FORBIDDEN_NAMES:
                    errors.append(
                        f"Forbidden name: {node.id}"
                    )

            elif isinstance(node, ast.Attribute):
                if node.attr in FORBIDDEN_NAMES:
                    errors.append(
                        f"Forbidden attribute: {node.attr}"
                    )

        if not found_class:
            errors.append(
                "NovaScanBlock class not found"
            )

        if not found_init:
            errors.append(
                "NovaScanBlock.__init__ not found"
            )

        if not found_forward:
            errors.append(
                "NovaScanBlock.forward not found"
            )

        # Runtime smoke testing is the authoritative check for
        # unresolved Python symbols. Static analysis above already
        # rejects non-PyTorch imports and dangerous operations.
        return (
            not errors,
            sorted(set(errors)),
        )

    # ------------------------------------------------------------------
    # SANDBOX / RUNTIME VALIDATION
    # ------------------------------------------------------------------

    def create_candidate(
        self,
        source: str,
        teacher: dict[str, Any],
        diagnosis: dict[str, Any],
    ) -> dict[str, Any]:

        generation = int(
            self.load_project_state().get(
                "current_generation",
                4,
            )
        )

        # Allocate IDs from persistent filesystem history as well as
        # autonomous_state.json. This prevents manual test candidates
        # from colliding with autonomous candidates.
        used_numbers = set()

        for base_dir in (
            CANDIDATE_DIR,
            RESULT_DIR,
            PASSED_DIR,
        ):
            if not base_dir.exists():
                continue

            for item in base_dir.iterdir():
                match = re.fullmatch(
                    rf"GEN{generation}-CORE-(\\d+)(?:\\.json)?",
                    item.name,
                )

                if match:
                    used_numbers.add(
                        int(match.group(1))
                    )

        next_number = max(
            used_numbers,
            default=0,
        ) + 1

        while (
            CANDIDATE_DIR
            / f"GEN{generation}-CORE-{next_number:03d}"
        ).exists():
            next_number += 1

        while (
            RESULT_DIR
            / f"GEN{generation}-CORE-{next_number:03d}.json"
        ).exists():
            next_number += 1

        self.state["attempt"] = next_number

        candidate_id = (
            f"GEN{generation}-CORE-{next_number:03d}"
        )

        # Reserve the number immediately.
        self.save_core_state()

        candidate_dir = (
            CANDIDATE_DIR / candidate_id
        )
        candidate_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        source_hash = hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest()

        candidate = {
            "candidate_id": candidate_id,
            "candidate": candidate_id,
            "type": "core",
            "target": "nova/blocks_scan.py",
            "parent": self.load_project_state().get(
                "primary_parent"
            ),
            "parent_generation": self.load_project_state().get(
                "parent_generation"
            ),
            "source_sha256": source_hash,
            "created_at": time.time(),
            "diagnosis": diagnosis,
            "teacher": teacher,
            "status": "GENERATED",
        }

        (candidate_dir / "blocks_scan.py").write_text(
            source,
            encoding="utf-8",
        )

        (candidate_dir / "candidate.json").write_text(
            json.dumps(
                candidate,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        candidate["local_source"] = str(
            candidate_dir / "blocks_scan.py"
        )

        return candidate

    def runtime_smoke_test(
        self,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:

        source = Path(
            candidate["local_source"]
        ).read_text(
            encoding="utf-8"
        )

        workspace = create_workspace(
            candidate["candidate_id"]
        )

        # Load the generated block into a completely separate module.
        write_file(
            candidate["candidate_id"],
            "nova/blocks_scan.py",
            source,
        )

        harness = r'''
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path("/opt/ai/work/nova-evo")
sys.path.insert(0, str(ROOT))

from nova.config import NovaConfig

ROOT = Path("/opt/ai/work/nova-evo")
BLOCK = Path.cwd() / "nova" / "blocks_scan.py"

spec = importlib.util.spec_from_file_location(
    "nova_generated_core",
    BLOCK,
)

if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load generated core")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

NovaScanBlock = module.NovaScanBlock


class CandidateModel(nn.Module):
    def __init__(self, config: NovaConfig):
        super().__init__()

        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )

        self.blocks = nn.ModuleList(
            [
                NovaScanBlock(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    conv_kernel=config.conv_kernel,
                    forget_bias=config.forget_bias,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(
            config.d_model
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        self.lm_head.weight = self.embedding.weight

    def forward(
        self,
        input_ids,
        states=None,
    ):
        x = self.embedding(input_ids)

        new_states = []

        for layer_idx, block in enumerate(self.blocks):
            state = (
                None
                if states is None
                else states[layer_idx]
            )

            x, state = block(
                x,
                state=state,
            )

            new_states.append(state)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, new_states


torch.manual_seed(1001)

config = NovaConfig(
    vocab_size=16384,
    d_model=384,
    d_state=384,
    num_layers=6,
    conv_kernel=5,
    forget_bias=1.125,
    learnable_initial_state=False,
    max_seq_len=512,
    pad_token_id=0,
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = CandidateModel(config).to(device)

ids = torch.randint(
    low=1,
    high=config.vocab_size,
    size=(2, 16),
    device=device,
)

# Forward/backward
model.train()

logits, states = model(ids)

assert torch.isfinite(logits).all()
assert len(states) == config.num_layers

loss = logits.float().mean()

model.zero_grad(set_to_none=True)
loss.backward()

gradient_tensors = 0
invalid_gradients = 0

for p in model.parameters():
    if p.grad is None:
        continue

    gradient_tensors += 1

    if not torch.isfinite(p.grad).all():
        invalid_gradients += 1

assert gradient_tensors > 0
assert invalid_gradients == 0

# Causality check:
# changing the suffix must not alter prefix outputs.
prefix = ids[:, :8]
suffix_a = torch.randint(
    1,
    config.vocab_size,
    (2, 8),
    device=device,
)
suffix_b = torch.randint(
    1,
    config.vocab_size,
    (2, 8),
    device=device,
)

with torch.no_grad():
    out_a, _ = model(
        torch.cat([prefix, suffix_a], dim=1)
    )
    out_b, _ = model(
        torch.cat([prefix, suffix_b], dim=1)
    )

causal_diff = (
    out_a[:, :8] - out_b[:, :8]
).abs().max().item()

assert causal_diff < 1e-6, (
    f"Causality failure: diff={causal_diff}"
)

parameters = sum(
    p.numel()
    for p in model.parameters()
)

result = {
    "status": "PASS",
    "device": str(device),
    "parameters": parameters,
    "gradient_tensors": gradient_tensors,
    "invalid_gradients": invalid_gradients,
    "causal_prefix_max_diff": causal_diff,
    "logits_shape": list(logits.shape),
}

print(json.dumps(result, ensure_ascii=False))
'''

        write_file(
            candidate["candidate_id"],
            "smoke_test.py",
            harness,
        )

        from evo.engine.experiment_runner import (
            ExperimentRunner,
        )

        runner = ExperimentRunner(
            default_timeout=120,
        )

        from nova.gpu_guard import ensure_vram

        ensure_vram(need_mb=1500)

        result = runner.run(
            candidate["candidate_id"],
            ["python", "smoke_test.py"],
            timeout=120,
        )

        if not result.success:
            info = result.to_dict()
            err = str(info.get("stderr") or "") + str(info.get("stdout") or "")
            oom = ("out of memory" in err.lower()) or ("OutOfMemoryError" in err)
            return {
                # GPU memory taken by a teacher is not the candidate's fault
                "status": "INFRA_ERROR" if oom else "FAIL",
                "runner": info,
            }

        try:
            parsed = json.loads(
                result.stdout.strip().splitlines()[-1]
            )
        except Exception as exc:
            return {
                "status": "FAIL",
                "error": (
                    f"Invalid smoke-test JSON: {exc}"
                ),
                "stdout": result.stdout,
            }

        return {
            "status": "PASS",
            "runner": result.to_dict(),
            "metrics": parsed,
        }

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------

    def write_training_harness(
        self,
        candidate: dict[str, Any],
        screening_steps: int,
        robust_steps: int,
    ) -> None:

        candidate_id = candidate["candidate_id"]

        resolve_training_metadata(
            candidate,
            self.load_project_state(),
            RESULT_DIR,
        )

        source = Path(
            candidate["local_source"]
        ).read_text(
            encoding="utf-8"
        )

        write_file(
            candidate_id,
            "nova/blocks_scan.py",
            source,
        )

        write_file(
            candidate_id,
            "candidate.json",
            json.dumps(
                candidate,
                indent=2,
                ensure_ascii=False,
            ),
        )

        harness = f'''
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path("/opt/ai/work/nova-evo").resolve()
sys.path.insert(0, str(ROOT))

from nova.config import NovaConfig
from evo.engine import training_runner as tr
from evo.engine.training_runner import TrainingRunner

BLOCK = Path.cwd() / "nova" / "blocks_scan.py"

import importlib.util

spec = importlib.util.spec_from_file_location(
    "nova_generated_core",
    BLOCK,
)

if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load generated core")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

NovaScanBlock = module.NovaScanBlock


class CandidateModel(nn.Module):
    def __init__(self, config: NovaConfig):
        super().__init__()

        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )

        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.02,
        )

        if config.pad_token_id is not None:
            with torch.no_grad():
                self.embedding.weight[
                    config.pad_token_id
                ].zero_()

        self.blocks = nn.ModuleList(
            [
                NovaScanBlock(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    conv_kernel=config.conv_kernel,
                    forget_bias=config.forget_bias,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(
            config.d_model
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        self.lm_head.weight = self.embedding.weight

    def forward(
        self,
        input_ids,
        states=None,
    ):
        if input_ids.ndim != 2:
            raise ValueError(
                f"Expected [batch, seq], got "
                f"{{tuple(input_ids.shape)}}"
            )

        x = self.embedding(input_ids)

        new_states = []

        for layer_idx, block in enumerate(self.blocks):
            state = (
                None
                if states is None
                else states[layer_idx]
            )

            x, state = block(
                x,
                state=state,
            )

            new_states.append(state)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, new_states

    def num_parameters(
        self,
        trainable_only=True,
    ):
        if trainable_only:
            return sum(
                p.numel()
                for p in self.parameters()
                if p.requires_grad
            )

        return sum(
            p.numel()
            for p in self.parameters()
        )


def build_candidate(config=None):
    # TrainingRunner passes the architecture config as a dict.
    if config is None:
        config = NovaConfig()
    elif isinstance(config, dict):
        known = set(NovaConfig.__dataclass_fields__)
        config = NovaConfig(
            **{{k: v for k, v in config.items() if k in known}}
        )
    return CandidateModel(config)


# Patch only this Python process.
# Nothing on disk is modified.
tr.build_model = build_candidate

runner = TrainingRunner()

candidate = json.loads(
    (Path.cwd() / "candidate.json").read_text(
        encoding="utf-8"
    )
)

sig = inspect.signature(
    runner.run_robust
)

kwargs = {{}}

for name, param in sig.parameters.items():

    if name in {{
        "candidate",
        "candidate_data",
        "candidate_record",
    }}:
        kwargs[name] = candidate

    elif name in {{
        "candidate_id",
        "id",
    }}:
        kwargs[name] = candidate["candidate_id"]

    elif name in {{
        "seed_list",
        "seeds",
        "robust_seeds",
    }}:
        kwargs[name] = {ROBUST_SEEDS!r}

    elif name in {{
        "max_steps",
        "steps",
        "training_steps",
        "robust_steps",
    }}:
        kwargs[name] = {robust_steps}

    elif name in {{
        "screening_steps",
    }}:
        kwargs[name] = {screening_steps}

    elif param.default is inspect.Parameter.empty:
        raise RuntimeError(
            f"Unsupported required run_robust parameter: {{name}}"
        )

result = runner.run_robust(**kwargs)

Path("robust_result.json").write_text(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print(
    json.dumps(
        result,
        ensure_ascii=False,
    )
)
'''

        write_file(
            candidate_id,
            "train_candidate.py",
            harness,
        )

    def robust_train(
        self,
        candidate: dict[str, Any],
        screening_steps: int = 100,
        robust_steps: int = 1000,
    ) -> dict[str, Any]:

        try:
            self.write_training_harness(
                candidate,
                screening_steps,
                robust_steps,
            )
        except CoreEvolutionError as exc:
            return {
                "status": "INFRA_ERROR",
                "error": str(exc),
            }

        from evo.engine.experiment_runner import (
            ExperimentRunner,
        )

        runner = ExperimentRunner(
            default_timeout=7200,
        )

        result = runner.run(
            candidate["candidate_id"],
            ["python", "train_candidate.py"],
            timeout=7200,
        )

        if not result.success:
            runner_info = result.to_dict()
            return {
                "status": (
                    "INFRA_ERROR"
                    if is_infrastructure_failure(
                        runner_info.get("stderr") or ""
                    )
                    else "TRAINING_FAILED"
                ),
                "runner": runner_info,
            }

        workspace = Path(
            result.workspace
        )

        robust_path = (
            workspace / "robust_result.json"
        )

        if not robust_path.exists():
            return {
                "status": "TRAINING_FAILED",
                "error": (
                    "Training completed but robust_result.json "
                    "was not produced"
                ),
                "runner": result.to_dict(),
            }

        robust = json.loads(
            robust_path.read_text(
                encoding="utf-8"
            )
        )

        if robust.get("status") != "COMPLETED":
            trace = robust.get("traceback") or ""
            return {
                "status": (
                    "INFRA_ERROR"
                    if is_infrastructure_failure(trace)
                    else "TRAINING_FAILED"
                ),
                "error": robust.get("error"),
                "traceback": trace[-3000:],
                "runner": result.to_dict(),
                "robust": robust,
            }

        return {
            "status": "COMPLETED",
            "runner": result.to_dict(),
            "robust": robust,
        }

    # ------------------------------------------------------------------
    # EVALUATION / LEARNING
    # ------------------------------------------------------------------

    def evaluate(
        self,
        candidate: dict[str, Any],
        smoke: dict[str, Any],
        training: dict[str, Any],
    ) -> dict[str, Any]:

        if smoke.get("status") == "INFRA_ERROR":
            return {
                "decision": "RETRY",
                "reason": "infrastructure_error",
                "error": "smoke test ran out of GPU memory",
            }

        if smoke.get("status") != "PASS":
            return {
                "decision": "REJECT",
                "reason": "runtime_smoke_failed",
            }

        if training.get("status") == "INFRA_ERROR":
            return {
                "decision": "RETRY",
                "reason": "infrastructure_error",
                "error": training.get("error")
                or (training.get("runner") or {}).get("stderr", "")[-500:],
            }

        if training.get("status") != "COMPLETED":
            return {
                "decision": "REJECT",
                "reason": "training_failed",
            }

        robust = training.get(
            "robust",
            {},
        )

        metrics = robust.get(
            "metrics",
            {},
        )

        candidate_loss = metrics.get(
            "validation_loss_mean"
        )

        project = self.load_project_state()

        parent_loss = (
            project.get("best_known", {})
            .get("validation_loss")
        )

        seeds = robust.get(
            "seeds",
            [],
        )

        if not isinstance(
            candidate_loss,
            (int, float),
        ):
            return {
                "decision": "REJECT",
                "reason": "missing_candidate_loss",
            }

        if not isinstance(
            parent_loss,
            (int, float),
        ):
            return {
                "decision": "REJECT",
                "reason": "missing_parent_baseline",
            }

        if len(seeds) < 3:
            return {
                "decision": "REJECT",
                "reason": "robust_requires_three_seeds",
            }

        best_known = project.get("best_known", {})
        current_dataset = (
            (project.get("data_policy") or {}).get("training_data")
            or ["data/gen4"]
        )[0]
        baseline_dataset = best_known.get("dataset", "data/gen4")
        candidate_dataset = robust.get("dataset", current_dataset)

        if not (baseline_dataset == current_dataset == candidate_dataset):
            return {
                "decision": "RETRY",
                "reason": "baseline_dataset_mismatch",
                "error": (
                    f"baseline={baseline_dataset} "
                    f"current={current_dataset} "
                    f"candidate={candidate_dataset}; "
                    "run the rebaseline first"
                ),
            }

        efficiency = efficiency_check(
            metrics.get("efficiency") or {},
            best_known.get("efficiency"),
            project.get("efficiency_policy"),
        )

        tolerance = float(
            (project.get("efficiency_policy") or {}).get(
                "equal_quality_tolerance", 0.005
            )
        )
        # Improvement must exceed seed noise, not just be "strictly lower".
        policy = project.get("efficiency_policy") or {}
        std_c = float(metrics.get("validation_loss_std") or 0.0)
        std_p = float(best_known.get("validation_loss_std") or 0.0)
        required = max(
            float(policy.get("min_improvement", 0.01)),
            2.0 * (std_c ** 2 + std_p ** 2) ** 0.5,
        )
        loss_better = float(candidate_loss) < float(parent_loss) - required
        loss_equal = float(candidate_loss) <= float(parent_loss) * (1 + tolerance)

        if loss_better and efficiency["ok"]:
            decision = "PASS"
            reason = (
                "Candidate robust validation loss is strictly below "
                "the parent baseline within the efficiency budget."
            )
        elif loss_equal and efficiency["gain"]:
            decision = "PASS"
            reason = (
                "Equal quality (within tolerance) with a clear "
                "CPU/size efficiency gain."
            )
        elif loss_better:
            decision = "REJECT"
            reason = (
                "Lower loss but efficiency budget violated: "
                + "; ".join(efficiency["violations"])
            )
        else:
            decision = "REJECT"
            reason = (
                "Candidate robust validation loss does not "
                "beat current parent baseline."
            )

        return {
            "decision": decision,
            "reason": reason,
            "efficiency": efficiency,
            "required_improvement": required,
            "calibration": metrics.get("calibration"),
            "parent_calibration": best_known.get("calibration"),
            "parent_loss": parent_loss,
            "candidate_loss": candidate_loss,
            "delta": (
                float(parent_loss)
                - float(candidate_loss)
            ),
            "seeds": seeds,
        }

    def learn(
        self,
        candidate: dict[str, Any],
        diagnosis: dict[str, Any],
        evaluation: dict[str, Any],
    ) -> None:

        lesson = {
            "candidate": candidate["candidate_id"],
            "decision": evaluation.get(
                "decision"
            ),
            "reason": evaluation.get(
                "reason"
            ),
            "parent_loss": evaluation.get(
                "parent_loss"
            ),
            "candidate_loss": evaluation.get(
                "candidate_loss"
            ),
            "delta": evaluation.get(
                "delta"
            ),
            "diagnosis": diagnosis,
            "timestamp": time.time(),
        }

        self.state.setdefault(
            "lessons",
            [],
        ).append(lesson)

    # ------------------------------------------------------------------
    # ONE AUTONOMOUS CORE CYCLE
    # ------------------------------------------------------------------

    def use_mentor(self, name: str) -> None:
        self.teacher_model = name
        self.teacher_timeout_factor = (
            self._registry.timeout_factor(name)
            if self._registry and name in self._registry.teachers
            else 1.0
        )

    def ask_teacher_with_fallback(
        self,
        diagnosis: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        """Try mentors in order until one returns a usable core source."""
        errors = []
        for name in self.mentors:
            self.use_mentor(name)
            try:
                teacher = self.ask_teacher(diagnosis)
                source = self.extract_source(teacher["response"])
                teacher["mentor"] = name
                teacher["mentor_errors"] = errors
                print(f"MENTOR: {name}", flush=True)
                return teacher, source
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}"[:300])
                print(f"MENTOR FAILED: {errors[-1]}", flush=True)
        raise CoreEvolutionError(
            "No mentor produced a core source: " + " | ".join(errors)
        )

    def _promoted_ids(self) -> set[str]:
        try:
            project = self.load_project_state()
        except Exception:
            return set()
        ids = {project.get("primary_parent")}
        ids.update(
            (entry or {}).get("candidate")
            for entry in project.get("lineage") or []
        )
        ids.update(
            (entry or {}).get("promoted")
            for entry in project.get("generation_history") or []
        )
        return {i for i in ids if i}

    def _recover_incomplete_candidate(
        self,
        diagnosis: dict[str, Any],
        screening_steps: int,
        robust_steps: int,
    ) -> dict[str, Any] | None:
        """
        Recover the newest CORE experiment that reached smoke PASS but
        failed before evaluation because of a training/infrastructure
        error.

        This prevents NOVA from wasting a valid experiment by inventing
        another architecture before the existing evidence is completed.
        """

        recoverable = None
        newest = -1.0

        if not RESULT_DIR.exists():
            return None

        for path in RESULT_DIR.glob(
            "GEN*.json"
        ):

            try:
                record = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )
            except Exception:
                continue

            training = record.get(
                "training",
                {},
            )

            evaluation = record.get(
                "evaluation",
                {},
            )

            smoke = record.get(
                "smoke",
                {},
            )

            if not isinstance(
                training,
                dict,
            ):
                continue

            if not isinstance(
                evaluation,
                dict,
            ):
                continue

            if not isinstance(
                smoke,
                dict,
            ):
                continue

            smoke_infra = (
                str((smoke or {}).get("status") or "").upper() == "INFRA_ERROR"
            )

            if (
                str(
                    training.get(
                        "status"
                    )
                    or ""
                ).upper()
                not in {"TRAINING_FAILED", "INFRA_ERROR"}
                and not smoke_infra
            ):
                continue

            if (
                str(
                    evaluation.get(
                        "reason"
                    )
                    or ""
                ).lower()
                not in {"training_failed", "infrastructure_error"}
            ):
                continue

            if (
                str(
                    smoke.get(
                        "status"
                    )
                    or ""
                ).upper()
                not in {"PASS", "INFRA_ERROR"}
            ):
                continue

            candidate_id = record.get(
                "candidate"
            )

            if not candidate_id:
                continue

            candidate_file = (
                CANDIDATE_DIR
                / str(candidate_id)
                / "candidate.json"
            )

            source_file = (
                CANDIDATE_DIR
                / str(candidate_id)
                / "blocks_scan.py"
            )

            if (
                not candidate_file.exists()
                or not source_file.exists()
            ):
                continue

            try:
                candidate = json.loads(
                    candidate_file.read_text(
                        encoding="utf-8"
                    )
                )
            except Exception:
                continue

            if candidate.get(
                "type"
            ) != "core":
                continue

            if int(
                candidate.get("recovery_attempts", 0)
            ) >= MAX_RECOVERY_ATTEMPTS:
                continue

            # never re-evaluate a core that is already active or promoted
            if candidate.get("status") == "PROMOTED" or candidate.get(
                "candidate_id"
            ) in self._promoted_ids():
                continue

            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0

            if mtime > newest:
                newest = mtime
                recoverable = candidate

        if recoverable is None:
            return None

        candidate_id = recoverable[
            "candidate_id"
        ]

        print()
        print(
            "=" * 72
        )
        print(
            "NOVA-EVO — RECOVERING INCOMPLETE EXPERIMENT"
        )
        print(
            "=" * 72
        )
        print(
            "CANDIDATE:",
            candidate_id,
        )

        recoverable[
            "generation"
        ] = diagnosis.get(
            "current_generation"
        )

        recoverable[
            "parent_generation"
        ] = diagnosis.get(
            "parent_generation"
        )

        recoverable[
            "parent"
        ] = (
            recoverable.get(
                "parent"
            )
            or diagnosis.get(
                "primary_parent"
            )
        )

        recoverable["recovery_attempts"] = (
            int(recoverable.get("recovery_attempts", 0)) + 1
        )

        recoverable[
            "local_source"
        ] = str(
            CANDIDATE_DIR
            / candidate_id
            / "blocks_scan.py"
        )

        candidate_file = (
            CANDIDATE_DIR
            / candidate_id
            / "candidate.json"
        )

        candidate_file.write_text(
            json.dumps(
                recoverable,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        smoke = self.runtime_smoke_test(
            recoverable
        )

        if smoke.get(
            "status"
        ) != "PASS":

            evaluation = {
                "decision":
                    "REJECT",

                "reason":
                    "runtime_smoke_failed",
            }

            result = {
                "status":
                    "REJECT",

                "decision":
                    "REJECT",

                "stage":
                    "runtime_smoke",

                "candidate":
                    candidate_id,

                "recovery":
                    True,

                "smoke":
                    smoke,

                "evaluation":
                    evaluation,
            }

        else:

            training = self.robust_train(
                recoverable,
                screening_steps=screening_steps,
                robust_steps=robust_steps,
            )

            evaluation = self.evaluate(
                recoverable,
                smoke,
                training,
            )

            result = {
                "status":
                    evaluation.get(
                        "decision",
                        "REJECT",
                    ),

                "decision":
                    evaluation.get(
                        "decision",
                        "REJECT",
                    ),

                "stage":
                    "evaluation",

                "candidate":
                    candidate_id,

                "recovery":
                    True,

                "smoke":
                    smoke,

                "training":
                    training,

                "evaluation":
                    evaluation,
            }

            if evaluation.get(
                "decision"
            ) in {
                "PASS",
                "PROMOTE",
            }:

                recoverable[
                    "status"
                ] = (
                    "PASSED_PENDING_PROMOTION"
                )

                promotion = {
                    "status":
                        "PROMOTION_UNAVAILABLE"
                }

                if hasattr(
                    self,
                    "_auto_promote",
                ):

                    promotion = (
                        self._auto_promote(
                            recoverable,
                            evaluation,
                            result,
                        )
                    )

                result[
                    "promotion"
                ] = promotion

                if promotion.get(
                    "status"
                ) == "PROMOTED":

                    recoverable[
                        "status"
                    ] = "PROMOTED"

            elif evaluation.get("decision") == "RETRY":

                recoverable[
                    "status"
                ] = "RETRY"

            else:

                recoverable[
                    "status"
                ] = "REJECT"

        self._write_result(
            recoverable,
            result,
        )

        if result.get("decision") != "RETRY":
            self.learn(
                recoverable,
                diagnosis,
                result.get(
                    "evaluation",
                    {},
                ),
            )

        self.state.setdefault(
            "candidates",
            [],
        ).append(
            {
                "candidate":
                    candidate_id,

                "status":
                    recoverable.get(
                        "status"
                    ),

                "decision":
                    result.get(
                        "decision"
                    ),

                "loss":
                    result.get(
                        "evaluation",
                        {},
                    ).get(
                        "candidate_loss"
                    ),

                "recovery":
                    True,

                "timestamp":
                    time.time(),
            }
        )

        self.save_core_state()

        return result

    def run_once(
        self,
        screening_steps: int = 100,
        robust_steps: int = 1000,
    ) -> dict[str, Any]:

        self.assert_allowed()
        self.assert_target_allowed(
            "nova/blocks_scan.py"
        )

        diagnosis = self.diagnose()

        recovered = (
            self._recover_incomplete_candidate(
                diagnosis,
                screening_steps,
                robust_steps,
            )
        )

        if recovered is not None:
            return recovered

        teacher, source = self.ask_teacher_with_fallback(
            diagnosis
        )

        safe, safety_errors = (
            self.source_safety_check(source)
        )

        if not safe:
            result = {
                "status": "REJECT",
                "stage": "source_safety",
                "errors": safety_errors,
            }

            self.state.setdefault(
                "lessons",
                [],
            ).append(
                {
                    "candidate": None,
                    "decision": "REJECT",
                    "reason": "source_safety",
                    "errors": safety_errors,
                    "diagnosis": diagnosis,
                    "timestamp": time.time(),
                }
            )

            self.save_core_state()
            return result

        contract_ok, contract_errors = (
            self.source_contract_check(
                source,
                teacher.get(
                    "plan",
                    "",
                ),
            )
        )

        if not contract_ok:

            candidate_hint = (
                f"GEN{diagnosis.get('current_generation', 0)}"
                f"-CORE-CONTRACT-{int(time.time())}"
            )

            result = {
                "status": "REJECT",
                "decision": "REJECT",
                "stage": "source_contract",
                "candidate": candidate_hint,
                "errors": contract_errors,
            }

            self.state.setdefault(
                "lessons",
                [],
            ).append(
                {
                    "candidate": candidate_hint,
                    "decision": "REJECT",
                    "reason": "source_contract",
                    "stage": "source_contract",
                    "errors": contract_errors,
                    "diagnosis": diagnosis,
                    "teacher": teacher,
                    "timestamp": time.time(),
                }
            )

            RESULT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            (
                RESULT_DIR
                / f"{candidate_hint}.json"
            ).write_text(
                json.dumps(
                    result,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            self.save_core_state()
            return result

        candidate = self.create_candidate(
            source,
            teacher,
            diagnosis,
        )

        # TrainingRunner requires generation metadata.
        # Add it before any training stage and persist it to candidate.json.
        candidate.setdefault(
            "generation",
            diagnosis.get(
                "current_generation"
            ),
        )

        candidate.setdefault(
            "parent_generation",
            diagnosis.get(
                "parent_generation"
            ),
        )

        candidate.setdefault(
            "parent",
            diagnosis.get(
                "primary_parent"
            ),
        )

        resolve_training_metadata(
            candidate,
            self.load_project_state(),
            RESULT_DIR,
        )

        candidate_file = (
            CANDIDATE_DIR
            / candidate["candidate_id"]
            / "candidate.json"
        )

        candidate_file.write_text(
            json.dumps(
                candidate,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        validation = self.validation.validate_candidate(
            candidate["candidate_id"],
            [
                candidate["local_source"]
            ],
        )

        if not validation.valid:
            result = {
                "status": "REJECT",
                "stage": "static_validation",
                "candidate": candidate["candidate_id"],
                "errors": validation.errors,
            }

            candidate["status"] = "REJECT"

            (CANDIDATE_DIR / candidate["candidate_id"]
             / "candidate.json").write_text(
                json.dumps(
                    candidate,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            self.learn(
                candidate,
                diagnosis,
                {
                    "decision": "REJECT",
                    "reason": "static_validation",
                    "errors": validation.errors,
                },
            )

            self.save_core_state()

            return result

        smoke = self.runtime_smoke_test(
            candidate
        )

        if smoke.get("status") != "PASS":
            infra = smoke.get("status") == "INFRA_ERROR"
            evaluation = {
                "decision": "RETRY" if infra else "REJECT",
                "reason": "infrastructure_error" if infra else "runtime_smoke_failed",
            }

            if not infra:
                self.learn(
                    candidate,
                    diagnosis,
                    evaluation,
                )

            candidate["status"] = evaluation["decision"]

            result = {
                "status": evaluation["decision"],
                "stage": "runtime_smoke",
                "candidate": candidate["candidate_id"],
                "smoke": smoke,
                "evaluation": evaluation,
            }

            self._write_result(
                candidate,
                result,
            )

            self.save_core_state()

            return result

        training = self.robust_train(
            candidate,
            screening_steps=screening_steps,
            robust_steps=robust_steps,
        )

        evaluation = self.evaluate(
            candidate,
            smoke,
            training,
        )

        promotion = None

        if evaluation.get(
            "decision"
        ) in {
            "PASS",
            "PROMOTE",
        }:

            candidate["status"] = (
                "PASSED_PENDING_PROMOTION"
            )

            promotion = self._auto_promote(
                candidate,
                evaluation,
                {
                    "candidate":
                        candidate["candidate_id"],
                    "evaluation":
                        evaluation,
                    "training":
                        training,
                },
            )

            evaluation[
                "promotion"
            ] = promotion

            if (
                promotion.get(
                    "status"
                )
                == "PROMOTED"
            ):

                candidate[
                    "status"
                ] = "PROMOTED"

        if (
            evaluation.get(
                "decision"
            )
            in {
                "PASS",
                "PROMOTE",
            }
        ):

            passed_dir = (
                PASSED_DIR
                / candidate["candidate_id"]
            )

            passed_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copy2(
                candidate["local_source"],
                passed_dir
                / "blocks_scan.py",
            )

            self.state.setdefault(
                "passed_candidates",
                [],
            ).append(
                candidate["candidate_id"]
            )

        elif evaluation.get("decision") == "RETRY":
            # Infrastructure failure: the candidate was never judged.
            candidate["status"] = "RETRY"

        else:
            candidate["status"] = "REJECT"

        result = {
            "status": evaluation["decision"],
            "candidate": candidate["candidate_id"],
            "diagnosis": diagnosis,
            "teacher": teacher,
            "validation": validation.to_dict(),
            "smoke": smoke,
            "training": training,
            "evaluation": evaluation,
            "promotion": promotion,
        }

        self._write_result(
            candidate,
            result,
        )

        if evaluation.get("decision") != "RETRY":
            self.learn(
                candidate,
                diagnosis,
                evaluation,
            )

        self.state.setdefault(
            "candidates",
            [],
        ).append(
            {
                "candidate": candidate["candidate_id"],
                "status": candidate["status"],
                "decision": evaluation["decision"],
                "loss": evaluation.get(
                    "candidate_loss"
                ),
                "timestamp": time.time(),
            }
        )

        self.save_core_state()

        return result

    def _auto_promote(
        self,
        candidate: dict[str, Any],
        evaluation: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Invoke the existing protected CorePromotionManager.

        The manager remains the authority for actual promotion.
        This method only connects the autonomous cycle to it.
        """

        try:
            import inspect

            from evo.engine.core_promotion_manager import (
                CorePromotionManager,
            )

            manager = CorePromotionManager()

            method = None
            method_name = None

            for name in (
                "promote",
                "promote_candidate",
                "execute_promotion",
                "execute",
            ):

                if hasattr(
                    manager,
                    name,
                ):

                    method = getattr(
                        manager,
                        name,
                    )

                    method_name = name
                    break

            if method is None:

                return {
                    "status": "PROMOTION_UNAVAILABLE",
                    "error": (
                        "CorePromotionManager has no "
                        "recognized promotion method."
                    ),
                }

            signature = inspect.signature(
                method
            )

            kwargs: dict[str, Any] = {}

            for parameter in signature.parameters.values():

                if parameter.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue

                name = parameter.name.lower()

                if "candidate_id" in name:

                    kwargs[
                        parameter.name
                    ] = candidate[
                        "candidate_id"
                    ]

                elif (
                    "candidate" in name
                    and "id" not in name
                ):

                    kwargs[
                        parameter.name
                    ] = candidate

                elif "evaluation" in name:

                    kwargs[
                        parameter.name
                    ] = evaluation

                elif "result" in name:

                    kwargs[
                        parameter.name
                    ] = result

                elif (
                    "parent" in name
                    and "id" in name
                ):

                    kwargs[
                        parameter.name
                    ] = candidate.get(
                        "parent"
                    )

                elif "source" in name:

                    kwargs[
                        parameter.name
                    ] = candidate.get(
                        "local_source"
                    )

                elif (
                    parameter.default
                    is inspect.Parameter.empty
                ):

                    return {
                        "status":
                            "PROMOTION_SIGNATURE_UNKNOWN",
                        "method":
                            method_name,
                        "error":
                            (
                                "Unable to map required "
                                f"promotion argument: "
                                f"{parameter.name}"
                            ),
                    }

            promoted = method(
                **kwargs
            )

            return {
                "status": "PROMOTED",
                "method": method_name,
                "result": promoted,
            }

        except Exception as exc:

            return {
                "status": "PROMOTION_ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    def _write_result(
        self,
        candidate: dict[str, Any],
        result: dict[str, Any],
    ) -> None:

        path = (
            RESULT_DIR
            / f"{candidate['candidate_id']}.json"
        )

        path.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # SELF TEST
    # ------------------------------------------------------------------

    @staticmethod
    def self_test() -> None:

        sample = """
import torch
import torch.nn as nn

class NovaScanBlock(nn.Module):
    def __init__(self, d_model, d_state, conv_kernel=5, forget_bias=1.5):
        super().__init__()

    def forward(self, x, state=None):
        return x, state
""".strip()

        ok, errors = (
            CoreEvolutionEngine.source_safety_check(
                sample
            )
        )

        assert ok, errors

        bad = """
import os
import torch

class NovaScanBlock:
    def __init__(self, d_model, d_state, conv_kernel=5, forget_bias=1.5):
        open("/tmp/x", "w")

    def forward(self, x, state=None):
        return x, state
""".strip()

        ok, errors = (
            CoreEvolutionEngine.source_safety_check(
                bad
            )
        )

        assert not ok
        assert any(
            "Forbidden import" in x
            for x in errors
        )

        assert any(
            "Forbidden call" in x
            for x in errors
        )

        print(
            "CORE EVOLUTION ENGINE SELFTEST: PASSED"
        )


if __name__ == "__main__":
    CoreEvolutionEngine.self_test()
