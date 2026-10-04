from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from evo.engine.oxcoder_mentor import OxCoderMentor, MentorRequest
from evo.learning.goal_store import LearningGoalStore
from evo.learning.learning_executor import LearningExecutor
from evo.learning.meta_learning import MetaLearningEngine
from evo.learning.task_compiler import TaskCompiler
from evo.learning.task_generator import TaskGenerator
from evo.learning.task_training_runner import TaskTrainingRunner


@dataclass
class LearningGoal:
    goal: str
    domain: str = ""
    status: str = "NEW"
    plan: list[str] = field(default_factory=list)
    knowledge: dict[str, Any] = field(default_factory=dict)


class LearningEngine:
    """NOVA-EVO autonomous learning coordinator."""

    def __init__(self) -> None:
        self.mentor = OxCoderMentor()
        self.goals = LearningGoalStore()
        self.compiler = TaskCompiler()
        self.generator = TaskGenerator()
        self.executor = LearningExecutor()
        self.meta_learning = MetaLearningEngine(
            knowledge=self.executor.knowledge
        )
        self.task_training = TaskTrainingRunner()

    def create_goal(self, goal: str) -> LearningGoal:
        goal = goal.strip()

        if not goal:
            raise ValueError("Learning goal cannot be empty.")

        stored = self.goals.load(goal)

        if stored is None:
            stored = self.goals.create(goal)

        return LearningGoal(
            goal=stored["goal"],
            status=stored["status"],
            plan=stored.get("plan", []),
            knowledge=stored,
        )

    @staticmethod
    def _extract_tasks(text: str) -> list[str]:
        tasks: list[str] = []

        if not text:
            return tasks

        for line in text.splitlines():
            line = line.strip()

            if not line.upper().startswith("TASK:"):
                continue

            task = line[5:].strip()

            if task and task not in tasks:
                tasks.append(task)

        if len(tasks) > 5:
            tasks = tasks[-5:]

        return tasks

    @staticmethod
    def _validate_plan_tasks(
        tasks: list[str],
        goal: str = "",
    ) -> list[str]:
        """Return deterministic validation errors for an autonomous plan."""

        errors: list[str] = []
        goal_lower = goal.lower()

        unsupported = (
            "no longer usable",
            "cannot outlive",
            "must fail to compile",
            "must not compile",
            "compiler rejects",
            "compiler reject",
            "compile-time error",
            "compilation error",
            "stack overflow",
            "memory profiler",
            "static analyzer",
            "real crate",
            "existing crate",
            "external crate",
            "hundreds of files",
            "thousands of files",
            "external dataset",
            "internet access",
            "network access",
            "filesystem",
            "file system",
            "dangling reference",
            "dangling references",
            "dangling referenci",
            "invalid reference",
            "invalid references",
            "neplatná referencia",
            "neplatné referencie",
            "neplatných referencií",
            "memory returned",
            "memory returned to the system",
            "allocated memory returned",
            "pamäť sa vrátila",
            "vrátila do systému",
            "vrátenie pamäte",
            "released bytes",
            "released memory",
            "prepustených bajtov",
            "uvoľnených bajtov",
            "memory leak",
            "memory leaks",
            "únik pamäte",
            "bez úniku pamäte",
            "leak-free",
            "memory state",
            "stav pamäte",
            "final memory state",
            "finálny stav pamäte",
            "compilation errors",
            "compilation error",
            "compilation would fail",
            "compilation will fail",
            "would fail to compile",
            "will fail to compile",
            "fails to compile",
            "fail to compile",
            "compile failure",
            "compiler failure",
            "compiler error",
            "compiler rejects",
            "compiler reject",
            "compile-time",
            "compile time",
            "compile-time error",
            "compilation would",
            "compilation will",
            "panic if",
            "program panics",
            "program panic",
            "must panic",
            "panics",
            "compiler enforces",
            "compiler enforces at runtime",
            "proving the compiler",
            "prove the compiler",
            "compiler rule at runtime",
            "compiler rules at runtime",
            "compile rules at runtime",
            "use-after-move",
            "use after move",
            "proving lifetime rules at runtime",
            "verifying lifetime rules at runtime",
            "compiler guarantees at runtime",
            "compiler guarantee at runtime",
            "heap-allocated buffer",
            "heap allocated buffer",
            "stav pamäte",
            "finálny stav pamäte",
            "uvoľnenej pamäte",
            "uvoľnenie pamäte",
            "allocated memory",
            "released memory",
            "prepustených bajtov",
            "dangling",
            "invalid reference",
            "invalid references",
            "load the file",
            "načíta súbor",
            "načítať súbor",
            "spustiť rust program",
            "spustit rust program",
        )

        for index, task in enumerate(tasks, 1):
            lowered = task.lower()

            for phrase in unsupported:
                if phrase in lowered:
                    errors.append(
                        f"TASK {index}: unsupported objective phrase: "
                        f"{phrase}"
                    )
                    break

            if "python" in lowered and "rust" in lowered:
                errors.append(
                    f"TASK {index}: multiple programming languages "
                    "are not supported in one autonomous task."
                )

            # Rc<T> is not Send and must not be proposed as a
            # cross-thread shared state mechanism.
            if "rc<" in lowered and (
                "thread" in lowered
                or "threads" in lowered
            ):
                errors.append(
                    f"TASK {index}: Rc cannot be used as the "
                    "cross-thread sharing mechanism."
                )

            # RefCell is for single-threaded interior mutability.
            # Cross-thread exercises must use a thread-safe primitive.
            if "refcell" in lowered and (
                "thread" in lowered
                or "threads" in lowered
            ):
                errors.append(
                    f"TASK {index}: RefCell is not an appropriate "
                    "cross-thread synchronization primitive."
                )

            # A learning task must not claim that a compile-time
            # property is observable as a runtime panic/assertion.
            runtime_compile_mix = (
                (
                    "compiler" in lowered
                    or "compile" in lowered
                    or "lifetime" in lowered
                    or "ownership" in lowered
                )
                and (
                    "at runtime" in lowered
                    or "runtime" in lowered
                    or "panic" in lowered
                    or "use-after" in lowered
                )
            )

            if runtime_compile_mix:
                errors.append(
                    f"TASK {index}: compile-time language rules "
                    "must not be presented as runtime behavior."
                )

            if any(
                word in lowered
                for word in (
                    "500 ",
                    "1000 ",
                    "200 ",
                    "hundreds",
                    "thousands",
                )
            ):
                errors.append(
                    f"TASK {index}: task requests too many cases "
                    "for a single local learning experiment."
                )

            if "rust" in goal_lower:
                rust_concepts = (
                    "rust",
                    "ownership",
                    "owning",
                    "vlastníct",
                    "vlastní",
                    "vlastni",
                    "borrow",
                    "borrowing",
                    "borrowed",
                    "referenc",
                    "reference",
                    "move",
                    "moved",
                    "presun",
                    "lifetime",
                    "životnosť",
                    "box<",
                    "rc<",
                    "arc<",
                    "mut ",
                    "mutability",
                    "mutable",
                    "struct",
                    "trait",
                    "generic",
                    "clone",
                    "drop",
                )

                if "python" in lowered:
                    errors.append(
                        f"TASK {index}: Python task is not allowed "
                        "inside a Rust learning goal."
                    )

                if not any(
                    concept in lowered
                    for concept in rust_concepts
                ):
                    errors.append(
                        f"TASK {index}: task is not directly related "
                        "to the Rust learning domain."
                    )

                ownership_focus = (
                    "ownership" in goal_lower
                    or "owning" in goal_lower
                    or "vlastníct" in goal_lower
                    or "vlastní" in goal_lower
                    or "borrow" in goal_lower
                    or "borrowing" in goal_lower
                    or "lifetime" in goal_lower
                    or "move semantics" in goal_lower
                )

                if ownership_focus:
                    ownership_concepts = (
                        "ownership",
                        "owner",
                        "owning",
                        "vlastníct",
                        "vlastní",
                        "vlastni",
                        "borrow",
                        "borrowing",
                        "borrowed",
                        "referenc",
                        "reference",
                        "move",
                        "moved",
                        "presun",
                        "lifetime",
                        "životnosť",
                        "box<",
                        "rc<",
                        "arc<",
                        "mutable borrow",
                        "mutable reference",
                        "mut ",
                        "clone",
                        "drop",
                    )

                    if not any(
                        concept in lowered
                        for concept in ownership_concepts
                    ):
                        errors.append(
                            f"TASK {index}: task does not directly "
                            "exercise the requested Rust ownership "
                            "domain."
                        )

            if "python" in goal_lower:
                if "rust" in lowered:
                    errors.append(
                        f"TASK {index}: Rust task is not allowed "
                        "inside a Python learning goal."
                    )

                python_concepts = (
                    "python",
                    "function",
                    "funkci",
                    "list",
                    "dictionary",
                    "dict",
                    "class",
                    "tried",
                    "exception",
                    "výnim",
                    "string",
                    "slovní",
                    "generátor",
                    "generator",
                )

                if not any(
                    concept in lowered
                    for concept in python_concepts
                ):
                    errors.append(
                        f"TASK {index}: task is not directly related "
                        "to the Python learning domain."
                    )

            if not any(
                marker in lowered
                for marker in (
                    "assert",
                    "aserc",
                    "overiť",
                    "overit",
                    "overenie",
                    "verify",
                    "verification",
                    "validated",
                    "validation",
                )
            ):
                errors.append(
                    f"TASK {index}: task has no explicit measurable "
                    "verification requirement."
                )

        if len(tasks) != 5:
            errors.append(
                f"Expected exactly 5 tasks, received {len(tasks)}."
            )

        return errors

    def plan(self, learning_goal: LearningGoal) -> LearningGoal:
        last_errors: list[str] = []

        for attempt in range(1, 4):
            repair_context = ""

            if last_errors:
                repair_context = (
                    " The previous plan was rejected by the local "
                    "validation gate. The exact rejected conditions are: "
                    + " | ".join(last_errors)
                    + ". Generate a completely new valid plan. "
                    "Do not use the rejected phrases or describe "
                    "compile-time rejection, compiler errors, dangling "
                    "references, memory-state verification, or external "
                    "analysis. Replace every rejected task with a small "
                    "valid Rust runtime exercise that directly tests "
                    "ownership, borrowing, move semantics, lifetimes, "
                    "mutability, Box, Rc, Arc, Clone, or Drop using "
                    "assertions and observable values. "
                    "Do NOT test compiler rejection. "
                    "Do NOT describe code that would fail to compile. "
                    "Do NOT ask the program to prove that invalid Rust "
                    "is rejected. "
                    "Use only valid Rust programs that compile and run "
                    "successfully."
                )

            request = MentorRequest(
                problem=(
                    f"Vytvor učebný plán pre NOVA-EVO.\n"
                    f"Cieľ: {learning_goal.goal}\n"
                    f"Pokus plánovania: {attempt}"
                ),
                context=(
                    "Vytvor presne 5 konkrétnych progresívnych učebných "
                    "úloh, ktoré NOVA-EVO dokáže okamžite vykonať sama "
                    "v aktuálnom lokálnom sandboxe. Každá úloha musí byť "
                    "realizovateľná jediným zdrojovým súborom Python alebo "
                    "Rust. Nesmie vyžadovať internet, sieť, externé API, "
                    "externé balíky, súbory mimo sandboxu, databázu, "
                    "compiler plugin, memory profiler, static analyzer "
                    "ani iný externý nástroj okrem štandardného Python "
                    "interpreteru alebo rustc. Nepoužívaj požiadavky na "
                    "reálne crate-y alebo stovky existujúcich súborov. "
                    "Použi malé syntetické dáta vytvorené priamo programom. "
                    "Každá úloha musí mať konkrétny cieľ overiteľný priamo "
                    "pomocou assertion v tom istom programe. Program musí "
                    "byť pri úspechu kompletne spustiteľný. Pre Rust "
                    "nevytváraj úlohy založené na tom, že compiler odmietne "
                    "neplatný kód, ani úlohy dokazujúce nemožnosť dangling "
                    "referencie alebo nemožnosť prežitia referencie iba "
                    "runtime pozorovaním. Používaj iba platný Rust a "
                    "pozorovateľné výsledky. Preferuj 1 až 10 malých "
                    "syntetických prípadov. Každá úloha musí byť "
                    "reprodukovateľná a musí priamo precvičovať cieľ."
                    + repair_context
                ),
                learning_plan=True,
                candidate_id=f"LEARNING-PLAN-{attempt}",
            )

            proposal = self.mentor.ask(request)

            if not proposal.success:
                last_errors = [
                    proposal.error
                    or "Mentor failed to generate a learning plan."
                ]
                continue

            plan: list[str] = []

            if proposal.raw_response:
                try:
                    raw = json.loads(proposal.raw_response)
                    message = raw.get(
                        "choices",
                        [{}],
                    )[0].get(
                        "message",
                        {},
                    )

                    content = message.get("content", "")
                    plan = self._extract_tasks(content)

                except (
                    json.JSONDecodeError,
                    IndexError,
                    TypeError,
                    AttributeError,
                ):
                    plan = []

            if not plan:
                plan = self._extract_tasks(
                    proposal.proposal
                )

            last_errors = self._validate_plan_tasks(
                plan,
                learning_goal.goal,
            )

            if last_errors:
                continue

            learning_goal.status = "PLANNED"
            learning_goal.plan = plan

            learning_goal.knowledge["status"] = "PLANNED"
            learning_goal.knowledge["current_phase"] = "PLAN"
            learning_goal.knowledge["plan"] = plan
            learning_goal.knowledge["next_task"] = plan[0]
            learning_goal.knowledge.setdefault(
                "completed_tasks",
                [],
            )
            learning_goal.knowledge.setdefault(
                "failed_tasks",
                [],
            )
            learning_goal.knowledge["plan_validation"] = {
                "status": "PASSED",
                "attempt": attempt,
                "errors": [],
            }

            self.goals.save(learning_goal.knowledge)

            return learning_goal

        learning_goal.status = "PLAN_ERROR"
        learning_goal.knowledge["status"] = "PLAN_ERROR"
        learning_goal.knowledge["current_phase"] = "PLAN"
        learning_goal.knowledge["error"] = (
            "Mentor did not produce a plan accepted by the "
            "local validation gate."
        )
        learning_goal.knowledge["plan_validation"] = {
            "status": "REJECTED",
            "attempts": 3,
            "errors": last_errors,
        }

        self.goals.save(learning_goal.knowledge)
        return learning_goal

    def _current_core_config(self) -> dict[str, Any]:
        state_file = self.executor.knowledge.root.parent / "engine" / "evo_state.json"

        try:
            data = json.loads(
                state_file.read_text(encoding="utf-8")
            )
            config = dict(
                data.get("primary_parent_config", {})
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
        except (
            OSError,
            json.JSONDecodeError,
        ):
            return {}

    def execute_next(self, goal: str) -> LearningGoal:
        learning_goal = self.create_goal(goal)

        if not learning_goal.plan:
            learning_goal = self.plan(learning_goal)

        if not learning_goal.plan:
            return learning_goal

        completed = learning_goal.knowledge.setdefault(
            "completed_tasks",
            [],
        )

        failed = learning_goal.knowledge.setdefault(
            "failed_tasks",
            [],
        )

        task = learning_goal.knowledge.get("next_task")

        if not task:
            remaining = [
                item
                for item in learning_goal.plan
                if item not in completed
            ]

            if not remaining:
                learning_goal.status = "COMPLETED"
                learning_goal.knowledge["status"] = "COMPLETED"
                learning_goal.knowledge["current_phase"] = "DONE"
                learning_goal.knowledge["next_task"] = None
                self.goals.save(learning_goal.knowledge)
                return learning_goal

            task = remaining[0]
            learning_goal.knowledge["next_task"] = task

        learning_goal.status = "COMPILING"
        learning_goal.knowledge["status"] = "COMPILING"
        learning_goal.knowledge["current_phase"] = "COMPILE"
        learning_goal.knowledge["last_task"] = task
        self.goals.save(learning_goal.knowledge)

        # Primary path: let the mentor generate a task-specific
        # executable experiment. Never silently replace a rejected
        # mentor task with an unrelated static exercise. That would
        # create false learning signals.
        compiled = self.generator.generate(
            goal=learning_goal.goal,
            task=task,
        )
        generated = compiled.success

        learning_goal.knowledge["last_compiled_task"] = (
            compiled.to_dict()
        )
        learning_goal.knowledge["task_generated_by_mentor"] = (
            generated
        )

        if not compiled.success:
            learning_goal.status = "TASK_UNSUPPORTED"
            learning_goal.knowledge["status"] = "TASK_UNSUPPORTED"
            learning_goal.knowledge["current_phase"] = "COMPILE"

            entry = {
                "task": task,
                "reason": compiled.reason,
            }

            if entry not in failed:
                failed.append(entry)

            learning_goal.knowledge["failed_tasks"] = failed
            learning_goal.knowledge["last_error"] = compiled.reason

            self.goals.save(learning_goal.knowledge)
            return learning_goal

        learning_goal.status = "EXECUTING"
        learning_goal.knowledge["status"] = "EXECUTING"
        learning_goal.knowledge["current_phase"] = "EXECUTE"
        self.goals.save(learning_goal.knowledge)

        result = self.executor.execute(
            goal=learning_goal.goal,
            task=task,
            filename=compiled.filename,
            content=compiled.content,
            command=compiled.command or [],
            expected_output=compiled.expected_output,
        )

        learning_goal.knowledge["last_execution"] = result

        meta_experience = self.meta_learning.record_experience(
            goal=learning_goal.goal,
            task=task,
            core_config=self._current_core_config(),
            success=bool(result["success"]),
            metrics={
                "duration_seconds": result.get(
                    "duration_seconds"
                ),
                "exit_code": result.get(
                    "returncode"
                ),
            },
            strategy=(
                compiled.task_type
                or "learning_executor"
            ),
            source="learning_engine",
        )

        learning_goal.knowledge["last_meta_experience"] = {
            "id": meta_experience["id"],
            "task_class": meta_experience["content"][
                "task_class"
            ],
            "core": meta_experience["content"]["core"],
            "success": meta_experience["content"]["success"],
        }

        if result["success"]:
            try:
                task_training = self.task_training.maybe_train()
            except Exception as exc:
                task_training = {
                    "status": "FAILED",
                    "error": str(exc),
                }

            learning_goal.knowledge[
                "last_task_training"
            ] = task_training

            if task not in completed:
                completed.append(task)

            remaining = [
                item
                for item in learning_goal.plan
                if item not in completed
            ]

            learning_goal.knowledge["completed_tasks"] = completed

            if remaining:
                learning_goal.status = "READY"
                learning_goal.knowledge["status"] = "READY"
                learning_goal.knowledge["current_phase"] = "LEARN"
                learning_goal.knowledge["next_task"] = remaining[0]
            else:
                learning_goal.status = "COMPLETED"
                learning_goal.knowledge["status"] = "COMPLETED"
                learning_goal.knowledge["current_phase"] = "DONE"
                learning_goal.knowledge["next_task"] = None

        else:
            learning_goal.status = "TASK_FAILED"
            learning_goal.knowledge["status"] = "TASK_FAILED"
            learning_goal.knowledge["current_phase"] = "REPAIR"
            learning_goal.knowledge["last_error"] = (
                result.get("stderr")
                or "Learning task execution failed"
            )

            failure_entry = {
                "task": task,
                "experiment_id": result.get("experiment_id"),
                "candidate_id": result.get("candidate_id"),
                "stderr": result.get("stderr", ""),
                "stdout": result.get("stdout", ""),
            }

            failed.append(failure_entry)
            learning_goal.knowledge["failed_tasks"] = failed

        self.goals.save(learning_goal.knowledge)

        return learning_goal

    def learn(self, goal: str) -> LearningGoal:
        learning_goal = self.create_goal(goal)

        if learning_goal.status in {
            "NEW",
            "PLAN_ERROR",
            "MENTOR_ERROR",
        }:
            return self.plan(learning_goal)

        return learning_goal

    def continue_learning(self, goal: str) -> LearningGoal:
        learning_goal = self.create_goal(goal)

        if learning_goal.status == "COMPLETED":
            return learning_goal

        return self.execute_next(goal)


if __name__ == "__main__":
    engine = LearningEngine()
    result = engine.learn("Python")

    print("STATUS:", result.status)
    print("GOAL:", result.goal)
    print("PLAN:")

    for index, step in enumerate(result.plan, 1):
        print(f"{index}. {step}")
