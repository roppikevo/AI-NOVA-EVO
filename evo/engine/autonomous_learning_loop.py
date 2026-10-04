from __future__ import annotations

import argparse
import json
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evo.engine.core_evolution_engine import CoreEvolutionEngine


ROOT = Path(__file__).resolve().parents[2]

MEMORY_DIR = (
    ROOT
    / "evo"
    / "core_evolution"
    / "learning"
)

MEMORY_FILE = (
    MEMORY_DIR
    / "learning_memory.json"
)


def now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def atomic_write_json(
    path: Path,
    data: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tmp.replace(path)


def load_memory() -> dict[str, Any]:
    if not MEMORY_FILE.exists():
        return {
            "version": 1,
            "lessons": [],
            "last_updated": None,
        }

    try:
        data = json.loads(
            MEMORY_FILE.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return {
            "version": 1,
            "lessons": [],
            "last_updated": None,
        }

    if not isinstance(
        data,
        dict,
    ):
        return {
            "version": 1,
            "lessons": [],
            "last_updated": None,
        }

    if not isinstance(
        data.get("lessons"),
        list,
    ):
        data["lessons"] = []

    return data


def compact_memory(
    memory: dict[str, Any],
    limit: int = 12,
) -> list[dict[str, Any]]:

    result: list[dict[str, Any]] = []

    for lesson in memory.get(
        "lessons",
        [],
    )[-limit:]:

        if not isinstance(
            lesson,
            dict,
        ):
            continue

        result.append(
            {
                "candidate":
                    lesson.get(
                        "candidate"
                    ),

                "generation":
                    lesson.get(
                        "generation"
                    ),

                "parent":
                    lesson.get(
                        "parent"
                    ),

                "decision":
                    lesson.get(
                        "decision"
                    ),

                "stage":
                    lesson.get(
                        "stage"
                    ),

                "validation_loss":
                    lesson.get(
                        "validation_loss"
                    ),

                "parent_loss":
                    lesson.get(
                        "parent_loss"
                    ),

                "delta":
                    lesson.get(
                        "delta"
                    ),

                "mechanism":
                    lesson.get(
                        "mechanism"
                    ),

                "lesson":
                    lesson.get(
                        "lesson"
                    ),

                "next_direction":
                    lesson.get(
                        "next_direction"
                    ),
            }
        )

    return result


def first_key(
    obj: Any,
    keys: set[str],
) -> Any:

    if isinstance(
        obj,
        dict,
    ):
        for key in keys:
            if (
                key in obj
                and obj[key]
                not in (None, "")
            ):
                return obj[key]

        for value in obj.values():
            found = first_key(
                value,
                keys,
            )

            if found not in (
                None,
                "",
            ):
                return found

    elif isinstance(
        obj,
        list,
    ):
        for value in obj:
            found = first_key(
                value,
                keys,
            )

            if found not in (
                None,
                "",
            ):
                return found

    return None


def extract_candidate_id(
    run_result: Any,
    before_candidates: set[str],
    root: Path,
) -> str | None:

    candidate = first_key(
        run_result,
        {
            "candidate_id",
            "candidate",
        },
    )

    if isinstance(
        candidate,
        dict,
    ):
        candidate = first_key(
            candidate,
            {
                "candidate_id",
                "candidate",
            },
        )

    if candidate:
        return str(candidate)

    candidates_dir = (
        root
        / "evo"
        / "core_evolution"
        / "candidates"
    )

    if candidates_dir.exists():

        created = [
            p
            for p in candidates_dir.iterdir()
            if (
                p.is_dir()
                and p.name
                not in before_candidates
            )
        ]

        if created:
            created.sort(
                key=lambda p:
                    p.stat().st_mtime,
                reverse=True,
            )

            return created[0].name

    return None


def find_candidate_records(
    candidate_id: str | None,
    root: Path,
) -> list[
    tuple[Path, dict[str, Any]]
]:

    if not candidate_id:
        return []

    roots = [
        (
            root
            / "evo"
            / "core_evolution"
            / "results"
        ),
        (
            root
            / "evo"
            / "engine"
            / "sandbox"
        ),
    ]

    records = []

    for base in roots:

        if not base.exists():
            continue

        for path in base.glob(
            "**/*.json"
        ):

            if (
                candidate_id
                not in path.name
                and candidate_id
                not in str(
                    path.parent
                )
            ):
                continue

            try:
                data = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )
            except Exception:
                continue

            if isinstance(
                data,
                dict,
            ):
                records.append(
                    (
                        path,
                        data,
                    )
                )

    records.sort(
        key=lambda item:
            (
                item[0].stat().st_mtime
                if item[0].exists()
                else 0
            ),
        reverse=True,
    )

    return records


def extract_metrics(
    records,
) -> dict[str, Any]:

    for path, data in records:

        robust = data.get(
            "robust"
        )

        if not isinstance(
            robust,
            dict,
        ):
            robust = {}

        evaluation = (
            data.get(
                "evaluation"
            )
        )

        if evaluation is None:
            evaluation = (
                robust.get(
                    "evaluation"
                )
            )

        metrics = data.get(
            "metrics"
        )

        if not isinstance(
            metrics,
            dict,
        ):
            metrics = {}

        robust_metrics = (
            robust.get(
                "metrics"
            )
        )

        if not isinstance(
            robust_metrics,
            dict,
        ):
            robust_metrics = {}

        merged = dict(
            robust_metrics
        )

        merged.update(
            {
                k: v
                for k, v
                in metrics.items()
                if v is not None
            }
        )

        loss = (
            merged.get(
                "validation_loss_mean"
            )
            or data.get(
                "validation_loss_mean"
            )
            or data.get(
                "val_loss"
            )
        )

        action = None

        if isinstance(
            evaluation,
            dict,
        ):
            action = (
                evaluation.get(
                    "action"
                )
                or evaluation.get(
                    "decision"
                )
            )

        elif isinstance(
            evaluation,
            str,
        ):
            action = evaluation

        status = data.get(
            "status"
        )

        if (
            status is None
            and action
            in {
                "PROMOTE",
                "PROMOTED",
                "PASS",
                "REJECT",
            }
        ):
            status = action

        stage = (
            data.get(
                "stage"
            )
            or robust.get(
                "stage"
            )
            or (
                "robust"
                if loss is not None
                else None
            )
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

        training_status = str(
            training.get(
                "status"
            )
            or ""
        ).upper()

        if (
            stage is None
            and training_status == "TRAINING_FAILED"
        ):
            stage = "training"

        evaluation_reason = ""

        evaluation_data = data.get(
            "evaluation"
        )

        if isinstance(
            evaluation_data,
            dict,
        ):
            evaluation_reason = str(
                evaluation_data.get(
                    "reason"
                )
                or ""
            ).lower()

        if (
            stage is None
            and evaluation_reason
            == "runtime_smoke_failed"
        ):
            stage = "runtime_smoke"

        if (
            stage is None
            and evaluation_reason
            == "training_failed"
        ):
            stage = "training"

        # Administrative records are ignored only when they contain
        # no outcome at all.
        if (
            loss is None
            and action is None
            and status is None
            and stage is None
        ):
            continue

        parent_loss = None
        delta = None

        if isinstance(
            evaluation,
            dict,
        ):
            parent_loss = (
                evaluation.get(
                    "parent_loss"
                )
            )

            delta = (
                evaluation.get(
                    "delta"
                )
            )

        training = robust.get(
            "training"
        )

        if not isinstance(
            training,
            dict,
        ):
            training = {}

        return {
            "result_file":
                str(path),

            "stage":
                stage,

            "status":
                status,

            "decision":
                action,

            "validation_loss":
                loss,

            "validation_loss_std":
                merged.get(
                    "validation_loss_std"
                ),

            "parent_loss":
                parent_loss,

            "delta":
                delta,

            "parameters":
                merged.get(
                    "parameters"
                ),

            "seeds":
                (
                    robust.get(
                        "seeds"
                    )
                    or data.get(
                        "seeds"
                    )
                    or []
                ),

            "steps":
                (
                    training.get(
                        "steps"
                    )
                    or data.get(
                        "steps"
                    )
                ),
        }

    return {}


def extract_mechanism(
    plan: str,
) -> str:

    if not plan:
        return ""

    marker = "MECHANISM:"

    if marker in plan:

        text = plan.split(
            marker,
            1,
        )[1]

        for stop in (
            "RATIONALE:",
            "RISKS:",
            "IMPLEMENTATION_PLAN:",
        ):

            if stop in text:
                text = text.split(
                    stop,
                    1,
                )[0]

        return (
            " ".join(
                text.split()
            )[:1200]
        )

    return (
        " ".join(
            plan.split()
        )[:1200]
    )


def build_lesson(
    before_state,
    after_state,
    candidate_id,
    run_result,
    teacher_result,
    records,
):

    metrics = extract_metrics(
        records
    )

    # run_once() is the primary source for the current cycle.
    # Result files are secondary evidence.
    if isinstance(
        run_result,
        dict,
    ):

        if (
            not metrics.get(
                "status"
            )
            and run_result.get(
                "status"
            )
        ):
            metrics["status"] = (
                run_result.get(
                    "status"
                )
            )

        if (
            not metrics.get(
                "decision"
            )
            and run_result.get(
                "decision"
            )
        ):
            metrics["decision"] = (
                run_result.get(
                    "decision"
                )
            )

        if (
            not metrics.get(
                "stage"
            )
            and run_result.get(
                "stage"
            )
        ):
            metrics["stage"] = (
                run_result.get(
                    "stage"
                )
            )

        evaluation = (
            run_result.get(
                "evaluation"
            )
        )

        if isinstance(
            evaluation,
            dict,
        ):

            metrics["decision"] = (
                evaluation.get(
                    "decision"
                )
                or evaluation.get(
                    "action"
                )
                or metrics.get(
                    "decision"
                )
            )

            if (
                metrics.get(
                    "validation_loss"
                )
                is None
            ):
                metrics[
                    "validation_loss"
                ] = evaluation.get(
                    "candidate_loss"
                )

            if (
                metrics.get(
                    "parent_loss"
                )
                is None
            ):
                metrics[
                    "parent_loss"
                ] = evaluation.get(
                    "parent_loss"
                )

            if (
                metrics.get(
                    "delta"
                )
                is None
            ):
                metrics[
                    "delta"
                ] = evaluation.get(
                    "delta"
                )


    parent = (
        before_state.get(
            "primary_parent"
        )
        or before_state.get(
            "active_core"
        )
    )

    generation = (
        before_state.get(
            "current_generation"
        )
    )

    if isinstance(
        teacher_result,
        dict,
    ):

        plan = str(
            teacher_result.get(
                "plan"
            )
            or teacher_result.get(
                "teacher_plan"
            )
            or ""
        )

    else:
        plan = str(
            teacher_result
            or ""
        )

    decision = (
        metrics.get(
            "decision"
        )
        or metrics.get(
            "status"
        )
    )

    if decision == "PROMOTED":
        decision = "PROMOTE"

    stage = metrics.get(
        "stage"
    )

    if decision == "PROMOTE":

        lesson_text = (
            "The proposed core improved the "
            "measured validation objective and "
            "passed the promotion gate."
        )

        next_direction = (
            "Build on the mechanism without "
            "assuming it is causal; test a "
            "complementary mechanism on the "
            "new active parent."
        )

    elif decision == "REJECT":

        lesson_text = (
            "The proposed core did not earn "
            "promotion. Preserve the observed "
            "failure mode and do not repeat "
            "the same change blindly."
        )

        next_direction = (
            "Diagnose why the mechanism failed, "
            "then formulate a different or "
            "corrected hypothesis."
        )

    else:

        lesson_text = (
            "The cycle did not reach a final "
            "promotion/rejection result. "
            "Record the last completed stage "
            "and continue from evidence."
        )

        next_direction = (
            "Inspect the failure stage before "
            "attempting another core."
        )

    if (
        stage
        and stage != "robust"
    ):
        lesson_text += (
            f" The cycle stopped or last "
            f"reported evidence at stage "
            f"'{stage}'."
        )

    return {
        "timestamp":
            now(),

        "candidate":
            candidate_id,

        "generation":
            generation,

        "parent":
            parent,

        "active_core_before":
            before_state.get(
                "active_core"
            ),

        "active_core_after":
            after_state.get(
                "active_core"
            ),

        "decision":
            decision,

        "stage":
            stage,

        "validation_loss":
            metrics.get(
                "validation_loss"
            ),

        "validation_loss_std":
            metrics.get(
                "validation_loss_std"
            ),

        "parent_loss":
            metrics.get(
                "parent_loss"
            ),

        "delta":
            metrics.get(
                "delta"
            ),

        "parameters":
            metrics.get(
                "parameters"
            ),

        "seeds":
            metrics.get(
                "seeds"
            ),

        "steps":
            metrics.get(
                "steps"
            ),

        "result_file":
            metrics.get(
                "result_file"
            ),

        "mechanism":
            extract_mechanism(
                plan
            ),

        "teacher_plan":
            plan[:6000],

        "lesson":
            lesson_text,

        "next_direction":
            next_direction,

        "run_result":
            (
                run_result
                if isinstance(
                    run_result,
                    (
                        dict,
                        list,
                        str,
                        int,
                        float,
                        bool,
                    ),
                )
                else repr(
                    run_result
                )
            ),
    }


def read_state(
    root: Path,
) -> dict[str, Any]:

    paths = [
        (
            root
            / "evo_state.json"
        ),
        (
            root
            / "evo"
            / "engine"
            / "evo_state.json"
        ),
        (
            root
            / "evo"
            / "core_evolution"
            / "autonomous_state.json"
        ),
    ]

    merged: dict[str, Any] = {}

    for path in paths:

        if not path.exists():
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

        merged.update(
            data
        )

    return merged


def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "NOVA-EVO autonomous "
            "learning/evolution loop"
        )
    )

    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    if args.cycles < 1:
        raise SystemExit(
            "--cycles must be >= 1"
        )

    root = ROOT

    memory = load_memory()

    engine = CoreEvolutionEngine()

    # Preserve the real teacher method once.
    # Never wrap a wrapper on later cycles.
    original_ask_teacher = engine.ask_teacher

    for cycle in range(
        1,
        args.cycles + 1,
    ):

        before_state = read_state(
            root
        )

        before_candidates_dir = (
            root
            / "evo"
            / "core_evolution"
            / "candidates"
        )

        if (
            before_candidates_dir.exists()
        ):

            before_candidates = {
                p.name
                for p in
                before_candidates_dir.iterdir()
                if p.is_dir()
            }

        else:
            before_candidates = set()

        captured: dict[str, Any] = {}

        def wrapped_ask_teacher(
            _self:
                CoreEvolutionEngine,
            diagnosis:
                dict[str, Any],
            *call_args,
            **call_kwargs,
        ):

            enriched = dict(
                diagnosis
            )

            compact = compact_memory(
                memory,
                limit=12,
            )

            existing_lessons = (
                diagnosis.get(
                    "lessons",
                    [],
                )
            )

            if not isinstance(
                existing_lessons,
                list,
            ):
                existing_lessons = []

            enriched[
                "learning_memory"
            ] = compact

            enriched[
                "lessons"
            ] = (
                existing_lessons
                + compact
            )

            enriched[
                "evolution_evidence"
            ] = diagnosis.get(
                "previous_results",
                [],
            )

            captured[
                "diagnosis"
            ] = enriched

            result = (
                original_ask_teacher(
                    enriched,
                    *call_args,
                    **call_kwargs,
                )
            )

            captured[
                "teacher_result"
            ] = result

            return result

        engine.ask_teacher = (
            types.MethodType(
                wrapped_ask_teacher,
                engine,
            )
        )

        print()
        print(
            "=" * 72
        )
        print(
            f"NOVA-EVO — "
            f"AUTONOMOUS CYCLE "
            f"{cycle}/{args.cycles}"
        )
        print(
            "=" * 72
        )

        print(
            "ACTIVE BEFORE :",
            before_state.get(
                "active_core"
            ),
        )

        print(
            "GENERATION    :",
            before_state.get(
                "current_generation"
            ),
        )

        try:

            run_result = (
                engine.run_once()
            )

            captured[
                "run_result"
            ] = run_result

            exit_code = 0

        except Exception as exc:

            run_result = {
                "error":
                    type(exc).__name__,

                "message":
                    str(exc),
            }

            captured[
                "run_result"
            ] = run_result

            exit_code = 1

        after_state = read_state(
            root
        )

        candidate_id = (
            extract_candidate_id(
                run_result,
                before_candidates,
                root,
            )
        )

        records = (
            find_candidate_records(
                candidate_id,
                root,
            )
        )

        lesson = build_lesson(
            before_state,
            after_state,
            candidate_id,
            run_result,
            captured.get(
                "teacher_result"
            ),
            records,
        )

        memory["lessons"] = [
            item
            for item
            in memory.get(
                "lessons",
                [],
            )
            if not (
                isinstance(
                    item,
                    dict,
                )
                and item.get(
                    "candidate"
                ) == candidate_id
                and item.get(
                    "result_file"
                )
                == lesson.get(
                    "result_file"
                )
            )
        ]

        memory[
            "lessons"
        ].append(
            lesson
        )

        memory[
            "last_updated"
        ] = now()

        atomic_write_json(
            MEMORY_FILE,
            memory,
        )

        print()
        print(
            "===== CYCLE RESULT ====="
        )

        print(
            "CANDIDATE     :",
            candidate_id,
        )

        print(
            "DECISION      :",
            lesson[
                "decision"
            ],
        )

        print(
            "STAGE         :",
            lesson[
                "stage"
            ],
        )

        print(
            "VALIDATION    :",
            lesson[
                "validation_loss"
            ],
        )

        print(
            "PARENT LOSS   :",
            lesson[
                "parent_loss"
            ],
        )

        print(
            "DELTA         :",
            lesson[
                "delta"
            ],
        )

        print(
            "ACTIVE AFTER  :",
            after_state.get(
                "active_core"
            ),
        )

        print(
            "LESSON SAVED  :",
            str(
                MEMORY_FILE
            ),
        )

        if exit_code != 0:

            print()
            print(
                "CYCLE ERROR   :",
                run_result,
            )

            print(
                "Autonomous loop "
                "stops after an "
                "infrastructure error."
            )

            return exit_code

        if cycle < args.cycles:
            time.sleep(
                max(
                    0.0,
                    args.pause,
                )
            )

    print()
    print(
        "AUTONOMOUS LOOP FINISHED"
    )

    print(
        "LESSONS       :",
        len(
            memory.get(
                "lessons",
                [],
            )
        ),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
