from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from evo.engine.experiment_record import ExperimentRecord
from evo.engine.experiment_runner import ExperimentRunner
from evo.engine.experiment_store import ExperimentStore
from evo.engine.knowledge_store import KnowledgeStore


class LearningExecutor:
    """Execute verified learning exercises inside the NOVA-EVO sandbox."""

    def __init__(self) -> None:
        self.runner = ExperimentRunner()
        self.experiments = ExperimentStore()
        self.knowledge = KnowledgeStore()

    @staticmethod
    def _slug(value: str) -> str:
        value = value.lower().strip()
        value = re.sub(r"[^a-z0-9]+", "-", value)
        value = value.strip("-")
        return value[:40] or "goal"

    @staticmethod
    def _allowed_command(command: list[str]) -> bool:
        if not command:
            return False

        binary = Path(command[0]).name

        if binary not in {
            "python",
            "python3",
            "rustc",
        }:
            return False

        if any(
            argument in {"-c", "--help", "--version"}
            for argument in command[1:]
        ):
            return binary in {"python", "python3"} and (
                "--version" in command[1:]
            )

        return True

    @staticmethod
    def _rust_binary(command: list[str]) -> str:
        if len(command) != 4:
            raise PermissionError(
                "Rust command must be: rustc <file.rs> -o <binary>"
            )

        if command[0] != "rustc":
            raise PermissionError(
                "Rust execution requires rustc as the first command."
            )

        if command[2] != "-o":
            raise PermissionError(
                "Rust command must use -o."
            )

        binary = command[3]

        if not re.fullmatch(
            r"[A-Za-z0-9_.-]+",
            binary,
        ):
            raise PermissionError(
                "Invalid Rust binary name."
            )

        if binary in {".", ".."}:
            raise PermissionError(
                "Invalid Rust binary name."
            )

        return binary

    def execute(
        self,
        goal: str,
        task: str,
        *,
        filename: str,
        content: str,
        command: list[str],
        expected_output: str | None = None,
    ) -> dict[str, Any]:

        if not goal.strip():
            raise ValueError("goal cannot be empty")

        if not task.strip():
            raise ValueError("task cannot be empty")

        if Path(filename).is_absolute():
            raise PermissionError(
                "Learning files must use relative paths"
            )

        if not self._allowed_command(command):
            raise PermissionError(
                f"Command is not allowed: {command}"
            )

        candidate_id = (
            f"LEARN-{self._slug(goal)}-"
            f"{int(time.time() * 1000)}"
        )

        experiment = ExperimentRecord(
            generation=0,
            candidate_id=candidate_id,
            hypothesis=task,
            objective=f"Autonomous learning task: {task}",
            dataset={"goal": goal},
            mentor={"source": "OxCoder"},
        )

        self.experiments.save(experiment)

        experiment.start()
        self.experiments.update(experiment)

        try:
            self.runner.write_file(
                candidate_id,
                filename,
                content,
            )

            compile_result = self.runner.run(
                candidate_id,
                command,
            )

            result = compile_result

            if command[0] == "rustc" and compile_result.success:
                binary = self._rust_binary(command)

                execution_result = self.runner.run(
                    candidate_id,
                    [f"./{binary}"],
                )

                execution_result.metadata.update(
                    {
                        "phase": "execution",
                        "compile_result": compile_result.to_dict(),
                    }
                )

                execution_result.duration_seconds = round(
                    compile_result.duration_seconds
                    + execution_result.duration_seconds,
                    4,
                )

                result = execution_result

            output_ok = (
                expected_output is None
                or expected_output in result.stdout
            )

            success = result.success and output_ok

            metrics = {
                "exit_code": result.returncode,
                "duration_seconds": result.duration_seconds,
                "stdout_length": len(result.stdout),
                "stderr_length": len(result.stderr),
                "output_check_passed": output_ok,
                "execution_phase": (
                    command[0] == "rustc"
                ),
            }

            if command[0] == "rustc":
                metrics["compile_exit_code"] = (
                    compile_result.returncode
                )
                metrics["compile_duration_seconds"] = (
                    compile_result.duration_seconds
                )
                metrics["compile_succeeded"] = (
                    compile_result.success
                )

            if success:
                experiment.complete(metrics)
            else:
                experiment.fail(
                    "Learning task execution failed",
                    {
                        **metrics,
                        "stdout": result.stdout[-4000:],
                        "stderr": result.stderr[-4000:],
                    },
                )

            self.experiments.update(experiment)

            knowledge_record = self.knowledge.create(
                knowledge_type=(
                    "success" if success else "failure"
                ),
                title=f"Learning task: {task}",
                content={
                    "goal": goal,
                    "task": task,
                    "candidate_id": candidate_id,
                    "experiment_id": experiment.experiment_id,
                    "language": (
                        "rust"
                        if command[0] == "rustc"
                        else "python"
                    ),
                    "task_type": (
                        "rust_generated"
                        if command[0] == "rustc"
                        else "python_generated"
                    ),
                    "filename": filename,
                    "source_code": content,
                    "command": command,
                    "expected_output": expected_output,
                    "actual_output": result.stdout,
                    "result": result.to_dict(),
                    "output_check_passed": output_ok,
                },
                status=(
                    "VALIDATED"
                    if success
                    else "EXPERIMENTAL"
                ),
                source="learning_executor",
                confidence=1.0 if success else 0.0,
            )

            knowledge_path = self.knowledge.save(
                knowledge_record
            )

            return {
                "success": success,
                "candidate_id": candidate_id,
                "experiment_id": experiment.experiment_id,
                "knowledge_id": knowledge_record["id"],
                "knowledge_path": str(knowledge_path),
                "status": experiment.state,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration_seconds": result.duration_seconds,
                "workspace": result.workspace,
                "compile_success": compile_result.success,
                "compile_returncode": compile_result.returncode,
                "execution_performed": command[0] == "rustc",
            }

        except Exception as exc:
            if experiment.state in {
                "CREATED",
                "RUNNING",
                "REPAIRING",
            }:
                experiment.fail(
                    "Executor exception",
                    {
                        "type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                self.experiments.update(experiment)

            raise


def self_test() -> None:
    executor = LearningExecutor()

    result = executor.execute(
        "executor-selftest",
        "Run a deterministic Python learning exercise.",
        filename="main.py",
        content=(
            'value = 21 * 2\n'
            'assert value == 42\n'
            'print("LEARNING_EXECUTOR_OK")\n'
        ),
        command=["python", "main.py"],
        expected_output="LEARNING_EXECUTOR_OK",
    )

    assert result["success"]
    assert result["status"] == "COMPLETED"
    assert "LEARNING_EXECUTOR_OK" in result["stdout"]
    assert result["execution_performed"] is False

    rust_result = executor.execute(
        "executor-rust-selftest",
        "Run a deterministic Rust learning exercise.",
        filename="main.rs",
        content=(
            'fn main() {\n'
            '    let mut value = 20;\n'
            '    let reference = &mut value;\n'
            '    *reference += 22;\n'
            '    assert_eq!(value, 42);\n'
            '    println!("RUST_EXECUTOR_OK");\n'
            '}\n'
        ),
        command=["rustc", "main.rs", "-o", "main"],
        expected_output="RUST_EXECUTOR_OK",
    )

    assert rust_result["success"]
    assert rust_result["status"] == "COMPLETED"
    assert rust_result["compile_success"] is True
    assert rust_result["execution_performed"] is True
    assert "RUST_EXECUTOR_OK" in rust_result["stdout"]

    print("LEARNING EXECUTOR SELFTEST: PASSED")
    print("RUST EXECUTION SELFTEST: PASSED")


if __name__ == "__main__":
    self_test()
