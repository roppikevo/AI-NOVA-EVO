from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from evo.engine import sandbox


@dataclass
class RunResult:
    candidate_id: str
    command: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    status: str
    timed_out: bool = False
    workspace: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == "success" and not self.timed_out

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "command": self.command,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "timed_out": self.timed_out,
            "workspace": self.workspace,
            "metadata": self.metadata,
            "success": self.success,
        }


class ExperimentRunner:
    """Execution boundary between NOVA-EVO and the sandbox."""

    def __init__(self, default_timeout: int | None = None) -> None:
        policy = sandbox.load_policy()

        if default_timeout is None:
            default_timeout = int(
                policy["limits"]["max_candidate_runtime_seconds"]
            )

        self.default_timeout = default_timeout

    def create_candidate(self, candidate_id: str):
        return sandbox.create_workspace(candidate_id)

    def write_file(
        self,
        candidate_id: str,
        relative_path: str,
        content: str,
    ):
        return sandbox.write_file(
            candidate_id,
            relative_path,
            content,
        )

    def run(
        self,
        candidate_id: str,
        command: list[str],
        timeout: int | None = None,
    ) -> RunResult:

        if not isinstance(command, list) or not command:
            raise ValueError("command must be a non-empty list")

        effective_timeout = (
            self.default_timeout
            if timeout is None
            else timeout
        )

        started = time.monotonic()

        result = sandbox.run(
            candidate_id,
            command,
            timeout=effective_timeout,
        )

        duration = time.monotonic() - started

        return RunResult(
            candidate_id=candidate_id,
            command=result["command"],
            returncode=result["returncode"],
            stdout=result["stdout"],
            stderr=result["stderr"],
            duration_seconds=duration,
            status=result["status"],
            timed_out=result["status"] == "timeout",
            workspace=result["workspace"],
            metadata={
                "timeout_seconds": effective_timeout,
            },
        )


def self_test() -> None:
    runner = ExperimentRunner(default_timeout=30)

    candidate_id = "RUNNER-SELFTEST-001"

    workspace = runner.create_candidate(candidate_id)

    assert workspace.exists()

    runner.write_file(
        candidate_id,
        "test.py",
        'print("RUNNER_OK")\n',
    )

    result = runner.run(
        candidate_id,
        ["python", "test.py"],
    )

    assert result.success
    assert result.returncode == 0
    assert "RUNNER_OK" in result.stdout
    assert result.duration_seconds >= 0
    assert result.workspace == str(workspace)

    print("EXPERIMENT RUNNER SELFTEST: PASSED")


if __name__ == "__main__":
    self_test()
