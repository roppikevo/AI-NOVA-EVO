from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/opt/ai/work/nova-evo").resolve()
SANDBOX = (ROOT / "evo/engine/sandbox").resolve()
POLICY_FILE = ROOT / "evo/engine/agent_policy.json"


def load_policy() -> dict:
    with POLICY_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_path(path: str | Path) -> Path:
    p = Path(path)

    if not p.is_absolute():
        p = ROOT / p

    p = p.resolve()

    try:
        p.relative_to(SANDBOX)
    except ValueError:
        raise PermissionError(
            f"Sandbox violation: {p} is outside {SANDBOX}"
        )

    return p


def create_workspace(candidate_id: str) -> Path:
    if not candidate_id or "/" in candidate_id or "\\" in candidate_id:
        raise ValueError("Invalid candidate_id")

    workspace = safe_path(SANDBOX / candidate_id)
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def write_file(candidate_id: str, relative_path: str, content: str) -> Path:
    workspace = create_workspace(candidate_id)

    target = safe_path(workspace / relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    target.write_text(content, encoding="utf-8")

    return target


def run(
    candidate_id: str,
    command: list[str],
    timeout: int | None = None,
) -> dict:

    policy = load_policy()

    if not isinstance(command, list) or not command:
        raise ValueError("command must be a non-empty list")

    workspace = create_workspace(candidate_id)

    if timeout is None:
        timeout = int(
            policy["limits"]["max_candidate_runtime_seconds"]
        )

    start = time.time()

    env = os.environ.copy()

    # Prevent accidental Python environment contamination.
    env["PYTHONUNBUFFERED"] = "1"

    try:
        result = subprocess.run(
            command,
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        status = "success" if result.returncode == 0 else "failed"

        return {
            "candidate_id": candidate_id,
            "status": status,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_seconds": round(time.time() - start, 4),
            "command": command,
            "workspace": str(workspace),
        }

    except subprocess.TimeoutExpired as exc:
        return {
            "candidate_id": candidate_id,
            "status": "timeout",
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "duration_seconds": round(time.time() - start, 4),
            "command": command,
            "workspace": str(workspace),
        }


def main():
    print("=" * 70)
    print("NOVA-EVO SANDBOX EXECUTOR")
    print("=" * 70)

    print(f"ROOT    : {ROOT}")
    print(f"SANDBOX : {SANDBOX}")

    policy = load_policy()

    print(
        "NETWORK :",
        "DISABLED BY POLICY" if not policy["network"]["allowed"] else "ALLOWED",
    )

    candidate = "SELFTEST-001"

    write_file(
        candidate,
        "test.py",
        'print("NOVA-EVO SANDBOX: EXECUTION OK")\n',
    )

    result = run(
        candidate,
        ["python", "test.py"],
        timeout=30,
    )

    print("\nRESULT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    assert result["status"] == "success"
    assert "SANDBOX: EXECUTION OK" in result["stdout"]

    # Security test: direct write outside sandbox must fail.
    try:
        safe_path(ROOT / "evo" / "gen3" / "LOCKED" / "forbidden.txt")
        raise AssertionError("SECURITY FAILURE: protected path accepted")
    except PermissionError:
        print("\nPROTECTED PATH TEST: OK")

    print("\nSANDBOX SELFTEST: PASSED")


if __name__ == "__main__":
    main()
