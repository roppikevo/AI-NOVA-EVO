from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from evo.engine.oxcoder_mentor import (
    MentorRequest,
    OxCoderMentor,
)
from evo.learning.task_compiler import CompiledTask


class TaskGenerator:
    """
    Generate, validate, and repair executable learning tasks.

    OxCoder proposes source code.
    TaskGenerator validates the executable specification.
    Invalid proposals are sent back for repair.
    Only validated local Python/Rust tasks can reach the executor.
    """

    MAX_CONTENT_SIZE = 100_000
    MAX_EXPECTED_OUTPUT = 500
    MAX_ATTEMPTS = 3

    FORBIDDEN_PATTERNS = (
        "import os",
        "import subprocess",
        "import socket",
        "import requests",
        "import urllib",
        "from os ",
        "from subprocess",
        "from socket",
        "from pathlib",
        "from urllib",
        "__import__(",
        "eval(",
        "exec(",
        "open(",
        "os.system",
        "os.popen",
        "subprocess.",
        "socket.",
        "std::fs",
        "std::process",
        "std::net",
        "std::env",
        "Command::",
        "TcpStream",
        "File::",
        "unsafe ",
        "include!(",
        "include_bytes!(",
        "include_str!(",
    )

    def __init__(
        self,
        mentor: OxCoderMentor | None = None,
    ) -> None:
        self.mentor = mentor or OxCoderMentor()

    def generate(
        self,
        goal: str,
        task: str,
    ) -> CompiledTask:

        last_reason = "Generation failed."

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            request = self._build_request(
                goal=goal,
                task=task,
                previous_error=(
                    last_reason
                    if attempt > 1
                    else ""
                ),
                attempt=attempt,
            )

            proposal = self.mentor.ask(request)

            if not proposal.success:
                last_reason = (
                    proposal.error
                    or "Executable task generation failed."
                )
                continue

            compiled = self._parse(
                task,
                proposal.proposal,
            )

            if compiled.success:
                return compiled

            last_reason = compiled.reason

        return CompiledTask(
            success=False,
            task=task,
            reason=(
                f"Task generation failed after "
                f"{self.MAX_ATTEMPTS} attempts: "
                f"{last_reason}"
            ),
        )

    @staticmethod
    def _build_request(
        *,
        goal: str,
        task: str,
        previous_error: str,
        attempt: int,
    ) -> MentorRequest:

        repair_context = ""

        if previous_error:
            repair_context = (
                "\nPrevious proposal was rejected.\n"
                f"Validation error: {previous_error}\n"
                "Correct that error in the new proposal.\n"
            )

        return MentorRequest(
            problem=(
                f"Learning goal: {goal}\n"
                f"Learning task: {task}\n"
                f"Generation attempt: {attempt}"
            ),
            context=(
                "Generate exactly one deterministic local learning "
                "exercise.\n"
                "The exercise must test the requested task directly.\n"
                "The program must contain at least one assertion.\n"
                "The program must print a short success marker.\n"
                "EXPECTED_OUTPUT must be exactly that success marker.\n"
                "Use only one local source file.\n"
                "Do not use files, network, subprocesses, shell commands, "
                "environment variables, unsafe code, or external packages.\n"
                "COMMAND must contain no shell chaining.\n"
                "For Rust use exactly: rustc <filename> -o <binary>\n"
                "For Python use exactly: python <filename> or "
                "python3 <filename>."
                f"{repair_context}"
            ),
            candidate_id=f"TASK-GENERATOR-{attempt}",
            executable_task=True,
        )

    def _parse(
        self,
        original_task: str,
        text: str,
    ) -> CompiledTask:

        if not text:
            return CompiledTask(
                success=False,
                task=original_task,
                reason="Empty executable task proposal.",
            )

        if "<think>" in text:
            if "</think>" in text:
                text = re.sub(
                    r"<think>.*?</think>",
                    "",
                    text,
                    flags=re.DOTALL,
                )
            else:
                return CompiledTask(
                    success=False,
                    task=original_task,
                    reason=(
                        "Incomplete <think> block in "
                        "generated response."
                    ),
                )

        text = text.strip()

        language = self._field(
            text,
            "LANGUAGE",
        )

        filename = self._field(
            text,
            "FILENAME",
        )

        command = self._field(
            text,
            "COMMAND",
        )

        expected = self._field(
            text,
            "EXPECTED_OUTPUT",
        )

        content_match = re.search(
            r"CONTENT_BEGIN:\s*\n(.*?)\nCONTENT_END:\s*$",
            text,
            flags=re.DOTALL,
        )

        if content_match is None:
            return CompiledTask(
                success=False,
                task=original_task,
                reason=(
                    "Missing CONTENT_BEGIN/CONTENT_END."
                ),
            )

        content = content_match.group(1)

        if not language or not filename or not command:
            return CompiledTask(
                success=False,
                task=original_task,
                reason=(
                    "Missing required executable-task fields."
                ),
            )

        language = language.lower().strip()
        filename = filename.strip()
        command = command.strip()
        expected = expected.strip()

        if not expected or expected.upper() == "NONE":
            return CompiledTask(
                success=False,
                task=original_task,
                reason=(
                    "Generated tasks must provide "
                    "EXPECTED_OUTPUT."
                ),
            )

        if len(expected) > self.MAX_EXPECTED_OUTPUT:
            return CompiledTask(
                success=False,
                task=original_task,
                reason="Expected output is too long.",
            )

        if len(content.encode("utf-8")) > self.MAX_CONTENT_SIZE:
            return CompiledTask(
                success=False,
                task=original_task,
                reason="Generated source is too large.",
            )

        path_error = self._validate_filename(
            filename,
        )

        if path_error:
            return CompiledTask(
                success=False,
                task=original_task,
                reason=path_error,
            )

        command_error = self._validate_command(
            language,
            filename,
            command,
        )

        if command_error:
            return CompiledTask(
                success=False,
                task=original_task,
                reason=command_error,
            )

        objective_error = self._validate_objective(
            original_task,
            language,
            content,
        )

        if objective_error:
            return CompiledTask(
                success=False,
                task=original_task,
                reason=objective_error,
            )

        source_error = self._validate_source(
            language,
            content,
            expected,
        )

        if source_error:
            return CompiledTask(
                success=False,
                task=original_task,
                reason=source_error,
            )

        compile_error = self._validate_compilation(
            language=language,
            filename=filename,
            content=content,
            command=command,
        )

        if compile_error:
            return CompiledTask(
                success=False,
                task=original_task,
                reason=compile_error,
            )

        return CompiledTask(
            success=True,
            task=original_task,
            language=language,
            task_type=(
                "python_generated"
                if language == "python"
                else "rust_generated"
            ),
            filename=filename,
            content=content,
            command=command.split(),
            expected_output=expected,
        )

    @staticmethod
    def _field(
        text: str,
        name: str,
    ) -> str:

        match = re.search(
            rf"^{re.escape(name)}:\s*(.+)$",
            text,
            flags=re.MULTILINE,
        )

        if match is None:
            return ""

        return match.group(1).strip()

    @staticmethod
    def _validate_filename(
        filename: str,
    ) -> str | None:

        if "/" in filename or "\\" in filename:
            return "Subdirectories are not allowed."

        if filename in {".", ".."}:
            return "Invalid filename."

        if ".." in filename:
            return "Parent traversal is not allowed."

        if not re.fullmatch(
            r"[A-Za-z0-9_.-]+",
            filename,
        ):
            return (
                "Filename contains forbidden characters."
            )

        return None

    @staticmethod
    def _validate_command(
        language: str,
        filename: str,
        command: str,
    ) -> str | None:

        forbidden = (
            "&&",
            "||",
            ";",
            "|",
            ">",
            "<",
            "$(",
            "`",
        )

        if any(
            token in command
            for token in forbidden
        ):
            return (
                "Command contains forbidden shell operators."
            )

        parts = command.split()

        if language == "python":
            if len(parts) != 2:
                return (
                    "Python command must have exactly "
                    "2 arguments."
                )

            if parts[0] not in {
                "python",
                "python3",
            }:
                return "Only python/python3 is allowed."

            if parts[1] != filename:
                return (
                    "Python command filename mismatch."
                )

            if not filename.endswith(".py"):
                return (
                    "Python task must use a .py file."
                )

            return None

        if language == "rust":
            if len(parts) != 4:
                return (
                    "Rust command must be: "
                    "rustc <file.rs> -o <binary>"
                )

            if parts[0] != "rustc":
                return "Only rustc is allowed."

            if parts[1] != filename:
                return (
                    "Rust command filename mismatch."
                )

            if parts[2] != "-o":
                return "Rust command must use -o."

            if not re.fullmatch(
                r"[A-Za-z0-9_.-]+",
                parts[3],
            ):
                return "Invalid Rust binary name."

            if not filename.endswith(".rs"):
                return (
                    "Rust task must use a .rs file."
                )

            return None

        return (
            f"Unsupported executable language: "
            f"{language}"
        )

    @staticmethod
    def _validate_compilation(
        *,
        language: str,
        filename: str,
        content: str,
        command: str,
        timeout_seconds: int = 20,
    ) -> str | None:
        """Compile generated source without executing it."""

        try:
            with tempfile.TemporaryDirectory(
                prefix="nova-task-"
            ) as temp_dir:
                root = Path(temp_dir)

                source_path = root / filename
                source_path.write_text(
                    content,
                    encoding="utf-8",
                )

                parts = command.split()

                if language == "python":
                    compiler = [
                        sys.executable,
                        "-m",
                        "py_compile",
                        filename,
                    ]

                elif language == "rust":
                    compiler = [
                        "rustc",
                        filename,
                        "-o",
                        parts[3],
                    ]

                else:
                    return (
                        f"Unsupported executable language: "
                        f"{language}"
                    )

                result = subprocess.run(
                    compiler,
                    cwd=root,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                    shell=False,
                    env={
                        "PATH":
                            "/usr/local/sbin:"
                            "/usr/local/bin:"
                            "/usr/sbin:"
                            "/usr/bin:"
                            "/sbin:"
                            "/bin"
                    },
                )

                if result.returncode != 0:
                    stderr = (result.stderr or "").strip()
                    stdout = (result.stdout or "").strip()

                    detail = (
                        stderr
                        or stdout
                        or (
                            "compiler exited with code "
                            f"{result.returncode}"
                        )
                    )

                    detail = detail[-2000:]

                    return (
                        f"Generated {language} task failed "
                        f"compilation: {detail}"
                    )

                return None

        except subprocess.TimeoutExpired:
            return (
                f"Generated {language} task compilation "
                f"timed out after {timeout_seconds} seconds."
            )

        except FileNotFoundError as exc:
            return (
                f"Required compiler is unavailable: "
                f"{exc.filename}"
            )

        except OSError as exc:
            return f"Compilation infrastructure error: {exc}"

    @staticmethod
    def _strip_comments(content: str) -> str:
        content = re.sub(
            r"/\\*.*?\\*/",
            "",
            content,
            flags=re.DOTALL,
        )

        content = re.sub(
            r"//.*?$",
            "",
            content,
            flags=re.MULTILINE,
        )

        return content

    @classmethod
    def _validate_objective(
        cls,
        task: str,
        language: str,
        content: str,
    ) -> str | None:
        """Check that generated code contains concrete evidence for the task."""

        task_lower = task.lower()

        unsupported = (
            "no longer usable",
            "cannot outlive",
            "must fail to compile",
            "must not compile",
            "compilation error count",
            "count compilation errors",
            "compile-time error",
            "static analyzer",
            "memory profiler",
            "real crate",
            "external crate",
            "existing files",
            "hundreds of files",
        )

        for phrase in unsupported:
            if phrase in task_lower:
                return (
                    "Learning objective requires an unsupported "
                    f"compile-time or external capability: {phrase}"
                )

        code = cls._strip_comments(content)

        # Concrete identifiers/code fragments written in backticks
        # by the task must really occur in executable source, not only
        # inside comments.
        literals = re.findall(r"`([^`]+)`", task)

        for literal in literals:
            if literal not in code:
                return (
                    "Generated source does not contain required "
                    f"task element: `{literal}`"
                )

        if language == "rust":
            if "struct " in task_lower and "struct " not in code:
                return (
                    "Rust task requires a struct, but no struct "
                    "definition was found."
                )

            if (
                "method" in task_lower
                and "impl " not in code
            ):
                return (
                    "Rust task requires a method, but no impl block "
                    "was found."
                )

            if (
                "lifetime" in task_lower
                and not re.search(
                    r"&'?[a-zA-Z_][a-zA-Z0-9_]*",
                    code,
                )
            ):
                return (
                    "Rust lifetime task does not contain an explicit "
                    "lifetime reference."
                )

            if (
                "borrowing" in task_lower
                and "&" not in code
            ):
                return (
                    "Rust borrowing task does not contain a borrow."
                )

            if (
                "mutable" in task_lower
                or "mutability" in task_lower
            ) and "mut" not in code:
                return (
                    "Rust mutability task does not contain `mut`."
                )

            if (
                "ownership" in task_lower
                and "Box<" in task_lower
                and "Box<" not in code
            ):
                return (
                    "Rust ownership task requires Box usage, "
                    "but no Box type was found."
                )

        return None

    @classmethod
    def _validate_source(
        cls,
        language: str,
        content: str,
        expected_output: str,
    ) -> str | None:

        lowered = content.lower()

        for pattern in cls.FORBIDDEN_PATTERNS:
            if pattern.lower() in lowered:
                return (
                    f"Generated source contains "
                    f"forbidden pattern: {pattern}"
                )

        if language == "python":
            if "assert " not in content:
                return (
                    "Python learning task must contain "
                    "at least one assertion."
                )

        elif language == "rust":
            if not any(
                token in content
                for token in (
                    "assert!(",
                    "assert_eq!(",
                    "assert_ne!(",
                )
            ):
                return (
                    "Rust learning task must contain "
                    "at least one assertion."
                )

        else:
            return (
                f"Unsupported executable language: "
                f"{language}"
            )

        if expected_output not in content:
            return (
                "EXPECTED_OUTPUT must be produced "
                "by the generated program."
            )

        return None


def self_test() -> None:
    generator = TaskGenerator()

    invalid_rust = generator._parse(
        "Rust invalid compilation",
        """
LANGUAGE: rust
FILENAME: main.rs
COMMAND: rustc main.rs -o main
EXPECTED_OUTPUT: NOVA_TASK_OK
CONTENT_BEGIN:
fn main() {
    let mut numbers = vec![1, 2, 3];
    let mut_ref = &mut numbers;
    *mut_ref.push(4);
    assert_eq!(numbers, vec![1, 2, 3, 4]);
    println!("NOVA_TASK_OK");
}
CONTENT_END:
""".strip(),
    )

    assert not invalid_rust.success
    assert "failed compilation" in invalid_rust.reason

    irrelevant_rust = generator._parse(
        "Rust task requires `BoxOwner` and `Box<i32>` ownership",
        """
LANGUAGE: rust
FILENAME: main.rs
COMMAND: rustc main.rs -o main
EXPECTED_OUTPUT: NOVA_TASK_OK
CONTENT_BEGIN:
fn main() {
    let value = 42;
    assert_eq!(value, 42);
    println!("NOVA_TASK_OK");
}
CONTENT_END:
""".strip(),
    )

    assert not irrelevant_rust.success
    assert "required task element" in irrelevant_rust.reason

    unsupported_lifetime = generator._parse(
        "Rust task asserting that a returned reference cannot outlive the original",
        """
LANGUAGE: rust
FILENAME: main.rs
COMMAND: rustc main.rs -o main
EXPECTED_OUTPUT: NOVA_TASK_OK
CONTENT_BEGIN:
fn main() {
    let value = "nova";
    assert_eq!(value, "nova");
    println!("NOVA_TASK_OK");
}
CONTENT_END:
""".strip(),
    )

    assert not unsupported_lifetime.success
    assert "unsupported compile-time" in unsupported_lifetime.reason

    valid_rust = generator._parse(
        "Rust assertion",
        """
LANGUAGE: rust
FILENAME: main.rs
COMMAND: rustc main.rs -o main
EXPECTED_OUTPUT: NOVA_TASK_OK
CONTENT_BEGIN:
fn main() {
    let value = 21 * 2;
    assert_eq!(value, 42);
    println!("NOVA_TASK_OK");
}
CONTENT_END:
""".strip(),
    )

    assert valid_rust.success
    assert valid_rust.language == "rust"
    assert valid_rust.command == [
        "rustc",
        "main.rs",
        "-o",
        "main",
    ]

    valid_python = generator._parse(
        "Python assertion",
        """
LANGUAGE: python
FILENAME: main.py
COMMAND: python main.py
EXPECTED_OUTPUT: NOVA_TASK_OK
CONTENT_BEGIN:
value = 21 * 2
assert value == 42
print("NOVA_TASK_OK")
CONTENT_END:
""".strip(),
    )

    assert valid_python.success
    assert valid_python.language == "python"

    blocked_shell = generator._parse(
        "Shell test",
        """
LANGUAGE: rust
FILENAME: main.rs
COMMAND: rustc main.rs -o main && ./main
EXPECTED_OUTPUT: NOVA_TASK_OK
CONTENT_BEGIN:
fn main() {
    assert_eq!(1, 1);
    println!("NOVA_TASK_OK");
}
CONTENT_END:
""".strip(),
    )

    assert not blocked_shell.success

    blocked_source = generator._parse(
        "Dangerous source",
        """
LANGUAGE: python
FILENAME: main.py
COMMAND: python main.py
EXPECTED_OUTPUT: NOVA_TASK_OK
CONTENT_BEGIN:
import os
assert True
print("NOVA_TASK_OK")
CONTENT_END:
""".strip(),
    )

    assert not blocked_source.success

    missing_assertion = generator._parse(
        "Missing assertion",
        """
LANGUAGE: python
FILENAME: main.py
COMMAND: python main.py
EXPECTED_OUTPUT: NOVA_TASK_OK
CONTENT_BEGIN:
print("NOVA_TASK_OK")
CONTENT_END:
""".strip(),
    )

    assert not missing_assertion.success

    think_removed = generator._parse(
        "Think filtering",
        """
<think>
internal reasoning
</think>
LANGUAGE: python
FILENAME: main.py
COMMAND: python main.py
EXPECTED_OUTPUT: NOVA_TASK_OK
CONTENT_BEGIN:
value = 42
assert value == 42
print("NOVA_TASK_OK")
CONTENT_END:
""".strip(),
    )

    assert think_removed.success

    malformed = generator._parse(
        "Malformed task",
        """
LANGUAGE: python
FILENAME: main.py
COMMAND: python main.py
EXPECTED_OUTPUT: NOVA_TASK_OK
CONTENT_BEGIN:
value = 42
assert value == 42
CONTENT_END:
""".strip(),
    )

    assert not malformed.success

    print(
        "TASK GENERATOR SELFTEST: PASSED"
    )


if __name__ == "__main__":
    self_test()
