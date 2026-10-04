from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class CompiledTask:
    """Safe executable representation of one learning task."""

    success: bool
    task: str
    language: str = ""
    task_type: str = ""
    filename: str = ""
    content: str = ""
    command: list[str] | None = None
    expected_output: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "task": self.task,
            "language": self.language,
            "task_type": self.task_type,
            "filename": self.filename,
            "content": self.content,
            "command": self.command or [],
            "expected_output": self.expected_output,
            "reason": self.reason,
        }


class TaskCompiler:
    """
    Convert a learning task into a safe, executable local exercise.

    This first version supports deterministic Python and Rust exercises.
    Unsupported tasks are rejected explicitly instead of being executed
    as an unrelated placeholder task.
    """

    def compile(self, task: str) -> CompiledTask:
        task = task.strip()

        if not task:
            return CompiledTask(
                success=False,
                task=task,
                reason="Empty learning task",
            )

        language = self._detect_language(task)

        if language == "rust":
            return self._compile_rust(task)

        if language == "python":
            return self._compile_python(task)

        return CompiledTask(
            success=False,
            task=task,
            reason=(
                "Unsupported learning task. "
                "No safe executable recipe exists yet."
            ),
        )

    @staticmethod
    def _detect_language(task: str) -> str:
        lowered = task.lower()

        rust_terms = (
            "rust",
            "rustový",
            "rustov",
            "cargo",
            "rustc",
            "ownership",
            "borrowing",
            "borrows",
            "lifetimes",
            "macro",
            "makrá",
        )

        python_terms = (
            "python",
            "pip",
            "pytest",
            "pythonov",
            "pythonový",
        )

        if any(term in lowered for term in rust_terms):
            return "rust"

        if any(term in lowered for term in python_terms):
            return "python"

        return ""

    def _compile_rust(self, task: str) -> CompiledTask:
        lowered = task.lower()

        if "ownership" in lowered or "vlastníct" in lowered:
            content = """fn main() {
    let text = String::from("nova");
    let moved = text;
    println!("{}", moved);
}
"""

        elif "syntax" in lowered:
            content = """fn main() {
    let value: i32 = 42;
    println!("{}", value);
}
"""

        elif "makr" in lowered or "macro" in lowered:
            content = """macro_rules! nova_value {
    ($value:expr) => {
        $value
    };
}

fn main() {
    let value = nova_value!(42);
    println!("{}", value);
}
"""

        else:
            return CompiledTask(
                success=False,
                task=task,
                language="rust",
                reason=(
                    "No executable Rust recipe exists "
                    "for this task yet."
                ),
            )

        return CompiledTask(
            success=True,
            task=task,
            language="rust",
            task_type="rust_compile",
            filename="main.rs",
            content=content,
            command=[
                "rustc",
                "main.rs",
                "-o",
                "main",
            ],
        )

    def _compile_python(self, task: str) -> CompiledTask:
        lowered = task.lower()

        if "syntax" in lowered:
            content = """def nova_learning_task(value: int) -> int:
    return value * 2

assert nova_learning_task(21) == 42
print("NOVA_PYTHON_TASK_OK")
"""

        elif "funkci" in lowered or "function" in lowered:
            content = """def add(a: int, b: int) -> int:
    return a + b

assert add(20, 22) == 42
print("NOVA_PYTHON_TASK_OK")
"""

        else:
            return CompiledTask(
                success=False,
                task=task,
                language="python",
                reason=(
                    "No executable Python recipe exists "
                    "for this task yet."
                ),
            )

        return CompiledTask(
            success=True,
            task=task,
            language="python",
            task_type="python_test",
            filename="main.py",
            content=content,
            command=[
                "python",
                "main.py",
            ],
            expected_output="NOVA_PYTHON_TASK_OK",
        )


def self_test() -> None:
    compiler = TaskCompiler()

    rust = compiler.compile(
        "Precvič ownership v Rust."
    )

    assert rust.success
    assert rust.language == "rust"
    assert rust.filename == "main.rs"
    assert rust.command == [
        "rustc",
        "main.rs",
        "-o",
        "main",
    ]

    python = compiler.compile(
        "Precvič syntax Pythonu."
    )

    assert python.success
    assert python.language == "python"
    assert python.filename == "main.py"
    assert python.command == [
        "python",
        "main.py",
    ]

    unsupported_rust = compiler.compile(
        "Načítať dataset s Rustovými chybami kompilácie, "
        "extrahovať vzory a vygenerovať 100 syntetických príkladov."
    )

    assert not unsupported_rust.success
    assert unsupported_rust.language == "rust"
    assert "No executable Rust recipe" in unsupported_rust.reason

    unsupported_python = compiler.compile(
        "Analyzuj dataset Pythonových chýb a vytvor klasifikátor."
    )

    assert not unsupported_python.success
    assert unsupported_python.language == "python"
    assert "No executable Python recipe" in unsupported_python.reason

    unsupported = compiler.compile(
        "Nauč sa históriu slovenského jazyka."
    )

    assert not unsupported.success
    assert "Unsupported learning task" in unsupported.reason

    print("TASK COMPILER SELFTEST: PASSED")


if __name__ == "__main__":
    self_test()
