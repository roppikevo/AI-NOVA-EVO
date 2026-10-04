from __future__ import annotations

import ast
import importlib.util
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ValidationResult:
    """Result of one NOVA-EVO candidate validation."""

    candidate_id: str
    valid: bool = False
    syntax_ok: bool = False
    imports_ok: bool = False
    build_ok: bool = False
    forward_ok: bool = False
    backward_ok: bool = False
    gradient_ok: bool = False

    parameter_count: int | None = None

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "valid": self.valid,
            "syntax_ok": self.syntax_ok,
            "imports_ok": self.imports_ok,
            "build_ok": self.build_ok,
            "forward_ok": self.forward_ok,
            "backward_ok": self.backward_ok,
            "gradient_ok": self.gradient_ok,
            "parameter_count": self.parameter_count,
            "errors": self.errors,
            "warnings": self.warnings,
            "checks": self.checks,
        }


class ValidationEngine:
    """
    Technical validation boundary for NOVA-EVO candidates.

    The first implementation validates Python source safely before
    any training experiment is allowed to proceed.
    """

    def __init__(
        self,
        project_root: str | Path = ".",
    ) -> None:
        self.project_root = Path(project_root).resolve()

    def validate_syntax(
        self,
        source: str,
    ) -> tuple[bool, str | None]:

        try:
            ast.parse(source)
            return True, None
        except SyntaxError as exc:
            return False, (
                f"SyntaxError: {exc.msg} "
                f"(line {exc.lineno}, column {exc.offset})"
            )

    def validate_python_file(
        self,
        path: str | Path,
    ) -> tuple[bool, str | None]:

        file_path = Path(path).resolve()

        try:
            file_path.relative_to(self.project_root)
        except ValueError:
            return False, "File is outside project root"

        if not file_path.exists():
            return False, f"File does not exist: {file_path}"

        if not file_path.is_file():
            return False, f"Not a regular file: {file_path}"

        try:
            source = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            return False, f"Cannot read file: {exc}"

        return self.validate_syntax(source)

    def validate_imports(
        self,
        module_path: str | Path,
    ) -> tuple[bool, str | None]:

        path = Path(module_path).resolve()

        try:
            path.relative_to(self.project_root)
        except ValueError:
            return False, "Module is outside project root"

        if not path.exists():
            return False, f"Module does not exist: {path}"

        module_name = (
            f"nova_evo_validation_"
            f"{abs(hash(str(path)))}"
        )

        try:
            spec = importlib.util.spec_from_file_location(
                module_name,
                path,
            )

            if spec is None or spec.loader is None:
                return False, "Unable to create import specification"

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module

            spec.loader.exec_module(module)

            return True, None

        except Exception as exc:
            return False, (
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            sys.modules.pop(module_name, None)

    def validate_candidate(
        self,
        candidate_id: str,
        files: list[str | Path],
    ) -> ValidationResult:

        result = ValidationResult(
            candidate_id=candidate_id,
        )

        if not files:
            result.errors.append("No candidate files supplied")
            return result

        for file_path in files:
            syntax_ok, syntax_error = self.validate_python_file(
                file_path
            )

            if not syntax_ok:
                result.errors.append(
                    f"{file_path}: {syntax_error}"
                )
                continue

            result.syntax_ok = True

            imports_ok, import_error = self.validate_imports(
                file_path
            )

            if not imports_ok:
                result.errors.append(
                    f"{file_path}: {import_error}"
                )
                continue

            result.imports_ok = True

        result.valid = (
            result.syntax_ok
            and result.imports_ok
            and not result.errors
        )

        return result

    @staticmethod
    def validate_parameter_count(
        model: Any,
        result: ValidationResult,
    ) -> None:

        try:
            import torch

            parameters = list(model.parameters())

            result.parameter_count = sum(
                parameter.numel()
                for parameter in parameters
            )

            result.checks["trainable_parameters"] = sum(
                parameter.numel()
                for parameter in parameters
                if parameter.requires_grad
            )

            result.build_ok = True

        except Exception as exc:
            result.errors.append(
                f"Parameter validation failed: "
                f"{type(exc).__name__}: {exc}"
            )

    @staticmethod
    def validate_forward_backward(
        model: Any,
        input_ids: Any,
        result: ValidationResult,
    ) -> None:

        try:
            import torch

            model.train()

            output = model(input_ids)

            if hasattr(output, "logits"):
                logits = output.logits
            elif isinstance(output, dict) and "logits" in output:
                logits = output["logits"]
            elif torch.is_tensor(output):
                logits = output
            else:
                raise TypeError(
                    "Model output does not expose logits"
                )

            result.forward_ok = True

            if not torch.isfinite(logits).all():
                raise ValueError(
                    "Forward output contains NaN or Inf"
                )

            loss = logits.float().mean()

            model.zero_grad(set_to_none=True)
            loss.backward()

            result.backward_ok = True

            gradient_count = 0
            invalid_gradients = 0

            for parameter in model.parameters():
                if parameter.grad is None:
                    continue

                gradient_count += 1

                if not torch.isfinite(parameter.grad).all():
                    invalid_gradients += 1

            result.checks["gradient_tensors"] = gradient_count
            result.checks["invalid_gradients"] = invalid_gradients

            result.gradient_ok = (
                gradient_count > 0
                and invalid_gradients == 0
            )

            if not result.gradient_ok:
                result.errors.append(
                    "Gradient validation failed"
                )

        except Exception as exc:
            result.errors.append(
                f"Forward/backward validation failed: "
                f"{type(exc).__name__}: {exc}"
            )

    def full_validation(
        self,
        candidate_id: str,
        files: list[str | Path],
    ) -> ValidationResult:

        result = self.validate_candidate(
            candidate_id,
            files,
        )

        if not result.valid:
            return result

        return result


def self_test() -> None:
    engine = ValidationEngine()

    valid_source = """
def add(a, b):
    return a + b
"""

    invalid_source = """
def broken(
    return 1
"""

    syntax_ok, error = engine.validate_syntax(
        valid_source
    )

    assert syntax_ok
    assert error is None

    syntax_ok, error = engine.validate_syntax(
        invalid_source
    )

    assert not syntax_ok
    assert error is not None

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        test_engine = ValidationEngine(root)

        valid_file = root / "valid.py"

        valid_file.write_text(
            "VALUE = 42\n",
            encoding="utf-8",
        )

        result = test_engine.validate_candidate(
            "VALIDATION-SELFTEST-001",
            [valid_file],
        )

        assert result.valid
        assert result.syntax_ok
        assert result.imports_ok
        assert not result.errors

        invalid_file = root / "invalid.py"

        invalid_file.write_text(
            "def broken(\n",
            encoding="utf-8",
        )

        result = test_engine.validate_candidate(
            "VALIDATION-SELFTEST-002",
            [invalid_file],
        )

        assert not result.valid
        assert not result.imports_ok
        assert result.errors

    print("VALIDATION ENGINE SELFTEST: PASSED")


if __name__ == "__main__":
    self_test()
