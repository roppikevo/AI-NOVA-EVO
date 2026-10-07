"""
A finer reading of the code exam, for tournaments and reports (the judge's exam - solved or not, 79 tasks - stays
as it is in the constitution).

The same held-out tasks, three numbers instead of one:

    score          0-100: for every task 20 points if the written body is valid Python, plus 80 points times the
                   share of its tests that pass (a function that is right for two tests of three is not worth zero)
    tests_passed   single tests passed, of all tests of all tasks (about 190 instead of 79 yes/no answers)
    nll            how unlikely the model finds the reference solutions (mean negative log-likelihood per token):
                   no writing involved, so it moves smoothly where "solved" jumps

    python -m evo.learning.fine_exam --models a.pt,b.pt
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from evo.learning.code_school import FORBIDDEN, _limits, solution_nll
from evo.learning.code_tasks import CodeTask, task_bank


def compiles(task: CodeTask, body: str) -> bool:
    if not body.strip():
        return False
    try:
        ast.parse(task.prompt + body.rstrip() + "\n")
        return True
    except (SyntaxError, ValueError):
        return False


def tests_passed(task: CodeTask, body: str, timeout: float = 5.0) -> int:
    """How many of the task's tests pass, each in its own sandboxed process (one endless loop costs one test)."""
    if FORBIDDEN.search(body) or not compiles(task, body):
        return 0
    passed = 0
    with tempfile.TemporaryDirectory() as d:
        for i, test in enumerate(task.tests):
            f = Path(d) / f"t{i}.py"
            f.write_text(task.prompt + body.rstrip() + "\n\n\n" + test + "\nprint('TEST_PASSED')\n", encoding="utf-8")
            try:
                p = subprocess.run([sys.executable, "-I", "-S", str(f)], cwd=d, capture_output=True, text=True, timeout=timeout,
                                   preexec_fn=_limits)
            except subprocess.TimeoutExpired:
                continue
            passed += "TEST_PASSED" in p.stdout
    return passed


def grade(tasks: list[CodeTask], bodies: dict[str, str]) -> dict:
    """Points for bodies already written ({task key: body}); tasks without a body count as empty."""
    points, passed, total, solved, valid = 0.0, 0, 0, 0, 0
    for t in tasks:
        body = bodies.get(t.key, "")
        ok = compiles(t, body)
        n = tests_passed(t, body) if ok else 0
        valid += ok
        passed += n
        total += len(t.tests)
        solved += bool(t.tests) and n == len(t.tests)
        points += 20.0 * ok + 80.0 * (n / len(t.tests) if t.tests else 0.0)
    return {"score": round(points / max(len(tasks), 1), 2), "tests_passed": passed, "tests": total, "solved": solved,
            "valid": valid, "tasks": len(tasks)}


def exam_tasks(keys: list[str] | None = None) -> list[CodeTask]:
    bank = task_bank()
    return [t for t in bank if t.key in set(keys)] if keys is not None else [t for t in bank if t.pool == "exam"]


def measure(model, tok, keys: list[str] | None = None, bodies: dict[str, str] | None = None) -> dict:
    """The fine exam of a model on the CPU. `bodies` (greedy, by task key) are reused when the caller already wrote them."""
    from evo.learning.code_school import write_body

    tasks = exam_tasks(keys)
    if bodies is None:
        bodies = {t.key: write_body(model, tok, t) for t in tasks}
    return {**grade(tasks, bodies), "nll": solution_nll(model, tok, tasks)}


def text(name: str, r: dict) -> str:
    return (f"{name}: code score {r['score']:.1f}/100, tests {r['tests_passed']}/{r['tests']}, solved {r['solved']}/{r['tasks']}, "
            f"valid Python {r['valid']}/{r['tasks']}, reference solutions {r['nll']:.4f} nats/token")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="comma list of checkpoints")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    import torch

    from nova.generate import load_checkpoint_model
    from nova.tokenizer import NovaTokenizer

    torch.set_num_threads(args.threads)
    dataset = Path(json.loads(Path("evo/engine/evo_state.json").read_text(encoding="utf-8"))["best_known"]["dataset"])
    tok = NovaTokenizer.load(dataset / "tokenizer.json")
    out = Path("evo/learning/fine_exam.json")
    results = json.loads(out.read_text()) if out.exists() else {}
    for path in [p for p in args.models.split(",") if p]:
        model, _ = load_checkpoint_model(path)
        results[Path(path).stem] = measure(model.float().eval(), tok)
        out.write_text(json.dumps(results, indent=1))
        print(text(Path(path).stem, results[Path(path).stem]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
