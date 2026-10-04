"""
Code school: NOVA learns to program from executed tests.

One round:
    1. NOVA writes function bodies for tasks of the current level
       (greedy + a few samples, CPU)
    2. every attempt is run against unit tests in a sandboxed subprocess
       (time/memory limits, no imports / file / dunder access)
    3. practice tasks it solved -> its own verified code is reinforced;
       practice tasks it failed -> learns the reference solution, with a
       <think> note about what failed
    4. short fine-tune (+ 5x normal data so language is not forgotten)
    5. exam on held-out tasks (never trained on): pass@1 before/after
    6. keep the weights only if exam pass@1 rises and language barely moves
    7. curriculum: level 2/3 unlock when exam pass@1 of the level >= 60 %

    python -m evo.learning.code_school --steps 200
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from evo.learning.code_tasks import CodeTask, task_bank
from nova.tokenizer import EOS_ID, NovaTokenizer, pack_sequences

STATE = Path("evo/learning/code_school_state.json")
LOG = Path("evo/learning/code_school_log.jsonl")
WEIGHTS_DIR = Path("evo/learning/checkpoints")
UNLOCK_AT = 0.6

# (min exam-NLL drop, max relative val-loss rise) - a bigger step towards
# correct code may cost a little more general loss; long_train recovers it.
KEEP_RULE = [(0.03, 0.005), (0.10, 0.01), (0.25, 0.02)]


def should_keep(gain: float, nll_drop: float, rise: float) -> bool:
    if gain < 0:
        return False
    if gain > 0:
        return rise <= (0.02 if gain >= 0.05 else 0.01)
    return any(nll_drop >= d and rise <= r for d, r in KEEP_RULE)

FORBIDDEN = re.compile(r"\bimport\b|\bopen\s*\(|__|\beval\b|\bexec\b|\bcompile\b|\bglobals\b|\blocals\b|\binput\s*\(")


# ---------------------------------------------------------------- sandbox

def _limits():  # pragma: no cover - runs in the child process
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))


def run_tests(task: CodeTask, body: str, timeout: float = 5.0) -> tuple[bool, str]:
    if FORBIDDEN.search(body):
        return False, "forbidden construct"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "solution.py"
        f.write_text(task.program(body), encoding="utf-8")
        try:
            p = subprocess.run([sys.executable, "-I", "-S", str(f)], cwd=d, capture_output=True,
                               text=True, timeout=timeout, preexec_fn=_limits)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if "ALL_TESTS_PASSED" in p.stdout:
        return True, "ok"
    err = (p.stderr or "").strip().splitlines()
    return False, (err[-1] if err else "no output")[:160]


# --------------------------------------------------------------- writing

@torch.no_grad()
def write_body(model, tok: NovaTokenizer, task: CodeTask, temperature: float = 0.0,
               max_tokens: int = 96) -> str:
    """NOVA continues the function after its signature + docstring.

    The prompt is encoded WITHOUT its final newline: in training text the
    newline and the indentation of the next line are usually one BPE token
    ("\n    "), so ending the prompt on a bare "\n" puts NOVA in a state it
    has never seen and it produces no indented body.
    """
    from nova.stepper import Writer

    model.eval()
    ids = [tok.lang_id("py")] + tok.encode(task.prompt.rstrip("\n"))
    writer = Writer(model, ids)   # NOVA core: constant cost per token; other models: 128-token window
    out: list[int] = []

    def body_text() -> str:
        text = tok.decode(out)
        return text[1:] if text.startswith("\n") else text

    for _ in range(max_tokens):
        logits = writer.logits.float()
        if temperature > 0:
            nxt = int(torch.multinomial(torch.softmax(logits / temperature, -1), 1))
        else:
            nxt = int(logits.argmax())
        if nxt == EOS_ID:
            break
        out.append(nxt)
        # stop when the function ends (a non-indented, non-empty line after the first)
        lines = body_text().split("\n")
        if len(lines) > 1 and any(l and not l.startswith((" ", "\t")) for l in lines[1:]):
            break
        writer.push(nxt)
    body = []
    for line in body_text().split("\n"):
        if line and not line.startswith((" ", "\t")):
            break
        body.append(line)
    return "\n".join(body).rstrip()


def solution_nll(model, tok: NovaTokenizer, tasks: list[CodeTask]) -> float:
    """Partial credit: mean NLL/token of the reference bodies of `tasks`.

    pass@1 is 0 until NOVA writes fully correct code; this number shows
    whether it is getting closer before that happens (lower = better).
    """
    from nova.generate import continuation_logprob

    if not tasks:
        return 0.0
    vals = [-continuation_logprob(model, tok, t.prompt.rstrip("\n"), "\n" + t.solution, "py") for t in tasks]
    return round(sum(vals) / len(vals), 4)


def attempt(model, tok, task: CodeTask, samples: int = 2) -> dict[str, Any]:
    bodies = [write_body(model, tok, task)] + [
        write_body(model, tok, task, temperature=0.7) for _ in range(samples)]
    results = [run_tests(task, b) for b in bodies]
    return {"key": task.key, "name": task.name, "level": task.level, "pool": task.pool,
            "pass1": results[0][0], "passk": any(ok for ok, _ in results),
            "best_body": next((b for b, (ok, _) in zip(bodies, results) if ok), bodies[0]),
            "error": results[0][1]}


def exam_score(results: list[dict], level: int | None = None) -> float:
    rs = [r for r in results if r["pool"] == "exam" and (level is None or r["level"] == level)]
    return round(sum(r["pass1"] for r in rs) / len(rs), 4) if rs else 0.0


# -------------------------------------------------------------- training

def training_sequences(tok: NovaTokenizer, tasks: dict[str, CodeTask], results: list[dict],
                       repeat: int = 3) -> list[list[int]]:
    docs = []
    for r in results:
        if r["pool"] != "practice":
            continue
        t = tasks[r["key"]]
        if r["passk"]:
            code = t.prompt + r["best_body"] + "\n"
            docs += [[tok.lang_id("py")] + tok.encode(code) + [EOS_ID]] * repeat
        else:
            code = t.prompt + t.solution + "\n"
            docs += [[tok.lang_id("py")] + tok.encode(code) + [EOS_ID]] * repeat
            if tok.has_think:
                docs.append(tok.encode_reasoned(
                    t.prompt.rstrip(),
                    f"Moje riešenie zlyhalo ({r['error']}). Správne riešenie:",
                    "\n" + t.solution, "py"))
    random.Random(0).shuffle(docs)
    return list(pack_sequences(docs, 128))


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"level": 1, "rounds": 0, "history": []}


def school_round(model, tok, dataset_dir: Path, state: dict, steps: int = 200,
                 log=print) -> dict[str, Any]:
    from nova.data import TokenSequenceDataset
    from nova.gpu_guard import ensure_vram
    from nova.training import TrainConfig, evaluate, train

    level = state["level"]
    bank = [t for t in task_bank() if t.level <= level]
    tasks = {t.key: t for t in bank}
    exam_tasks = [t for t in bank if t.pool == "exam"]
    model.to("cpu")
    before = [attempt(model, tok, t) for t in bank]
    report = {"time": time.time(), "level": level, "tasks": len(bank),
              "before": {"exam_pass1": exam_score(before), "exam_nll": solution_nll(model, tok, exam_tasks),
                         "practice_passk":
                         round(sum(r["passk"] for r in before if r["pool"] == "practice")
                               / max(1, sum(r["pool"] == "practice" for r in before)), 4),
                         "by_level": {l: exam_score(before, l) for l in range(1, level + 1)}}}
    log(f"code before: {report['before']}")

    seqs = training_sequences(tok, tasks, before)
    lines = (dataset_dir / "train.txt").read_text(encoding="utf-8").splitlines()
    rng = random.Random(len(seqs) + state.get("rounds", 0))
    normal = [list(map(int, l.split())) for l in rng.sample(lines, min(len(lines), 8 * len(seqs) + 500))]
    mix = seqs + normal
    rng.shuffle(mix)
    tmp = WEIGHTS_DIR / "code_tmp.txt"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text("\n".join(" ".join(map(str, s)) for s in mix) + "\n", encoding="utf-8")

    ensure_vram()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    val = TokenSequenceDataset(dataset_dir / "val.txt", seq_len=128)
    loader = torch.utils.data.DataLoader(val, batch_size=32)
    model.to(dev)
    v0 = evaluate(model, loader, dev, max_batches=60)["loss"]
    original = copy.deepcopy(model.state_dict())
    cfg = TrainConfig(seed=7, batch_size=8, learning_rate=5e-5, max_steps=steps,
                      eval_every=steps, log_every=steps, device=dev)
    model, _ = train(model, TokenSequenceDataset(tmp, seq_len=128), val, cfg)
    v1 = evaluate(model, loader, dev, max_batches=60)["loss"]
    tmp.unlink(missing_ok=True)

    model.to("cpu")
    after = [attempt(model, tok, t) for t in bank]
    report["after"] = {"exam_pass1": exam_score(after), "exam_nll": solution_nll(model, tok, exam_tasks),
                       "by_level": {l: exam_score(after, l) for l in range(1, level + 1)}}
    report["val_loss_before"], report["val_loss_after"] = round(v0, 4), round(v1, 4)
    report["train_sequences"] = {"code": len(seqs), "normal": len(normal)}
    gain = report["after"]["exam_pass1"] - report["before"]["exam_pass1"]
    rise = (v1 - v0) / max(v0, 1e-6)
    nll0, nll1 = report["before"]["exam_nll"], report["after"]["exam_nll"]
    nll_drop = (nll0 - nll1) / max(nll0, 1e-6)  # relative, on held-out exam tasks
    report["exam_nll_drop"] = round(nll_drop, 4)
    report["keep_rule"] = KEEP_RULE
    keep = should_keep(gain, nll_drop, rise)
    report["decision"] = "KEEP" if keep else "DISCARD"
    if not keep:
        model.load_state_dict(original)
    level_score = report["after" if keep else "before"]["by_level"].get(level, 0.0)
    if level_score >= UNLOCK_AT and level < 3:
        state["level"] = level + 1
        report["unlocked_level"] = level + 1
    state["rounds"] = state.get("rounds", 0) + 1
    log(f"code after: {report['after']} val {v0:.4f}->{v1:.4f} => {report['decision']}")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=1)
    args = ap.parse_args(argv)

    from nova.generate import load_checkpoint_model
    from nova.weights import current_weights, set_active

    evo = json.loads(Path("evo/engine/evo_state.json").read_text(encoding="utf-8"))
    best = evo["best_known"]
    dataset_dir = Path(best["dataset"])
    ckpt_path = current_weights(best)
    model, ckpt = load_checkpoint_model(ckpt_path)
    tok = NovaTokenizer.load(dataset_dir / "tokenizer.json")
    state = load_state()
    print(f"weights: {ckpt_path} | level {state['level']}")
    for i in range(args.rounds):
        report = school_round(model, tok, dataset_dir, state, steps=args.steps)
        if report["decision"] == "KEEP":
            out = WEIGHTS_DIR / f"codeschool-{time.strftime('%Y%m%d-%H%M%S')}.pt"
            torch.save({**{k: v for k, v in ckpt.items() if k not in ("model_state_dict", "optimizer")},
                        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "code_school": report}, out)
            set_active(str(out), best, "code_school")
            from evo.learning.self_correction import prune_checkpoints
            prune_checkpoints("codeschool-", keep=3)
            report["weights_out"] = str(out)
        state.setdefault("history", []).append({k: report.get(k) for k in ("time", "level", "before", "after", "exam_nll_drop", "decision")})
        state["history"] = state["history"][-50:]
        STATE.write_text(json.dumps(state, indent=2))
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False, default=str) + "\n")
        print(json.dumps({k: report.get(k) for k in ("level", "before", "after", "val_loss_before",
                                                      "val_loss_after", "exam_nll_drop", "decision", "unlocked_level")},
                         ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
