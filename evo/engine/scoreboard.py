"""
Scoreboard: one row that says where NOVA stands - per language, in code,
and about itself - instead of a single validation number.

    val            loss on the dataset's validation split (the number used so far)
    lang.<x>       the same loss split by language tag (sk cs pl en py rs)
    code           held-out code-school exam: tasks solved (greedy) + solution NLL, per level
    creator        log-probability of " roppik" after "Môj tvorca je"
    weakest        the language that is furthest behind its own best value so far

Rows are appended to evo/learning/scoreboard.jsonl; the autopilot reads the
last row to decide how much code practice the next training block gets.

    python -m evo.engine.scoreboard
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BOARD = Path("evo/learning/scoreboard.jsonl")
METHOD = 2  # 2: language carried across rows, same 3200-sequence sample as long_train
LANGS = {4: "sk", 5: "cs", 6: "pl", 7: "en", 8: "py", 9: "rs"}
EOS = 3


def language_of_tokens(seqs: np.ndarray) -> np.ndarray:
    """For every token: id of the language tag it belongs to (0 = unknown).

    A document is `<lang> text <eos>`; sequences are packed from one continuous
    stream, so a row usually continues the document of the row before it."""
    out = np.zeros_like(seqs)
    cur = 0  # rows are consecutive pieces of one token stream: the language carries over
    for i, row in enumerate(seqs):
        for j, t in enumerate(row):
            if t in LANGS:
                cur = int(t)
            out[i, j] = cur
            if t == EOS:
                cur = 0
    return out


@torch.no_grad()
def per_language_loss(model, val: np.ndarray, device: str, batch_size: int = 32,
                      max_sequences: int = 3200) -> dict:  # same sample as long_train's evaluate()
    model = model.to(device).eval()
    val = val[:max_sequences]
    lang = language_of_tokens(val)[:, 1:]
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    total, n = 0.0, 0
    for i in range(0, len(val), batch_size):
        b = torch.from_numpy(val[i:i + batch_size]).long().to(device)
        out = model(b[:, :-1])
        logits = out[0] if isinstance(out, (tuple, list)) else out
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), b[:, 1:].reshape(-1),
                               reduction="none").reshape(b.size(0), -1).cpu().numpy()
        la = lang[i:i + batch_size]
        total += float(loss.sum())
        n += loss.size
        for lid in LANGS:
            m = la == lid
            if m.any():
                sums[lid] = sums.get(lid, 0.0) + float(loss[m].sum())
                counts[lid] = counts.get(lid, 0) + int(m.sum())
    return {"val": round(total / max(n, 1), 4),
            "lang": {LANGS[k]: round(sums[k] / counts[k], 4) for k in sums},
            "lang_tokens": {LANGS[k]: counts[k] for k in counts}}


def code_exam(model, tok) -> dict:
    from evo.learning.code_school import run_tests, solution_nll, write_body
    from evo.learning.code_tasks import task_bank

    model = model.to("cpu").eval()
    exam = [t for t in task_bank() if t.pool == "exam"]
    by_level: dict[int, list[bool]] = {}
    for t in exam:
        ok, _ = run_tests(t, write_body(model, tok, t))
        by_level.setdefault(t.level, []).append(ok)
    solved = sum(sum(v) for v in by_level.values())
    return {"solved": solved, "tasks": len(exam), "pass1": round(solved / max(1, len(exam)), 4),
            "by_level": {str(l): f"{sum(v)}/{len(v)}" for l, v in sorted(by_level.items())},
            "nll": solution_nll(model, tok, exam)}


def weakest_language(row: dict, history: list[dict]) -> str | None:
    """Language whose loss is furthest above its own best so far (relative)."""
    best: dict[str, float] = {}
    for h in history + [row]:
        if h.get("method") != row.get("method"):
            continue  # numbers of an older measuring method are not comparable
        for k, v in (h.get("lang") or {}).items():
            best[k] = min(best.get(k, v), v)
    gaps = {k: (v - best[k]) / best[k] for k, v in (row.get("lang") or {}).items() if best.get(k)}
    if not gaps:
        return None
    k = max(gaps, key=gaps.get)
    return k if gaps[k] > 0.002 else None


def read_board(path: Path = BOARD) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def measure(model, tok, val: np.ndarray, device: str, with_code: bool = True) -> dict:
    from nova.generate import continuation_logprob

    row = {"time": time.time(), "date": time.strftime("%Y-%m-%d %H:%M"), "method": METHOD}
    row.update(per_language_loss(model, val, device))
    if with_code:
        row["code"] = code_exam(model, tok)
    row["creator"] = round(continuation_logprob(model, tok, "Môj tvorca je", " roppik", "sk"), 3)
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-code", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="do not touch the GPU")
    args = ap.parse_args(argv)

    from evo.engine.long_train import load_tokens
    from nova.generate import load_checkpoint_model
    from nova.tokenizer import NovaTokenizer
    from nova.weights import current_weights

    best = json.loads(Path("evo/engine/evo_state.json").read_text(encoding="utf-8"))["best_known"]
    dataset = Path(best["dataset"])
    weights = current_weights(best)
    model, _ = load_checkpoint_model(weights)
    tok = NovaTokenizer.load(dataset / "tokenizer.json")
    val = load_tokens(dataset / "val.txt")
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    if device == "cuda":
        from nova.gpu_guard import ensure_vram

        ensure_vram()  # a teacher model may hold the VRAM
    try:
        row = measure(model, tok, val, device, with_code=not args.no_code)
    except torch.OutOfMemoryError:  # still no room: same sample on CPU (slower, same numbers)
        torch.cuda.empty_cache()
        row = measure(model, tok, val, "cpu", with_code=not args.no_code)
        row["device"] = "cpu-fallback"
    history = read_board()
    row["weights"] = weights
    row["core"] = best.get("candidate")
    row["weakest"] = weakest_language(row, history)
    BOARD.parent.mkdir(parents=True, exist_ok=True)
    with BOARD.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    prev = history[-1] if history else {}
    print(f"SCOREBOARD {row['date']}  val {row['val']}" + (f" (prev {prev['val']})" if prev.get("val") else ""))
    print("  languages: " + "  ".join(f"{k} {v}" for k, v in row["lang"].items()))
    if "code" in row:
        c = row["code"]
        print(f"  code exam: {c['solved']}/{c['tasks']} solved, by level {c['by_level']}, NLL {c['nll']}")
    print(f"  creator recall: {row['creator']}   weakest: {row['weakest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
