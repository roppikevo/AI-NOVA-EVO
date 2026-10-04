"""
Entrance exam for a new teacher model: is it served by the router, does it answer in the right
language, does it write code that parses - and how fast is it?

    python -m evo.learning.teacher_check --name Qwen3.6-35B-A3B --enable

  * one explanation in every natural language (sk, cs, pl, en) and one program in Python and Rust,
    judged by the same quality check that filters the teacher corpus
  * speed in tokens per second (taken from the router, or from the clock)
  * --enable: when the exam is passed, the teacher is switched on in teachers.json and the measured
    numbers are written next to it; a teacher that fails stays off

Calling it loads the model in the router (the router keeps one model in memory).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from evo.learning import teacher_corpus as tc
from evo.learning.teacher_registry import REGISTRY_FILE

EXAM = [("explain", "sk", "ako funguje fotosyntéza"), ("explain", "cs", "proč je obloha modrá"),
        ("explain", "pl", "czym jest grawitacja"), ("explain", "en", "how the internet works"),
        ("code", "py", "check whether a string is a palindrome"), ("code", "rs", "compute the greatest common divisor")]
PASS_SHARE = 0.66       # at least 4 of 6 answers must be usable ...
MUST_PASS = ("sk",)     # ... and Slovak, the model's first language, must be one of them


def ask(router: str, name: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    body = json.dumps({"model": name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7,
                       "max_tokens": max_tokens}).encode("utf-8")
    req = urllib.request.Request(f"{router}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    seconds = time.time() - t0
    msg = data["choices"][0]["message"]
    tokens = int((data.get("usage") or {}).get("completion_tokens") or 0)
    speed = (data.get("timings") or {}).get("predicted_per_second") or (tokens / seconds if seconds > 0 else 0.0)
    return {"answer": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or "", "tokens": tokens,
            "seconds": round(seconds, 1), "tok_s": round(float(speed), 2)}


def served(router: str, timeout: float = 10.0) -> list[str]:
    with urllib.request.urlopen(f"{router}/v1/models", timeout=timeout) as r:
        return [m["id"] for m in json.load(r).get("data", [])]


def exam(router: str, name: str, max_tokens: int = 3072, timeout: float = 1800.0) -> dict:
    rows = []
    for kind, lang, topic in EXAM:
        task = tc.Task(name, kind, lang, topic)
        try:
            got = ask(router, name, task.prompt, max_tokens, timeout)
            ok, why = tc.quality_check(task, got["answer"])
            rows.append({"lang": lang, "kind": kind, "ok": ok, "why": why, "tokens": got["tokens"], "seconds": got["seconds"],
                         "tok_s": got["tok_s"], "reasoning": bool(got["reasoning"] or "<think>" in got["answer"]),
                         "sample": tc.strip_think(got["answer"])[:240]})
        except Exception as e:  # a model that cannot be loaded or times out simply fails this question
            rows.append({"lang": lang, "kind": kind, "ok": False, "why": f"{type(e).__name__}: {e}"[:200], "tokens": 0,
                         "seconds": 0.0, "tok_s": 0.0, "reasoning": False, "sample": ""})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    good = [r for r in rows if r["ok"]]
    speeds = [r["tok_s"] for r in rows if r["tok_s"] > 0]
    passed = len(good) >= PASS_SHARE * len(rows) and all(any(r["ok"] for r in rows if r["lang"] == m) for m in MUST_PASS)
    return {"teacher": name, "passed": passed, "ok": len(good), "questions": len(rows),
            "tok_s": round(sum(speeds) / len(speeds), 2) if speeds else 0.0,
            "reasoning": any(r["reasoning"] for r in rows), "failed": {r["lang"]: r["why"] for r in rows if not r["ok"]},
            "date": time.strftime("%Y-%m-%d %H:%M")}


def record(result: dict, enable: bool, path: Path = REGISTRY_FILE) -> bool:
    """Write the measured numbers to the registry; switch the teacher on only after a passed exam."""
    data = json.loads(path.read_text(encoding="utf-8"))
    entry = data["teachers"].get(result["teacher"])
    if entry is None:
        return False
    entry["measured"] = {k: result[k] for k in ("passed", "ok", "questions", "tok_s", "reasoning", "date")}
    if result["tok_s"] > 0:
        entry["speed"] = "fast" if result["tok_s"] >= 12 else "medium" if result["tok_s"] >= 5 else "slow"
    if enable and result["passed"]:
        entry["enabled"] = True
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return bool(entry.get("enabled"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--enable", action="store_true", help="switch the teacher on when it passes")
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args(argv)

    router = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))["router"].rstrip("/")
    try:
        names = served(router)
    except Exception as e:
        print(f"router does not answer: {e}")
        return 2
    if args.name not in names:
        print(f"the router does not serve {args.name} (it serves: {', '.join(names)}) - "
              f"add it to /etc/ai/models.ini and restart llama-server")
        return 2
    result = exam(router, args.name, args.max_tokens, args.timeout)
    enabled = record(result, args.enable)
    print("=== TEACHER CHECK ===")
    print(json.dumps({**result, "enabled": enabled}, indent=1, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
