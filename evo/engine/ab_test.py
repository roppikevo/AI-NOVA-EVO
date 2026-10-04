"""
A/B test of a training setting, decided by measurement instead of opinion.

Both variants start from the SAME active weights, train the same number of
tokens with --no-activate (the active model is never touched) and are
compared on the same validation data. If B is better than A by more than
the margin, its settings become an autopilot override
(evo/engine/autopilot_overrides.json) - NOVA adjusts its own learning rules,
within what the Creator's scripts allow.

    python -m evo.engine.ab_test --name ctx256 --steps 8000 \
        --a "--batch-size 64" --b "--batch-size 32 --seq-mult 2" \
        --apply '{"--batch-size": "32", "--seq-mult": "2"}'
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

LOG = Path("evo/learning/ab_tests.jsonl")
OVERRIDES = Path("evo/engine/autopilot_overrides.json")
COMMON = ["--warmup", "100", "--lr", "1e-4", "--eval-every", "2000", "--patience", "9", "--max-hours", "1.0",
          "--extra-dirs", "data/teacher_v1,data/web_v1", "--bulk-dir", "data/bulk_v1", "--bulk-frac", "0.7",
          "--code-frac", "0.05", "--no-activate"]


def parse_report(output: str) -> dict:
    marker = "=== LONG TRAIN REPORT ==="
    if marker not in output:
        raise RuntimeError("no report in output: " + output[-400:])
    return json.loads(output[output.rindex(marker) + len(marker):])


def run_variant(extra: str, steps: int, runner=subprocess.run) -> dict:
    cmd = [sys.executable, "-m", "evo.engine.long_train", "--steps", str(steps)] + COMMON + shlex.split(extra)
    p = runner(cmd, capture_output=True, text=True, timeout=5400)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "")[-600:])
    return parse_report(p.stdout)


def decide(a: dict, b: dict, margin: float) -> str:
    return "B" if a["best_val"] - b["best_val"] > margin else "A"


def apply_override(activity: str, settings: dict, path: Path = OVERRIDES) -> None:
    cur = json.loads(path.read_text()) if path.exists() else {}
    cur.setdefault(activity, {}).update(settings)
    path.write_text(json.dumps(cur, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--a", default="", help="extra long_train arguments of the current setting")
    ap.add_argument("--b", required=True, help="extra long_train arguments of the candidate setting")
    ap.add_argument("--apply", default="{}", help="JSON: long_train arguments to override if B wins")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--margin", type=float, default=0.004)
    args = ap.parse_args(argv)

    rec = {"time": time.time(), "date": time.strftime("%Y-%m-%d %H:%M"), "name": args.name,
           "a": args.a, "b": args.b, "steps": args.steps, "margin": args.margin}
    try:
        ra = run_variant(args.a, args.steps)
        rb = run_variant(args.b, args.steps)
        rec.update({"a_val": ra["best_val"], "b_val": rb["best_val"], "start_val": ra["val_start"],
                    "a_hours": ra["hours"], "b_hours": rb["hours"], "winner": decide(ra, rb, args.margin)})
        if rec["winner"] == "B":
            apply_override("long_train", json.loads(args.apply))
            rec["applied"] = json.loads(args.apply)
    except Exception as exc:
        rec.update({"winner": "A", "error": f"{type(exc).__name__}: {exc}"[:500]})
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("AB TEST", json.dumps(rec, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
