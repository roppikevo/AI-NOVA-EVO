"""
Autopilot: NOVA builds its own learning schedule.

Runs activities one after another (never two at once) and chooses the
next one by measured benefit per hour (UCB: exploit what helped,
still try the others now and then):

    self_correct  quiz -> mistakes -> fix -> held-out exam (+ free web lookup)
    evolution     core-evolution cycles (mentor Devstral -> Qwen -> OxCoder)
    teachers      light teacher run (small teachers, low priority, time box)
    long_train    continue training the active weights (+ teacher/web texts + bulk web corpus)
                  (+ code-school practice 5 %; fetches the next web batch in parallel on CPU)

Benefit of an activity (normalised, >= 0):
    self_correct  exam-accuracy gain of kept rounds
    evolution     number of promoted cores
    teachers      accepted items / 40
    long_train    validation-loss drop x 100 (the model itself getting better counts most)

Stops on --hours or when evo/engine/STOP_AUTOPILOT exists.
State: evo/engine/autopilot_state.json   Log: evo/engine/autopilot_log.jsonl

    python -m evo.engine.autopilot --hours 5.5
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

STATE = Path("evo/engine/autopilot_state.json")
LOG = Path("evo/engine/autopilot_log.jsonl")
STOP = Path("evo/engine/STOP_AUTOPILOT")
OVERRIDES = Path("evo/engine/autopilot_overrides.json")

PY = sys.executable

# Measured on 2026-10-01: long training on the bulk web corpus gave the biggest gain by far
# (val 3.80 -> 3.52 in one night); standalone code school always cost too much language
# (code practice now rides inside long_train via --code-frac); evolution rarely passes.
ACTIVITIES: dict[str, dict[str, Any]] = {
    "self_correct": {"cmd": [PY, "-m", "evo.learning.self_correction", "--rounds", "2", "--steps", "200", "--web"],
                     "timeout_h": 1.0, "nice": 10},
    "evolution": {"cmd": [PY, "-m", "evo.engine.core_evolution_loop", "--cycles", "2"],
                  "timeout_h": 1.5, "nice": 0},
    "teachers": {"cmd": [PY, "-m", "evo.learning.teacher_corpus", "--per-task", "3", "--max-minutes", "30"],
                 "timeout_h": 0.75, "nice": 19, "seeded": True},
    # GPU: bigger batches (GPU was ~45 % busy at batch 16); CPU/network in parallel: next web batch
    "long_train": {"cmd": [PY, "-m", "evo.engine.long_train", "--steps", "12000", "--batch-size", "64",
                           "--warmup", "100", "--lr", "1e-4", "--eval-every", "2000", "--patience", "4",
                           "--max-hours", "1.0", "--extra-dirs", "data/teacher_v1,data/web_v1",
                           "--bulk-dir", "data/bulk_v1", "--bulk-frac", "0.7", "--code-frac", "0.05"],
                   "timeout_h": 1.3, "nice": 0,
                   "parallel": [PY, "-m", "evo.corpus.bulk_web", "--mchars", "sk=40,cs=25,pl=25,en=40"],
                   "post": [PY, "-m", "evo.engine.scoreboard"]},
}


# ------------------------------------------------------------------ metrics

def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def _bulk_tokens() -> int:
    try:
        import numpy as np

        return int(sum(np.load(p, mmap_mode="r").shape[0] for p in Path("data/bulk_v1").glob("*.npy")) * 128)
    except Exception:
        return 0


def snapshot() -> dict[str, Any]:
    """Counters read before/after an activity to measure its benefit."""
    teacher_items = sum(len(_jsonl(p)) for p in Path("data/teacher_v1").glob("*.jsonl"))
    decisions = _jsonl(Path("evo/core_evolution/decisions.jsonl"))
    rounds = _jsonl(Path("evo/learning/self_correction_log.jsonl"))
    progress = _jsonl(Path("evo/learning/long_train_progress.jsonl"))
    return {
        "teacher_items": teacher_items,
        "promotions": sum(1 for d in decisions if d.get("outcome") == "PASS"),
        "sc_rounds": len(rounds),
        "lt_rows": len(progress),
        "cs_rounds": len(_jsonl(Path("evo/learning/code_school_log.jsonl"))),
        "bulk_tokens": _bulk_tokens(),
    }


def benefit(name: str, before: dict, after: dict) -> float:
    if name == "teachers":
        return max(0, after["teacher_items"] - before["teacher_items"]) / 40
    if name == "web_bulk":
        # new data is only worth something once it is learned: small, capped value
        return min(1.0, max(0, after.get("bulk_tokens", 0) - before.get("bulk_tokens", 0)) / 50e6)
    if name == "evolution":
        return float(max(0, after["promotions"] - before["promotions"]))
    if name == "self_correct":
        rounds = _jsonl(Path("evo/learning/self_correction_log.jsonl"))[before["sc_rounds"]:]
        return sum(max(0.0, r["after"]["exam"] - r["before"]["exam"])
                   for r in rounds if r.get("decision") == "KEEP" and "after" in r) * 10
    if name == "code_school":
        rounds = _jsonl(Path("evo/learning/code_school_log.jsonl"))[before.get("cs_rounds", 0):]
        return sum(max(0.0, r["after"]["exam_pass1"] - r["before"]["exam_pass1"])
                   + max(0.0, r.get("exam_nll_drop", 0.0))
                   for r in rounds if r.get("decision") == "KEEP" and "after" in r) * 10
    if name == "long_train":
        rows = _jsonl(Path("evo/learning/long_train_progress.jsonl"))[before["lt_rows"]:]
        starts = [r for r in rows if r.get("start")]
        evals = [r for r in rows if not r.get("start")]
        if starts and evals:
            # real gain over the weights we started from (not recovery from a warm-up bump)
            return max(0.0, starts[0]["val_loss"] - min(r["val_loss"] for r in evals)) * 100
        return 0.0
    return 0.0


# ----------------------------------------------------------------- choosing

def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"stats": {n: {"runs": 0, "benefit": 0.0, "hours": 0.0} for n in ACTIVITIES},
            "last": [], "seed": 100}


DISCOUNT = 0.9   # old results fade: what helped (or failed) 30 runs ago should not decide today
MIN_RUNS = 0.2   # an activity whose (discounted) run count fell this low is tried again


def discount(state: dict, gamma: float = DISCOUNT) -> None:
    for s in state["stats"].values():
        for k in ("runs", "benefit", "hours"):
            s[k] = s.get(k, 0.0) * gamma


def choose(state: dict, exploration: float = 0.6) -> str:
    stats = state["stats"]
    total = sum(stats.get(n, {}).get("runs", 0) for n in ACTIVITIES) + 1
    # every activity runs at least once (and again once its history has faded)
    for name in ACTIVITIES:
        if stats.get(name, {}).get("runs", 0) < MIN_RUNS:
            return name
    last = state.get("last", [])[-2:]

    def ucb(name: str) -> float:
        s = stats[name]
        rate = s["benefit"] / max(s["hours"], 0.05)
        bonus = exploration * math.sqrt(math.log(total + 1) / s["runs"])
        penalty = 0.5 if last.count(name) == 2 else 0.0  # avoid 3x the same in a row
        return rate + bonus - penalty

    return max(ACTIVITIES, key=ucb)


def tune(name: str, cmd: list[str], board: Path = Path("evo/learning/scoreboard.jsonl")) -> list[str]:
    """Adapt the next training block: A/B-test winners (overrides) + the last scoreboard row."""
    cmd = list(cmd)
    if OVERRIDES.exists():
        try:
            for flag, value in json.loads(OVERRIDES.read_text()).get(name, {}).items():
                if flag in cmd:
                    cmd[cmd.index(flag) + 1] = str(value)
                else:
                    cmd += [flag, str(value)]
        except (json.JSONDecodeError, AttributeError):
            pass
    if name != "long_train":
        return cmd
    rows = _jsonl(board)
    if not rows:
        return cmd
    row = rows[-1]
    pass1 = (row.get("code") or {}).get("pass1")
    if pass1 is not None and "--code-frac" in cmd:
        # little code solved -> more practice; most solved -> language gets the room back
        cmd[cmd.index("--code-frac") + 1] = "0.10" if pass1 < 0.3 else ("0.05" if pass1 < 0.7 else "0.03")
    if row.get("weakest") in ("sk", "cs", "pl", "en"):
        cmd += ["--boost-lang", row["weakest"]]
    return cmd


def run_activity(name: str, state: dict, runner: Callable = subprocess.run, log=print) -> dict:
    spec = ACTIVITIES[name]
    cmd = tune(name, list(spec["cmd"]))
    if spec.get("seeded"):
        state["seed"] = state.get("seed", 100) + 1
        cmd += ["--seed", str(state["seed"])]
    if spec.get("nice"):
        cmd = ["nice", "-n", str(spec["nice"])] + cmd
    before = snapshot()
    t0 = time.time()
    side = None
    if spec.get("parallel") and runner is subprocess.run:
        # CPU/network side job (e.g. fetching the next web batch) while the GPU trains
        side = subprocess.Popen(["nice", "-n", "19"] + list(spec["parallel"]),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        proc = runner(cmd, timeout=spec["timeout_h"] * 3600, capture_output=True, text=True)
        rc = proc.returncode
        tail = (proc.stdout or "")[-1500:] + (proc.stderr or "")[-800:]
    except subprocess.TimeoutExpired:
        rc, tail = "timeout", ""
    if side is not None:
        try:
            side.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            side.kill()
    if spec.get("post") and rc == 0 and runner is subprocess.run:
        try:  # measurement after the block (scoreboard); never fails the activity
            subprocess.run(["nice", "-n", "10"] + list(spec["post"]), timeout=900,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    hours = (time.time() - t0) / 3600
    gain = benefit(name, before, snapshot()) if rc == 0 else 0.0
    discount(state)
    s = state["stats"].setdefault(name, {"runs": 0, "benefit": 0.0, "hours": 0.0})
    s["runs"] += 1
    s["benefit"] += gain
    s["hours"] += hours
    state["last"] = (state.get("last", []) + [name])[-10:]
    rec = {"time": time.time(), "activity": name, "rc": rc, "hours": round(hours, 3),
           "benefit": round(gain, 4), "tail": tail[-600:]}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"[autopilot] {name}: rc={rc} {hours:.2f} h benefit={gain:.3f}")
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=5.5)
    args = ap.parse_args(argv)
    deadline = time.time() + args.hours * 3600
    state = load_state()
    while time.time() < deadline:
        if STOP.exists():
            print("[autopilot] STOP_AUTOPILOT found - stopping")
            return 3
        name = choose(state)
        left_h = (deadline - time.time()) / 3600
        if ACTIVITIES[name]["timeout_h"] > left_h + 0.25:
            # not enough time left for this activity -> pick the shortest one or finish
            short = min(ACTIVITIES, key=lambda n: ACTIVITIES[n]["timeout_h"])
            if ACTIVITIES[short]["timeout_h"] > left_h + 0.25:
                break
            name = short
        print(f"[autopilot] next: {name} ({left_h:.2f} h left)", flush=True)
        run_activity(name, state)
        STATE.write_text(json.dumps(state, indent=2))
    print("[autopilot] summary:", json.dumps(state["stats"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
