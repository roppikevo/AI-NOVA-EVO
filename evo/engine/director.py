"""
The director: NOVA runs its own improvement loop, day after day, without anybody queueing jobs.

    python -m evo.engine.director                 # runs until evo/director/STOP exists
    python -m evo.engine.director --smoke         # one tiny attempt end to end (nothing is released)

One cycle:
    1. pick a recipe - a way to learn, or a change of its own structure (one more layer, a wider or narrower
       local view: nova/surgery.py); the ones that produced accepted champions are tried more often, and a
       winner gets variations of itself (other learning rate, data mix, length)
    2. train a challenger from the current champion (the champion itself is never modified)
    3. the judge compares challenger and champion on text nobody trained on (evo.engine.judge)
    4. accepted -> frozen as the next release (backup, git tag), it becomes the champion
       rejected -> thrown away
    5. a released champion is on probation: checked once more against its predecessor on fresh text; if it is
       significantly worse, the director steps back to the predecessor
    6. report for the Creator (Slovak) in evo/director/report.txt and in the agent outbox
While the GPU trains, the CPU works on the side: a teacher writes texts for the weakest language, or the
next part of clean web text is fetched.

When PLATEAU attempts in a row are rejected, the champion has taken what this size and data can give:
the director starts the next generation - a bigger model trained from scratch (in resumable segments) -
and makes it the champion if the judge says it is better.

Robust by construction:
  * the whole state is in evo/director/state.json and is written after every step; after a restart of the
    server the loop simply continues (an attempt that was cut off counts as failed)
  * a recipe that fails twice in a row rests for a day
  * the GPU is never taken from somebody else: if another job holds it, the director waits
  * evo/director/STOP ends the loop after the running step; the Creator's word stands above it
  * the director chooses the strategy, never the rules: what "better" means is the sealed constitution
    (evo.engine.constitution); if the rules or the held-out texts change, it halts and releases nothing
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

DIR = Path("evo/director")
STATE, LOG, STOP, REPORT = DIR / "state.json", DIR / "log.jsonl", DIR / "STOP", DIR / "report.txt"
OUTBOX = Path.home() / "nova-agent" / "outbox"
PY = sys.executable

PLATEAU = 16           # rejected attempts in a row before a new generation is trained from scratch (the slow way)
COOLDOWN_H = 24.0      # a recipe that failed twice in a row rests this long
DISCOUNT = 0.93
GPU_BUSY_MB = 1500     # more than this in use by somebody else -> wait
SIDE_RAM_GB = 24       # free RAM a teacher on the CPU needs next to a training run
TEACHER = "Qwen3.6-35B-A3B"
MAX_OWN_RECIPES = 8    # recipes the director derived itself (variations of what worked)

TRAIN_COMMON = ["--no-activate", "--batch-size", "64", "--warmup", "100", "--eval-every", "2000", "--patience", "99",
                "--extra-dirs", "data/teacher_v1,data/web_v1", "--bulk-dir", "data/bulk_v1"]

# Trial and error, fast: every attempt is 15-30 minutes. WEAK is replaced by the champion's weakest language.
RECIPES: dict[str, dict[str, Any]] = {
    # --- ways to learn
    "gentle": {"steps": 4000, "flags": {"--lr": "2e-5", "--bulk-frac": "0.95", "--code-frac": "0.03"}},
    "fresh-web": {"steps": 5000, "flags": {"--lr": "5e-5", "--bulk-frac": "0.97", "--code-frac": "0.02"}},
    "weak-language": {"steps": 4000, "flags": {"--lr": "3e-5", "--bulk-frac": "0.9", "--code-frac": "0.03",
                                               "--boost-lang": "WEAK", "--boost-frac": "0.5"}},
    "teachers": {"steps": 3000, "flags": {"--lr": "3e-5", "--bulk-frac": "0.6", "--code-frac": "0.03",
                                          "--focus": "teacher", "--focus-frac": "0.5"}},
    "code": {"steps": 4000, "flags": {"--lr": "3e-5", "--bulk-frac": "0.8", "--code-frac": "0.25"}},
    # --- changes of its own structure, in place (nova/surgery.py): the edited model starts as an exact copy
    "one-more-layer": {"surgery": {"op": "add_layer"}, "steps": 6000,
                       "flags": {"--lr": "5e-5", "--warmup": "300", "--bulk-frac": "0.95", "--code-frac": "0.03"}},
    "wider-view": {"surgery": {"op": "kernel", "delta": 2}, "steps": 4000,
                   "flags": {"--lr": "4e-5", "--warmup": "300", "--bulk-frac": "0.95", "--code-frac": "0.03"}},
    "narrower-view": {"surgery": {"op": "kernel", "delta": -2}, "steps": 5000,
                      "flags": {"--lr": "4e-5", "--warmup": "300", "--bulk-frac": "0.95", "--code-frac": "0.03"}},
    # --- several models
    "collective": {"collective": True, "rounds": 2, "steps": 3000},
    # control for the collective: five clones with the SAME training (the collective's own settings, other seeds),
    # simply averaged - no specialities, no votes. Tells whether the organisation adds anything to plain averaging.
    "average-of-5": {"soup": 5, "steps": 3000, "flags": {"--lr": "1e-4", "--bulk-frac": "0.7", "--code-frac": "0.05"}},
}

# Next generations (bigger cores), tried in this order when the champion stops improving.
LADDER = [{"line": "NOVA-53M", "override": {"d_model": 896, "d_state": 896, "num_layers": 12}, "steps": 300000}]


# ------------------------------------------------------------------ state

def new_state(champion: str, name: str) -> dict:
    return {"champion": {"name": name, "checkpoint": champion}, "started": time.time(), "attempts": 0, "accepted": 0,
            "rejected_in_a_row": 0, "recipes": {}, "history": [], "releases": [name], "grow": None, "grown": [],
            "current": None, "seed": 7000, "interventions": []}


def load_state(path: Path | None = None) -> dict | None:
    path = path or STATE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_state(state: dict, path: Path | None = None) -> None:
    path = path or STATE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def log_event(rec: dict, path: Path | None = None) -> None:
    path = path or LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.time(), "date": time.strftime("%Y-%m-%d %H:%M"), **rec}, ensure_ascii=False) + "\n")


def next_release_name(name: str) -> str:
    m = re.match(r"^(.*)-v(\d+)$", name)
    return f"{m.group(1)}-v{int(m.group(2)) + 1}" if m else f"{name}-v2"


def free_release_name(name: str, exists: Callable[[str], bool] | None = None, params: int | None = None) -> str:
    """Next version of `name` that has no release folder yet (a release is never overwritten).

    A champion that changed its size starts a new line: NOVA-24M-v3 with 25.3 M parameters -> NOVA-25M-v1."""
    exists = exists or (lambda n: (Path("evo/releases") / n).exists())
    m = re.match(r"^(.*?)-(\d+)M-v\d+$", name)
    if params and m and int(round(params / 1e6)) != int(m.group(2)):
        name = f"{m.group(1)}-{int(round(params / 1e6))}M-v0"
    new = next_release_name(name)
    while exists(new):
        new = next_release_name(new)
    return new


# ------------------------------------------------------------------ choosing

def recipe_stats(state: dict, name: str) -> dict:
    return state["recipes"].setdefault(name, {"runs": 0.0, "gain": 0.0, "fails": 0, "rest_until": 0.0})


def all_recipes(state: dict) -> dict[str, dict]:
    """The built-in recipes plus the ones the director derived itself."""
    return {**RECIPES, **state.get("own_recipes", {})}


def mutate_recipe(name: str, recipe: dict, rng) -> tuple[str, dict] | None:
    """A variation of a recipe that just produced a champion: one knob turned (trial and error on its own method)."""
    if "flags" not in recipe or recipe.get("collective"):
        return None
    new = {**recipe, "flags": dict(recipe["flags"]), "parent": name.split("~")[0]}
    knob = rng.choice(["lr", "lr", "steps", "bulk", "code"])
    up = rng.random() < 0.5
    if knob == "lr":
        v = float(new["flags"]["--lr"]) * (2.0 if up else 0.5)
        v = min(2e-4, max(5e-6, v))
        new["flags"]["--lr"] = f"{v:.1e}".replace("e-0", "e-")
        tag = f"lr{new['flags']['--lr']}"
    elif knob == "steps":
        new["steps"] = int(min(12000, max(2000, round(recipe["steps"] * (1.5 if up else 0.67), -2))))
        tag = f"steps{new['steps']}"
    elif knob == "bulk":
        v = min(0.98, max(0.5, float(new["flags"].get("--bulk-frac", "0.9")) + (0.07 if up else -0.1)))
        new["flags"]["--bulk-frac"] = f"{v:.2f}"
        tag = f"web{new['flags']['--bulk-frac']}"
    else:
        v = min(0.4, max(0.01, float(new["flags"].get("--code-frac", "0.03")) * (2.0 if up else 0.5)))
        new["flags"]["--code-frac"] = f"{v:.3f}".rstrip("0")
        tag = f"code{new['flags']['--code-frac']}"
    if new["flags"] == recipe["flags"] and new["steps"] == recipe["steps"]:
        return None
    return f"{new['parent']}~{tag}", new


def learn_from(state: dict, name: str, accepted: bool) -> str | None:
    """After an attempt: a winner gets a variation of itself; a home-made recipe that keeps losing is dropped."""
    import random

    own = state.setdefault("own_recipes", {})
    made = None
    if accepted:
        m = mutate_recipe(name, all_recipes(state)[name], random.Random(state["seed"] * 31 + state["attempts"]))
        if m and m[0] not in all_recipes(state):
            own[m[0]] = m[1]
            made = m[0]
    for n in list(own):
        st = state["recipes"].get(n)
        if st and st["runs"] >= 3.0 and st["gain"] <= 0:          # tried enough, never produced a champion
            del own[n]
    while len(own) > MAX_OWN_RECIPES:                              # keep the ones with the best record
        worst = min(own, key=lambda n: (recipe_stats(state, n)["gain"] / max(recipe_stats(state, n)["runs"], 0.2), n != made))
        del own[worst]
    return made


def available(state: dict, now: float | None = None, recipes: dict | None = None) -> list[str]:
    now = time.time() if now is None else now
    return [n for n in (recipes or all_recipes(state)) if recipe_stats(state, n)["rest_until"] <= now]


def choose(state: dict, now: float | None = None, recipes: dict | None = None, exploration: float = 0.5) -> str | None:
    """Discounted UCB over the recipes: what produced champions is used more, everything is retried sometimes."""
    names = available(state, now, recipes)
    if not names:
        return None
    for n in names:
        if recipe_stats(state, n)["runs"] < 0.2:
            return n
    total = sum(recipe_stats(state, n)["runs"] for n in names) + 1
    last = [h["recipe"] for h in state["history"][-2:]]

    def ucb(n: str) -> float:
        s = recipe_stats(state, n)
        return s["gain"] / s["runs"] + exploration * math.sqrt(math.log(total + 1) / s["runs"]) - (0.5 if last.count(n) == 2 else 0)

    return max(names, key=ucb)


def record(state: dict, name: str, gain: float, failed: bool, now: float | None = None) -> None:
    now = time.time() if now is None else now
    for s in state["recipes"].values():
        s["runs"] *= DISCOUNT
        s["gain"] *= DISCOUNT
    s = recipe_stats(state, name)
    if failed:
        s["fails"] += 1
        if s["fails"] >= 2:
            s["rest_until"], s["fails"] = now + COOLDOWN_H * 3600, 0
    else:
        s["fails"] = 0
        s["runs"] += 1
        s["gain"] += max(0.0, gain)


def weakest_language(scores: dict | None) -> str:
    """Natural language with the highest held-out web loss (the champion's weakest side)."""
    lang = (scores or {}).get("web_by_language") or {}
    return max(lang, key=lang.get) if lang else "en"


# ------------------------------------------------------------------ the outside world (replaced in tests)

class World:
    """Everything the director does outside its own state. Tests use a fake one."""

    def __init__(self, dataset: Path, web_dir: Path = Path("data/bulk_val_v1")) -> None:
        self.dataset, self.web_dir = dataset, web_dir
        self.child: subprocess.Popen | None = None
        self.side: subprocess.Popen | None = None

    # --- processes
    def run(self, cmd: list[str], timeout_h: float, nice: int = 0) -> tuple[int | str, str]:
        full = (["nice", "-n", str(nice)] if nice else []) + cmd
        out = DIR / "last_run.out"
        with out.open("w") as f:
            self.child = subprocess.Popen(full, stdout=f, stderr=subprocess.STDOUT, text=True)
            try:
                rc: int | str = self.child.wait(timeout=timeout_h * 3600)
            except subprocess.TimeoutExpired:
                self.child.terminate()
                try:
                    self.child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                rc = "timeout"
            finally:
                self.child = None
        text = out.read_text(errors="replace")
        return rc, text[-3000:]

    def start_side(self, cmd: list[str]) -> None:
        if self.side is None or self.side.poll() is not None:
            self.side = subprocess.Popen(["nice", "-n", "19"] + cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def wait_side(self, timeout_s: float = 2400) -> None:
        if self.side is not None and self.side.poll() is None:
            try:
                self.side.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self.side.terminate()
        self.side = None

    def stop_children(self) -> None:
        for p in (self.child, self.side):
            if p is not None and p.poll() is None:
                p.terminate()

    # --- machine
    def gpu_used_mb(self) -> int:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=30).stdout
            return int(out.strip().splitlines()[0])
        except Exception:
            return 0

    def free_ram_gb(self) -> float:
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / 1e6
        except Exception:
            pass
        return 0.0

    def unload_teachers(self) -> None:
        try:
            from nova.gpu_guard import loaded_models, unload

            for m in loaded_models():
                unload(m)
        except Exception:
            pass

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def average(self, paths: list[str], out: Path) -> None:
        """Plain mean of the weights of several checkpoints."""
        import torch

        from evo.collective.ensemble import average_states

        cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
        ck = {k: v for k, v in cks[0].items() if k != "optimizer"}
        ck["model_state_dict"] = average_states([c["model_state_dict"] for c in cks])
        ck["kind"] = "average"
        torch.save(ck, out)

    # --- judging and releasing
    def scores(self, checkpoint: str) -> dict:
        """Numbers for the report and for choosing the weak language."""
        from evo.engine import judge

        sets = judge.held_out_sets(self.dataset, self.web_dir)
        s = judge.score(checkpoint, sets, str(self.dataset / "tokenizer.json"))
        out = {k: round(float(v.mean()), 4) for k, v in s["seq"].items()}
        web = s["seq"].get("web")
        if web is not None and len(web) % 4 == 0:
            n = len(web) // 4
            out["web_by_language"] = {l: round(float(web[i * n:(i + 1) * n].mean()), 4) for i, l in enumerate(("sk", "cs", "pl", "en"))}
        out.update({"code": len(s["code"] or []), "creator": s["creator"], "params": s.get("params")})
        try:
            import torch

            from nova import surgery

            out["genome"] = surgery.genome(torch.load(checkpoint, map_location="cpu", weights_only=False))
        except Exception:
            pass
        return out

    def judge(self, champion: str, challenger: str) -> dict:
        from evo.engine import judge

        return judge.judge(champion, challenger, self.dataset, self.web_dir)

    def release_name(self, champion_name: str, params: int | None = None) -> str:
        return free_release_name(champion_name, params=params)

    def surgery(self, checkpoint: str, op: dict, out: Path, seed: int) -> str | None:
        """Edit the champion's structure in place (exact copy at first); None if the change is not possible."""
        import torch

        from evo.engine import constitution
        from nova import surgery

        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        new = surgery.apply(ck, op, max_parameters=int(constitution.load().get("max_parameters", 0)) or None, seed=seed)
        if new is None:
            return None
        torch.save(new, out)
        return str(out)

    def probation(self, previous: str, new: str, index: int, per_lang: int = 400) -> dict:
        """A released champion against its predecessor on web text that no decision has used (a fresh window each time)."""
        import numpy as np

        from evo.collective.node import load_node
        from evo.collective.stats import paired_bootstrap, seq_loss
        import torch

        rows = []
        for name in ("sk", "cs", "pl", "en"):
            files = sorted(self.web_dir.glob(f"{name}-*.npy"))
            if files:
                data = np.load(files[0], mmap_mode="r")
                start = 2000 + per_lang * (index % max(1, (len(data) - 2000) // per_lang))
                rows.append(np.asarray(data[start:start + per_lang]))
        if not rows:
            return {"diff": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
        seqs = np.concatenate(rows)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = str(self.dataset / "tokenizer.json")
        a = seq_loss(load_node("previous", previous, tok, "", device), seqs)
        b = seq_loss(load_node("new", new, tok, "", device), seqs)
        return paired_bootstrap(a, b)

    def release(self, name: str, checkpoint: str) -> str | None:
        """Freeze the accepted challenger; returns the released full-precision weights (or None)."""
        rc, tail = self.run([PY, "-m", "evo.engine.release", "--name", name, "--candidates", checkpoint, "--only-candidates",
                             "--web-val-dir", str(self.web_dir)], timeout_h=0.5)
        out = Path("evo/releases") / name / "nova_model_fp32.pt"
        if rc != 0 or not out.exists():
            log_event({"event": "release_failed", "name": name, "rc": rc, "tail": tail[-400:]})
            return None
        rel = out.parent
        small = [str(rel / f) for f in ("MODEL.json", "SHA256SUMS", "tokenizer.json", "blocks_scan.py", "model_scan.py", "config.py")]
        if (rel / "nova_model.pt").stat().st_size < 95e6:        # GitHub refuses files of 100 MB and more
            small.append(str(rel / "nova_model.pt"))
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        subprocess.run(["git", "add", "-f"] + [f for f in small if Path(f).exists()], env=env)
        subprocess.run(["git", "add", "-f", str(STATE), str(LOG), str(REPORT)], env=env, capture_output=True)
        subprocess.run(["git", "commit", "-qm", f"Director: release {name} (accepted by the judge)"], env=env)
        subprocess.run(["git", "tag", "-a", name.lower(), "-m", f"{name}: released by the director"], env=env, capture_output=True)
        subprocess.run(["git", "push", "-q", "origin", "master"], env=env, capture_output=True, timeout=600)
        subprocess.run(["git", "push", "-q", "origin", name.lower()], env=env, capture_output=True, timeout=600)
        tar = Path.home() / "nova-evo-backups" / "releases" / f"{name}.tar.gz"
        if tar.exists() and OUTBOX.exists():                      # a copy for the Creator's PC, offered for 20 minutes
            dst = OUTBOX / f"release-{name}.tar.gz"
            subprocess.run(["cp", str(tar), str(dst)])
            subprocess.Popen(["bash", "-c", f"sleep 1200; rm -f '{dst}'"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(out)


def halt(state: dict, reason: str) -> None:
    """The constitution does not hold: stop everything, release nothing, wait for the Creator."""
    state["halted"] = {"reason": reason, "date": time.strftime("%Y-%m-%d %H:%M")}
    state["current"] = None
    STOP.parent.mkdir(parents=True, exist_ok=True)
    STOP.write_text(f"constitution: {reason}\n")
    save_state(state)
    log_event({"event": "halted", "reason": reason})


def _broken(exc: Exception) -> bool:
    return type(exc).__name__ == "ConstitutionError"


# ------------------------------------------------------------------ one attempt

def train_command(recipe: dict, champion: str, out: Path, seed: int, weak: str, steps: int | None = None) -> list[str]:
    n = steps or recipe["steps"]
    cmd = [PY, "-m", "evo.engine.long_train", "--init", champion, "--out-checkpoint", str(out), "--steps", str(n),
           "--seed", str(seed), "--max-hours", str(round(n / 9000 + 0.3, 2))] + TRAIN_COMMON
    for k, v in recipe["flags"].items():
        cmd += [k, weak if v == "WEAK" else v]
    return cmd


def collective_challenger(run_dir: Path, base: str) -> str | None:
    """The core the nodes adopted (None if they kept the base)."""
    f = run_dir / "report.json"
    if not f.exists():
        return None
    bases = [r.get("base") for r in json.loads(f.read_text()).get("node_reports", {}).values() if r.get("base")]
    core = max(set(bases), key=bases.count) if bases else None
    return core if core and core != base and Path(core).exists() else None


def side_command(state: dict, world: World) -> list[str] | None:
    """CPU work next to a training run: teacher texts for the weak language, else the next web batch."""
    weak = weakest_language(state["champion"].get("scores"))
    if world.free_ram_gb() >= SIDE_RAM_GB:
        state["seed"] += 1
        return [PY, "-m", "evo.learning.teacher_corpus", "--teachers", TEACHER, "--per-task", "4", "--max-minutes", "25",
                "--seed", str(state["seed"]), "--langs", f"{weak},py,rs", "--new-topics", "12"]
    return [PY, "-m", "evo.corpus.bulk_web", "--mchars", "sk=40,cs=25,pl=25,en=40", "--max-gb", "24"]


def attempt(state: dict, world: World, name: str, smoke: bool = False, recipes: dict | None = None) -> dict:
    """Train one challenger with recipe `name`, judge it, release it if accepted."""
    recipe = (recipes or all_recipes(state))[name]
    champ = state["champion"]
    state["seed"] += 1
    n = state["attempts"] + 1
    out = DIR / "challengers" / f"attempt-{n}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    weak = weakest_language(champ.get("scores"))
    rec: dict[str, Any] = {"attempt": n, "recipe": name, "champion": champ["name"], "date": time.strftime("%Y-%m-%d %H:%M"),
                           "start": champ["checkpoint"], "steps": recipe.get("steps"), "flags": recipe.get("flags"),
                           "clones": recipe.get("soup") or (5 if recipe.get("collective") else 1)}
    state["current"] = {"what": "attempt", "recipe": name, "since": time.time()}
    save_state(state)
    t0 = time.time()
    challenger: str | None = None
    if recipe.get("collective"):
        world.wait_side()
        world.unload_teachers()
        run = f"director-{n}"
        rc, tail = world.run([PY, "-m", "evo.collective.collective_v2", "--name", run, "--base", champ["checkpoint"], "--steps",
                              str(30 if smoke else recipe["steps"]), "--rounds", str(1 if smoke else recipe["rounds"]),
                              "--core-candidates", "[]", "--baseline-rounds", "0", "--max-hours", "5"], timeout_h=6)
        subprocess.run(["pkill", "-f", "evo.collective.agent"], capture_output=True)
        run_dir = Path("evo/collective/runs") / run
        challenger = collective_challenger(run_dir, champ["checkpoint"]) if rc == 0 else None
        rec["note"] = "the nodes kept the base" if rc == 0 and challenger is None else ""
    elif recipe.get("soup"):
        side = side_command(state, world)
        if side:
            world.start_side(side)
        parts, rc, tail = [], 0, ""
        for i in range(int(recipe["soup"])):
            part = out.with_name(f"attempt-{n}-clone{i}.pt")
            rc, tail = world.run(train_command(recipe, champ["checkpoint"], part, state["seed"] * 10 + i, weak),
                                 timeout_h=recipe["steps"] / 9000 + 0.8)
            if rc != 0 or not part.exists():
                rc = rc or 1
                break
            parts.append(str(part))
        if rc == 0 and len(parts) >= 2:
            world.average(parts, out)
        for f in out.parent.glob(f"attempt-{n}-clone*.pt"):
            f.unlink(missing_ok=True)
        challenger = str(out) if rc == 0 and out.exists() else None
    else:
        init: str | None = champ["checkpoint"]
        edited = out.with_name(f"attempt-{n}-init.pt")
        if recipe.get("surgery"):                      # change its own structure first; the copy starts out identical
            init = world.surgery(champ["checkpoint"], recipe["surgery"], edited, state["seed"])
            rec["surgery"] = recipe["surgery"]
        if init is None:
            rc, tail = 1, "this change of structure is not possible for the champion (size limit or range)"
            rec["note"] = tail
        else:
            if not smoke:
                side = side_command(state, world)
                if side:
                    world.start_side(side)
            rc, tail = world.run(train_command(recipe, init, out, state["seed"], weak, 30 if smoke else None),
                                 timeout_h=recipe["steps"] / 9000 + 0.8)
        edited.unlink(missing_ok=True)
        challenger = str(out) if rc == 0 and out.exists() else None
    rec.update({"rc": rc, "hours": round((time.time() - t0) / 3600, 2), "weak": weak})
    failed = rc != 0
    gain = 0.0
    if challenger:
        try:
            v = world.judge(champ["checkpoint"], challenger)
            rec["verdict"] = {k: v[k] for k in ("accept", "reasons", "decision", "vault", "code", "creator") if k in v}
            rec["sets"] = {k: [s["before"], s["after"], s["percent"]] for k, s in v["sets"].items()}
            gain = v["decision"]["gain_percent"] if v["accept"] else 0.0
            if v["accept"] and not smoke:
                new = world.release_name(champ["name"], v.get("params"))
                released = world.release(new, challenger)
                if released:
                    state["champion"] = {"name": new, "checkpoint": released, "since": time.time()}
                    state["releases"].append(new)
                    state["accepted"] += 1
                    rec["released"] = new
                    # on probation: checked once more on text no decision has used; worst case it steps back
                    state["probation"] = {"name": new, "recipe": name, "gain": gain,
                                          "previous": {k: champ[k] for k in ("name", "checkpoint", "scores") if k in champ}}
                else:
                    rec["note"] = "accepted, but the release failed"
                    failed = True
        except Exception as exc:  # a broken judge must never promote anything
            rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
            failed = not _broken(exc)          # a broken constitution is not the recipe's fault
            if _broken(exc):
                halt(state, str(exc))
    elif failed:
        rec["tail"] = tail[-500:]
    if challenger and Path(challenger).exists() and str(Path(challenger).parent) == str(out.parent):
        Path(challenger).unlink(missing_ok=True)       # released weights live in evo/releases; challengers are thrown away
    if recipe.get("collective"):
        for f in (Path("evo/collective/runs") / f"director-{n}").glob("*.pt"):
            f.unlink(missing_ok=True)
    state["attempts"] = n
    if not state.get("halted"):
        record(state, name, gain, failed)
        made = learn_from(state, name, bool(rec.get("released")))
        if made:
            rec["new_recipe"] = made
    if rec.get("released"):
        state["rejected_in_a_row"] = 0
    elif not failed and not state.get("halted"):
        state["rejected_in_a_row"] += 1
    state["history"] = (state["history"] + [rec])[-200:]
    state["current"] = None
    save_state(state)
    log_event({"event": "attempt", **rec})
    return rec


def challenge(state: dict, world: World, checkpoint: str, label: str = "external") -> dict:
    """A model made elsewhere (a collective core, a confirmed new core) asks for the title: the judge decides."""
    champ = state["champion"]
    rec: dict[str, Any] = {"attempt": state["attempts"], "recipe": label, "champion": champ["name"],
                           "date": time.strftime("%Y-%m-%d %H:%M"), "rc": 0, "challenger": checkpoint}
    try:
        v = world.judge(champ["checkpoint"], checkpoint)
        rec["verdict"] = {k: v[k] for k in ("accept", "reasons", "decision", "vault", "code", "creator") if k in v}
        rec["sets"] = {k: [s["before"], s["after"], s["percent"]] for k, s in v["sets"].items()}
        if v["accept"]:
            new = world.release_name(champ["name"])
            released = world.release(new, checkpoint)
            if released:
                state["champion"] = {"name": new, "checkpoint": released, "since": time.time()}
                state["releases"].append(new)
                state["accepted"] += 1
                state["rejected_in_a_row"] = 0
                rec["released"] = new
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
        if _broken(exc):
            halt(state, str(exc))
    state["history"] = (state["history"] + [rec])[-200:]
    save_state(state)
    log_event({"event": "challenge", **rec})
    return rec


def probation_step(state: dict, world: World) -> str:
    """The newest champion once more against its predecessor, on fresh text. Significantly worse -> step back."""
    pr = state["probation"]
    state["probation"] = None
    champ = state["champion"]
    if champ["name"] != pr["name"]:
        return "probation:skipped"
    try:
        r = world.probation(pr["previous"]["checkpoint"], champ["checkpoint"], len(state["releases"]))
    except Exception as exc:
        log_event({"event": "probation_error", "name": pr["name"], "error": f"{type(exc).__name__}: {exc}"[:300]})
        save_state(state)
        return "probation:error"
    back = r.get("n", 0) > 0 and r["lo"] > 0                      # the whole interval says: worse than before
    rec = {"event": "probation", "name": pr["name"], "previous": pr["previous"]["name"], "result": r, "reverted": back}
    if back:
        state["champion"] = dict(pr["previous"])
        state.setdefault("reverted", []).append(pr["name"])
        state["rejected_in_a_row"] += 1
        s = recipe_stats(state, pr["recipe"])                      # the recipe loses the credit it got
        s["gain"] = max(0.0, s["gain"] - float(pr.get("gain", 0.0)))
        state["history"] = (state["history"] + [{"attempt": state["attempts"], "recipe": pr["recipe"], "date": time.strftime("%Y-%m-%d %H:%M"),
                                                 "rc": 0, "reverted": pr["name"], "probation": r}])[-200:]
    save_state(state)
    log_event(rec)
    return "probation:reverted" if back else "probation:kept"


# ------------------------------------------------------------------ next generation

def grow_step(state: dict, world: World, hours: float = 3.0, ladder: list[dict] | None = None) -> dict | None:
    """Train the next (bigger) generation for one segment; when it is finished, let the judge decide."""
    ladder = LADDER if ladder is None else ladder
    g = state.get("grow")
    if g is None:
        nxt = next((s for s in ladder if s["line"] not in state["grown"]), None)
        if nxt is None:
            return None
        g = state["grow"] = {**nxt, "since": time.time(), "segments": 0}
        log_event({"event": "grow_start", "line": g["line"], "override": g["override"]})
    state["current"] = {"what": "grow", "line": g["line"], "since": time.time()}
    save_state(state)
    side = side_command(state, world)
    if side:
        world.start_side(side)
    rc, tail = world.run([PY, "-m", "evo.engine.train_line", "--name", g["line"], "--config-override", json.dumps(g["override"]),
                          "--total-steps", str(g["steps"]), "--max-hours", str(hours), "--bulk-frac", "0.95", "--code-frac", "0.03",
                          "--val-extra-dir", str(world.web_dir)], timeout_h=hours + 1.0)
    g["segments"] += 1
    st_file = Path("evo/lines") / g["line"] / "state.json"
    line = json.loads(st_file.read_text()) if st_file.exists() else {}
    g.update({"steps_done": line.get("steps_done", 0), "best_val": line.get("best_val"), "last_rc": rc})
    rec: dict[str, Any] = {"event": "grow_segment", "line": g["line"], "rc": rc, "steps_done": g["steps_done"], "best_val": g["best_val"]}
    if rc != 0:
        g["fails"] = g.get("fails", 0) + 1
        rec["tail"] = tail[-400:]
        if g["fails"] >= 3:                                   # this size does not run here: give it up
            state["grown"].append(g["line"])
            state["grow"] = None
            rec["given_up"] = True
    else:
        g["fails"] = 0
    if line.get("finished") and line.get("best_checkpoint") and state.get("grow"):
        champ = state["champion"]
        try:
            v = world.judge(champ["checkpoint"], line["best_checkpoint"])
            rec["verdict"] = {k: v[k] for k in ("accept", "reasons", "decision", "vault", "code", "creator") if k in v}
            if v["accept"]:
                name = f"{g['line']}-v1"
                released = world.release(name, line["best_checkpoint"])
                if released:
                    state["champion"] = {"name": name, "checkpoint": released, "since": time.time()}
                    state["releases"].append(name)
                    state["accepted"] += 1
                    state["rejected_in_a_row"] = 0
                    rec["released"] = name
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
            if _broken(exc):                   # keep the finished generation: it is judged again once the Creator has looked
                halt(state, str(exc))
                log_event(rec)
                return rec
        state["grown"].append(g["line"])
        state["grow"] = None
        if not rec.get("released"):
            state["rejected_in_a_row"] = 0                     # the champion stays; go back to learning attempts
    state["current"] = None
    save_state(state)
    log_event(rec)
    return rec


# ------------------------------------------------------------------ report (for the Creator, in Slovak)

def report(state: dict, now: float | None = None) -> str:
    now = time.time() if now is None else now
    c = state["champion"]
    s = c.get("scores") or {}
    days = (now - state["started"]) / 86400
    L = [f"NOVA: hlásenie riaditeľa ({time.strftime('%Y-%m-%d %H:%M', time.localtime(now))})", ""]
    if state.get("halted"):
        L += [f"ZASTAVENÉ {state['halted']['date']}: ústava nesedí ({state['halted']['reason']}). Nič sa nevydáva, čaká sa na tvorcu.", ""]
    L += [
         f"Súčasný model (šampión): {c['name']}",
         f"Riaditeľ beží {days:.1f} dňa; bez zásahu zvonka {(now - (state['interventions'][-1]['time'] if state['interventions'] else state['started'])) / 86400:.1f} dňa "
         f"(zásahov spolu: {len(state['interventions'])})",
         f"Pokusy o zlepšenie: {state['attempts']}, prijaté: {state['accepted']}, odmietnuté za sebou: {state['rejected_in_a_row']} (pri {PLATEAU} sa začne generácia od nuly)",
         f"Vydania: {', '.join(state['releases'])}"]
    if s:
        L.append(f"Strata na odloženom texte: dataset {s.get('dataset')}, web {s.get('web')}; programovanie {s.get('code')}/79; "
                 f"pozná tvorcu: {'áno' if s.get('creator', -9) >= -0.7 else 'NIE'}")
        if s.get("web_by_language"):
            L.append("Web po jazykoch: " + ", ".join(f"{k} {v}" for k, v in s["web_by_language"].items())
                     + f" (najslabší: {weakest_language(s)})")
    ge = s.get("genome")
    if ge:
        L.append(f"Stavba: {ge['layers']} vrstiev, šírka {ge['width']}, pohľad na {ge['kernel']} susedných slov, "
                 f"{ge['parameters'] / 1e6:.1f} M parametrov")
    if state.get("reverted"):
        L.append(f"Vrátené späť po skúšobnej lehote: {', '.join(state['reverted'])}")
    if state.get("probation"):
        L.append(f"V skúšobnej lehote: {state['probation']['name']}")
    if state.get("own_recipes"):
        L.append("Vlastné recepty (obmeny úspešných): " + ", ".join(state["own_recipes"]))
    g = state.get("grow")
    if g:
        L.append(f"Nová generácia {g['line']}: {g.get('steps_done', 0)} z {g['steps']} krokov, najlepšia strata {g.get('best_val')}")
    cur = state.get("current")
    if cur:
        L.append(f"Práve beží: {cur.get('what')} {cur.get('recipe') or cur.get('line') or ''}")
    L += ["", "Posledné pokusy:"]
    for h in state["history"][-8:]:
        v = h.get("verdict") or {}
        if h.get("reverted"):
            res = f"VRÁTENÉ SPÄŤ: {h['reverted']} neobstál v skúšobnej lehote ({h['probation'].get('diff'):+.4f})"
        elif h.get("released"):
            res = f"PRIJATÉ, vydané ako {h['released']} (zisk {v.get('decision', {}).get('gain_percent')} %)"
        elif v:
            res = f"odmietnuté (zisk {v.get('decision', {}).get('gain_percent')} %): " + "; ".join(v.get("reasons", []))[:140]
        else:
            res = f"zlyhalo (rc {h.get('rc')}) {h.get('note', '')}"
        L.append(f"  {h['date']}  {h['recipe']:<14} {res}")
    rest = [n for n, r in state["recipes"].items() if r.get("rest_until", 0) > now]
    if rest:
        L.append("Recepty, ktoré oddychujú po dvoch zlyhaniach: " + ", ".join(rest))
    return "\n".join(L) + "\n"


def write_report(state: dict) -> None:
    txt = report(state)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(txt, encoding="utf-8")
    if OUTBOX.exists():
        (OUTBOX / "nova-hlasenie.txt").write_text(txt, encoding="utf-8")


# ------------------------------------------------------------------ the loop

def recover(state: dict) -> None:
    """After a restart: an activity that was cut off counts as a failure of its recipe."""
    cur = state.get("current")
    if cur:
        if cur.get("what") == "attempt":
            record(state, cur["recipe"], 0.0, failed=True)
        log_event({"event": "recovered", "was": cur})
        state["current"] = None
    for f in (DIR / "challengers").glob("*.pt") if (DIR / "challengers").exists() else []:
        f.unlink(missing_ok=True)


def gpu_is_free(world: World) -> bool:
    if world.gpu_used_mb() <= GPU_BUSY_MB:
        return True
    world.unload_teachers()          # a teacher model on the GPU is ours to move; anything else is not
    world.sleep(20)
    return world.gpu_used_mb() <= GPU_BUSY_MB


def cycle(state: dict, world: World, recipes: dict | None = None) -> str:
    """One step of the loop. Returns what happened (for the log and the tests)."""
    if not gpu_is_free(world):
        return "gpu_busy"
    if state.get("probation"):
        return probation_step(state, world)
    if "scores" not in state["champion"]:
        state["champion"]["scores"] = world.scores(state["champion"]["checkpoint"])
        save_state(state)
    if state.get("grow") or state["rejected_in_a_row"] >= PLATEAU:
        rec = grow_step(state, world)
        if rec is not None:
            return "grow"
        state["rejected_in_a_row"] = 0     # nothing bigger left to try: keep learning
    name = choose(state, recipes=recipes)
    if name is None:
        return "all_recipes_resting"
    attempt(state, world, name, recipes=recipes)
    return f"attempt:{name}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--champion", default="", help="first champion (a released nova_model_fp32.pt); only used when there is no state yet")
    ap.add_argument("--name", default="", help="its release name, e.g. NOVA-24M-v1")
    ap.add_argument("--smoke", action="store_true", help="one tiny attempt end to end; nothing is released, no state is kept")
    ap.add_argument("--max-cycles", type=int, default=0)
    ap.add_argument("--challenge", default="", help="comma list of checkpoints the judge compares with the champion before the loop starts")
    ap.add_argument("--intervention", default="", help="note that somebody had to step in (resets the days-without-help counter)")
    args = ap.parse_args(argv)
    if args.intervention:
        st = load_state()
        if st is None:
            print("no state yet")
            return 2
        st["interventions"].append({"time": time.time(), "date": time.strftime("%Y-%m-%d %H:%M"), "note": args.intervention})
        save_state(st)
        log_event({"event": "intervention", "note": args.intervention})
        return 0

    dataset = Path(json.loads(Path("evo/engine/evo_state.json").read_text(encoding="utf-8"))["best_known"]["dataset"])
    world = World(dataset)
    DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        state = new_state(args.champion, args.name or "SMOKE-v1")
        global STATE, LOG
        STATE, LOG = DIR / "smoke_state.json", DIR / "smoke_log.jsonl"
        state["champion"]["scores"] = world.scores(args.champion)
        print("champion:", json.dumps(state["champion"]["scores"]))
        ok = True
        for name in ("gentle", "one-more-layer"):       # one way to learn and one change of its own structure
            rec = attempt(state, world, name, smoke=True)
            print(json.dumps(rec, ensure_ascii=False))
            ok = ok and rec.get("rc") == 0 and "verdict" in rec
        print(report(state))
        print("SMOKE OK" if ok else "SMOKE FAILED")
        return 0 if ok else 1

    import fcntl

    lock = (DIR / "lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("the director is already running")
        return 0
    state = load_state()
    if state is None:
        if not args.champion or not Path(args.champion).exists():
            print("no state yet: give the first champion with --champion and --name")
            return 2
        state = new_state(args.champion, args.name or Path(args.champion).parent.name)
        log_event({"event": "start", "champion": state["champion"]})
    recover(state)
    state.pop("halted", None)            # started again (by the Creator): the seal is checked at the next verdict
    save_state(state)

    def stop_now(*_):
        world.stop_children()
        state["current"] = None          # stopped on purpose: nobody is to blame
        save_state(state)
        log_event({"event": "stopped_by_signal"})
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop_now)
    for ck in [c for c in args.challenge.split(",") if c]:
        if Path(ck).exists() and gpu_is_free(world):
            rec = challenge(state, world, ck)
            print(f"challenge {ck}: {'released as ' + rec['released'] if rec.get('released') else 'rejected'}", flush=True)

    cycles = 0
    while not STOP.exists():
        what = cycle(state, world)
        write_report(state)
        print(f"{time.strftime('%Y-%m-%d %H:%M')} {what}", flush=True)
        if what in ("gpu_busy", "all_recipes_resting"):
            world.sleep(300)
        cycles += 1
        if args.max_cycles and cycles >= args.max_cycles:
            break
    world.wait_side(60)
    log_event({"event": "stopped"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
