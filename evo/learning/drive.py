"""
Drive: what NOVA does about a weak or lost ability (evo.engine.abilities) - it does not rest while it got worse.

For the weakest abilities of the champion it builds verified practice material (the tasks are made by a program that
knows the answer: evo.learning.skills, evo.learning.rulelab) plus a replay of every other ability, so that what it
learns does not push out what it can already do, and turns it into an ordinary training recipe for the director.
The judge decides as always; with the constitution's "abilities" rules a significant new ability may cost a small,
bounded amount of text quality, and no ability may get significantly worse.

Each way of practising is a hypothesis about why the ability is weak:
  practice   the kind of data was missing (it worked for 5 of 7 kinds on 14.10)
  longer     it was there, but too little of it (twice the steps, a lower rate)
  gentle     the learning pushed other things out (lower rate, smaller share, more replay)
A way that failed for an ability on a champion is not tried again on it; the ways that worked before go first.
When every way failed, the director says so in its report - the ability needs a new hypothesis.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path("data/drive_v1")

STRATEGIES = {
    "practice": {"steps": 6000, "lr": "1e-4", "focus": "0.5", "bulk": "0.5", "n_target": 20000, "n_replay": 2500},
    "longer": {"steps": 12000, "lr": "7e-5", "focus": "0.5", "bulk": "0.5", "n_target": 40000, "n_replay": 4000},
    "gentle": {"steps": 6000, "lr": "5e-5", "focus": "0.3", "bulk": "0.6", "n_target": 20000, "n_replay": 6000},
    # the core's own errors decide the material: the items where it fails (a feature value with a significantly
    # higher error rate - evo.engine.abilities.diagnose) make up most of the practice for that ability
    "focus-errors": {"steps": 6000, "lr": "1e-4", "focus": "0.5", "bulk": "0.5", "n_target": 20000, "n_replay": 2500,
                     "error_share": 0.7},
}
MAX_TARGETS = 3


def family(metric: str) -> tuple[str, str]:
    """'skills/ordering/form' -> ('skills', 'ordering'); 'rulelab/deeper' -> ('rulelab', 'deeper')."""
    parts = metric.split("/")
    return parts[0], parts[1]


def targets_of(deficits: list[dict]) -> list[str]:
    """The abilities to practise now: the first deficits, one entry per skill kind (new and form share the kind)."""
    out: list[str] = []
    for d in deficits:
        fam, name = family(d["metric"])
        key = f"{fam}/{name}" if fam == "skills" else "rulelab"
        if key not in out:
            out.append(key)
        if len(out) >= MAX_TARGETS:
            break
    return out


def _accept_pattern(pattern: dict | None, share: float):
    """Keep every item with the failing feature value and only enough of the others that it makes `share`."""
    if not pattern:
        return None
    key, value = pattern["feature"], str(pattern["value"])
    rate = pattern.get("items", 50) / 200                  # how common the value is among exam items
    keep_other = min(1.0, rate * (1 - share) / max(1e-6, share * (1 - rate)))

    def accept(item: dict, seed: int) -> bool:
        feats = {"lang": item.get("lang"), **item.get("feat", {})} if "feat" in item else item
        if str(feats.get(key)) == value:
            return True
        return random.Random(seed * 7919).random() < keep_other
    return accept


def build_material(targets: list[str], strategy: str, root: Path = ROOT, seed: int = 0, pattern: dict | None = None,
                   pattern_target: str | None = None) -> Path:
    """Practice for the targets + replay of every other skill kind and of RuleLab; verified by construction.
    With a pattern (focus-errors) the practice for `pattern_target` is drawn mostly from the failing cases."""
    from evo.learning import rulelab, skills

    s = STRATEGIES[strategy]
    name = f"{strategy}-{'-'.join(t.replace('/', '_') for t in targets)}"
    if pattern:
        name += f"-{pattern['feature']}-{pattern['value']}".replace(" ", "")
    out_dir = Path(root) / name
    out = out_dir / "material.jsonl"
    if out.exists() and out.stat().st_size > 0:
        return out_dir
    tmp = Path(root) / "_parts" / name
    rows: list[str] = []
    for kind in skills.KINDS:
        if kind == "deduction":
            continue                                  # kept out of every material: it measures transfer
        n = s["n_target"] if f"skills/{kind}" in targets else s["n_replay"]
        acc = _accept_pattern(pattern, s.get("error_share", 0.7)) if pattern_target == f"skills/{kind}" else None
        p = skills.material(kind, n, tmp, start=seed * 1_000_000 % 500_000, forms=skills.TRAIN_FORMS, accept=acc)
        rows += p.read_text(encoding="utf-8").splitlines()
    n = s["n_target"] if "rulelab" in targets else s["n_replay"]
    acc = _accept_pattern(pattern, s.get("error_share", 0.7)) if pattern_target == "rulelab" else None
    p = rulelab.material("qa", n, tmp, accept=acc)
    rows += p.read_text(encoding="utf-8").splitlines()
    random.Random(seed).shuffle(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return out_dir


def recipe(targets: list[str], strategy: str, material_dir: Path) -> tuple[str, dict]:
    s = STRATEGIES[strategy]
    name = f"drive:{strategy}:{'+'.join(t.split('/')[-1] for t in targets)}"
    return name, {"steps": s["steps"], "drive": {"strategy": strategy, "targets": targets},
                  "flags": {"--lr": s["lr"], "--bulk-frac": s["bulk"], "--code-frac": "0.03",
                            "--extra-dirs": f"data/teacher_v1,data/web_v1,data/self_v2,{material_dir}",
                            "--focus": f"dir:{material_dir}", "--focus-frac": s["focus"]}}


def order(state: dict) -> list[str]:
    """Ways that produced a champion before go first."""
    wins = state.get("drive_wins", {})
    return sorted(STRATEGIES, key=lambda k: (-int(wins.get(k, 0)), list(STRATEGIES).index(k)))


def next_try(state: dict, champion: str, targets: list[str]) -> str | None:
    tried = state.setdefault("drive", {}).setdefault(champion, {})
    key = "+".join(targets)
    done = set(tried.get(key, []))
    return next((s for s in order(state) if s not in done), None)


def note_try(state: dict, champion: str, targets: list[str], strategy: str) -> None:
    state.setdefault("drive", {}).setdefault(champion, {}).setdefault("+".join(targets), []).append(strategy)


def summary(state: dict, champion: str) -> dict:
    return {"tried": state.get("drive", {}).get(champion, {}), "wins": state.get("drive_wins", {})}


if __name__ == "__main__":
    print(json.dumps(STRATEGIES, indent=1))
