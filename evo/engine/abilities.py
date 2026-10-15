"""
Abilities: what NOVA can do, measured the same way every time, kept for every champion - the director's measure of
whether it is getting better or worse, and the reason it does not rest while something got worse.

The suite (fixed; it never changes with what was learned):
  skills/<kind>/new        8 kinds of thinking, fresh items in the material's form (evo.learning.skills)
  skills/<kind>/form       the same in a form of the question never trained (calibrated scoring)
  rulelab/<part>           relations and order: new problems, unseen phrasing, unseen words, deeper chains
Every metric keeps the answer of every item, so two cores are compared item by item (exact McNemar test) with
Holm's correction over the whole suite - a difference that chance makes in one of twenty metrics is not a result.

    python -m evo.engine.abilities measure <checkpoint> --name NOVA8-53M-v3     # adds to evo/director/abilities.jsonl
    python -m evo.engine.abilities show                                           # the champion's standing, weak spots
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

BOARD = Path("evo/director/abilities.jsonl")
N_ITEMS = 200
TARGET = 0.90            # what "can do it" means for a skill that can be mastered; below it the metric is a weak spot

# The Creator (15.10): 90-99 % is the goal for single, masterable skills - not a universal bar for all reasoning.
#   mastery   a defined skill on new items of the trained kind (skills/<kind>/new, rulelab/same): goal >= TARGET
#   frontier  the edges: an unknown form of the question, unseen words, longer chains, a kind never trained
#             (deduction): no fixed bar - the goal is a confirmed improvement and never a loss
FRONTIER = ("/form", "rulelab/phrasing", "rulelab/words", "rulelab/deeper", "skills/deduction/")


def kind_of(metric: str) -> str:
    return "frontier" if any(f in metric for f in FRONTIER) else "mastery"


SAMPLES = {"select": 0, "confirm": 200}   # reserved item ranges: the judge decides on "select"; "confirm" is never
                                         # used for any decision - it only confirms a result (the Creator's rule 5/9)
KEEP = ("accuracy", "chance", "correct", "feat", "margin")


def measure(checkpoint: str, tokenizer: str, n: int = N_ITEMS, device: str = "auto", sample: str = "select") -> dict[str, dict]:
    from evo.learning import rulelab, skills

    start = SAMPLES[sample]
    out: dict[str, dict] = {}
    s = skills.exam(checkpoint, tokenizer, n, device=device, parts=("new", "format_cal"), start=start)
    for kind, row in s["kinds"].items():
        out[f"skills/{kind}/new"] = {k: row[k] for k in KEEP if k in row}
        out[f"skills/{kind}/form"] = {k: row["format_cal"][k] for k in KEEP if k in row["format_cal"]}
    r = rulelab.exam(checkpoint, tokenizer, n, device=device, start=start)
    for part, row in r["kinds"].items():
        out[f"rulelab/{part}"] = {k: row[k] for k in KEEP if k in row}
    return out


def diagnose(metric: dict, min_items: int = 20) -> dict:
    """Where the errors of one ability concentrate, from the core's own answers: for every feature of the items
    (language, form of the rule, number of people, ...) the error rate with that value against the rest
    (two-proportion z-test), and how sure the core was when it was wrong - confidently wrong points to a wrong
    rule it has learned, unsure to something it has not learned yet."""
    import math

    correct = metric.get("correct") or ""
    feats = metric.get("feat") or []
    margin = metric.get("margin") or []
    n = len(correct)
    out: dict[str, Any] = {"items": n, "errors": correct.count("0"), "patterns": []}
    if not n or len(feats) != n:
        return out
    wrong = [c == "0" for c in correct]
    keys = sorted({k for f in feats for k in f})
    for key in keys:
        values = sorted({str(f.get(key)) for f in feats})
        if len(values) < 2:
            continue
        for v in values:
            ins = [w for w, f in zip(wrong, feats) if str(f.get(key)) == v]
            outs = [w for w, f in zip(wrong, feats) if str(f.get(key)) != v]
            if len(ins) < min_items or len(outs) < min_items:
                continue
            p1, p2 = sum(ins) / len(ins), sum(outs) / len(outs)
            pp = (sum(ins) + sum(outs)) / (len(ins) + len(outs))
            se = math.sqrt(max(1e-12, pp * (1 - pp) * (1 / len(ins) + 1 / len(outs))))
            z = (p1 - p2) / se
            if z >= 2.5:
                out["patterns"].append({"feature": key, "value": v, "error_rate": round(p1, 3), "elsewhere": round(p2, 3),
                                        "items": len(ins), "z": round(z, 2)})
    out["patterns"].sort(key=lambda d: -d["z"])
    if len(margin) == n and out["errors"]:
        right_m = sorted(m for m, w in zip(margin, wrong) if not w)
        typical = right_m[len(right_m) // 2] if right_m else 0.0
        conf_wrong = sum(1 for m, w in zip(margin, wrong) if w and m >= typical)
        out["confidently_wrong"] = round(conf_wrong / out["errors"], 3)      # wrong as surely as it is usually right
    return out


def load(path: Path | None = None) -> list[dict]:
    path = path or BOARD
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def record(name: str, checkpoint: str, board: dict, path: Path | None = None, role: str = "champion") -> dict:
    path = path or BOARD
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": time.time(), "date": time.strftime("%Y-%m-%d %H:%M"), "name": name, "checkpoint": checkpoint,
           "role": role, "board": board}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def board_of(name: str, rows: list[dict] | None = None, role: str = "champion") -> dict | None:
    for r in reversed(rows if rows is not None else load()):
        if r["name"] == name and r.get("role", "champion") == role:
            return r["board"]
    return None


def compare(before: dict, after: dict, alpha: float = 0.05) -> dict[str, Any]:
    """Item by item on every metric both boards share; Holm over all of them."""
    from evo.learning.skills import holm, mcnemar

    rows = {}
    for key in sorted(set(before) & set(after)):
        a, b = before[key].get("correct"), after[key].get("correct")
        if not a or not b or len(a) != len(b):
            continue
        rows[key] = {"before": before[key]["accuracy"], "after": after[key]["accuracy"], **mcnemar(a, b)}
    sig = holm({k: r["p"] for k, r in rows.items()}, alpha)
    better, worse = [], []
    for k, r in rows.items():
        r["significant"] = sig.get(k, False)
        if r["significant"]:
            (better if r["only_b"] > r["only_a"] else worse).append(k)
    return {"metrics": rows, "better": better, "worse": worse}


def deficits(rows: list[dict], champion: str) -> list[dict]:
    """Where the champion stands below what NOVA once reached (a loss - first) or below the target (a weak spot).
    Losses first (largest first); then weak spots, the half-learned ones before those still at chance (what was
    half learned is the cheapest to finish)."""
    current = board_of(champion, rows)
    if not current:
        return []
    best: dict[str, tuple[float, str]] = {}
    for r in rows:
        if r.get("role", "champion") != "champion":
            continue
        for k, v in r["board"].items():
            if k not in best or v["accuracy"] > best[k][0]:
                best[k] = (v["accuracy"], r["name"])
    out = []
    for k, v in current.items():
        b, who = best.get(k, (v["accuracy"], champion))
        above = (v["accuracy"] - v["chance"]) / max(1e-9, 1 - v["chance"])
        if b - v["accuracy"] >= 0.05 and who != champion:
            out.append({"metric": k, "kind": "loss", "now": v["accuracy"], "best": b, "best_by": who, "gap": round(b - v["accuracy"], 3)})
        elif kind_of(k) == "mastery" and v["accuracy"] < TARGET:
            out.append({"metric": k, "kind": "weak", "now": v["accuracy"], "chance": v["chance"], "above_chance": round(above, 3),
                        "gap": round(TARGET - v["accuracy"], 3)})
        elif kind_of(k) == "frontier" and v["accuracy"] < 0.99:
            out.append({"metric": k, "kind": "frontier", "now": v["accuracy"], "chance": v["chance"], "above_chance": round(above, 3),
                        "gap": round(1 - v["accuracy"], 3)})
    losses = sorted((d for d in out if d["kind"] == "loss"), key=lambda d: -d["gap"])
    weak = sorted((d for d in out if d["kind"] == "weak"), key=lambda d: (d["above_chance"] < 0.1, d["gap"]))
    frontier = sorted((d for d in out if d["kind"] == "frontier"), key=lambda d: d["gap"])
    return losses + weak + frontier


def text(rows: list[dict], champion: str) -> list[str]:
    cur = board_of(champion, rows)
    if not cur:
        return [f"Schopnosti: pre {champion} ešte nezmerané."]
    L = [f"Schopnosti ({champion}, {len(cur)} meraní po 200 úloh):"]
    for k in sorted(cur):
        L.append(f"  {k:28} {cur[k]['accuracy']:.2f} (náhoda {cur[k]['chance']:.2f})")
    d = deficits(rows, champion)
    losses = [x for x in d if x["kind"] == "loss"]
    if losses:
        L.append("ZHORŠENIA (riešim prednostne): " + "; ".join(f"{x['metric']} {x['now']:.2f} (najlepšie {x['best']:.2f}, {x['best_by']})" for x in losses))
    weak = [x for x in d if x["kind"] == "weak"][:5]
    if weak:
        L.append(f"Zvládnuteľné schopnosti pod cieľom {TARGET:.2f}: " + "; ".join(f"{x['metric']} {x['now']:.2f}" for x in weak))
    front = [x for x in d if x["kind"] == "frontier"][:5]
    if front:
        L.append("Hranice (cieľ: potvrdené zlepšenie, nie pevná hranica): " + "; ".join(f"{x['metric']} {x['now']:.2f}" for x in front))
    return L


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    me = sub.add_parser("measure")
    me.add_argument("checkpoint")
    me.add_argument("--name", required=True)
    me.add_argument("--role", default="champion", choices=["champion", "challenger", "confirm"])
    me.add_argument("--tokenizer", default=None)
    me.add_argument("--n", type=int, default=N_ITEMS)
    me.add_argument("--out", default=None, help="also write the board to this file (json)")
    me.add_argument("--sample", default="select", choices=list(SAMPLES))
    sh = sub.add_parser("show")
    sh.add_argument("--champion", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "measure":
        tok = args.tokenizer
        if tok is None:
            st = json.loads(Path("evo/engine/evo_state.json").read_text(encoding="utf-8"))
            tok = str(Path(st["best_known"]["dataset"]) / "tokenizer.json")
        b = measure(args.checkpoint, tok, args.n, sample=args.sample)
        record(args.name, args.checkpoint, b, role=args.role if args.sample == "select" else "confirm")
        if args.out:
            Path(args.out).write_text(json.dumps(b), encoding="utf-8")
        print(json.dumps({k: v["accuracy"] for k, v in b.items()}, ensure_ascii=False))
    else:
        champ = args.champion
        if champ is None:
            champ = json.loads(Path("evo/director/state.json").read_text(encoding="utf-8"))["champion"]["name"]
        print("\n".join(text(load(), champ)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
