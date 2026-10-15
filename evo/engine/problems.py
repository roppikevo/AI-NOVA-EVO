"""
Problems: NOVA's long memory of what it cannot do yet (the Creator's conditions 15.10, points 2, 3, 6, 7, 11).

Every weak or lost ability is a problem with a record: what it is, where it started, every experiment made on it
(the hypothesis, its prediction, the data, the cost, what happened to the ability and to all the others), the
hypotheses that failed, the ways that worked, how sure the latest number is, and when it is opened again.

    open      not at the target; there are untried ways
    parked    every way tried so far failed - it waits for a reason to come back (below), not for a new attempt
              with the same means
    solved    at the target on the selection sample AND on the confirmation sample no decision ever looked at

Two classes (the Creator, 15.10): "mastery" - a single masterable skill, solved at >= 90 % on both samples;
"frontier" - an edge of reasoning (unknown form, unseen words, longer chains, an untrained kind), never solved by a
bar: its goal is a confirmed improvement and no loss.

A problem comes back (is reopened) when: it got worse again (regression); a way of practising exists that was never
tried on it; the core changed (a new champion) and another ability improved since it was parked; new material for
its family was registered. It never comes back to repeat what it already tried under the same conditions.

The next problem is chosen by: losses before weak spots, then expected gain per hour from the record of the ways
(what worked on similar problems), then the gap to the target.

    python -m evo.engine.problems              # the register, as the director sees it
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

REGISTER = Path("evo/director/problems.json")
SOLVED_AT = 0.90            # the Creator's 90-99 %: below it nothing is solved
NAMES = {"arith": "počítanie", "compare": "porovnávanie čísel", "sequence": "postupnosti", "deduction": "logické vyvodzovanie",
         "logic-if": "ak–potom", "ordering": "poradie a vzťahy", "recall": "vybavenie z kontextu", "code": "čítanie kódu",
         "same": "vzťahy – nové úlohy", "phrasing": "vzťahy – nová formulácia", "words": "vzťahy – nové slová",
         "deeper": "vzťahy – dlhšie reťaze"}


def describe(metric: str) -> str:
    parts = metric.split("/")
    name = NAMES.get(parts[1], parts[1])
    if parts[0] == "skills":
        return f"{name} ({'nové úlohy' if parts[2] == 'new' else 'neznámy tvar otázky'})"
    return name


def ci_half(acc: float, n: int = 200) -> float:
    return round(1.96 * math.sqrt(max(acc * (1 - acc), 1e-6) / n), 3)


def load(path: Path | None = None) -> dict:
    path = path or REGISTER
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"problems": {}, "materials": {}}


def save(reg: dict, path: Path | None = None) -> None:
    path = path or REGISTER
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, indent=1, ensure_ascii=False), encoding="utf-8")


def sync(reg: dict, champion: str, board: dict, confirm: dict | None, deficits: list[dict], strategies: list[str],
         now: float | None = None) -> list[str]:
    """Bring the register up to date with the champion's measurements. Returns what changed (for the log)."""
    now = time.time() if now is None else now
    date = time.strftime("%Y-%m-%d %H:%M", time.localtime(now))
    probs = reg.setdefault("problems", {})
    notes = []
    gaps = {d["metric"]: d for d in deficits}
    for metric, v in board.items():
        acc = v["accuracy"]
        conf = (confirm or {}).get(metric, {}).get("accuracy")
        p = probs.get(metric)
        if p is None:
            if metric not in gaps:
                continue
            p = probs[metric] = {"metric": metric, "description": describe(metric), "status": "open", "opened": date,
                                 "baseline": acc, "history": [], "tried": [], "failed_hypotheses": [], "worked": [],
                                 "reopened": [], "best": acc}
            notes.append(f"opened {metric} ({acc:.2f})")
        prev = p.get("last", {})
        p["last"] = {"accuracy": acc, "confirm": conf, "champion": champion, "date": date, "uncertainty": ci_half(acc)}
        p["best"] = max(p.get("best", acc), acc)
        from evo.engine.abilities import kind_of

        p.setdefault("class", kind_of(metric))
        # only a masterable skill can be "solved" (the Creator: 90-99 % is for single skills, not for all reasoning);
        # a frontier stays open as long as there is a reasonable way to push it further
        solved = p["class"] == "mastery" and acc >= SOLVED_AT and conf is not None and conf >= SOLVED_AT
        if solved and p["status"] != "solved":
            p["status"] = "solved"
            p["solved"] = {"date": date, "champion": champion, "accuracy": acc, "confirm": conf}
            notes.append(f"solved {metric} ({acc:.2f}, confirmed {conf:.2f})")
            continue
        if p["status"] == "solved" and not solved:
            reopen(p, f"regression: {acc:.2f} (confirmation {conf})", date, keep_tried=False)
            notes.append(f"reopened {metric}: regression")
            continue
        if p["status"] == "parked":
            reason = None
            if prev.get("accuracy") is not None and prev["accuracy"] - acc >= 0.05:
                reason = f"regression {prev['accuracy']:.2f} -> {acc:.2f}"
            elif any(s not in p["tried"] for s in strategies):
                reason = "a way of practising never tried on it is available: " + ", ".join(s for s in strategies if s not in p["tried"])
            elif prev.get("champion") and prev["champion"] != champion and p.get("parked_champion") != champion:
                reason = f"the core changed ({prev['champion']} -> {champion}) - abilities it builds on may have improved"
            elif any(m.get("family") == metric.split("/")[1] and m.get("date", "") > p.get("parked", "") for m in reg.get("materials", {}).values()):
                reason = "new material for its family"
            if reason:
                reopen(p, reason, date, keep_tried=reason.startswith("a way"))
                notes.append(f"reopened {metric}: {reason}")
    return notes


def reopen(p: dict, reason: str, date: str, keep_tried: bool) -> None:
    p["status"] = "open"
    p["reopened"].append({"date": date, "reason": reason})
    if not keep_tried:
        p["tried_before"] = p.get("tried_before", []) + [p["tried"]]
        p["tried"] = []                      # conditions changed: the ways may be tried again under the new ones


def park_if_exhausted(p: dict, strategies: list[str], champion: str, date: str) -> bool:
    if p["status"] == "open" and all(s in p["tried"] for s in strategies):
        p["status"] = "parked"
        p["parked"] = date
        p["parked_champion"] = champion
        return True
    return False


def choose(reg: dict, deficits: list[dict], strategies: list[str], order: list[str]) -> tuple[str, str] | None:
    """(metric, strategy): the first open problem in the deficits' order that still has an untried way; the way is
    the first untried one in `order` (ways that worked first)."""
    probs = reg.get("problems", {})
    for d in deficits:
        p = probs.get(d["metric"])
        if not p or p["status"] != "open":
            continue
        for s in order:
            if s in strategies and s not in p["tried"]:
                return d["metric"], s
    return None


def hypothesis(strategy: str, metric: str, diag: dict | None) -> dict:
    """What the way assumes about the cause, and what it predicts - so that the result can refute it."""
    d = describe(metric)
    if strategy == "focus-errors" and diag and diag.get("patterns"):
        pt = diag["patterns"][0]
        return {"cause": f"chyby v „{d}“ sa sústreďujú na {pt['feature']}={pt['value']} (chybovosť {pt['error_rate']:.2f} vs {pt['elsewhere']:.2f}, z={pt['z']})",
                "prediction": f"chybovosť pri {pt['feature']}={pt['value']} klesne aspoň o 0,10 a schopnosť stúpne aspoň o 0,05", "pattern": pt}
    text = {"practice": ("chýbali cvičenia tohto druhu", "schopnosť stúpne aspoň o 0,05 na výberovej aj potvrdzovacej vzorke"),
            "longer": ("cvičenia boli, ale príliš málo/krátko", "schopnosť stúpne aspoň o 0,05; viac krokov pomôže viac než bežné cvičenie"),
            "gentle": ("učenie vytláča iné schopnosti (zabúdanie)", "schopnosť stúpne a žiadna iná neklesne o viac než 0,05"),
            "focus-errors": ("chyby majú vzorec", "chybovosť vo vzorci klesne")}[strategy]
    conf = (diag or {}).get("confidently_wrong")
    cause = text[0] + (f"; {int(conf * 100)} % chýb s plnou istotou – pravdepodobne naučené zlé pravidlo" if conf is not None and conf >= 0.5 else "")
    return {"cause": cause, "prediction": text[1]}


def record(reg: dict, metric: str, entry: dict, strategies: list[str], champion: str) -> None:
    """One experiment on one problem: supported or refuted, side effects, cost, efficiency."""
    p = reg["problems"][metric]
    before, after = entry.get("before"), entry.get("after")
    gain = None if before is None or after is None else round(after - before, 3)
    entry["gain"] = gain
    entry["supported"] = bool(gain is not None and gain >= 0.05 and not entry.get("crashed"))
    if entry.get("steps") and gain is not None:
        entry["gain_per_1k_steps"] = round(1000 * gain / entry["steps"], 4)
    if entry.get("hours") and gain is not None:
        entry["gain_per_hour"] = round(gain / max(entry["hours"], 1e-3), 3)
    p["history"].append(entry)
    if not entry.get("crashed"):
        if entry["strategy"] not in p["tried"]:
            p["tried"].append(entry["strategy"])
        (p["worked"] if entry["supported"] else p["failed_hypotheses"]).append(
            {"strategy": entry["strategy"], "cause": entry.get("hypothesis", {}).get("cause"), "gain": gain, "date": entry.get("date")})
        if not entry["supported"]:
            park_if_exhausted(p, strategies, champion, entry.get("date", ""))
    if after is not None and after > p.get("best", 0):
        p["best"] = after


def efficiency(reg: dict) -> dict[str, dict]:
    """Per way of practising over all problems: tries, supported, mean gain, gain per hour (point 7)."""
    out: dict[str, dict] = {}
    for p in reg.get("problems", {}).values():
        for e in p["history"]:
            if e.get("crashed"):
                continue
            s = out.setdefault(e["strategy"], {"tries": 0, "supported": 0, "gain": 0.0, "hours": 0.0})
            s["tries"] += 1
            s["supported"] += int(e.get("supported", False))
            s["gain"] += e.get("gain") or 0.0
            s["hours"] += e.get("hours") or 0.0
    for s in out.values():
        s["mean_gain"] = round(s["gain"] / s["tries"], 3) if s["tries"] else 0.0
        s["gain_per_hour"] = round(s["gain"] / s["hours"], 3) if s["hours"] else 0.0
    return out


def text(reg: dict) -> list[str]:
    probs = reg.get("problems", {})
    if not probs:
        return ["Register problémov: prázdny."]
    L = ["Register problémov (otvorené / odložené / vyriešené):"]
    for st in ("open", "parked", "solved"):
        for m, p in sorted(probs.items()):
            if p["status"] != st:
                continue
            last = p.get("last", {})
            L.append(f"  [{st}{'' if p.get('class', 'mastery') == 'mastery' else ', hranica'}] {p['description']}: {p['baseline']:.2f} -> {last.get('accuracy', 0):.2f} (±{last.get('uncertainty', 0):.2f}"
                     f"{', potvrdenie ' + format(last['confirm'], '.2f') if last.get('confirm') is not None else ''}); "
                     f"skúšané: {', '.join(p['tried']) or '-'}; pokusov {len(p['history'])}")
    eff = efficiency(reg)
    if eff:
        L.append("Účinnosť spôsobov: " + "; ".join(f"{k} {v['supported']}/{v['tries']} (zisk/h {v['gain_per_hour']:+.3f})" for k, v in eff.items()))
    return L


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    reg = load()
    print(json.dumps(reg, indent=1, ensure_ascii=False) if args.json else "\n".join(text(reg)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
