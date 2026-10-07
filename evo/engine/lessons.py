"""
Lessons: what the director's own attempts say when they are taken together.

The director's memory of attempts is episodic ("this recipe was rejected on this champion"). This module is the
summary on top of it: evidence about recipes and about single settings, drawn from every judged attempt in the
director's log. It is evidence, not conclusions, and it keeps three rules:

  1. An attempt repeated with the same recipe, the same settings and the same champion is ONE experiment measured
     several times. The repeats say how much the judge's numbers move by chance (also for the code exam); they say
     nothing more about the recipe.
  2. A setting that came with better results is a correlation. Values of one setting usually came with different
     recipes, so the comparison is marked as confounded and its confidence stays low or medium. Only controlled
     pairs - the same recipe on the same champion with this one setting changed - can make it high.
  3. "Known", "looks promising" and "not enough data" are kept apart: every line carries the number of
     experiments, of champions and a confidence.

The loop it closes: experiments -> statistics -> lesson -> a controlled test of the lesson (one knob of a known
recipe turned) -> experiment. Hypotheses about new architectures still come from outside.

    python -m evo.engine.lessons            # the summary from evo/director/log.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

LOG = Path("evo/director/log.jsonl")
OUT = Path("evo/director/lessons.json")

# settings of a learning attempt that are compared; everything else in the flags makes the attempt "special"
KNOBS = {"web": "--bulk-frac", "lr": "--lr", "code": "--code-frac"}
SETTINGS = ("web", "lr", "code", "steps", "clones")
IGNORED_FLAGS = {"--warmup"}
MIN_EFFECT = 0.05        # percent: a difference below this (or below twice the repeat noise) is not read as a sign
JUDGE_NEEDS = 0.3        # percent: what the judge asks for (only to word a lesson; the constitution decides, not this)
EFFECTS = ("gain", "dataset", "web")
LIMITS = {"web": (0.5, 0.98), "lr": (5e-6, 2e-4), "code": (0.01, 0.4), "steps": (2000, 12000)}


def _pct(pair: Any) -> float | None:
    try:
        before, after = float(pair[0]), float(pair[1])
        return round(100 * (before - after) / before, 3) if before else None
    except (TypeError, ValueError, IndexError):
        return None


def observations(rows: list[dict]) -> list[dict]:
    """One record per judged attempt: what was changed and what the judge measured (positive = better)."""
    reverted = {r.get("name") for r in rows if r.get("event") == "probation" and r.get("reverted")}
    out = []
    for r in rows:
        v = r.get("verdict")
        if r.get("event") != "attempt" or not v or not r.get("recipe"):
            continue
        flags = r.get("flags") or {}
        settings: dict[str, float] = {}
        for name, flag in KNOBS.items():
            if flag in flags:
                try:
                    settings[name] = float(flags[flag])
                except (TypeError, ValueError):
                    pass
        if r.get("steps"):
            settings["steps"] = float(r["steps"])
        settings["clones"] = float(r.get("clones") or 1)
        other = sorted(f"{k}={flags[k]}" for k in flags if k not in KNOBS.values() and k not in IGNORED_FLAGS)
        if r.get("surgery"):
            other.append("surgery=" + str(r["surgery"].get("op")))
        if not flags:
            other.append("collective")
        sets = r.get("sets") or {}
        code = v.get("code") if isinstance(v.get("code"), dict) else {}
        out.append({
            "recipe": r["recipe"], "parent": r["recipe"].split("~")[0], "champion": r.get("champion"),
            "line": re.sub(r"-v\d+$", "", str(r.get("champion"))),
            "gain": (v.get("decision") or {}).get("gain_percent"), "dataset": _pct(sets.get("dataset")), "web": _pct(sets.get("web")),
            "code": (code["after"] - code["before"]) if isinstance(code.get("after"), (int, float)) and isinstance(code.get("before"), (int, float)) else None,
            "accepted": bool(v.get("accept")), "released": r.get("released"), "reverted": r.get("released") in reverted if r.get("released") else False,
            "settings": settings, "other": other, "date": r.get("date"),
        })
    return [o for o in out if o["gain"] is not None]


def experiments(obs: list[dict]) -> list[dict]:
    """Repeats of the same attempt on the same champion folded into one experiment (rule 1)."""
    groups: dict[str, list[dict]] = {}
    for o in obs:
        key = json.dumps([o["champion"], o["recipe"], o["settings"], o["other"]], sort_keys=True)
        groups.setdefault(key, []).append(o)
    out = []
    for g in groups.values():
        e = {k: g[0][k] for k in ("recipe", "parent", "champion", "line", "settings", "other")}
        e["runs"] = len(g)
        for k in EFFECTS + ("code",):
            vals = [o[k] for o in g if o[k] is not None]
            e[k] = round(mean(vals), 3) if vals else None
            e[k + "_runs"] = vals
        e["accepted"] = sum(o["accepted"] for o in g)
        e["reverted"] = sum(o["reverted"] for o in g)
        out.append(e)
    return out


def repeat_noise(exps: list[dict]) -> dict[str, Any]:
    """How much the judge's numbers differ between repeats of the same experiment (pooled over the repeated ones)."""
    out: dict[str, Any] = {"experiments": 0, "runs": 0}
    for k in ("gain", "code"):
        ss, dof, widest = 0.0, 0, 0.0
        for e in exps:
            vals = e[k + "_runs"]
            if len(vals) < 2:
                continue
            m = mean(vals)
            ss += sum((x - m) ** 2 for x in vals)
            dof += len(vals) - 1
            widest = max(widest, max(vals) - min(vals))
            if k == "gain":
                out["experiments"] += 1
                out["runs"] += len(vals)
        out[k + "_std"] = round((ss / dof) ** 0.5, 3) if dof else None
        out[k + "_widest"] = round(widest, 3) if dof else None
    return out


def threshold(noise: dict | None) -> float:
    std = (noise or {}).get("gain_std")
    return max(MIN_EFFECT, 2 * std) if std is not None else MIN_EFFECT


def by_recipe(exps: list[dict], thr: float = MIN_EFFECT) -> dict[str, dict]:
    """Evidence about each recipe (variations of a recipe count with it): how often, on how many champions, with what sign."""
    out: dict[str, dict] = {}
    for parent in sorted({e["parent"] for e in exps}):
        own = [e for e in exps if e["parent"] == parent]
        gains = [e["gain"] for e in own]
        pos, neg = sum(g > thr for g in gains), sum(g < -thr for g in gains)
        champions = len({e["champion"] for e in own})
        accepted, back = sum(e["accepted"] for e in own), sum(e["reverted"] for e in own)
        if accepted and back >= accepted:
            reading = "does not hold"                            # got through the judge, stepped back after probation
        elif pos and neg:
            reading = "depends on conditions"
        elif neg and neg >= 2 * len(own) / 3:
            reading = "negative"
        elif pos and pos >= 2 * len(own) / 3:
            reading = "positive" if accepted or max(gains) >= JUDGE_NEEDS else "small gain"
        else:
            reading = "no clear effect"
        runs = sum(e["runs"] for e in own)
        confidence = "high" if len(own) >= 3 and champions >= 2 and reading != "depends on conditions" else "medium" if len(own) >= 2 or runs >= 3 else "low"
        out[parent] = {"experiments": len(own), "runs": runs, "champions": champions, "gain_mean": round(mean(gains), 3),
                       "gain_best": max(gains), "gain_worst": min(gains), "positive": pos, "negative": neg,
                       "accepted": accepted, "stepped_back": back,
                       "reading": reading, "confidence": confidence}
    return out


def _plain(exps: list[dict]) -> list[dict]:
    return [e for e in exps if not e["other"]]


def controlled_pairs(exps: list[dict], setting: str) -> list[dict]:
    """Two experiments with the same recipe on the same champion that differ in this one setting only (rule 2)."""
    pairs = []
    for i, a in enumerate(exps):
        for b in exps[i + 1:]:
            if a["champion"] != b["champion"] or a["parent"] != b["parent"] or a["other"] != b["other"]:
                continue
            sa, sb = a["settings"], b["settings"]
            if setting not in sa or setting not in sb or sa[setting] == sb[setting]:
                continue
            if any(sa.get(k) != sb.get(k) for k in set(sa) | set(sb) if k != setting):
                continue
            lo, hi = (a, b) if sa[setting] < sb[setting] else (b, a)
            pairs.append({"champion": a["champion"], "recipe": a["parent"], "low": lo["settings"][setting], "high": hi["settings"][setting],
                          **{k: round(hi[k] - lo[k], 3) for k in EFFECTS if hi[k] is not None and lo[k] is not None}})
    return pairs


def by_setting(exps: list[dict], thr: float = MIN_EFFECT) -> dict[str, dict]:
    """Evidence about single settings, from plain learning attempts only: the lower half of the values against the
    upper half (medians, so one odd recipe does not decide), plus the controlled pairs."""
    plain = _plain(exps)
    out: dict[str, dict] = {}
    for s in SETTINGS:
        have = [e for e in plain if s in e["settings"]]
        values = sorted({e["settings"][s] for e in have})
        if len(values) < 2:
            continue
        cut = values[(len(values) - 1) // 2]                     # the lower half includes the middle value
        halves = {"low": [e for e in have if e["settings"][s] <= cut], "high": [e for e in have if e["settings"][s] > cut]}
        row: dict[str, Any] = {}
        for name, half in halves.items():
            row[name] = {"values": [min(e["settings"][s] for e in half), max(e["settings"][s] for e in half)], "experiments": len(half),
                         "recipes": sorted({e["parent"] for e in half}), "champions": len({e["champion"] for e in half}),
                         **{k: round(median([e[k] for e in half if e[k] is not None]), 3) for k in EFFECTS if any(e[k] is not None for e in half)}}
        row["high_minus_low"] = {k: round(row["high"][k] - row["low"][k], 3) for k in EFFECTS if k in row["high"] and k in row["low"]}
        pairs = controlled_pairs(have, s)
        row["controlled_pairs"] = pairs
        diff = row["high_minus_low"].get("gain", 0.0)
        agree = [p for p in pairs if abs(p.get("gain", 0.0)) > thr]
        row["pairs_with_a_sign"] = len(agree)
        row["confounded"] = not agree                            # no pair that separates the setting from the recipe
        same_sign = bool(agree) and all((p["gain"] > 0) == (agree[0]["gain"] > 0) for p in agree)
        broad = all(h["experiments"] >= 3 and len(h["recipes"]) >= 2 and h["champions"] >= 2 for h in (row["low"], row["high"]))
        if len(agree) >= 2 and same_sign:
            row["confidence"], row["better"] = "high", "high" if agree[0]["gain"] > 0 else "low"
        elif (len(agree) == 1) or (broad and abs(diff) > thr):
            row["confidence"] = "medium"
            row["better"] = ("high" if agree[0]["gain"] > 0 else "low") if agree else ("high" if diff > 0 else "low")
        else:
            row["confidence"] = "low"
            row["better"] = ("high" if diff > 0 else "low") if abs(diff) > thr else None
        out[s] = row
    return out


def _tag(setting: str, value: float) -> tuple[str, str]:
    """(text of the value for the recipe, tag for its name) - the same spelling the director's own variations use."""
    if setting == "lr":
        text = f"{value:.1e}".replace("e-0", "e-")
        return text, f"lr{text}"
    if setting == "web":
        return f"{value:.2f}", f"web{value:.2f}"
    if setting == "code":
        text = f"{value:.3f}".rstrip("0")
        return text, f"code{text}"
    return str(int(value)), f"steps{int(value)}"


def suggest(exps: list[dict], recipes: dict[str, dict], champion: str, taken: set[str] | None = None, thr: float = MIN_EFFECT) -> dict | None:
    """The next controlled test of a lesson: a known recipe with ONE setting moved to the side that looked better.

    Only lessons that are not settled (confidence below high) are tested; the recipe is the best plain one already
    judged on this champion from the side that looked worse, so the result makes a controlled pair with it."""
    taken = taken or set()
    lessons = by_setting(exps, thr)
    order = sorted((s for s in lessons if s in ("web", "lr", "code", "steps") and lessons[s]["confidence"] != "high" and lessons[s].get("better")),
                   key=lambda s: -abs(lessons[s]["high_minus_low"].get("gain", 0.0)))
    plain_here = [e for e in _plain(exps) if e["champion"] == champion and e["recipe"] in recipes and "flags" in recipes[e["recipe"]]]
    for s in order:
        row = lessons[s]
        better, worse = row["better"], "low" if row["better"] == "high" else "high"
        lo, hi = row[worse]["values"]
        bases = sorted((e for e in plain_here if s in e["settings"] and lo <= e["settings"][s] <= hi), key=lambda e: -e["gain"])
        target_vals = sorted(e["settings"][s] for e in _plain(exps) if s in e["settings"]
                             and row[better]["values"][0] <= e["settings"][s] <= row[better]["values"][1])
        if not bases or not target_vals:
            continue
        low_limit, high_limit = LIMITS[s]                        # the same range the director's own variations keep to
        target = min(high_limit, max(low_limit, target_vals[len(target_vals) // 2]))
        for base in bases:
            text, tag = _tag(s, target)
            name = f"{base['parent']}~{tag}"
            if name in recipes or name in taken or target == base["settings"][s]:
                continue
            old = recipes[base["recipe"]]
            new = {**old, "flags": dict(old["flags"]), "parent": base["parent"], "lesson": s}
            if s == "steps":
                new["steps"] = int(target)
            else:
                new["flags"][KNOBS[s]] = text
            return {"name": name, "recipe": new, "setting": s, "from": base["settings"][s], "to": target, "base": base["recipe"],
                    "reason": f"{s}: the {better} values came with a better result ({row['high_minus_low'].get('gain', 0.0):+.2f} % high minus low, "
                              f"confidence {row['confidence']}{', confounded' if row['confounded'] else ''}); "
                              f"testing it alone on {base['recipe']} ({base['settings'][s]:g} -> {target:g})"}
    return None


def summary(rows: list[dict]) -> dict[str, Any]:
    exps = experiments(observations(rows))
    noise = repeat_noise(exps)
    thr = threshold(noise)
    return {"attempts": sum(e["runs"] for e in exps), "experiments": len(exps), "champions": len({e["champion"] for e in exps}),
            "repeat_noise": noise, "threshold_percent": round(thr, 3), "recipes": by_recipe(exps, thr), "settings": by_setting(exps, thr)}


def load_rows(path: Path | None = None) -> list[dict]:
    path = path or LOG
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def text(s: dict) -> str:
    n = s["repeat_noise"]
    L = [f"LESSONS from {s['attempts']} judged attempts = {s['experiments']} different experiments on {s['champions']} champions",
         f"  repeats of the same experiment ({n['experiments']} experiments, {n['runs']} runs): gain differs by {n.get('gain_std')} % (std), "
         f"widest {n.get('gain_widest')} %; code exam by {n.get('code_std')} tasks (std), widest {n.get('code_widest')}",
         f"  a difference is read as a sign above {s['threshold_percent']} %", "", "  recipes:"]
    for name, r in sorted(s["recipes"].items(), key=lambda kv: kv[1]["gain_mean"]):
        L.append(f"    {name:<16} {r['reading']:<22} confidence {r['confidence']:<6} | {r['experiments']} experiments ({r['runs']} runs) on {r['champions']} champions, "
                 f"gain {r['gain_worst']:+.2f} .. {r['gain_best']:+.2f} %"
                 + (f", accepted {r['accepted']}" if r["accepted"] else "") + (f", stepped back {r['stepped_back']}" if r["stepped_back"] else ""))
    L += ["", "  settings (plain learning attempts; medians; high minus low):"]
    for name, r in s["settings"].items():
        d = r["high_minus_low"]
        L.append(f"    {name:<6} low {r['low']['values'][0]:g}-{r['low']['values'][1]:g} (n {r['low']['experiments']}, {len(r['low']['recipes'])} recipes) "
                 f"vs high {r['high']['values'][0]:g}-{r['high']['values'][1]:g} (n {r['high']['experiments']}, {len(r['high']['recipes'])} recipes): "
                 + ", ".join(f"{k} {d[k]:+.2f} %" for k in EFFECTS if k in d)
                 + f" | controlled pairs {len(r['controlled_pairs'])} ({r['pairs_with_a_sign']} with a clear sign), confidence {r['confidence']}"
                 + (", confounded" if r["confounded"] else "")
                 + (f", better: {r['better']}" if r.get("better") else ", no side better"))
    return "\n".join(L)


READINGS_SK = {"negative": "škodí", "positive": "pomáha", "small gain": "malý zisk, sudcovi nestačí", "does not hold": "prejde sudcom, neobstojí v skúšobnej lehote",
               "depends on conditions": "závisí od podmienok", "no clear effect": "bez zreteľného účinku"}
CONFIDENCE_SK = {"high": "vysoká", "medium": "stredná", "low": "nízka"}
SETTINGS_SK = {"web": "podiel webu", "lr": "rýchlosť učenia", "code": "podiel kódu", "steps": "počet krokov", "clones": "počet klonov"}


def lines_sk(s: dict, limit: int = 6) -> list[str]:
    """A few lines for the director's report to the Creator."""
    if not s["experiments"]:
        return []
    n = s["repeat_noise"]
    L = [f"Poučenia z vlastných pokusov ({s['attempts']} pokusov = {s['experiments']} rôznych experimentov na {s['champions']} šampiónoch):"]
    if n.get("gain_std") is not None:
        L.append(f"  opakovanie toho istého pokusu: zisk kolíše o {n['gain_std']} %, skúška z kódu o {n.get('code_std')} úlohy (najviac {n.get('code_widest')})")
    known = sorted(s["recipes"].items(), key=lambda kv: ({"high": 0, "medium": 1, "low": 2}[kv[1]["confidence"]], kv[1]["gain_mean"]))
    for name, r in known[:limit]:
        L.append(f"  {name}: {READINGS_SK[r['reading']]} (istota {CONFIDENCE_SK[r['confidence']]}; experimenty: {r['experiments']}, šampióni: {r['champions']}, "
                 f"zisk {r['gain_worst']:+.2f} až {r['gain_best']:+.2f} %)")
    for name, r in s["settings"].items():
        if r.get("better"):
            side = "vyššie" if r["better"] == "high" else "nižšie"
            L.append(f"  {SETTINGS_SK[name]}: {side} hodnoty vyšli lepšie (o {abs(r['high_minus_low'].get('gain', 0.0)):.2f} %), istota {CONFIDENCE_SK[r['confidence']]}"
                     + ("; nie je to oddelené od vplyvu receptu, treba overiť" if r["confounded"] else f"; potvrdené riadenými pármi: {r['pairs_with_a_sign']}"))
    return L


def write(path: Path | None = None, rows: list[dict] | None = None) -> dict:
    s = summary(load_rows() if rows is None else rows)
    path = path or OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=str(LOG))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    s = summary(load_rows(Path(args.log)))
    print(json.dumps(s, indent=1, ensure_ascii=False) if args.json else text(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
