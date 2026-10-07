"""
The experiment ledger: one record for everything that was tried, under which conditions, and how it ended.

Tournament runs (evo.engine.compare_arch), the director's attempts, challenges and new generations all land in
evo/ledger.jsonl, one line each:

    id, date, kind (tournament / attempt / challenge / generation), name, parent, change, config,
    train (steps, lr, seed, running text), metrics (dataset, web, web as running text, code exam),
    cost (state, training and writing speed), identity class, verdict, reason, gain, source

The ledger is the system's memory of its own experiments: before something is tried again, `tried()` says
whether it was tried and how it ended; `noise()` says how much the same experiment differs between seeds, so a
difference smaller than that is not taken for an improvement.

    python -m evo.engine.ledger --backfill          # take in the existing reports and the director's log
    python -m evo.engine.ledger --show 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

LEDGER = Path("evo/ledger.jsonl")
REPORT = Path("evo/learning/arch_compare/report.json")
DIRECTOR_LOG = Path("evo/director/log.jsonl")


def load(path: Path | None = None) -> list[dict]:
    path = path or LEDGER
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def record(entry: dict | None, path: Path | None = None) -> dict | None:
    """Append an entry (None and entries already in the ledger are skipped). Returns what was written."""
    if not entry:
        return None
    path = path or LEDGER
    rows = load(path)
    if any(r.get("key") == entry["key"] for r in rows):
        return None
    out = {"id": f"EXP-{len(rows) + 1:05d}", "date": entry.pop("date", None) or time.strftime("%Y-%m-%d %H:%M"), **entry}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return out


def from_report_row(v: dict, date: str | None = None, steps: int | None = None) -> dict | None:
    """A measured run of the comparison tool (one model trained from scratch)."""
    if not v.get("name"):
        return None
    loss, cpu = v.get("loss") or {}, v.get("cpu") or {}
    entry: dict[str, Any] = {
        "key": f"tournament:{v['name']}:{v.get('seed', 1001)}", "date": date, "kind": "tournament", "name": v["name"], "parent": "scratch",
        "change": {"nova8": "generation-8 candidate " + str((v.get("override") or {}).get("pattern", "")),
                   "transformer": "transformer", "nova": "generation-7 core"}.get(v.get("arch"), str(v.get("arch"))),
        "config": v.get("override"),
        "train": {k: x for k, x in {"steps": steps, "lr": v.get("lr"), "seed": v.get("seed", 1001), "carry": v.get("carry"),
                                    "carry_share": v.get("carry_share"), "compiled": v.get("compile")}.items() if x is not None},
        "source": f"{REPORT}:{v['name']}",
    }
    if not loss:
        entry.update({"verdict": "failed", "reason": str(v.get("error") or "no result")[:300]})
        return entry
    entry["metrics"] = {k: x for k, x in {"dataset": loss.get("dataset"), "web": loss.get("web"), "web_carried": v.get("web_carried"),
                                          "code": f"{v.get('code_solved')}/{v.get('code_tasks')}" if v.get("code_tasks") else None}.items() if x is not None}
    fine = v.get("code_fine") or {}
    if fine:
        entry["metrics"].update({"code_score": fine.get("score"), "code_nll": fine.get("nll")})
    entry["cost"] = {k: x for k, x in {"state_kb": cpu.get("state_kb_after_writing"), "train_tok_s": v.get("train_tokens_per_s"),
                                       "cpu_write_tok_s": cpu.get("write_tokens_per_s"), "train_hours": v.get("train_hours")}.items() if x is not None}
    if v.get("identity"):
        entry["identity"] = v["identity"].get("class")
    entry["verdict"] = "measured"
    return entry


def from_director_event(rec: dict) -> dict | None:
    """An attempt, a challenge or a finished generation from the director's log (events without a verdict are skipped)."""
    event = rec.get("event")
    v = rec.get("verdict")
    if event not in ("attempt", "grow_segment", "challenge") or not (v or rec.get("error") or (event == "attempt" and rec.get("rc") not in (0, None))):
        return None
    when = rec.get("date") or time.strftime("%Y-%m-%d %H:%M")
    name = rec.get("line") or rec.get("recipe") or "?"
    kind = "generation" if event == "grow_segment" else "challenge" if rec.get("challenger") else "attempt"
    entry: dict[str, Any] = {
        "key": f"{kind}:{name}:{rec.get('attempt', '')}:{when}", "date": when, "kind": kind, "name": name, "parent": rec.get("champion"),
        "change": rec.get("note") or (json.dumps(rec.get("flags")) if rec.get("flags") else name),
        "train": {k: x for k, x in {"steps": rec.get("steps") or rec.get("steps_done"), "hours": rec.get("hours")}.items() if x is not None},
        "source": f"{DIRECTOR_LOG}",
    }
    if rec.get("identity"):
        entry["identity"] = rec["identity"].get("class")
    if v:
        sets = rec.get("sets") or {}
        entry["metrics"] = {k: s[1] for k, s in sets.items() if isinstance(s, (list, tuple)) and len(s) > 1}
        if isinstance(v.get("code"), dict):
            entry["metrics"]["code"] = v["code"].get("after")
        entry["gain_percent"] = (v.get("decision") or {}).get("gain_percent")
        entry["verdict"] = "accepted" if v.get("accept") else "rejected"
        entry["reason"] = "; ".join(v.get("reasons") or []) or ("released as " + str(rec.get("released")) if rec.get("released") else "")
    else:
        entry["verdict"] = "failed"
        entry["reason"] = str(rec.get("error") or rec.get("tail") or f"exit code {rec.get('rc')}")[:300]
    return entry


def backfill(report: Path | None = None, director_log: Path | None = None, path: Path | None = None) -> int:
    n = 0
    report = report or REPORT
    if report.exists():
        rep = json.loads(report.read_text(encoding="utf-8"))
        for v in (rep.get("results") or {}).values():
            n += bool(record(from_report_row(v, rep.get("date"), rep.get("steps")), path))
    director_log = director_log or DIRECTOR_LOG
    if director_log.exists():
        for line in director_log.read_text(encoding="utf-8").splitlines():
            try:
                n += bool(record(from_director_event(json.loads(line)), path))
            except ValueError:
                continue
    return n


def _same(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def tried(config: dict | None = None, name: str | None = None, rows: list[dict] | None = None) -> list[dict]:
    """Earlier experiments with this architecture (config) or this name, oldest first."""
    rows = load() if rows is None else rows
    return [r for r in rows if (config is not None and r.get("config") is not None and _same(r["config"], config))
            or (name is not None and r.get("name") == name)]


def noise(rows: list[dict] | None = None, metrics: tuple[str, ...] = ("dataset", "web")) -> dict[str, dict]:
    """Spread between seeds: runs with the same architecture and training, differing only in the seed.

    Returns {name of the first run: {"runs": n, metric: {"mean", "std", "min", "max"}}} for groups of two or more."""
    rows = load() if rows is None else rows
    groups: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("kind") != "tournament" or r.get("verdict") != "measured" or not r.get("config"):
            continue
        train = {k: v for k, v in (r.get("train") or {}).items() if k != "seed"}
        groups.setdefault(json.dumps([r["config"], train], sort_keys=True), []).append(r)
    out = {}
    for g in groups.values():
        if len({(r.get("train") or {}).get("seed") for r in g}) < 2:
            continue
        stats: dict[str, Any] = {"runs": len(g)}
        for m in metrics:
            vals = [r["metrics"][m] for r in g if m in (r.get("metrics") or {})]
            if len(vals) >= 2:
                stats[m] = {"mean": round(mean(vals), 4), "std": round(pstdev(vals), 4), "min": min(vals), "max": max(vals)}
        out[g[0]["name"]] = stats
    return out


def noise_percent(rows: list[dict] | None = None, metric: str = "dataset") -> float | None:
    """The largest relative spread (std / mean, in percent) seen between seeds; None while nothing was repeated."""
    spreads = [100 * s[metric]["std"] / s[metric]["mean"] for s in noise(rows).values() if metric in s and s[metric]["mean"]]
    return round(max(spreads), 3) if spreads else None


def text(rows: list[dict], last: int = 30) -> str:
    L = [f"EXPERIMENT LEDGER: {len(rows)} records"]
    for r in rows[-last:]:
        m, c = r.get("metrics") or {}, r.get("cost") or {}
        parts = [f"{r.get('id')} {str(r.get('date'))[:16]} {r.get('kind', ''):<10} {str(r.get('name'))[:28]:<28} {r.get('verdict', ''):<8}"]
        if m:
            parts.append(" ".join(f"{k} {v}" for k, v in m.items()))
        if c.get("state_kb") is not None:
            parts.append(f"state {c['state_kb']} kB")
        if r.get("identity"):
            parts.append(r["identity"])
        if r.get("gain_percent") is not None:
            parts.append(f"gain {r['gain_percent']:+.2f} %")
        if r.get("reason"):
            parts.append("- " + str(r["reason"])[:110])
        L.append("  " + " | ".join(parts))
    spread = noise(rows)
    for name, s in spread.items():
        L.append(f"  between seeds ({name}, {s['runs']} runs): " + ", ".join(f"{k} {v['mean']} +- {v['std']}" for k, v in s.items() if k != "runs"))
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backfill", action="store_true", help="take in the comparison report and the director's log")
    ap.add_argument("--show", type=int, default=0, help="print the last N records")
    args = ap.parse_args(argv)
    if args.backfill:
        print(f"{backfill()} new records")
    if args.show or not args.backfill:
        print(text(load(), args.show or 30))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
