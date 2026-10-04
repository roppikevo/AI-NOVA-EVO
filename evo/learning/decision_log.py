"""
Decision log: every core-evolution attempt becomes a labelled example.

Later NOVA learns (with choose()) to predict an attempt's outcome before
spending GPU time on it. Records are appended to
evo/core_evolution/decisions.jsonl:

    {"candidate", "mentor", "source_chars", "outcome", "stage",
     "candidate_loss", "parent_loss", "timestamp"}

outcome is one of: PASS, REJECT_LOSS, REJECT_EFFICIENCY, SMOKE_FAILED,
CONTRACT_FAILED, SAFETY_FAILED, NO_SOURCE, INFRA_RETRY, ERROR.

    python -m evo.learning.decision_log --backfill   # from results/*.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

LOG = Path("evo/core_evolution/decisions.jsonl")
RESULTS = Path("evo/core_evolution/results")
CANDIDATES = Path("evo/core_evolution/candidates")


def outcome_of(result: dict[str, Any]) -> tuple[str, str | None]:
    if result.get("error") or result.get("status") == "CYCLE_ERROR":
        err = str(result.get("error", ""))
        return ("NO_SOURCE" if "core source" in err else "ERROR"), None
    ev = result.get("evaluation") or {}
    decision = ev.get("decision") or result.get("decision") or result.get("status")
    reason = str(ev.get("reason") or "")
    stage = result.get("stage")
    if decision in ("PASS", "PROMOTE"):
        return "PASS", stage
    if decision == "RETRY":
        return "INFRA_RETRY", stage
    if stage == "source_contract":
        return "CONTRACT_FAILED", stage
    if stage in ("source_safety", "safety"):
        return "SAFETY_FAILED", stage
    if "smoke" in reason or stage == "runtime_smoke":
        return "SMOKE_FAILED", stage
    if "efficiency" in reason:
        return "REJECT_EFFICIENCY", stage
    return "REJECT_LOSS", stage


def record_from_result(result: dict[str, Any], mentor: str | None = None) -> dict[str, Any]:
    cid = result.get("candidate")
    outcome, stage = outcome_of(result)
    teacher = result.get("teacher") or {}
    src = CANDIDATES / str(cid) / "blocks_scan.py"
    ev = result.get("evaluation") or {}
    return {
        "candidate": cid,
        "mentor": mentor or teacher.get("mentor") or teacher.get("model"),
        "source_chars": src.stat().st_size if cid and src.exists() else None,
        "outcome": outcome,
        "stage": stage,
        "candidate_loss": ev.get("candidate_loss"),
        "parent_loss": ev.get("parent_loss"),
        "timestamp": time.time(),
    }


def append(record: dict[str, Any], log: Path = LOG) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def backfill(results: Path = RESULTS, log: Path = LOG) -> Counter:
    seen = set()
    if log.exists():
        seen = {json.loads(l).get("candidate") for l in log.read_text().splitlines() if l.strip()}
    counts: Counter = Counter()
    for path in sorted(results.glob("GEN*.json")):
        if path.name.endswith(".official.json"):
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rec = record_from_result(result)
        if rec["candidate"] in seen:
            continue
        append(rec, log)
        counts[rec["outcome"]] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    args = ap.parse_args(argv)
    if args.backfill:
        print("backfilled:", dict(backfill()))
    if LOG.exists():
        rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
        print(f"{len(rows)} decisions:", dict(Counter(r["outcome"] for r in rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
