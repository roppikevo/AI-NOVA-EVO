"""
Run several autonomous core-evolution cycles in a row.

    python -m evo.engine.core_evolution_loop --cycles 8

Each cycle = CoreEvolutionEngine.run_once(): teacher proposes a new core,
safety + contract checks, smoke test, robust training (3 seeds) on the
current dataset, efficiency-aware evaluation, auto-promotion on PASS.

Stop early by creating  evo/core_evolution/STOP_LOOP  (checked between
cycles). A failing cycle is logged and the loop continues.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

STOP_FILE = Path("evo/core_evolution/STOP_LOOP")


def summarize(r: dict) -> dict:
    ev = r.get("evaluation") or {}
    eff = (((r.get("training") or {}).get("robust") or {}).get("metrics") or {}).get("efficiency") or {}
    return {
        "candidate": r.get("candidate"),
        "status": r.get("status") or r.get("decision"),
        "reason": ev.get("reason") or r.get("stage") or "; ".join(map(str, r.get("errors") or []))[:200] or None,
        "candidate_loss": ev.get("candidate_loss"),
        "parent_loss": ev.get("parent_loss"),
        "params": eff.get("parameters"),
        "cpu_tok_s": eff.get("cpu_tokens_per_sec"),
        "vram_mb": eff.get("peak_vram_mb"),
        "promotion": (r.get("promotion") or {}).get("status") if isinstance(r.get("promotion"), dict) else r.get("promotion"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--robust-steps", type=int, default=1000)
    args = ap.parse_args(argv)

    from evo.engine.core_evolution_engine import CoreEvolutionEngine

    history = []
    for i in range(1, args.cycles + 1):
        if STOP_FILE.exists():
            print("STOP_LOOP found - stopping")
            STOP_FILE.unlink()
            break
        t0 = time.time()
        print(f"\n######## CYCLE {i}/{args.cycles} ########", flush=True)
        try:
            result = CoreEvolutionEngine().run_once(robust_steps=args.robust_steps)
            s = summarize(result)
        except Exception as exc:
            s = {"status": "CYCLE_ERROR", "error": f"{type(exc).__name__}: {exc}",
                 "trace": traceback.format_exc()[-1500:]}
        try:
            from evo.learning.decision_log import append, record_from_result

            src = result if s.get("status") != "CYCLE_ERROR" else s
            append(record_from_result(src))
        except Exception as exc:
            print("decision log failed:", exc)
        s["cycle"] = i
        s["minutes"] = round((time.time() - t0) / 60, 1)
        history.append(s)
        print("CYCLE RESULT:", json.dumps(s, default=str), flush=True)

    print("\n======== LOOP SUMMARY ========")
    for s in history:
        print(json.dumps({k: s.get(k) for k in ("cycle", "candidate", "status", "reason",
                                                 "candidate_loss", "parent_loss", "params",
                                                 "cpu_tok_s", "promotion", "minutes", "error")},
                         default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
