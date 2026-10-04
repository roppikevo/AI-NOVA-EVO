"""
Architecture tournament: which way of spending ~10 M parameters learns best?

Every variant is trained FROM SCRATCH with the same data mix, the same number
of training tokens and the same schedule, with --no-activate (the deployed
model is never touched), and measured on the same validation data.
The current core takes part as variant "A", so the comparison is fair: the
deployed weights have seen far more data than a tournament run.

Why: in the current core 6.3 M of 9.9 M parameters are the token table
(16384 x 384, tied with the output layer). A factorized embedding
(16384 x d_embed + small projections) frees parameters for more layers at the
same model size.

    python -m evo.engine.arch_tournament --steps 18000

Result: evo/learning/arch_tournament.jsonl (one row per variant) + a ranking.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

LOG = Path("evo/learning/arch_tournament.jsonl")

VARIANTS: dict[str, dict] = {
    "A-current": {},
    "B1-e128-L12": {"d_embed": 128, "num_layers": 12},
    "B2-e192-L10": {"d_embed": 192, "num_layers": 10},
    "B3-e128-d512-L7": {"d_embed": 128, "d_model": 512, "d_state": 512, "num_layers": 7},
}

COMMON = ["--from-scratch", "--no-activate", "--batch-size", "64", "--lr", "3e-4", "--warmup", "500",
          "--eval-every", "3000", "--patience", "99", "--max-hours", "3.0",
          "--extra-dirs", "data/teacher_v1,data/web_v1", "--bulk-dir", "data/bulk_v1", "--bulk-frac", "0.7",
          "--code-frac", "0.05"]


def cpu_speed(config: dict) -> dict:
    from evo.engine.architecture_factory import build_model
    from nova.efficiency import measure_cpu

    m = measure_cpu(build_model(config), int(config["vocab_size"]))
    return {k: m.get(k) for k in ("prompt_tokens_per_s", "gen_tokens_per_s", "params") if k in m} or m


def run_variant(name: str, override: dict, steps: int, runner=subprocess.run) -> dict:
    from evo.engine.ab_test import parse_report

    cmd = [sys.executable, "-m", "evo.engine.long_train", "--steps", str(steps)] + COMMON
    if override:
        cmd += ["--config-override", json.dumps(override)]
    t0 = time.time()
    p = runner(cmd, capture_output=True, text=True, timeout=4 * 3600)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "")[-800:])
    rep = parse_report(p.stdout)
    curve = [l for l in p.stdout.splitlines() if l.startswith("step ")]
    return {"variant": name, "override": override, "best_val": rep["best_val"], "params": rep.get("params"),
            "train_hours": rep["hours"], "steps_done": rep["steps_done"], "stop": rep["stop_reason"],
            "config": rep.get("config"), "curve": [c.split(" best")[0] for c in curve],
            "wall_minutes": round((time.time() - t0) / 60, 1)}


def ranking(rows: list[dict]) -> list[dict]:
    return sorted([r for r in rows if r.get("best_val") is not None], key=lambda r: r["best_val"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=18000)
    ap.add_argument("--only", default="", help="comma list of variant names")
    ap.add_argument("--variants", default="", help='JSON {name: config override} to run instead of the built-in list')
    ap.add_argument("--max-hours", type=float, default=3.0, help="time limit per variant")
    args = ap.parse_args(argv)

    if args.variants:
        VARIANTS.clear()
        VARIANTS.update(json.loads(args.variants))
    COMMON[COMMON.index("--max-hours") + 1] = str(args.max_hours)
    names = [n for n in VARIANTS if not args.only or n in args.only.split(",")]
    run_id = time.strftime("%Y%m%d-%H%M")
    rows = []
    for name in names:
        print(f"=== {name} {VARIANTS[name]} ===", flush=True)
        try:
            row = run_variant(name, VARIANTS[name], args.steps)
            try:
                row["cpu"] = cpu_speed(row["config"])
            except Exception as exc:
                row["cpu"] = {"error": str(exc)[:200]}
        except Exception as exc:
            row = {"variant": name, "override": VARIANTS[name], "best_val": None,
                   "error": f"{type(exc).__name__}: {exc}"[:600]}
        row.update({"run": run_id, "steps": args.steps, "time": time.time()})
        rows.append(row)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k not in ("curve", "config")}, ensure_ascii=False), flush=True)
        for c in row.get("curve", [])[-3:]:
            print("   ", c)
    print("=== RANKING (same tokens, from scratch; lower val = better) ===")
    for i, r in enumerate(ranking(rows), 1):
        print(f"{i}. {r['variant']:<18} val {r['best_val']:.4f}  params {r.get('params')}  "
              f"train {r.get('train_hours')} h  cpu {r.get('cpu')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
