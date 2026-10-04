"""
NOVA collective v2 - runner: starts the autonomous nodes, watches (does not steer), measures.

    python -m evo.collective.collective_v2 --name coll-v2 --base evo/releases/NOVA-10M-v1/nova_model_fp32.pt

Nodes: nova0 = the frozen release (first leader, never trained) + five clones (sk, cs, pl, en, code).
The nodes organise themselves (see evo.collective.agent). This runner only
  - writes one config per node and starts the processes,
  - optionally stops the leader before the last round ("drill": does the collective survive?),
  - collects every node's own report and measures the result on validation data nobody trained on.
"""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

RUNS = Path("evo/collective/runs")
PORT0 = 8111
FOUNDER = "nova0"
CORE_CANDIDATES = [{"forget_bias": 1.5}, {"conv_kernel": 7}, {"conv_kernel": 3}]


def node_configs(run_dir: Path, base: str, dataset: str, key: str, steps: int, rounds: int, core_steps: int,
                 focuses: list[str], core_candidates: list[dict] | None = None) -> dict[str, dict]:
    ids = [FOUNDER] + focuses
    peers = {nid: f"http://127.0.0.1:{PORT0 + i}" for i, nid in enumerate(ids)}
    cfgs = {}
    for i, nid in enumerate(ids):
        cfgs[nid] = {"id": nid, "port": PORT0 + i, "peers": peers, "key": key, "run_dir": str(run_dir),
                     "weights": base, "base": base, "dataset": dataset, "tokenizer": f"{dataset}/tokenizer.json",
                     "focus": "" if nid == FOUNDER else nid, "frozen": nid == FOUNDER, "first_leader": FOUNDER,
                     "rounds": rounds, "steps": steps, "core_steps": core_steps, "seed": i, "threads": 2,
                     "challenges": 30, "core_margin": 0.01,
                     "core_candidates": CORE_CANDIDATES if core_candidates is None else core_candidates,
                     "poll": 3.0, "leader_fails": 4, "exam_timeout": 1800, "round_timeout": 5 * 3600, "linger": 900}
    return cfgs


def states(peers: dict[str, Any]) -> dict[str, dict | None]:
    return {nid: p.state() for nid, p in peers.items()}


def summarise(reports: dict[str, dict], drill: dict | None) -> dict[str, Any]:
    """One picture of what the nodes did, built only from their own reports."""
    rounds: dict[int, dict] = {}
    for nid, rep in reports.items():
        for h in rep.get("history", []):
            r = rounds.setdefault(h["round"], {"round": h["round"], "elected_by": {}})
            if "trained" in h:
                r.update({"led_by": h["leader"], "trained": h["trained"]})
                if "merge" in h:
                    m = h["merge"]
                    r["core"] = {"yes": m["yes"], "of": m["of"], "accepted": m["accepted"],
                                 "votes": {k: [v["before"], v["after"], v["yes"]] for k, v in m["votes"].items()}}
                if "core_change" in h:
                    c = h["core_change"]
                    r["core_change"] = {"candidate": c["candidate"], "yes": c["yes"], "of": c["of"], "accepted": c["accepted"],
                                        "results": {k: [round(v["parent"], 4), round(v["candidate"], 4)] for k, v in c["results"].items()}}
            if "election" in h:
                r["elected_by"][nid] = h["election"]["leader"]
                r["scores"] = h["election"]["scores"]
            if "failover" in h:
                r.setdefault("failover", {})[nid] = h["failover"]
    out = []
    for r in sorted(rounds):
        row = rounds[r]
        views = set(row["elected_by"].values())
        row["next_leader"] = views.pop() if len(views) == 1 else None
        row["agreement"] = len(set(row["elected_by"].values())) == 1
        out.append(row)
    return {"rounds": out, "messages": {n: r.get("messages_sent") for n, r in reports.items()},
            "total_messages": sum(r.get("messages_sent", 0) for r in reports.values()), "drill": drill,
            "final_base": sorted({str(r.get("base")) for r in reports.values()})}


def run(args, log=print) -> dict:
    from evo.collective import ensemble
    from evo.collective import experiment as ex
    from evo.collective.agent import Peer
    from evo.engine.long_train import load_tokens
    from evo.learning.code_tasks import task_bank
    from nova.tokenizer import NovaTokenizer

    run_dir = RUNS / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ALL_DONE").unlink(missing_ok=True)
    best = json.loads(Path("evo/engine/evo_state.json").read_text(encoding="utf-8"))["best_known"]
    dataset = best["dataset"]
    focuses = [f for f in args.focuses.split(",") if f] if args.focuses else list(ex.SPECIALTIES)
    unknown = [f for f in focuses if f not in ex.known_focuses()]
    if unknown or len(set(focuses)) != len(focuses):
        raise SystemExit(f"bad --focuses {focuses}: unknown {unknown}; known: {ex.known_focuses()}")
    key = secrets.token_hex(16)
    pool = json.loads(args.core_candidates) if args.core_candidates else None
    cfgs = node_configs(run_dir, args.base, dataset, key, args.steps, args.rounds, args.core_steps, focuses, pool)
    procs = {}
    for nid, cfg in cfgs.items():
        p = run_dir / f"{nid}.json"
        p.write_text(json.dumps(cfg, indent=1))
        procs[nid] = subprocess.Popen([sys.executable, "-m", "evo.collective.agent", "--config", str(p)],
                                      stdout=subprocess.DEVNULL, stderr=(run_dir / f"{nid}.err").open("w"))
    peers = {nid: Peer(cfg["peers"][nid], key=key) for nid, cfg in cfgs.items()}
    log(f"=== {len(procs)} nodes started; they organise themselves from here ===")

    drill, t0, last = None, time.time(), ""
    reports: dict[str, dict] = {}
    try:
        while time.time() - t0 < args.max_hours * 3600:
            st = states(peers)
            alive = {n: s for n, s in st.items() if s}
            line = " ".join(f"{n}:{s['round']}/{s['phase']}" for n, s in alive.items())
            if line != last:
                log(f"  {time.strftime('%H:%M')} {line}")
                last = line
            if args.drill and drill is None and alive and args.rounds >= 2 and \
                    all(s["elect_round"] >= args.rounds - 2 for s in alive.values()):
                leaders = {s["leader"] for s in alive.values()}
                if len(leaders) == 1 and (leader := leaders.pop()) in procs:
                    procs[leader].terminate()
                    drill = {"stopped_leader": leader, "before_round": args.rounds - 1, "time": time.strftime("%H:%M")}
                    log(f"=== DRILL: leader {leader} was stopped before the last round ===")
            for nid, s in alive.items():
                if s.get("finished") and nid not in reports:
                    try:
                        reports[nid] = peers[nid].call("/report", timeout=30)
                    except Exception:
                        pass
            running = [n for n in procs if procs[n].poll() is None]
            if all(n in reports for n in alive) and alive and all(s.get("finished") for s in alive.values()):
                break
            if not running:
                break
            time.sleep(15)
    finally:
        (run_dir / "ALL_DONE").write_text("done")
        time.sleep(5)
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
    for nid in cfgs:  # reports written by the nodes themselves (if the API answer was missed)
        f = run_dir / f"{nid}_report.json"
        if nid not in reports and f.exists():
            reports[nid] = json.loads(f.read_text())

    report: dict[str, Any] = {"name": args.name, "date": time.strftime("%Y-%m-%d %H:%M"), "base": args.base,
                              "focuses": focuses, "steps_per_round": args.steps, "rounds": args.rounds, "hours": round((time.time() - t0) / 3600, 2),
                              "nodes": summarise(reports, drill), "node_reports": reports}
    (run_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))

    # ---- measurement on validation data nobody trained on
    log("=== measuring ===")
    tokenizer = Path(dataset) / "tokenizer.json"
    tok = NovaTokenizer.load(tokenizer)
    split = ex.split_validation(load_tokens(Path(dataset) / "val.txt"))
    files: dict[str, Path] = {"release": Path(args.base)}
    bases = [r.get("base") for r in reports.values() if r.get("base")]
    core = max(set(bases), key=bases.count) if bases else None
    if core and Path(core).exists() and core != args.base:
        files["collective_core"] = Path(core)
    for nid, r in reports.items():
        if nid != FOUNDER and r.get("personal") and Path(r["personal"]).exists():
            files[nid] = Path(r["personal"])
    k = min(args.baseline_rounds, args.rounds)
    if k > 0:
        # fair comparison: ONE model that gets the same number of training tokens as all clones in k rounds
        base_path = run_dir / "baseline.pt"
        if not base_path.exists():
            log(f"=== baseline: one model, {args.steps * len(focuses) * k} steps (= {k} rounds of {len(focuses)} clones) ===")
            ex.train(args.base, base_path, args.steps * len(focuses) * k, {}, 1999, log)
        files[f"baseline_single_{k}_rounds"] = base_path
        core_k = run_dir / f"core_round{k - 1}.pt"
        if core_k.exists():
            files[f"collective_core_after_{k}_rounds"] = core_k
    else:
        v1_baseline = RUNS / "coll-v1" / "baseline.pt"
        if v1_baseline.exists():
            files["baseline_single_same_tokens"] = v1_baseline
    for r in range(args.rounds):
        f = run_dir / f"core_round{r}.pt"
        if f.exists():
            files.setdefault(f"core_round{r}", f)
    nodes = {n: ex.local_node(n, p, tokenizer) for n, p in files.items()}
    lp = {n: ex.nll_of(node, split["eval"]) for n, node in nodes.items()}
    measure: dict[str, Any] = {"loss": {n: round(float(-v.mean()), 4) for n, v in lp.items() if not n.startswith("core_round")},
                               "core_by_round": {n: round(float(-v.mean()), 4) for n, v in lp.items() if n.startswith("core_round")}}
    members = [n for n in lp if n in focuses or n == "release"]
    if len(members) >= 2:
        cal = np.stack([ex.nll_of(nodes[n], split["cal"]) for n in members])
        ev = np.stack([lp[n] for n in members])
        w = ensemble.fit_weights(cal)
        measure["collective_answer"] = {"members": members, "equal_votes": round(ensemble.mixture_nll(ev), 4),
                                        "leader_weights": round(ensemble.mixture_nll(ev, w), 4),
                                        "weights": {n: round(float(x), 3) for n, x in zip(members, w)}}
    exam = [t.key for t in task_bank() if t.pool == "exam"]
    team = [nodes[n] for n in members if n != "release"]
    if team:
        tg = ensemble.solve_together(team, exam)
        measure["code"] = {"collective": {k: tg[k] for k in ("solved", "tasks", "by_level", "per_node")},
                           "release_alone": ensemble.solve_together([nodes["release"]], exam)["solved"]}
        if "collective_core" in nodes:
            measure["code"]["collective_core_alone"] = ensemble.solve_together([nodes["collective_core"]], exam)["solved"]
    measure["creator"] = ex.creator_report([v for n, v in nodes.items() if not n.startswith("core_round")], tok)
    report["measure"] = measure
    (run_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    return report


def summary(report: dict) -> str:
    n, m = report["nodes"], report.get("measure", {})
    lines = [f"COLLECTIVE v2 {report['name']} ({report['date']}, {report.get('hours')} h, "
             f"{n['total_messages']} messages between nodes)"]
    for r in n["rounds"]:
        lines.append(f"round {r['round']}: led by {r.get('led_by')}  trained {r.get('trained')}")
        if "core" in r:
            c = r["core"]
            lines.append(f"   collective core: {c['yes']}/{c['of']} yes -> {'ADOPTED' if c['accepted'] else 'rejected'}   "
                         f"[before, after, yes] {c['votes']}")
        if "core_change" in r:
            c = r["core_change"]
            lines.append(f"   core change {c['candidate']}: {c['yes']}/{c['of']} yes -> "
                         f"{'ACCEPTED' if c['accepted'] else 'rejected'}   [parent, candidate] {c['results']}")
        if "failover" in r:
            lines.append(f"   leader lost -> {r['failover']}")
        lines.append(f"   election: next leader {r['next_leader']} (all nodes agree: {r['agreement']})  scores {r.get('scores')}")
    if n.get("drill"):
        lines.append(f"drill: {n['drill']}")
    if m:
        lines.append(f"loss (lower = better): {m.get('loss')}")
        if m.get("core_by_round"):
            lines.append(f"collective core, round by round: {m['core_by_round']}")
        lines.append(f"collective answer: {m.get('collective_answer')}")
        lines.append(f"code exam: {m.get('code')}")
        lines.append(f"know the Creator: {m.get('creator', {}).get('know_creator')}  recall {m.get('creator', {}).get('recall')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--steps", type=int, default=2000, help="training steps per clone per round")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--core-steps", type=int, default=1000, help="steps of each from-scratch run when a core change is judged")
    ap.add_argument("--drill", action="store_true", help="stop the leader before the last round")
    ap.add_argument("--core-candidates", default="", help="JSON list of core changes to judge together ([] = none)")
    ap.add_argument("--baseline-rounds", type=int, default=0,
                    help="after the run, train ONE model with the tokens of this many rounds and compare it with the collective core")
    ap.add_argument("--focuses", default="",
                    help="comma list of clone specialities (default: sk,cs,pl,en,code); more: py,rs,teach,web,data")
    ap.add_argument("--max-hours", type=float, default=5.0)
    args = ap.parse_args(argv)
    report = run(args)
    print(summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
