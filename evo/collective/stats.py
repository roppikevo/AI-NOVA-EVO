"""
Statistics of a collective run: did the collective core get better or worse, by how much, and is the
change real or noise?

    python -m evo.collective.stats --run coll-24m
    python -m evo.collective.stats --run scale-10 --compare scale-5

What is measured (always on text nobody trained on):
  * loss of the start model, of the collective core after every round, of the single-model baseline and
    of every clone - on the held-out part of the dataset and on held-out web text
  * for every step a paired bootstrap over the test sequences: mean change, 95 % interval and a verdict
    (improvement / decline / no clear change)
  * the same per language (sk, cs, pl, en, py, rs)
  * code exam after every round: tasks solved, tasks gained and lost against the start
  * efficiency: improvement per 100 M training tokens, collective vs one model
  * trend: is the core still improving in the last rounds?
  * how the nodes voted and how they did in the exams (from their own reports)

Writes stats.json, stats.txt, stats.csv and stats_seq.npz (per-sequence losses, used by --compare) into
the run directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

RUNS = Path("evo/collective/runs")
BATCH, PREDICTED = 64, 127          # training batch and predicted tokens per sequence (see experiment.TRAIN_COMMON)
LANG_NAMES = {4: "sk", 5: "cs", 6: "pl", 7: "en", 8: "py", 9: "rs"}


# ------------------------------------------------------------------ statistics (pure)

def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> dict[str, float]:
    """Change b - a of the mean loss, measured on the same sequences, with a 95 % bootstrap interval."""
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    if len(d) == 0:
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), size=len(d))].mean() for _ in range(n)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"diff": round(float(d.mean()), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4), "n": int(len(d))}


def verdict(ci: dict[str, float]) -> str:
    """Lower loss is better: the whole interval below zero is an improvement, above zero a decline."""
    if ci["n"] == 0:
        return "no data"
    if ci["hi"] < 0:
        return "improvement"
    if ci["lo"] > 0:
        return "decline"
    return "no clear change"


def percent(before: float, after: float) -> float:
    return round(100.0 * (after - before) / before, 2) if before else 0.0


def trend(values: list[float]) -> dict[str, Any]:
    """Slope of the loss per round (least squares) and what the last three rounds did."""
    v = [float(x) for x in values]
    if len(v) < 2:
        return {"slope_per_round": 0.0, "last3": 0.0, "state": "too short"}
    slope = float(np.polyfit(np.arange(len(v)), v, 1)[0])
    last3 = v[-1] - v[max(0, len(v) - 4)]
    state = "still improving" if last3 < -0.002 else "getting worse" if last3 > 0.002 else "flat"
    return {"slope_per_round": round(slope, 5), "last3": round(last3, 4), "state": state}


def exam_change(before: set[str], after: set[str]) -> dict[str, Any]:
    return {"solved": len(after), "gained": len(after - before), "lost": len(before - after),
            "net": len(after) - len(before)}


def efficiency(loss_start: float, loss_end: float, tokens: float) -> float:
    """Loss improvement per 100 M training tokens (positive = better)."""
    return round((loss_start - loss_end) / (tokens / 1e8), 5) if tokens > 0 else 0.0


def by_language(a: np.ndarray, b: np.ndarray, lang: np.ndarray) -> dict[str, dict]:
    out = {}
    for i, name in LANG_NAMES.items():
        rows = lang == i
        if rows.sum() >= 20:
            ci = paired_bootstrap(a[rows], b[rows], n=1000, seed=i)
            out[name] = {"before": round(float(a[rows].mean()), 4), "after": round(float(b[rows].mean()), 4),
                         "percent": percent(float(a[rows].mean()), float(b[rows].mean())), **ci, "verdict": verdict(ci)}
    return out


def vote_stats(rounds: list[dict]) -> list[dict]:
    """What the nodes themselves said: votes on the merged core and exam marks."""
    out = []
    for r in rounds:
        row: dict[str, Any] = {"round": r["round"], "led_by": r.get("led_by"), "next_leader": r.get("next_leader"),
                               "agreement": r.get("agreement")}
        c = r.get("core")
        if c:
            ch = [v[1] - v[0] for v in c["votes"].values()]
            row.update({"yes": c["yes"], "of": c["of"], "accepted": c["accepted"],
                        "mean_change_seen_by_nodes": round(float(np.mean(ch)), 4) if ch else None})
        if r.get("scores"):
            s = list(r["scores"].values())
            row.update({"exam_mean": round(float(np.mean(s)), 4), "exam_best": round(float(max(s)), 4)})
        if r.get("core_change"):
            cc = r["core_change"]
            row["core_change"] = f"{cc['candidate']} {cc['yes']}/{cc['of']} {'accepted' if cc['accepted'] else 'rejected'}"
        out.append(row)
    return out


def analyse(names: list[str], losses: dict[str, dict[str, np.ndarray]], langs: dict[str, np.ndarray],
            exams: dict[str, set[str]], report: dict) -> dict[str, Any]:
    """All tables from per-sequence losses. `names` are in order: start, core_round0.., then the others."""
    start = names[0]
    cores = [n for n in names if n.startswith("core_round")]
    n_clones = len(report.get("focuses") or [k for k, v in report.get("node_reports", {}).items() if not v.get("frozen")])
    tokens_round = float(report.get("steps_per_round", 0)) * BATCH * PREDICTED * n_clones
    out: dict[str, Any] = {"run": report.get("name"), "start": start, "clones": n_clones,
                           "tokens_per_round_M": round(tokens_round / 1e6, 1), "sets": {}, "rounds": []}
    for set_name, table in losses.items():
        s: dict[str, Any] = {"sequences": int(len(table[start])),
                             "loss": {n: round(float(table[n].mean()), 4) for n in names if n in table}}
        if cores:
            ci = paired_bootstrap(table[start], table[cores[-1]])
            s["start_to_final_core"] = {**ci, "percent": percent(s["loss"][start], s["loss"][cores[-1]]), "verdict": verdict(ci)}
            s["by_language"] = by_language(table[start], table[cores[-1]], langs[set_name])
            s["trend"] = trend([s["loss"][c] for c in cores])
        base = next((n for n in names if n.startswith("baseline")), None)
        if base and cores:
            k = int(report.get("baseline_rounds") or 0) or len(cores)
            core_k = f"core_round{k - 1}" if f"core_round{k - 1}" in table else cores[-1]
            ci = paired_bootstrap(table[base], table[core_k])
            s["collective_vs_single_model"] = {
                "single": s["loss"][base], "collective": s["loss"][core_k], "rounds": k, **ci, "verdict": verdict(ci),
                "note": "negative = the collective core is better than one model with the same training",
                "efficiency_collective": efficiency(s["loss"][start], s["loss"][core_k], tokens_round * k),
                "efficiency_single": efficiency(s["loss"][start], s["loss"][base], tokens_round * k)}
        out["sets"][set_name] = s
    prev = start
    for i, c in enumerate(cores):
        row: dict[str, Any] = {"round": i, "tokens_M": round(tokens_round * (i + 1) / 1e6, 1)}
        for set_name, table in losses.items():
            step = paired_bootstrap(table[prev], table[c], n=1000, seed=i)
            total = paired_bootstrap(table[start], table[c], n=1000, seed=100 + i)
            row[set_name] = {"loss": round(float(table[c].mean()), 4), "step": step["diff"], "step_verdict": verdict(step),
                             "total": total["diff"], "total_lo": total["lo"], "total_hi": total["hi"],
                             "total_percent": percent(float(table[start].mean()), float(table[c].mean())),
                             "total_verdict": verdict(total)}
        if c in exams and start in exams:
            row["code"] = exam_change(exams[start], exams[c])
        out["rounds"].append(row)
        prev = c
    if exams:
        out["code"] = {n: len(v) for n, v in exams.items()}
        if cores and cores[-1] in exams and start in exams:
            out["code_start_to_final"] = exam_change(exams[start], exams[cores[-1]])
    out["nodes"] = vote_stats(report.get("nodes", {}).get("rounds", []))
    return out


def text(st: dict) -> str:
    L = [f"STATISTICS {st['run']}: {st['clones']} clones, {st['tokens_per_round_M']} M training tokens per round",
         "(loss: lower = better; change: negative = improvement; interval = 95 % bootstrap over test sequences)"]
    for name, s in st["sets"].items():
        L.append(f"--- test set '{name}' ({s['sequences']} sequences nobody trained on)")
        if "start_to_final_core" in s:
            c = s["start_to_final_core"]
            L.append(f"  start {s['loss'][st['start']]} -> final core: change {c['diff']:+.4f} ({c['percent']:+.2f} %), "
                     f"interval [{c['lo']:+.4f}, {c['hi']:+.4f}] -> {c['verdict'].upper()}")
            L.append(f"  trend: {s['trend']['state']} (slope {s['trend']['slope_per_round']:+.5f} per round, last 3 rounds {s['trend']['last3']:+.4f})")
            for lang, v in s.get("by_language", {}).items():
                L.append(f"    {lang}: {v['before']} -> {v['after']} ({v['percent']:+.2f} %) [{v['lo']:+.4f}, {v['hi']:+.4f}] {v['verdict']}")
        if "collective_vs_single_model" in s:
            c = s["collective_vs_single_model"]
            L.append(f"  after {c['rounds']} rounds: collective core {c['collective']} vs one model with the same training {c['single']}: "
                     f"{c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] -> {c['verdict'].upper()}")
            L.append(f"  efficiency (improvement per 100 M tokens): collective {c['efficiency_collective']}, one model {c['efficiency_single']}")
        others = {n: v for n, v in s["loss"].items() if not n.startswith("core_round") and n != st["start"]}
        if others:
            L.append(f"  others: {others}")
    if st["rounds"]:
        L.append("--- round by round (collective core)")
        for r in st["rounds"]:
            parts = [f"  round {r['round']:2d} ({r['tokens_M']:6.0f} M tok)"]
            for name in st["sets"]:
                v = r[name]
                parts.append(f"{name}: {v['loss']} step {v['step']:+.4f} total {v['total']:+.4f} ({v['total_percent']:+.2f} %) {v['total_verdict']}")
            if "code" in r:
                c = r["code"]
                parts.append(f"code {c['solved']} (+{c['gained']} -{c['lost']})")
            L.append(" | ".join(parts))
    if st.get("code"):
        L.append(f"--- code exam (tasks solved): {st['code']}")
    if st.get("nodes"):
        L.append("--- what the nodes decided")
        for r in st["nodes"]:
            L.append(f"  round {r['round']}: led by {r.get('led_by')}, core {r.get('yes')}/{r.get('of')} "
                     f"{'adopted' if r.get('accepted') else 'rejected'}, nodes saw {r.get('mean_change_seen_by_nodes')}, "
                     f"exam mean {r.get('exam_mean')}, next leader {r.get('next_leader')}"
                     + (f", core change {r['core_change']}" if r.get("core_change") else ""))
    if st.get("compare"):
        c = st["compare"]
        L.append(f"--- {c['a']} vs {c['b']} (final collective cores, same start, same total training)")
        for name, v in c["sets"].items():
            L.append(f"  {name}: {c['a']} {v['a']} vs {c['b']} {v['b']}: {v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] -> "
                     f"{v['winner']}")
    return "\n".join(L)


def compare(a_name: str, a: dict[str, np.ndarray], b_name: str, b: dict[str, np.ndarray]) -> dict:
    """Final cores of two runs on the same test sequences (negative = run a is better)."""
    out: dict[str, Any] = {"a": a_name, "b": b_name, "sets": {}}
    for set_name in ("dataset", "web"):
        ka, kb = f"{set_name}/final_core", f"{set_name}/final_core"
        if ka in a and kb in b and len(a[ka]) == len(b[kb]):
            ci = paired_bootstrap(b[kb], a[ka])
            v = verdict(ci)
            out["sets"][set_name] = {"a": round(float(a[ka].mean()), 4), "b": round(float(b[kb].mean()), 4), **ci,
                                     "winner": a_name if v == "improvement" else b_name if v == "decline" else "no clear difference"}
    return out


# ------------------------------------------------------------------ measuring (server)

def seq_loss(node, seqs: np.ndarray, batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(seqs), batch):
        out += [-float(np.mean(r)) for r in node.score(seqs[i:i + batch].tolist())]
    return np.array(out, dtype=np.float32)


def solved_tasks(node, keys: list[str]) -> set[str]:
    from evo.learning.code_school import run_tests
    from evo.learning.code_tasks import task_bank

    tasks = {t.key: t for t in task_bank()}
    bodies = node.solve(keys, 0)
    return {k for k in keys if any(run_tests(tasks[k], b)[0] for b in bodies.get(k, []))}


def held_out_web(directory: Path, per_lang: int = 400) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Held-out web text (parts no training used) and the language of every row."""
    rows, lang = [], []
    for i, name in ((4, "sk"), (5, "cs"), (6, "pl"), (7, "en")):
        files = sorted(directory.glob(f"{name}-*.npy")) if directory.exists() else []
        if files:
            part = np.asarray(np.load(files[0], mmap_mode="r")[:per_lang])
            rows.append(part)
            lang.append(np.full(len(part), i))
    return (np.concatenate(rows), np.concatenate(lang)) if rows else (None, None)


def measure(run: str, web_dir: str, device: str, code: bool, log=print) -> dict:
    import torch

    from evo.collective import experiment as ex
    from evo.collective.node import load_node
    from evo.engine.long_train import load_tokens, row_languages
    from evo.learning.code_tasks import task_bank

    run_dir = RUNS / run
    report = json.loads((run_dir / "report.json").read_text())
    dataset = Path(json.loads(Path("evo/engine/evo_state.json").read_text(encoding="utf-8"))["best_known"]["dataset"])
    tokenizer = str(dataset / "tokenizer.json")
    val = load_tokens(dataset / "val.txt")
    split = ex.split_validation(val)
    n_cal = len(split["cal"])
    sets = {"dataset": split["eval"]}
    langs = {"dataset": row_languages(val)[n_cal:n_cal + len(split["eval"])]}
    web, web_lang = held_out_web(Path(web_dir)) if web_dir else (None, None)
    if web is not None:
        sets["web"], langs["web"] = web, web_lang

    files: dict[str, Path] = {"start": Path(report["base"])}
    r = 0
    while (run_dir / f"core_round{r}.pt").exists():
        files[f"core_round{r}"] = run_dir / f"core_round{r}.pt"
        r += 1
    if (run_dir / "baseline.pt").exists():
        files["baseline_single"] = run_dir / "baseline.pt"
    for nid, rep in report.get("node_reports", {}).items():
        if not rep.get("frozen") and rep.get("personal") and Path(rep["personal"]).exists():
            files[f"clone_{nid}"] = Path(rep["personal"])
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    exam_keys = [t.key for t in task_bank() if t.pool == "exam"]
    losses: dict[str, dict[str, np.ndarray]] = {s: {} for s in sets}
    exams: dict[str, set[str]] = {}
    for name, path in files.items():
        t0 = time.time()
        node = load_node(name, str(path), tokenizer, "", device)
        for s, seqs in sets.items():
            losses[s][name] = seq_loss(node, seqs)
        if code and (name == "start" or name.startswith("core_round") or name.startswith("baseline")):
            exams[name] = solved_tasks(node, exam_keys)
        log(f"  {name}: " + ", ".join(f"{s} {losses[s][name].mean():.4f}" for s in sets)
            + (f", code {len(exams[name])}/{len(exam_keys)}" if name in exams else "") + f" ({time.time() - t0:.0f}s)")
        del node
    report["baseline_rounds"] = _baseline_rounds(report)
    st = analyse(list(files), losses, langs, exams, report)
    cores = [n for n in files if n.startswith("core_round")]
    seq = {f"{s}/{n}": v for s, t in losses.items() for n, v in t.items()}
    if cores:
        for s in sets:
            seq[f"{s}/final_core"] = losses[s][cores[-1]]
    np.savez_compressed(run_dir / "stats_seq.npz", **seq)
    return st


def _baseline_rounds(report: dict) -> int:
    for k in report.get("measure", {}).get("loss", {}):
        if k.startswith("baseline_single_") and k.endswith("_rounds"):
            return int(k.split("_")[2])
    return 0


def write(run: str, st: dict) -> None:
    run_dir = RUNS / run
    (run_dir / "stats.json").write_text(json.dumps(st, indent=1, ensure_ascii=False))
    (run_dir / "stats.txt").write_text(text(st) + "\n", encoding="utf-8")
    with (run_dir / "stats.csv").open("w", newline="") as f:
        w = csv.writer(f)
        sets = list(st["sets"])
        w.writerow(["round", "tokens_M"] + [f"{s}_{c}" for s in sets for c in ("loss", "step", "total", "total_lo", "total_hi", "total_percent", "verdict")]
                   + ["code_solved", "code_gained", "code_lost"])
        for r in st["rounds"]:
            row = [r["round"], r["tokens_M"]]
            for s in sets:
                v = r[s]
                row += [v["loss"], v["step"], v["total"], v["total_lo"], v["total_hi"], v["total_percent"], v["total_verdict"]]
            c = r.get("code", {})
            w.writerow(row + [c.get("solved", ""), c.get("gained", ""), c.get("lost", "")])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--compare", default="", help="another run with the same start and the same total training")
    ap.add_argument("--web-val-dir", default="data/bulk_val_v1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-code", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="do not measure again: take stats.json of the run (to add --compare)")
    args = ap.parse_args(argv)
    done = RUNS / args.run / "stats.json"
    if args.reuse and done.exists():
        st = json.loads(done.read_text())
    else:
        st = measure(args.run, args.web_val_dir, args.device, not args.no_code)
    if args.compare:
        other = RUNS / args.compare / "stats_seq.npz"
        if other.exists():
            st["compare"] = compare(args.run, dict(np.load(RUNS / args.run / "stats_seq.npz")), args.compare, dict(np.load(other)))
        else:
            print(f"(no statistics of {args.compare} yet - comparison skipped)")
    write(args.run, st)
    print(text(st))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
