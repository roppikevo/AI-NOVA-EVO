"""
NOVA core vs a standard transformer of the same size: same tokenizer, same data mix, same number of
training tokens, same seed - trained from scratch, measured on text nobody trained on.

    python -m evo.engine.compare_arch --steps 18000

  * two sizes (about 10 M and 24 M parameters); the NOVA core runs once per size with its usual
    settings, the transformer gets THREE attempts per size (two shapes, then the better shape again with
    a higher learning rate) - the yardstick must not lose because it was set up badly
  * measured: loss on held-out dataset text and on held-out web text (with a paired bootstrap between the
    NOVA core and the best transformer), code exam, CPU speed for reading a prompt and for writing token
    by token after a 127-token prompt, size of the state carried while writing, training speed
  * resumable: a variant whose weights exist is not trained again

Writes evo/learning/arch_compare/{report.json, report.txt, seq.npz}. The deployed model is not touched.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

OUT = Path("evo/learning/arch_compare")

NOVA = {"10M": {}, "24M": {"d_model": 640, "d_state": 640, "num_layers": 8}}
TRANSFORMER = {
    "10M": {"a": {"d_model": 384, "num_layers": 2, "n_heads": 6}, "b": {"d_model": 288, "num_layers": 5, "n_heads": 6}},
    "24M": {"a": {"d_model": 512, "num_layers": 5, "n_heads": 8}, "b": {"d_model": 448, "num_layers": 7, "n_heads": 7}},
}
LR, LR_HIGH = "3e-4", "1e-3"
LONG_CONTEXTS = (1024, 4096)   # speed only: both models are trained on 128-token sequences
COMMON = ["--from-scratch", "--no-activate", "--batch-size", "64", "--warmup", "500", "--eval-every", "3000",
          "--patience", "99", "--extra-dirs", "data/teacher_v1,data/web_v1", "--bulk-dir", "data/bulk_v1",
          "--bulk-frac", "0.7", "--code-frac", "0.05", "--seed", "1001"]


def first_round(groups: list[str]) -> list[dict]:
    rows = []
    for g in groups:
        rows.append({"name": f"nova-{g}", "group": g, "arch": "nova", "override": NOVA[g], "lr": LR})
        for k, shape in TRANSFORMER[g].items():
            rows.append({"name": f"tf-{g}-{k}", "group": g, "arch": "transformer",
                         "override": {"arch": "transformer", **shape}, "lr": LR})
    return rows


def second_round(results: dict[str, dict], groups: list[str]) -> list[dict]:
    """The better transformer shape of each size once more, with a higher learning rate."""
    rows = []
    for g in groups:
        done = [r for r in results.values() if r["group"] == g and r["arch"] == "transformer" and r.get("loss")]
        if done:
            best = min(done, key=lambda r: r["loss"]["dataset"])
            rows.append({"name": f"tf-{g}-lr{LR_HIGH}", "group": g, "arch": "transformer", "override": best["override"],
                         "lr": LR_HIGH})
    return rows


def train(v: dict, steps: int, max_hours: float, runner=subprocess.run) -> dict:
    from evo.engine.ab_test import parse_report

    out = OUT / f"{v['name']}.pt"
    cmd = [sys.executable, "-m", "evo.engine.long_train", "--steps", str(steps), "--lr", v["lr"], "--max-hours",
           str(max_hours), "--out-checkpoint", str(out)] + COMMON
    if v["override"]:
        cmd += ["--config-override", json.dumps(v["override"])]
    p = runner(cmd, capture_output=True, text=True, timeout=int(max_hours * 3600) + 1800)
    if p.returncode != 0 or not out.exists():
        raise RuntimeError((p.stderr or p.stdout or "")[-800:])
    rep = parse_report(p.stdout)
    return {"params": rep.get("params"), "train_hours": rep["hours"], "steps_done": rep["steps_done"],
            "best_val_during_training": rep["best_val"], "tokens_seen": rep.get("tokens_seen"),
            "train_tokens_per_s": round(rep.get("tokens_seen", 0) / max(rep["hours"] * 3600, 1e-9)),
            "curve": [l.split(" best")[0] for l in p.stdout.splitlines() if l.startswith("step ")][-8:]}


def cpu_speed(model, vocab: int, prompt_len: int = 127, gen: int = 64, threads: int = 8, repeats: int = 3) -> dict:
    """Reading a prompt and then writing token by token (batch 1, CPU) - the way the model is used.

    Each architecture writes the way it is meant to: the NOVA core with its stepper (recurrent state +
    convolution window), the transformer with its key/value cache."""
    import torch

    from nova.stepper import Stepper, supports

    m = copy.deepcopy(model).to("cpu").float().eval()
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        with torch.no_grad():
            x = torch.randint(12, vocab, (1, prompt_len))
            m(x)
            read = float("inf")
            for _ in range(repeats):
                t0 = time.perf_counter()
                m(x)
                read = min(read, time.perf_counter() - t0)
            write, state = float("inf"), 0
            for _ in range(repeats):
                if supports(m):
                    st = Stepper(m)
                    tok = st.prime(x).argmax(-1)
                    t0 = time.perf_counter()
                    for _ in range(gen):
                        tok = st.step(tok).argmax(-1)
                    write = min(write, time.perf_counter() - t0)
                    state = st.state_bytes()
                else:
                    logits, cache = m(x)[:2]
                    tok = logits[:, -1:].argmax(-1)
                    t0 = time.perf_counter()
                    for _ in range(gen):
                        logits, cache = m(tok, cache)[:2]
                        tok = logits[:, -1:].argmax(-1)
                    write = min(write, time.perf_counter() - t0)
                    state = _nbytes(cache)
        return {"read_tokens_per_s": round(prompt_len / read), "write_tokens_per_s": round(gen / write, 1),
                "state_kb_after_writing": round(state / 1024, 1), "threads": threads}
    finally:
        torch.set_num_threads(prev)


def _nbytes(obj: Any) -> int:
    import torch

    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(_nbytes(o) for o in obj)
    return 0


def evaluate(v: dict, sets: dict[str, np.ndarray], tokenizer: str, exam_keys: list[str], device: str) -> tuple[dict, dict]:
    from evo.collective.node import load_node
    from evo.collective.stats import seq_loss, solved_tasks

    node = load_node(v["name"], str(OUT / f"{v['name']}.pt"), tokenizer, "", device)
    seq = {s: seq_loss(node, rows) for s, rows in sets.items()}
    res = {"loss": {s: round(float(a.mean()), 4) for s, a in seq.items()},
           "code_solved": len(solved_tasks(node, exam_keys)), "code_tasks": len(exam_keys),
           "cpu": cpu_speed(node.model, int(node.model.embedding.num_embeddings))}
    # writing speed and memory as the text behind the model grows (the NOVA state does not grow, a key/value cache does)
    res["cpu_by_context"] = {str(n): cpu_speed(node.model, int(node.model.embedding.num_embeddings), prompt_len=n, gen=32, repeats=1)
                             for n in LONG_CONTEXTS}
    return res, seq


def verdicts(results: dict[str, dict], seq: dict[str, np.ndarray], groups: list[str]) -> dict[str, dict]:
    """Per size: the NOVA core against the best transformer attempt, on the same test sequences."""
    from evo.collective.stats import paired_bootstrap, verdict

    out = {}
    for g in groups:
        nova = results.get(f"nova-{g}")
        tfs = [r for r in results.values() if r["group"] == g and r["arch"] == "transformer" and r.get("loss")]
        if not nova or not nova.get("loss") or not tfs:
            continue
        best = min(tfs, key=lambda r: r["loss"]["dataset"])
        row: dict[str, Any] = {"nova": nova["name"], "transformer": best["name"], "transformer_attempts": len(tfs), "sets": {}}
        for s in nova["loss"]:
            a, b = seq[f"{s}/{best['name']}"], seq[f"{s}/{nova['name']}"]
            ci = paired_bootstrap(a, b)                       # NOVA minus transformer: negative = NOVA better
            v = verdict(ci)
            row["sets"][s] = {"nova": nova["loss"][s], "transformer": best["loss"][s], **ci,
                              "percent": round(100 * ci["diff"] / best["loss"][s], 2),
                              "winner": "NOVA" if v == "improvement" else "transformer" if v == "decline" else "no clear difference"}
        row["code"] = {"nova": nova["code_solved"], "transformer": best["code_solved"], "tasks": nova["code_tasks"]}
        row["cpu_write_speedup"] = round(nova["cpu"]["write_tokens_per_s"] / max(best["cpu"]["write_tokens_per_s"], 1e-9), 2)
        row["cpu_read_speedup"] = round(nova["cpu"]["read_tokens_per_s"] / max(best["cpu"]["read_tokens_per_s"], 1e-9), 2)
        out[g] = row
    return out


def text(report: dict) -> str:
    L = [f"NOVA CORE vs TRANSFORMER ({report['date']}): from scratch, {report['steps']} steps = "
         f"{report['steps'] * 64 * 127 / 1e6:.0f} M tokens each, same data, same seed",
         "(loss: lower = better; the transformer has three attempts per size, the NOVA core one)"]
    for r in report["results"].values():
        if not r.get("loss"):
            L.append(f"  {r['name']:<16} FAILED: {r.get('error', '')[:160]}")
            continue
        c = r["cpu"]
        L.append(f"  {r['name']:<16} {r['params'] / 1e6:5.2f} M  lr {r['lr']:<5} loss dataset {r['loss']['dataset']:.4f}"
                 + (f"  web {r['loss']['web']:.4f}" if "web" in r["loss"] else "")
                 + f"  code {r['code_solved']}/{r['code_tasks']}  CPU read {c['read_tokens_per_s']} tok/s, "
                   f"write {c['write_tokens_per_s']} tok/s, state {c['state_kb_after_writing']} kB  "
                   f"train {r['train_tokens_per_s']} tok/s  {r['override'] or 'NOVA core'}")
    for g, v in report.get("verdicts", {}).items():
        L.append(f"--- {g}: {v['nova']} vs best transformer {v['transformer']} (of {v['transformer_attempts']} attempts)")
        for s, x in v["sets"].items():
            L.append(f"  {s}: NOVA {x['nova']} vs transformer {x['transformer']}: {x['diff']:+.4f} ({x['percent']:+.2f} %) "
                     f"[{x['lo']:+.4f}, {x['hi']:+.4f}] -> {x['winner'].upper()}")
        L.append(f"  code exam: NOVA {v['code']['nova']} vs transformer {v['code']['transformer']} of {v['code']['tasks']}")
        L.append(f"  CPU after a 127-token prompt: NOVA writes {v['cpu_write_speedup']}x and reads {v['cpu_read_speedup']}x as fast as the transformer")
        a, b = report["results"][v["nova"]], report["results"][v["transformer"]]
        for n in sorted(set(a.get("cpu_by_context", {})) & set(b.get("cpu_by_context", {})), key=int):
            x, y = a["cpu_by_context"][n], b["cpu_by_context"][n]
            L.append(f"  CPU after {n} tokens: NOVA writes {x['write_tokens_per_s']} tok/s with {x['state_kb_after_writing']} kB of state, "
                     f"transformer {y['write_tokens_per_s']} tok/s with {y['state_kb_after_writing']} kB")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=18000)
    ap.add_argument("--groups", default="10M,24M")
    ap.add_argument("--max-hours", type=float, default=3.0, help="time limit per variant")
    ap.add_argument("--web-val-dir", default="data/bulk_val_v1")
    ap.add_argument("--no-second-round", action="store_true")
    args = ap.parse_args(argv)

    import torch

    from evo.collective import experiment as ex
    from evo.collective.stats import held_out_web
    from evo.engine.long_train import load_tokens
    from evo.learning.code_tasks import task_bank

    OUT.mkdir(parents=True, exist_ok=True)
    groups = [g for g in args.groups.split(",") if g in NOVA]
    dataset = Path(json.loads(Path("evo/engine/evo_state.json").read_text(encoding="utf-8"))["best_known"]["dataset"])
    tokenizer = str(dataset / "tokenizer.json")
    sets = {"dataset": ex.split_validation(load_tokens(dataset / "val.txt"))["eval"]}
    web, _ = held_out_web(Path(args.web_val_dir))
    if web is not None:
        sets["web"] = web
    exam_keys = [t.key for t in task_bank() if t.pool == "exam"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rep_file = OUT / "report.json"
    report = json.loads(rep_file.read_text()) if rep_file.exists() else {"results": {}}
    report.update({"date": time.strftime("%Y-%m-%d %H:%M"), "steps": args.steps, "groups": groups})
    seq: dict[str, np.ndarray] = dict(np.load(OUT / "seq.npz")) if (OUT / "seq.npz").exists() else {}

    def run(variants: list[dict]) -> None:
        for v in variants:
            row = report["results"].get(v["name"], {})
            if row.get("loss") and row.get("steps") == args.steps and all(f"{s}/{v['name']}" in seq for s in sets):
                print(f"=== {v['name']}: already measured ===", flush=True)
                continue
            print(f"=== {v['name']} {v['override'] or 'NOVA core'} lr {v['lr']} ===", flush=True)
            row = {**v, "steps": args.steps}
            try:
                row.update(train(v, args.steps, args.max_hours))
                res, s = evaluate(v, sets, tokenizer, exam_keys, device)
                row.update(res)
                seq.update({f"{k}/{v['name']}": a for k, a in s.items()})
                np.savez_compressed(OUT / "seq.npz", **seq)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"[:600]
            report["results"][v["name"]] = row
            rep_file.write_text(json.dumps(report, indent=1, ensure_ascii=False))
            print(json.dumps({k: x for k, x in row.items() if k != "curve"}, ensure_ascii=False), flush=True)

    run(first_round(groups))
    if not args.no_second_round:
        run(second_round(report["results"], groups))
    report["verdicts"] = verdicts(report["results"], seq, groups)
    rep_file.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (OUT / "report.txt").write_text(text(report) + "\n", encoding="utf-8")
    print(text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
