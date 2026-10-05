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
# Generation 8 of the NOVA core (nova/core8.py): candidates of about the same size, every one with a state that
# does not grow with the text. One change at a time, so that the result says what helped:
#   mlp   the generation-7 memory + a non-linear layer        (was the missing non-linearity the problem?)
#   lru   a gated linear recurrent unit + the non-linear layer (was the memory itself too weak?)
#   slot  lru blocks alternating with a slot memory read by content (does recall by content help?)
#   hash  lru blocks alternating with a hash-table memory: write under an address, read by address
#   mem   lru blocks with two matrix-memory blocks              (another form of recall by content)
#   win   lru blocks with two blocks that see the last 32 tokens (what a small fixed window buys)
#   nslot, nwin, nwin16, nwin3, nsw   the same ideas on the generation-7 memory instead of the lru
N8 = {"arch": "nova8", "d_model": 448, "heads": 8}
CANDIDATES = {
    "24M": {
        "mlp": {**N8, "pattern": "NNNNNNN", "mlp_hidden": 1152},
        "lru": {**N8, "pattern": "LLLLLLL", "mlp_hidden": 1280},
        "slot": {**N8, "pattern": "LSLSLSL", "mlp_hidden": 1344, "slots": 16},
        "hash": {**N8, "pattern": "LHLHLHL", "mlp_hidden": 1344, "hash_slots": 128},
        "mem": {**N8, "pattern": "LLMLLML", "mlp_hidden": 1280},
        "win": {**N8, "pattern": "LLWLLWL", "mlp_hidden": 1280, "window": 32},
        # second round: the generation-7 memory (N: better and cheaper than the lru in the first round, and it reads
        # running text well without being trained for it) combined with what helped
        "nslot": {**N8, "pattern": "NSNSNSN", "mlp_hidden": 1296, "slots": 16},
        "nwin": {**N8, "pattern": "NNWNNWN", "mlp_hidden": 1184, "window": 32},
        "nwin16": {**N8, "pattern": "NNWNNWN", "mlp_hidden": 1184, "window": 16},
        "nwin3": {**N8, "pattern": "NWNWNWN", "mlp_hidden": 1184, "window": 32},
        "nsw": {**N8, "pattern": "NSWNSWN", "mlp_hidden": 1264, "slots": 16, "window": 32},
    },
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


def extra_round(results: dict[str, dict], groups: list[str], nova_lrs: list[str], tf_lrs: list[str]) -> list[dict]:
    """More learning rates for both sides, so that neither architecture is judged by one setting only.

    The NOVA core is run at every rate in nova_lrs; the transformer shape that was better at the common
    rate is run at every rate in tf_lrs. Rates that were already run are skipped by name."""
    rows = []
    for g in groups:
        for lr in nova_lrs:
            if lr != LR:
                rows.append({"name": f"nova-{g}-lr{lr}", "group": g, "arch": "nova", "override": NOVA[g], "lr": lr})
        first = [r for r in results.values() if r["group"] == g and r["arch"] == "transformer" and r.get("loss") and r["lr"] == LR]
        if first:
            best = min(first, key=lambda r: r["loss"]["dataset"])
            for lr in tf_lrs:
                if lr != LR:
                    rows.append({"name": f"tf-{g}-lr{lr}", "group": g, "arch": "transformer", "override": best["override"], "lr": lr})
    return rows


def candidate_round(groups: list[str], names: list[str], lr: str, compiled: bool = False, carry: int = 0, share: float = 0.0) -> list[dict]:
    """Generation-8 candidates of the NOVA core, each trained once at `lr` (carry > 1: on running text)."""
    rows = []
    for g in groups:
        for n in names:
            shape = CANDIDATES.get(g, {}).get(n)
            if shape:
                name = f"n8-{n}-{g}" + ("" if lr == LR_HIGH else f"-lr{lr}") + (f"-carry{carry}" + ("mix" if share > 0 else "") if carry > 1 else "")
                rows.append({"name": name, "group": g, "arch": "nova8", "override": shape, "lr": lr,
                             **({"compile": True} if compiled else {}), **({"carry": carry} if carry > 1 else {}), **({"carry_share": share} if carry > 1 and share > 0 else {})})
    return rows


def best_candidates(results: dict[str, dict], groups: list[str], n: int) -> list[str]:
    """Names (as in CANDIDATES) of the n generation-8 candidates with the lowest mean loss, trained the plain way."""
    names = []
    for g in groups:
        rows = [r for r in results.values() if r["group"] == g and r["arch"] == "nova8" and r.get("loss") and not r.get("carry")]
        rows.sort(key=lambda r: sum(r["loss"].values()) / len(r["loss"]))
        for r in rows[:n]:
            short = r["name"][len("n8-"):].split(f"-{g}")[0]
            if short in CANDIDATES.get(g, {}) and short not in names:
                names.append(short)
    return names


def carried_loss(model, web: np.ndarray, device: str) -> float | None:
    """Loss on the held-out web text read as running text (state carried from row to row); None for a model
    whose state grows with the text (a transformer would have to keep everything it has read)."""
    from evo.engine.long_train import evaluate_carried

    if hasattr(model, "blocks") and model.blocks and hasattr(model.blocks[0], "qkv"):
        return None
    was = model.training
    try:
        return round(float(evaluate_carried(model, web, device, streams=32)), 4)
    finally:
        model.train(was)


def train(v: dict, steps: int, max_hours: float, runner=subprocess.run) -> dict:
    from evo.engine.ab_test import parse_report

    out = OUT / f"{v['name']}.pt"
    cmd = [sys.executable, "-m", "evo.engine.long_train", "--steps", str(steps), "--lr", v["lr"], "--max-hours",
           str(max_hours), "--out-checkpoint", str(out)] + COMMON
    if v["override"]:
        cmd += ["--config-override", json.dumps(v["override"])]
    if v.get("compile"):
        cmd += ["--compile"]         # same arithmetic, fewer kernels: only the training speed changes
    if v.get("carry"):
        cmd += ["--carry", str(v["carry"])]   # the web text as running text, the state carried from row to row
        if v.get("carry_share"):
            cmd += ["--carry-share", str(v["carry_share"])]   # only this share of a batch; the rest starts cold
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
    if "web" in sets:
        res["web_carried"] = carried_loss(node.model, sets["web"], device)
    # writing speed and memory as the text behind the model grows (the NOVA state does not grow, a key/value cache does)
    res["cpu_by_context"] = {str(n): cpu_speed(node.model, int(node.model.embedding.num_embeddings), prompt_len=n, gen=32, repeats=1)
                             for n in LONG_CONTEXTS}
    del node
    free_card()
    return res, seq


def free_card() -> None:
    """Give back what measuring left on the graphics card: the next training run needs the whole card."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def verdicts(results: dict[str, dict], seq: dict[str, np.ndarray], groups: list[str]) -> dict[str, dict]:
    """Per size: the best NOVA attempt against the best transformer attempt, on the same test sequences."""
    from evo.collective.stats import paired_bootstrap, verdict

    out = {}
    for g in groups:
        novas = [r for r in results.values() if r["group"] == g and r["arch"] == "nova" and r.get("loss")]
        tfs = [r for r in results.values() if r["group"] == g and r["arch"] == "transformer" and r.get("loss")]
        if not novas or not tfs:
            continue
        nova = min(novas, key=lambda r: r["loss"]["dataset"])
        best = min(tfs, key=lambda r: r["loss"]["dataset"])
        row: dict[str, Any] = {"nova": nova["name"], "transformer": best["name"], "transformer_attempts": len(tfs),
                               "nova_attempts": len(novas), "sets": {}}
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
        # every generation-8 candidate against the best transformer and against the best generation-7 core
        row["candidates"] = {}
        for c in sorted((r for r in results.values() if r["group"] == g and r["arch"] == "nova8" and r.get("loss")),
                        key=lambda r: r["loss"]["dataset"]):
            entry: dict[str, Any] = {}
            for other, label in ((best, "vs_transformer"), (nova, "vs_gen7")):
                entry[label] = {}
                for s in c["loss"]:
                    ci = paired_bootstrap(seq[f"{s}/{other['name']}"], seq[f"{s}/{c['name']}"])   # candidate minus other
                    v = verdict(ci)
                    entry[label][s] = {**ci, "percent": round(100 * ci["diff"] / other["loss"][s], 2),
                                       "result": "better" if v == "improvement" else "worse" if v == "decline" else "no clear difference"}
            row["candidates"][c["name"]] = entry
        out[g] = row
    return out


def text(report: dict) -> str:
    L = [f"NOVA CORE vs TRANSFORMER ({report['date']}): from scratch, {report['steps']} steps = "
         f"{report['steps'] * 64 * 127 / 1e6:.0f} M tokens each, same data, same seed",
         "(loss: lower = better; the best attempt of each architecture is compared, the number of attempts is given below)"]
    for r in report["results"].values():
        if not r.get("loss"):
            L.append(f"  {r['name']:<16} FAILED: {r.get('error', '')[:160]}")
            continue
        c = r["cpu"]
        L.append(f"  {r['name']:<16} {r['params'] / 1e6:5.2f} M  lr {r['lr']:<5} loss dataset {r['loss']['dataset']:.4f}"
                 + (f"  web {r['loss']['web']:.4f}" if "web" in r["loss"] else "")
                 + (f" (as running text {r['web_carried']:.4f})" if r.get("web_carried") else "")
                 + f"  code {r['code_solved']}/{r['code_tasks']}  CPU read {c['read_tokens_per_s']} tok/s, "
                   f"write {c['write_tokens_per_s']} tok/s, state {c['state_kb_after_writing']} kB  "
                   f"train {r['train_tokens_per_s']} tok/s  {r['override'] or 'NOVA core'}")
    for g, v in report.get("verdicts", {}).items():
        L.append(f"--- {g}: best NOVA {v['nova']} (of {v.get('nova_attempts', 1)} attempts) vs best transformer "
                 f"{v['transformer']} (of {v['transformer_attempts']} attempts)")
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
        for name, e in v.get("candidates", {}).items():
            c = report["results"][name]
            long = c.get("cpu_by_context", {}).get("4096", {})
            L.append(f"  candidate {name}: " + "; ".join(
                f"{s} {c['loss'][s]:.4f} ({e['vs_transformer'][s]['percent']:+.2f} % vs transformer: {e['vs_transformer'][s]['result']}, "
                f"{e['vs_gen7'][s]['percent']:+.2f} % vs generation 7: {e['vs_gen7'][s]['result']})" for s in c["loss"]))
            L.append(f"      code {c['code_solved']}/{c['code_tasks']}, training {c['train_tokens_per_s']} tok/s"
                     f"{' (compiled)' if c.get('compile') else ''}, CPU write "
                     f"{c['cpu']['write_tokens_per_s']} tok/s, after 4096 tokens {long.get('write_tokens_per_s', '-')} tok/s "
                     f"with {long.get('state_kb_after_writing', '-')} kB of state")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=18000)
    ap.add_argument("--groups", default="10M,24M")
    ap.add_argument("--max-hours", type=float, default=3.0, help="time limit per variant")
    ap.add_argument("--web-val-dir", default="data/bulk_val_v1")
    ap.add_argument("--no-second-round", action="store_true")
    ap.add_argument("--nova-lrs", default="", help="more learning rates for the NOVA core, e.g. 1e-3,2e-3")
    ap.add_argument("--tf-lrs", default="", help="more learning rates for the better transformer shape, e.g. 2e-3")
    ap.add_argument("--candidates", default="", help="generation-8 cores to train, e.g. mlp,lru,slot,win (see CANDIDATES)")
    ap.add_argument("--candidate-lr", default=LR_HIGH)
    ap.add_argument("--only-candidates", action="store_true", help="skip the rounds that are not measured yet (NOVA 7, transformer)")
    ap.add_argument("--compile-candidates", action="store_true", help="train the candidates with the compiled forward pass")
    ap.add_argument("--carry-best", type=int, default=0,
                    help="afterwards train the N best candidates once more on running text (--candidate-carry rows per stream)")
    ap.add_argument("--candidate-carry", type=int, default=0,
                    help="train the candidates on running text: this many consecutive rows with the state carried over")
    ap.add_argument("--candidate-carry-share", type=float, default=0.0,
                    help="with --candidate-carry: the share of every batch that is running text (mixed batches)")
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
            free_card()
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

    lrs = lambda text: [x.strip() for x in text.split(",") if x.strip()]   # noqa: E731
    if not args.only_candidates:
        run(first_round(groups))
        if not args.no_second_round:
            run(second_round(report["results"], groups))
        if args.nova_lrs or args.tf_lrs:
            run(extra_round(report["results"], groups, lrs(args.nova_lrs), lrs(args.tf_lrs)))
    if args.candidates:
        carry, share = args.candidate_carry, args.candidate_carry_share
        run(candidate_round(groups, lrs(args.candidates), args.candidate_lr, args.compile_candidates, 0 if args.carry_best else carry, share))
        if args.carry_best and carry > 1:
            run(candidate_round(groups, best_candidates(report["results"], groups, args.carry_best), args.candidate_lr,
                                args.compile_candidates, carry, share))
    report["verdicts"] = verdicts(report["results"], seq, groups)
    rep_file.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (OUT / "report.txt").write_text(text(report) + "\n", encoding="utf-8")
    print(text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
