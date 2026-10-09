"""
Build a core of the size you choose, fitted to the machine it runs on.

    python -m nova.build                                   # what this machine is and what it can do (nothing is trained)
    python -m nova.build --size auto --text my.txt         # the plan for the largest core this machine trains well on the text
    python -m nova.build --size 100M --text my.txt --run   # build and train a 100 M core on my.txt
    python -m nova.build --size 53M                        # a size we released: the finished core is used, nothing is trained

Steps:
  1. the machine: graphics card (memory, bf16), processor threads, memory
  2. the plan: the shape of the core for the size (nova.sizes); a size we released is taken as it is (its weights are
     joined from their parts and checked); any other size is built new from the same design, and the largest core we
     released is its teacher for the first 30 % of the training (the new core learns the teacher's next-token
     distribution beside the text itself - both use the same tokenizer)
  3. a trial run on the card: a few real training steps (with the teacher) at the batch the rules suggest; when the
     card runs out of memory the next smaller batch is tried; the measured speed gives the time estimate
  4. with --run: the training on your text, held-out loss before and after, the core saved as a file that
     nova.generate loads

The time estimate is measured, not promised: the trial run is without compilation (training with compilation, as our
runs had, was faster) and a long run on a warm card can be slower. Training on several cards is not implemented.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

from nova import sizes as S

TEACHER_SHARE = 0.3          # the teacher's weight falls linearly to zero over this share of the steps (as in our 53 M run)
TEACHER_WEIGHT = 0.5
SAME_SIZE = 0.10             # a target within 10 % of a released core's parameters means that core
TOKENS_PER_PARAM = 20.0
SEQ = S.SEQ


# ---------------------------------------------------------------- 1. the machine

def ram_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1)
    except (ValueError, OSError, AttributeError):
        pass
    try:                                            # Windows
        import ctypes

        class Mem(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong)] + \
                       [(f"x{i}", ctypes.c_ulonglong) for i in range(6)]
        m = Mem()
        m.len = ctypes.sizeof(Mem)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return round(m.total / 2 ** 30, 1)
    except Exception:
        return None


def detect() -> dict[str, Any]:
    """What this machine has. The card's memory is what PyTorch can use (a little less than printed on the box)."""
    import torch

    m: dict[str, Any] = {"system": f"{platform.system()} {platform.machine()}", "cpu_threads": os.cpu_count() or 1,
                         "ram_gb": ram_gb(), "torch": torch.__version__, "gpu": None}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        m["gpu"] = {"name": p.name, "vram_gb": round(p.total_memory / 2 ** 30, 2), "count": torch.cuda.device_count(),
                    "bf16": bool(torch.cuda.is_bf16_supported()), "capability": f"{p.major}.{p.minor}"}
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        m["gpu"] = {"name": "Apple GPU (mps)", "vram_gb": None, "count": 1, "bf16": False, "capability": "mps"}
    return m


# ---------------------------------------------------------------- 2. the plan

def parse_size(text: str) -> float | None:
    """'auto' -> None; '24M', '1.5B', '3e8', '100000000' -> parameters."""
    t = text.strip().upper().replace("_", "")
    if t == "AUTO":
        return None
    mult = {"K": 1e3, "M": 1e6, "B": 1e9, "G": 1e9}.get(t[-1:], 1.0)
    value = float(t[:-1] if t[-1:] in "KMBG" else t) * mult
    if value < 1e6:
        raise ValueError(f"{text}: a core needs at least about a million parameters")
    return value


def released(root: Path | None = None) -> list[dict]:
    """Released cores with weights: name, folder, parameters, architecture, vocabulary - smallest first."""
    from nova import demo

    out = []
    for rel in demo.releases(root):
        info = json.loads((rel / "MODEL.json").read_text(encoding="utf-8"))
        cfg = info.get("config") or {}
        p = info.get("parameters")
        if not p and cfg.get("arch") == "nova8":
            p = S.parameters(cfg)
        if not p:
            continue
        out.append({"name": rel.name, "path": rel, "params": int(p), "arch": cfg.get("arch", "nova7"),
                    "vocab": cfg.get("vocab_size", S.VOCAB), "frozen": info.get("frozen", "")})
    # the newest of each size wins, then by size
    best: dict[int, dict] = {}
    for r in sorted(out, key=lambda r: r["frozen"]):
        best[round(r["params"], -5)] = r
    return sorted(best.values(), key=lambda r: r["params"])


def largest_for_hours(hours: float, tflops: float) -> float:
    """Parameters P whose compute-optimal run (20 tokens per parameter, 6 P tokens FLOP) takes `hours`."""
    return math.sqrt(hours * 3600 * tflops * 1e12 / (6 * TOKENS_PER_PARAM))


def plan(target: float | None, machine: dict, releases: list[dict], text_tokens: int | None = None,
         hours: float | None = None, tflops: float | None = None) -> dict[str, Any]:
    """What to do for a target size on this machine. Pure arithmetic: nothing is loaded or trained here."""
    notes: list[str] = []
    gpu = machine.get("gpu") or {}
    vram = gpu.get("vram_gb")
    tflops = tflops or S.RTX4060_TFLOPS
    vocab = releases[-1]["vocab"] if releases else S.VOCAB

    if target is None:                               # auto: the largest core the card, the text and the time allow
        if not vram:
            if releases:
                r = releases[-1]
                notes.append("no graphics card for training: the largest released core is used as it is "
                             "(training on the processor is possible for small experiments, see examples/train_on_text.py)")
                return {"action": "use", "release": r["name"], "params": r["params"], "notes": notes}
            raise SystemExit("no graphics card and no released core - nothing sensible can be built here")
        fit = S.for_card(vram, min_batch=32, exact=False)
        cap = fit["params"] if fit else 0
        why = f"the card ({vram} GB)"
        if text_tokens:
            by_text = text_tokens / TOKENS_PER_PARAM * 4          # up to 4 passes over the text
            if by_text < cap:
                cap, why = by_text, f"the text ({text_tokens / 1e6:.1f} M tokens, up to 4 passes)"
        if hours:
            by_time = largest_for_hours(hours, tflops)
            if by_time < cap:
                cap, why = by_time, f"the time ({hours:g} h)"
        if cap < 2e6:
            raise SystemExit(f"{why} allows no useful core - give more text or more time")
        target = cap
        notes.append(f"size chosen by {why}")

    for r in releases:
        if abs(r["params"] - target) / target <= SAME_SIZE:
            notes.append(f"{r['name']} has this size: the finished core is used, nothing is trained")
            return {"action": "use", "release": r["name"], "params": r["params"], "notes": notes}

    found = S.for_params(target, exact=False)
    cfg = {**found["config"], "vocab_size": vocab}
    p = S.approx_parameters(cfg)
    d, n = cfg["d_model"], len(cfg["pattern"])
    out: dict[str, Any] = {"action": "build", "config": cfg, "params": p, "params_m": round(p / 1e6, 1), "d_model": d,
                           "blocks": n, "state_kb": S.state_kb(d, n), "notes": notes}
    teachers = [r for r in releases if r["vocab"] == vocab]
    if teachers:
        t = teachers[-1]
        out["teacher"] = t["name"]
        if t["params"] > p:
            notes.append(f"the teacher {t['name']} is larger than the new core - the new core learns from it, it will not pass it")
    else:
        notes.append("no released core with the same tokenizer: trained without a teacher")
    tokens = TOKENS_PER_PARAM * p
    if text_tokens:
        passes = tokens / text_tokens
        if passes > 4:
            notes.append(f"the text is small for this size: {passes:.0f} passes over it would be needed; the core will "
                         f"memorise rather than learn - a smaller size (about {text_tokens * 4 / TOKENS_PER_PARAM / 1e6:.0f} M) "
                         "or more text is better")
            tokens = text_tokens * 4
    out["tokens"] = int(tokens)
    if vram:
        b = S.batch_for(p, d, n, vram)
        out["batch"] = b
        if b is None:
            notes.append(f"by the rules, a training step of this core does not fit {vram} GB even at batch 8")
        out["hours_by_rule"] = round(S.train_hours(p, tokens, tflops), 1)
    else:
        out["batch"] = None
        notes.append("no graphics card: training this size on the processor would take very long")
    if hours:
        out["hours_limit"] = hours
    return out


# ---------------------------------------------------------------- 3. a trial run

def trial(cfg: dict, batch: int, device: str, teacher=None, steps: int = 6, smaller: bool = True) -> dict[str, Any]:
    """A few real training steps (AdamW, bf16 on a card that has it) with the teacher if given. On out-of-memory the next
    smaller batch is tried. Returns the batch that ran, peak memory (MB, on a card), seconds per step and tokens per second."""
    import torch
    import torch.nn.functional as F

    from evo.engine.architecture_factory import build_model
    from evo.engine.long_train import teacher_loss

    batches = [b for b in (64, 48, 32, 24, 16, 8, 4, 2, 1) if b <= batch] if smaller else [batch]
    last_error = ""
    for b in batches:
        model = opt = None
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(0)
            model = build_model({**cfg, "max_seq_len": SEQ}).to(device).train()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
            x = torch.randint(0, cfg["vocab_size"], (b, SEQ), device=device)
            times = []
            for i in range(steps):
                t0 = time.time()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                    out = model(x)
                    logits = out[0] if isinstance(out, (tuple, list)) else out
                    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), x[:, 1:].reshape(-1))
                    if teacher is not None:
                        loss = loss + TEACHER_WEIGHT * teacher_loss(teacher, x[:, :-1], logits[:, :-1])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if device == "cuda":
                    torch.cuda.synchronize()
                if i >= min(2, steps - 1):                  # the first steps warm up
                    times.append(time.time() - t0)
            sec = sum(times) / len(times)
            res = {"batch": b, "seconds_per_step": round(sec, 4), "tokens_per_s": round(b * (SEQ - 1) / sec),
                   "peak_mb": round(torch.cuda.max_memory_reserved() / 2 ** 20) if device == "cuda" else None,
                   "params": sum(p.numel() for p in model.parameters())}
            if last_error:
                res["larger_batches"] = last_error
            return res
        except RuntimeError as e:                       # torch.OutOfMemoryError is a RuntimeError too
            if "out of memory" not in str(e).lower():
                raise
            last_error = f"batch {b} and larger ran out of memory"
        finally:
            del model, opt
            if device == "cuda":
                torch.cuda.empty_cache()
    return {"batch": None, "error": last_error or "no batch ran"}


# ---------------------------------------------------------------- 4. training on a text

def text_rows(path: Path, tok, lang: str):
    import torch

    from nova.tokenizer import EOS_ID, pack_sequences

    docs = [[tok.lang_id(lang)] + tok.encode(p.strip()) + [EOS_ID]
            for p in path.read_text(encoding="utf-8").split("\n\n") if p.strip()]
    rows = list(pack_sequences(docs, SEQ))
    if len(rows) < 20:
        raise SystemExit(f"too little text: {len(rows)} sequences of {SEQ} tokens (at least 20 are needed)")
    return torch.tensor(rows, dtype=torch.long)


def held_out_loss(model, rows, device: str, batch: int = 16) -> float:
    import torch
    import torch.nn.functional as F

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            x = rows[i:i + batch].to(device)
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            total += float(F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), x[:, 1:].reshape(-1), reduction="sum"))
            count += x[:, 1:].numel()
    return total / count


def train(cfg: dict, rows, device: str, batch: int, steps: int, out: Path, teacher=None, hours: float | None = None,
          lr: float = 1e-3, seed: int = 0, log=print) -> dict[str, Any]:
    """Train a new core on `rows` (the last tenth held out). The teacher's weight falls linearly to zero at 30 % of
    the steps. Stops at `steps` or after `hours`; saves {config, model_state_dict} to `out`."""
    import torch
    import torch.nn.functional as F

    from evo.engine.architecture_factory import build_model
    from evo.engine.long_train import lr_at, teacher_loss

    torch.manual_seed(seed)
    rows = rows[torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed))]
    held = max(2, len(rows) // 10)
    train_rows, val = rows[held:], rows[:held]
    model = build_model({**cfg, "max_seq_len": SEQ}).to(device)
    before = held_out_loss(model, val, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    until = int(TEACHER_SHARE * steps) if teacher is not None else 0
    warmup = max(1, min(1000, steps // 10))
    began, done = time.time(), 0
    for step in range(steps):
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr, warmup)
        x = train_rows[torch.randint(len(train_rows), (min(batch, len(train_rows)),))].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
            o = model(x)
            logits = o[0] if isinstance(o, (tuple, list)) else o
            ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), x[:, 1:].reshape(-1))
            loss = ce
            if step < until:
                loss = ce + TEACHER_WEIGHT * (1 - step / until) * teacher_loss(teacher, x[:, :-1], logits[:, :-1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        done = step + 1
        if done % max(1, steps // 20) == 0 or done == steps:
            log(f"step {done:>7}/{steps}  loss {float(ce.detach()):.4f}  {(time.time() - began) / 60:.1f} min")
        if hours and time.time() - began > hours * 3600:
            log(f"time limit of {hours:g} h reached at step {done}")
            break
    after = held_out_loss(model, val, device)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "model_state_dict": model.state_dict(), "steps": done, "teacher_until": until,
                "held_out_loss": after, "held_out_loss_before": before}, out)
    return {"steps": done, "held_out_before": round(before, 4), "held_out_after": round(after, 4),
            "hours": round((time.time() - began) / 3600, 2), "out": str(out)}


# ---------------------------------------------------------------- the command

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="", help="auto, or parameters: 24M, 100M, 1.5B, 3e8 (default: only describe the machine)")
    ap.add_argument("--text", default="", help="a UTF-8 text file to train on; blank lines separate documents")
    ap.add_argument("--lang", default="en", choices=["sk", "cs", "pl", "en", "py", "rs"])
    ap.add_argument("--hours", type=float, default=0.0, help="time available for training (also limits the size with auto)")
    ap.add_argument("--tflops", type=float, default=0.0, help="useful bf16 throughput, when known (else measured by the trial)")
    ap.add_argument("--no-trial", action="store_true", help="plan by the rules only, without a trial run on the card")
    ap.add_argument("--no-teacher", action="store_true")
    ap.add_argument("--run", action="store_true", help="train after the plan (needs --text)")
    ap.add_argument("--out", default="my_cores/core.pt")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    machine = detect()
    gpu = machine["gpu"]
    report: dict[str, Any] = {"machine": machine}
    if not args.json:
        print(f"machine: {machine['system']}, {machine['cpu_threads']} threads, {machine['ram_gb']} GB memory, PyTorch {machine['torch']}")
        print("card:", f"{gpu['name']}, {gpu['vram_gb']} GB, bf16 {gpu['bf16']}" if gpu else "none (processor only)")
    if not args.size:
        if gpu and gpu.get("vram_gb"):
            fit = S.for_card(gpu["vram_gb"], exact=False)
            report["largest_by_rules"] = fit and {"params_m": fit["params_m"], "batch": fit["batch_for_card"]}
            if not args.json and fit:
                print(f"by our rules the largest core this card trains at batch 32 or more: {fit['params_m']} M parameters "
                      f"(batch {fit['batch_for_card']}); see python -m nova.sizes")
        if args.json:
            print(json.dumps(report, default=str))
        return 0

    rel = released()
    tok = None
    rows = None
    text_tokens = None
    if args.text:
        from nova.tokenizer import NovaTokenizer

        if not rel:
            raise SystemExit("a released core is needed for its tokenizer (evo/releases/)")
        tok = NovaTokenizer.load(rel[-1]["path"] / "tokenizer.json")
        rows = text_rows(Path(args.text), tok, args.lang)
        text_tokens = int(rows.numel())
    p = plan(parse_size(args.size), machine, rel, text_tokens, args.hours or None, args.tflops or None)
    report["plan"] = p
    if not args.json:
        if p["action"] == "use":
            print(f"plan: use {p['release']} ({p['params'] / 1e6:.1f} M parameters) - python -m nova.demo --release {p['release']}")
        else:
            print(f"plan: new core {p['params_m']} M parameters, width {p['d_model']} x {p['blocks']} blocks ({p['config']['pattern']}), "
                  f"state {p['state_kb']} kB, teacher {p.get('teacher', 'none')}, {p['tokens'] / 1e6:.0f} M tokens"
                  + (f", batch {p['batch']} by the rules, ~{p['hours_by_rule']} h by the rules" if p.get("batch") else ""))
        for n in p["notes"]:
            print("  -", n)

    if p["action"] == "use":
        from nova import demo

        r = next(x for x in rel if x["name"] == p["release"])
        demo.load(r["path"])                              # joins the parts and checks the checksums
        report["ready"] = p["release"]
        if not args.json:
            print(f"{p['release']}: weights joined and checked")
        else:
            print(json.dumps(report, default=str))
        return 0

    import torch

    from evo.engine.long_train import load_teacher

    device = "cuda" if gpu and gpu.get("vram_gb") else "cpu"
    teacher = None
    if p.get("teacher") and not args.no_teacher and (args.run or not args.no_trial):
        from nova.parts import join

        tpath = next(x for x in rel if x["name"] == p["teacher"])["path"]
        join(tpath / "nova_model.pt")
        teacher = load_teacher(tpath / "nova_model.pt").to(device)
        if device == "cuda" and torch.cuda.is_bf16_supported():
            teacher = teacher.to(torch.bfloat16)
    batch = p.get("batch") or (16 if device == "cuda" else 4)
    if not args.no_trial:
        t = trial(p["config"], batch, device, teacher)
        report["trial"] = t
        if t.get("batch") is None:
            raise SystemExit(f"the trial run failed: {t.get('error')} - choose a smaller size")
        batch = t["batch"]
        steps = max(1, int(p["tokens"] / (batch * (SEQ - 1))))
        hours = steps * t["seconds_per_step"] / 3600
        report["estimate"] = {"batch": batch, "steps": steps, "hours": round(hours, 1)}
        if not args.json:
            print(f"trial on {device}: batch {batch}, {t['tokens_per_s']} tokens/s" + (f", peak {t['peak_mb']} MB" if t["peak_mb"] else "")
                  + (f" ({t['larger_batches']})" if t.get("larger_batches") else ""))
            print(f"estimate: {steps} steps, about {hours:.1f} h" + (f" (limited to {args.hours:g} h)" if args.hours else ""))
    else:
        steps = max(1, int(p["tokens"] / (batch * (SEQ - 1))))

    if args.run:
        if rows is None:
            raise SystemExit("--run needs --text")
        res = train(p["config"], rows, device, batch, steps, Path(args.out), teacher, args.hours or None)
        report["trained"] = res
        tok.save(Path(args.out).with_suffix(".tokenizer.json"))
        if not args.json:
            print(f"held-out loss {res['held_out_before']} -> {res['held_out_after']}; saved {res['out']} (+ tokenizer)")
    if args.json:
        print(json.dumps(report, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
