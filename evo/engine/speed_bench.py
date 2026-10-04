"""
Where does the NOVA core lose time? Measures the current code against the faster arithmetic
(nova/fast_scan.py, nova/stepper.py) - same weights, same results - and against the yardstick transformer.

    python -m evo.engine.speed_bench                 # GPU part is small (< 1.5 GB) and short

  * training: forward + backward of the state recurrence for one layer at the real training shape, and a
    whole training step of the 24 M model (small batch), current code vs FastScan
  * writing on the CPU: tokens per second after a prompt for (a) how generation works today (the whole
    window is read again for every token), (b) the old one-token call, (c) the stepper, (d) the
    transformer with a key/value cache - at several lengths of text

Nothing is changed in the model; the result goes to evo/learning/speed_bench.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

OUT = Path("evo/learning/speed_bench.json")
BASE = {"vocab_size": 16384, "d_model": 384, "d_state": 384, "num_layers": 6, "conv_kernel": 5, "forget_bias": 1.125,
        "learnable_initial_state": False}
NOVA_24M = {**BASE, "d_model": 640, "d_state": 640, "num_layers": 8}
TF_24M = {**BASE, "arch": "transformer", "d_model": 512, "num_layers": 5, "n_heads": 8}


def _transformer(config: dict):
    """The yardstick transformer; None if it is not installed (NOVA_TRANSFORMER_FILE may point to its source)."""
    try:
        from nova.transformer_lm import build_transformer
    except ImportError:
        import importlib.util
        import os

        path = os.environ.get("NOVA_TRANSFORMER_FILE", "")
        if not path or not Path(path).exists():
            return None
        spec = importlib.util.spec_from_file_location("transformer_lm_bench", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        build_transformer = mod.build_transformer
    return build_transformer(config).eval()


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _time(fn, device: str, repeats: int) -> float:
    fn()
    _sync(device)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def scan_bench(device: str, shape=(64, 127, 640), dtype=torch.float32, repeats: int = 5) -> dict:
    """One layer's state recurrence, forward + backward: current core vs FastScan."""
    from nova.blocks_scan import ScanRefBackward
    from nova.fast_scan import FastScan

    g = torch.Generator().manual_seed(0)
    f = torch.sigmoid(torch.randn(shape, generator=g)).to(device=device, dtype=dtype).requires_grad_()
    i = torch.sigmoid(torch.randn(shape, generator=g)).to(device=device, dtype=dtype).requires_grad_()
    u = torch.randn(shape, generator=g).to(device=device, dtype=dtype).requires_grad_()
    w = torch.randn(shape, generator=g).to(device=device, dtype=dtype)

    def run(fn):
        out = fn.apply(f, i, u, None)
        grads = torch.autograd.grad((out * w).sum(), [f, i, u])
        return out.detach(), grads

    a, b = run(ScanRefBackward), run(FastScan)
    diff = max(float((a[0] - b[0]).abs().max()), *[float((x - y).abs().max()) for x, y in zip(a[1], b[1])])
    ref = _time(lambda: run(ScanRefBackward), device, repeats)
    fast = _time(lambda: run(FastScan), device, repeats)
    return {"shape": list(shape), "dtype": str(dtype).replace("torch.", ""), "reference_ms": round(ref, 2),
            "fast_ms": round(fast, 2), "speedup": round(ref / fast, 2), "max_abs_diff": diff}


def train_step_bench(config: dict, batch: int, device: str, repeats: int = 4, seq: int = 128) -> dict:
    """A whole training step (forward, loss, backward, optimizer) with the current scan and with FastScan."""
    import nova.blocks_scan as bs
    from evo.engine.architecture_factory import build_model
    from nova.fast_scan import FastScan

    original = bs.ScanRefBackward
    res = {}
    try:
        for name, fn in (("reference", original), ("fast", FastScan)):
            bs.ScanRefBackward = fn
            torch.manual_seed(0)
            model = build_model(config).to(device).train()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            x = torch.randint(12, int(config["vocab_size"]), (batch, seq), device=device)
            use_bf16 = device.startswith("cuda")

            def step():
                with torch.autocast(device_type="cuda" if use_bf16 else "cpu", dtype=torch.bfloat16, enabled=use_bf16):
                    logits = model(x[:, :-1])[0]
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), x[:, 1:].reshape(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                return float(loss)

            first = step()
            res[name + "_ms"] = round(_time(step, device, repeats), 1)
            res[name + "_first_loss"] = round(first, 4)
            del model, opt
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    finally:
        bs.ScanRefBackward = original
    res.update({"batch": batch, "speedup": round(res["reference_ms"] / res["fast_ms"], 2)})
    return res


@torch.no_grad()
def write_bench(contexts=(127, 1024, 4096), threads: int = 8, gen: int = 48) -> dict:
    """Tokens per second when writing on the CPU after a prompt of the given length (24 M models, batch 1)."""
    from evo.engine.architecture_factory import build_model
    from nova.stepper import Stepper

    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        torch.manual_seed(0)
        nova, tf = build_model(NOVA_24M).eval(), _transformer(TF_24M)
        out: dict = {"threads": threads, "by_context": {}}
        for n in contexts:
            x = torch.randint(12, 16384, (1, n))
            row = {}
            st = Stepper(nova)
            st.prime(x)
            tok = torch.tensor([5])
            t0 = time.perf_counter()
            for _ in range(gen):
                st.step(tok)
            row["nova_stepper_tok_s"] = round(gen / (time.perf_counter() - t0), 1)
            row["nova_state_kb"] = round(st.state_bytes() / 1024, 1)
            _, states = nova(x)
            tok2 = torch.tensor([[5]])
            t0 = time.perf_counter()
            for _ in range(16):
                _, states = nova(tok2, states)
            row["nova_old_step_tok_s"] = round(16 / (time.perf_counter() - t0), 1)
            window = x[:, -128:]
            t0 = time.perf_counter()
            for _ in range(4):
                nova(window)
            row["nova_today_window_tok_s"] = round(4 / (time.perf_counter() - t0), 1)
            if tf is not None:
                _, cache = tf(x)
                t0 = time.perf_counter()
                for _ in range(gen):
                    _, cache = tf(tok2, cache)
                row["transformer_cache_tok_s"] = round(gen / (time.perf_counter() - t0), 1)
                row["transformer_state_kb"] = round(sum(t.numel() * t.element_size() for kv in cache for t in kv) / 1024, 1)
            out["by_context"][str(n)] = row
        x = torch.randint(12, 16384, (1, 127))
        out["read_tok_s"] = {}
        for name, m in (("nova", nova), ("transformer", tf)):
            if m is None:
                continue
            m(x)
            t0 = time.perf_counter()
            for _ in range(5):
                m(x)
            out["read_tok_s"][name] = round(5 * 127 / (time.perf_counter() - t0))
        return out
    finally:
        torch.set_num_threads(prev)


def text(r: dict) -> str:
    L = ["SPEED: the current NOVA code vs faster arithmetic (same weights, same results) and vs the transformer"]
    for s in r.get("scan", []):
        L.append(f"  training, state recurrence of one layer {s['shape']} {s['dtype']} on {r['device']}: now {s['reference_ms']} ms, "
                 f"fast {s['fast_ms']} ms = {s['speedup']}x (largest difference {s['max_abs_diff']:.1e})")
    for s in r.get("train_step", []):
        L.append(f"  whole training step, 24 M model, batch {s['batch']} on {r['device']}: now {s['reference_ms']} ms, "
                 f"fast {s['fast_ms']} ms = {s['speedup']}x")
    w = r.get("write")
    if w:
        L.append(f"  reading a 127-token prompt on the CPU ({w['threads']} threads): NOVA {w['read_tok_s'].get('nova')} tok/s, "
                 f"transformer {w['read_tok_s'].get('transformer')} tok/s")
        for n, row in w["by_context"].items():
            L.append(f"  writing after {n} tokens: NOVA today {row['nova_today_window_tok_s']} tok/s, old one-token call "
                     f"{row['nova_old_step_tok_s']} tok/s, NOVA stepper {row['nova_stepper_tok_s']} tok/s ({row['nova_state_kb']} kB), "
                     f"transformer {row.get('transformer_cache_tok_s')} tok/s ({row.get('transformer_state_kb')} kB)")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--train-batch", type=int, default=8)
    ap.add_argument("--cpu-only", action="store_true")
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() and not args.cpu_only else "cpu"
    r: dict = {"date": time.strftime("%Y-%m-%d %H:%M"), "device": device, "scan": [], "train_step": []}
    free_gb = torch.cuda.mem_get_info()[0] / 1e9 if device == "cuda" else 0.0
    r["gpu_free_gb"] = round(free_gb, 2)
    if device == "cpu" or free_gb > 0.8:
        r["scan"].append(scan_bench(device))
        if device == "cuda":
            r["scan"].append(scan_bench(device, dtype=torch.bfloat16))
    if device == "cpu" or free_gb > 2.0:
        r["train_step"].append(train_step_bench(NOVA_24M, args.train_batch, device))
    if device == "cuda" and free_gb > 6.0:   # only when nothing else uses the card
        r["train_step"].append(train_step_bench(NOVA_24M, 64, device))
    r["write"] = write_bench(threads=args.threads)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, indent=1))
    print(text(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
