"""
What do the slot memories of a generation-8 core actually do?

For every slot block (mixer S) of a trained core, on held-out text:

    writing    how open the write gate is, how sharply a token picks its slots, how many of the slots are in
               real use, how long a slot keeps what was written into it (half-life in tokens)
    reading    how sharply a query picks among the slots, how big the block's output is next to the stream
    ablation   the loss when the block is switched off, when it reads all slots equally (no choice by content),
               and when it writes into all slots equally (the table becomes one running average)

The ablations say how much of the block's worth is the content addressing and how much is "one more memory".

    python -m evo.engine.slot_probe --models a.pt,b.pt [--rows 100] [--threads 4]
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT = Path("evo/learning/slot_probe.json")
MODES = ("normal", "off", "uniform_read", "uniform_write")


def slot_forward(mixer, u: torch.Tensor, mode: str = "normal", stats: dict | None = None) -> torch.Tensor:
    """The slot mixer's forward pass for a fresh state, with one part replaced (mode) and its inner numbers kept (stats)."""
    from nova.core8 import MAX_DECAY, lru_scan

    b, t, _ = u.shape
    w = mixer.write(u).float()
    q = mixer.query(u).view(b, t, mixer.heads, mixer.ds).float()
    value, gate = w[..., :mixer.ds], torch.sigmoid(w[..., -1:])
    address = torch.softmax(w[..., mixer.ds:mixer.ds + mixer.slots], dim=-1)
    if mode == "uniform_write":
        address = torch.full_like(address, 1.0 / mixer.slots)
    share = address * gate
    log_keep = torch.log1p(-share.clamp(max=1.0 - math.exp(-MAX_DECAY))).unsqueeze(-1)
    table = lru_scan(log_keep, share.unsqueeze(-1) * value.unsqueeze(-2), None)
    match = torch.einsum("bthd,btsd->bths", q, table) / math.sqrt(mixer.ds)
    attention = torch.softmax(match, dim=-1)
    if mode == "uniform_read":
        attention = torch.full_like(attention, 1.0 / mixer.slots)
    read = torch.einsum("bths,btsd->bthd", attention, table)
    y = mixer.out(read.reshape(b, t, mixer.heads * mixer.ds).to(u.dtype))
    if mode == "off":
        y = torch.zeros_like(y)
    if stats is not None:
        eps = 1e-9
        usage = share.mean(dim=(0, 1))                                              # how much is written into each slot
        usage_p = usage / usage.sum().clamp(min=eps)
        stats.update({
            "gate_mean": float(gate.mean()), "gate_open_share": float((gate > 0.5).float().mean()),
            "address_slots_per_token": float(torch.exp(-(address * (address + eps).log()).sum(-1)).mean()),
            "slots_in_use": float(torch.exp(-(usage_p * (usage_p + eps).log()).sum())),
            "busiest_slot_share": float(usage_p.max()),
            "half_life_tokens": [round(float(math.log(0.5) / min(-1e-9, float(v))), 1) for v in log_keep.mean(dim=(0, 1, 3))],
            "read_slots_per_query": float(torch.exp(-(attention * (attention + eps).log()).sum(-1)).mean()),
            "read_top_slot_weight": float(attention.max(dim=-1).values.mean()),
            "output_norm": float(y.float().norm(dim=-1).mean()),
        })
    return y


@contextmanager
def patched(model, modes: dict[int, str], stats: dict[int, dict] | None = None):
    """Run the model with some slot blocks in another mode ({block index: mode})."""
    saved = {}
    for i, mode in modes.items():
        mixer = model.blocks[i].mixer
        saved[i] = mixer.__dict__.get("forward")
        st = None if stats is None else stats.setdefault(i, {})
        mixer.forward = (lambda u, state=None, _m=mixer, _mode=mode, _st=st: (slot_forward(_m, u, _mode, _st), None))
    try:
        yield
    finally:
        for i, f in saved.items():
            if f is None:
                del model.blocks[i].mixer.forward
            else:
                model.blocks[i].mixer.forward = f


@torch.no_grad()
def loss(model, rows: np.ndarray, batch: int = 32) -> float:
    total, n = 0.0, 0
    for i in range(0, len(rows), batch):
        b = torch.from_numpy(np.ascontiguousarray(rows[i:i + batch])).long()
        out = model(b[:, :-1])
        logits = out[0] if isinstance(out, (tuple, list)) else out
        total += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), b[:, 1:].reshape(-1), reduction="sum"))
        n += b[:, 1:].numel()
    return total / max(n, 1)


def slot_blocks(model) -> list[int]:
    from nova.core8 import SlotMixer

    return [i for i, blk in enumerate(getattr(model, "blocks", [])) if isinstance(getattr(blk, "mixer", None), SlotMixer)]


@torch.no_grad()
def probe(model, rows: np.ndarray, batch: int = 32) -> dict:
    model = model.float().eval()
    blocks = slot_blocks(model)
    if not blocks:
        return {"slot_blocks": []}
    base = loss(model, rows, batch)
    stats: dict[int, dict] = {}
    with patched(model, {i: "normal" for i in blocks}, stats):
        same = loss(model, rows[:batch], batch)                    # the patched pass must be the model's own pass
        plain_check = abs(same - loss_plain(model, rows[:batch], batch, blocks))
        streams = stream_norms(model, rows[:batch], blocks)
    out: dict = {"loss": round(base, 4), "slot_blocks": blocks, "patched_pass_differs_by": round(plain_check, 6), "blocks": {}}
    for i in blocks:
        s = dict(stats[i])
        s["output_vs_stream"] = round(s.pop("output_norm") / max(streams[i], 1e-9), 3)
        for mode in MODES[1:]:
            with patched(model, {i: mode}):
                s[f"loss_{mode}"] = round(loss(model, rows, batch), 4)
        out["blocks"][str(i)] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}
    for mode in MODES[1:]:
        with patched(model, {i: mode for i in blocks}):
            out[f"loss_all_{mode}"] = round(loss(model, rows, batch), 4)
    return out


def loss_plain(model, rows: np.ndarray, batch: int, blocks: list[int]) -> float:
    """The loss with the model's own slot code (the patches taken off for a moment)."""
    saved = {i: model.blocks[i].mixer.forward for i in blocks}
    for i in blocks:
        del model.blocks[i].mixer.forward
    try:
        return loss(model, rows, batch)
    finally:
        for i, f in saved.items():
            model.blocks[i].mixer.forward = f


def stream_norms(model, rows: np.ndarray, blocks: list[int]) -> dict[int, float]:
    """Mean size of the residual stream entering each slot block."""
    norms: dict[int, float] = {}
    hooks = [model.blocks[i].register_forward_pre_hook(lambda m, args, _i=i: norms.__setitem__(_i, float(args[0].float().norm(dim=-1).mean())))
             for i in blocks]
    try:
        model(torch.from_numpy(np.ascontiguousarray(rows)).long()[:, :-1])
    finally:
        for h in hooks:
            h.remove()
    return norms


def text(name: str, r: dict) -> str:
    if not r.get("slot_blocks"):
        return f"{name}: no slot blocks"
    pct = lambda v: f"{100 * (v / r['loss'] - 1):+.2f} %"
    L = [f"{name}: loss {r['loss']:.4f}; all slot blocks off {r['loss_all_off']:.4f} ({pct(r['loss_all_off'])}), "
         f"all reading every slot equally {r['loss_all_uniform_read']:.4f} ({pct(r['loss_all_uniform_read'])}), "
         f"all writing into every slot equally {r['loss_all_uniform_write']:.4f} ({pct(r['loss_all_uniform_write'])})"]
    for i, s in r["blocks"].items():
        hl = s["half_life_tokens"]
        L.append(f"  block {i}: write gate {s['gate_mean']:.2f} (open for {100 * s['gate_open_share']:.0f} % of tokens), a token writes into "
                 f"{s['address_slots_per_token']:.1f} slots, {s['slots_in_use']:.1f} of {len(hl)} slots in use (busiest takes {100 * s['busiest_slot_share']:.0f} %), "
                 f"half-life {min(hl):.0f}-{max(hl):.0f} tokens (median {float(np.median(hl)):.0f})")
        L.append(f"           a query reads {s['read_slots_per_query']:.1f} slots (top slot weight {s['read_top_slot_weight']:.2f}), output {s['output_vs_stream']:.2f} of the stream; "
                 f"off {pct(s['loss_off'])}, uniform read {pct(s['loss_uniform_read'])}, uniform write {pct(s['loss_uniform_write'])}")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="comma list of checkpoints")
    ap.add_argument("--web-dir", default="data/bulk_val_v1")
    ap.add_argument("--rows", type=int, default=100, help="rows per language")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)

    from evo.engine.long_context import web_rows
    from nova.generate import load_checkpoint_model

    torch.set_num_threads(args.threads)
    rows = web_rows(Path(args.web_dir), args.rows)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for path in [p for p in args.models.split(",") if p]:
        model, _ = load_checkpoint_model(path)
        results[Path(path).stem] = probe(model, rows)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=1))
        print(text(Path(path).stem, results[Path(path).stem]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
