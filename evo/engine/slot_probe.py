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
    q = mixer.query(u).view(b, t, mixer.heads, mixer.dk).float()
    value, gate = w[..., :mixer.ds], torch.sigmoid(w[..., -1:])
    address = torch.softmax(w[..., mixer.ds:mixer.ds + mixer.slots], dim=-1)
    if mode == "uniform_write":
        address = torch.full_like(address, 1.0 / mixer.slots)
    share = address * gate
    log_keep = torch.log1p(-share.clamp(max=1.0 - math.exp(-MAX_DECAY))).unsqueeze(-1)
    table = lru_scan(log_keep, share.unsqueeze(-1) * value.unsqueeze(-2), None)
    read, attention = mixer.read(q, table)
    if mode == "uniform_read":
        attention = torch.full_like(attention, 1.0 / mixer.slots)
        read = attention @ (table[..., mixer.dk:] if mixer.split else table)
    y = mixer.out(read.reshape(b, t, mixer.heads * mixer.dv).to(u.dtype))
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


KINDS = ("word start", "capital word start", "inside a word", "number", "punctuation", "line break", "other")


def kind_of(piece: str) -> str:
    """A rough class of a token from its text (as the tokenizer decodes it)."""
    body = piece.strip(" ")
    if "\n" in piece:
        return "line break"
    if not body:
        return "other"
    if body[0].isdigit():
        return "number"
    if not any(c.isalnum() for c in body):
        return "punctuation"
    if piece[:1] == " ":
        return "capital word start" if body[0].isupper() else "word start"
    return "inside a word"


def _information(joint: torch.Tensor) -> float:
    """Mutual information (bits) between the rows and the columns of a table of weights."""
    p = joint / joint.sum().clamp(min=1e-12)
    outer = p.sum(1, keepdim=True) * p.sum(0, keepdim=True)
    mask = p > 0
    return float((p[mask] * (p[mask] / outer[mask]).log2()).sum())


@torch.no_grad()
def contents(model, rows: np.ndarray, decode, langs: tuple[str, ...] = ("sk", "cs", "pl", "en"), batch: int = 32, top: int = 6) -> dict:
    """What is written into each slot: which kinds of tokens, which languages, which tokens, where in the row.

    `rows` hold equal shares of the languages in the order of `langs`; `decode(token_id)` gives a token's text."""
    model = model.float().eval()
    blocks = slot_blocks(model)
    vocab, seq = model.embedding.num_embeddings, rows.shape[1] - 1
    per_lang = max(1, len(rows) // len(langs))
    kinds = torch.tensor([KINDS.index(kind_of(decode(i))) for i in range(vocab)])
    by_token = {i: torch.zeros(model.blocks[i].mixer.slots, vocab, dtype=torch.float64) for i in blocks}
    by_lang = {i: torch.zeros(model.blocks[i].mixer.slots, len(langs), dtype=torch.float64) for i in blocks}
    by_pos = {i: torch.zeros(model.blocks[i].mixer.slots, seq, dtype=torch.float64) for i in blocks}
    seen: dict[int, torch.Tensor] = {}
    hooks = [model.blocks[i].mixer.register_forward_pre_hook(lambda m, args, _i=i: seen.__setitem__(_i, args[0])) for i in blocks]
    try:
        for start in range(0, len(rows), batch):
            ids = torch.from_numpy(np.ascontiguousarray(rows[start:start + batch])).long()[:, :-1]
            model(ids)
            lang = torch.clamp(torch.arange(start, start + len(ids)) // per_lang, max=len(langs) - 1)
            for i in blocks:
                mx = model.blocks[i].mixer
                w = mx.write(seen[i]).float()
                share = (torch.softmax(w[..., mx.ds:mx.ds + mx.slots], dim=-1) * torch.sigmoid(w[..., -1:])).double()      # [b, t, slots]
                by_token[i].index_add_(1, ids.reshape(-1), share.reshape(-1, mx.slots).t())
                by_lang[i].index_add_(1, lang, share.sum(1).t())
                by_pos[i] += share.sum(0).t()
    finally:
        for h in hooks:
            h.remove()
    out: dict = {}
    for i in blocks:
        mass = by_token[i]                                                    # [slots, vocab]
        by_kind = torch.zeros(mass.shape[0], len(KINDS), dtype=torch.float64).index_add_(1, kinds, mass)
        total = mass.sum().clamp(min=1e-12)
        kind_all, lang_all = by_kind.sum(0) / total, by_lang[i].sum(0) / total
        slots = []
        for s in range(mass.shape[0]):
            m = mass[s].sum().clamp(min=1e-12)
            kind_share, lang_share = by_kind[s] / m, by_lang[i][s] / m
            k = int((kind_share / kind_all.clamp(min=1e-9)).argmax())
            best = torch.topk(mass[s], top)
            slots.append({"slot": s, "share_of_writing": round(float(m / total), 4),
                          "kind": KINDS[int(kind_share.argmax())], "kind_share": round(float(kind_share.max()), 3),
                          "most_typical_kind": KINDS[k], "typical_lift": round(float(kind_share[k] / kind_all[k].clamp(min=1e-9)), 2),
                          "languages": {l: round(float(v), 3) for l, v in zip(langs, lang_share)},
                          "mean_position": round(float((by_pos[i][s] * torch.arange(seq)).sum() / m), 1),
                          "tokens": [[decode(int(t)), round(float(v / m), 3)] for v, t in zip(best.values, best.indices)]})
        slots.sort(key=lambda r: -r["share_of_writing"])
        out[str(i)] = {"all_writing": {"kinds": {k: round(float(v), 3) for k, v in zip(KINDS, kind_all)},
                                       "languages": {l: round(float(v), 3) for l, v in zip(langs, lang_all)}},
                       "bits_slot_tells_about": {"kind of token": round(_information(by_kind), 3), "language": round(_information(by_lang[i]), 3),
                                                 "token": round(_information(mass), 3), "position in the row": round(_information(by_pos[i]), 3)},
                       "slots": slots}
    return out


def contents_text(name: str, c: dict, show: int = 8) -> str:
    L = [f"{name}: what is written into the slots"]
    for i, b in c.items():
        info = b["bits_slot_tells_about"]
        L.append(f"  block {i}: knowing the slot tells " + ", ".join(f"{v} bits about the {k}" for k, v in info.items())
                 + "; all writing: " + ", ".join(f"{k} {100 * v:.0f} %" for k, v in b["all_writing"]["kinds"].items() if v >= 0.005))
        for r in b["slots"][:show]:
            langs = " ".join(f"{l} {100 * v:.0f}" for l, v in r["languages"].items())
            toks = " ".join(repr(t) for t, _ in r["tokens"])
            L.append(f"    slot {r['slot']:>2}: {100 * r['share_of_writing']:4.1f} % of writing | mostly {r['kind']} ({100 * r['kind_share']:.0f} %), "
                     f"typical: {r['most_typical_kind']} x{r['typical_lift']} | {langs} | position {r['mean_position']:.0f} | {toks}")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="comma list of checkpoints")
    ap.add_argument("--web-dir", default="data/bulk_val_v1")
    ap.add_argument("--rows", type=int, default=100, help="rows per language")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--contents", action="store_true", help="also: what is written into each slot (kinds of tokens, languages, tokens)")
    ap.add_argument("--only-contents", action="store_true", help="skip the ablations")
    args = ap.parse_args(argv)

    from evo.engine.long_context import web_rows
    from nova.generate import load_checkpoint_model

    torch.set_num_threads(args.threads)
    rows = web_rows(Path(args.web_dir), args.rows)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    decode = None
    if args.contents or args.only_contents:
        from nova.tokenizer import NovaTokenizer

        dataset = Path(json.loads(Path("evo/engine/evo_state.json").read_text(encoding="utf-8"))["best_known"]["dataset"])
        tok = NovaTokenizer.load(dataset / "tokenizer.json")
        decode = lambda i: tok.decode([i], skip_special=False)   # noqa: E731
    for path in [p for p in args.models.split(",") if p]:
        model, _ = load_checkpoint_model(path)
        name = Path(path).stem
        if not args.only_contents:
            results[name] = {**results.get(name, {}), **probe(model, rows)}
            print(text(name, results[name]), flush=True)
        if decode is not None:
            results.setdefault(name, {})["contents"] = contents(model, rows, decode)
            print(contents_text(name, results[name]["contents"]), flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
