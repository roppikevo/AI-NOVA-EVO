"""
Sizes of the generation-8 core: from a target (number of parameters, or the memory of a graphics card) to the
shape of the core, its state, and what training it would take.

    python -m nova.sizes --presets                    # the table from 24 M to 3 B parameters
    python -m nova.sizes --params 1e9                 # the layout closest to one billion parameters
    python -m nova.sizes --vram-gb 24                 # the largest core one card of 24 GB can train
    python -m nova.sizes --params 3e8 --json          # the settings for evo.engine.train_line --config-override

What is measured and what is extrapolated. Our own runs end at 53 M parameters on one 8 GB card (RTX 4060):
24 M (width 448) and 53 M (width 704) were trained in full, a 37 M core was measured in a training step. The
rules below reproduce those runs; everything larger follows the same rules and has not been trained by us:

  * the build stays the same at every size: blocks alternate the generation-7 memory (N) and the slot memory (S),
    starting and ending with N; 16 slots; the channel mixer is 2.89 times the width (as in both trained cores)
  * deeper as well as wider when growing: 7 blocks up to about 60 M, then more blocks as the width grows
  * memory of a training step = 16 bytes per parameter (weights, gradients, two optimizer moments)
    + 12 bytes per logit (batch x tokens x vocabulary) + 122 bytes per activation (batch x tokens x width x blocks)
    + 300 MB for the card's own use - fitted to the peaks measured in the steps of the 24 M and 37 M cores
    (within 1 %). The memory the card has to hold is about 18 % more than the peak (measured: 17-19 %), and
    about 95 % of a card is usable; with that, 45 M does not fit 8 GB at batch 64 and 53 M fits at 48, as found
  * tokens: 20 per parameter is the usual compute-optimal rule of thumb; our cores saw 46 (53 M) and 100 (24 M)
    tokens per parameter and still improved, so treat it as a minimum
  * speed = about 9 TFLOP/s of useful work on the RTX 4060 (6 x parameters x tokens per second, measured on the
    trained runs); other cards scale with their bf16 throughput (--tflops)
  * the state the core carries grows with width and depth, never with the length of the text
    (56 kB at 448 x 7, 88 kB at 704 x 7)

These are plans: before a long run, a short trial run on the card itself (as the 53 M core had) confirms the
memory and the speed.

A billion-parameter core does not fit one 8 GB card for training at any useful batch; it needs a larger card or
several, and training on several cards is not implemented here yet.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

VOCAB = 16384
SEQ = 128
MLP_RATIO = 2.89
SLOTS = 16

# memory model of one training step (bytes), fitted to the measured steps
BYTES_PER_PARAM = 16
BYTES_PER_LOGIT = 12
BYTES_PER_ACT = 122
CARD_OVERHEAD_MB = 300
HELD = 1.18                 # memory the card holds per byte of peak use (measured 1.17-1.19)
USABLE = 0.95               # share of a card's memory a training run can have
RTX4060_TFLOPS = 9.0        # useful training throughput measured on the trained runs

# the two cores trained in full and one measured step - the yardstick for the rules above
MEASURED = {
    "24M": {"d_model": 448, "blocks": 7, "params": 24_167_975, "state_kb": 56.0, "step_mb_b64": 5142},
    "37M": {"d_model": 576, "blocks": 7, "params": 37_200_000, "step_mb_b64": 6180},
    "53M": {"d_model": 704, "blocks": 7, "params": 52_959_271, "state_kb": 88.0, "trial_mb_b48": 6531},
}

PRESETS = ["24M", "53M", "100M", "300M", "1B", "3B"]


def pattern(blocks: int) -> str:
    """N S N S ... N: an odd number of blocks, memory of generation 7 on both ends."""
    blocks = max(3, blocks | 1)
    return "".join("N" if i % 2 == 0 else "S" for i in range(blocks))


def blocks_for(d_model: int) -> int:
    """Depth for a width: 7 blocks up to width 768 (our trained cores), then about one more pair per 256 of width."""
    if d_model <= 768:
        return 7
    return 7 + 2 * math.ceil((d_model - 768) / 256)


def config(d_model: int, blocks: int | None = None, vocab: int = VOCAB) -> dict[str, Any]:
    blocks = blocks_for(d_model) if blocks is None else blocks
    return {"arch": "nova8", "vocab_size": vocab, "d_model": d_model, "heads": max(8, d_model // 128),
            "pattern": pattern(blocks), "mlp_hidden": int(round(MLP_RATIO * d_model / 16)) * 16, "slots": SLOTS}


def parameters(cfg: dict) -> int:
    """Exact number of parameters, counted on a model built without memory (the 'meta' device)."""
    import torch

    from evo.engine.architecture_factory import build_model

    with torch.device("meta"):
        m = build_model({**cfg, "max_seq_len": SEQ})
    return int(sum(p.numel() for p in m.parameters()))


def approx_parameters(cfg: dict) -> int:
    """Quick estimate (within 1 % of the exact count for the trained cores): embedding + 12.0 x width^2 per block."""
    d, n = cfg["d_model"], len(cfg["pattern"])
    return int(cfg.get("vocab_size", VOCAB) * d + 12.0 * d * d * n)


def step_memory_mb(params: int, d_model: int, blocks: int, batch: int, seq: int = SEQ, vocab: int = VOCAB) -> float:
    b = BYTES_PER_PARAM * params + BYTES_PER_LOGIT * batch * (seq - 1) * vocab + BYTES_PER_ACT * batch * (seq - 1) * d_model * blocks
    return b / 2 ** 20 + CARD_OVERHEAD_MB


def state_kb(d_model: int, blocks: int) -> float:
    return round(56.0 * (d_model / 448) * (blocks / 7), 1)


def train_hours(params: int, tokens: float, tflops: float = RTX4060_TFLOPS) -> float:
    return 6 * params * tokens / (tflops * 1e12) / 3600


def describe(cfg: dict, exact: bool = True, vram_gb: float | None = None, tflops: float = RTX4060_TFLOPS,
             tokens_per_param: float = 20.0) -> dict[str, Any]:
    p = parameters(cfg) if exact else approx_parameters(cfg)
    d, n = cfg["d_model"], len(cfg["pattern"])
    row: dict[str, Any] = {"params": p, "params_m": round(p / 1e6, 1), "d_model": d, "blocks": n, "pattern": cfg["pattern"],
                           "mlp_hidden": cfg["mlp_hidden"], "heads": cfg["heads"], "slots": cfg["slots"], "state_kb": state_kb(d, n),
                           # a release file holds the token table twice (reading and writing share it in the model)
                           "weights_fp16_mb": round(2 * (p + cfg.get("vocab_size", VOCAB) * d) / 1e6),
                           "weights_fp32_mb": round(4 * (p + cfg.get("vocab_size", VOCAB) * d) / 1e6),
                           "step_mb": {b: round(step_memory_mb(p, d, n, b)) for b in (64, 48, 32, 16, 8)}}
    tokens = tokens_per_param * p
    row["tokens_suggested"] = int(tokens)
    row["hours_at_tflops"] = {"tflops": tflops, "hours": round(train_hours(p, tokens, tflops), 1)}
    if vram_gb:
        row["batch_for_card"] = batch_for(p, d, n, vram_gb)
    return row


def batch_for(params: int, d_model: int, blocks: int, vram_gb: float) -> int | None:
    """The largest batch (64 down to 8) whose step fits the card with the reserve; None when none does."""
    limit = vram_gb * 1024 * USABLE
    for b in (64, 48, 32, 24, 16, 8):
        if step_memory_mb(params, d_model, blocks, b) * HELD <= limit:
            return b
    return None


def for_params(target: float, exact: bool = True) -> dict:
    """The width (a multiple of 64) whose core comes closest to `target` parameters."""
    best = None
    for d in range(256, 16385, 64):
        cfg = config(d)
        p = approx_parameters(cfg)
        if best is None or abs(p - target) < abs(best[1] - target):
            best = (cfg, p)
        if p > 2 * target:
            break
    cfg = best[0]
    return {"config": cfg, **describe(cfg, exact)}


def for_card(vram_gb: float, min_batch: int = 32, exact: bool = True) -> dict | None:
    """The largest core whose training step fits a card at batch `min_batch` or more."""
    best = None
    for d in range(256, 16385, 64):
        cfg = config(d)
        p = approx_parameters(cfg)
        b = batch_for(p, d, len(cfg["pattern"]), vram_gb)
        if b is None or b < min_batch:
            break
        best = cfg
    return None if best is None else {"config": best, **describe(best, exact, vram_gb=vram_gb)}


def preset(name: str) -> dict:
    if name in MEASURED:
        m = MEASURED[name]
        return config(m["d_model"], m["blocks"])
    target = float(name.upper().replace("M", "e6").replace("B", "e9"))
    return for_params(target, exact=False)["config"]


def table(exact: bool = True, tflops: float = RTX4060_TFLOPS) -> str:
    L = ["| Core | Parameters | Width x blocks | State | Weights (16-bit) | Training step, batch 64 / 16 | 20 tokens per parameter | Trained by us |",
         "|---|---|---|---|---|---|---|---|"]
    for name in PRESETS:
        r = describe(preset(name), exact, tflops=tflops)
        L.append(f"| {name} | {r['params_m']} M | {r['d_model']} x {r['blocks']} | {r['state_kb']} kB | {r['weights_fp16_mb']} MB "
                 f"| {r['step_mb'][64] / 1024:.1f} / {r['step_mb'][16] / 1024:.1f} GB | {r['tokens_suggested'] / 1e9:.1f} B tokens, "
                 f"~{r['hours_at_tflops']['hours']:.0f} h at {tflops:g} TFLOP/s | {'yes' if name in ('24M', '53M') else 'no'} |")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", type=float, help="target number of parameters, e.g. 1e9")
    ap.add_argument("--vram-gb", type=float, help="memory of the card to train on")
    ap.add_argument("--min-batch", type=int, default=32)
    ap.add_argument("--tflops", type=float, default=RTX4060_TFLOPS, help="useful bf16 throughput of the card (RTX 4060: about 9)")
    ap.add_argument("--presets", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quick", action="store_true", help="estimate the parameters instead of counting them")
    args = ap.parse_args(argv)
    exact = not args.quick
    if args.presets or (args.params is None and args.vram_gb is None):
        print(table(exact, args.tflops))
        return 0
    if args.vram_gb is not None and args.params is None:
        r = for_card(args.vram_gb, args.min_batch, exact)
        if r is None:
            print(f"no core trains on {args.vram_gb} GB at batch {args.min_batch} or more")
            return 1
    else:
        r = for_params(args.params, exact)
        if args.vram_gb:
            r["batch_for_card"] = batch_for(r["params"], r["d_model"], r["blocks"], args.vram_gb)
    r["hours_at_tflops"] = {"tflops": args.tflops, "hours": round(train_hours(r["params"], r["tokens_suggested"], args.tflops), 1)}
    if args.json:
        print(json.dumps(r["config"]))
    else:
        print(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
