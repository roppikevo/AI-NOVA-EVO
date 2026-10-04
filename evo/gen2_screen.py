from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch

ROOT = Path("/opt/ai/work/nova-evo")
CANDIDATES_FILE = ROOT / "evo/gen2/candidates.json"
DATA_ROOT = ROOT / "data/gen0_unseen2"
OUT_FILE = ROOT / "evo/gen2/screen_results.json"

import sys
sys.path.insert(0, str(ROOT))

from nova.config import NovaConfig
from nova.model import NovaModel


SEED = 1001
STEPS = 300
BATCH_SIZE = 8
SEQ_LEN = 128
LR = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data():
    from nova.data import TokenSequenceDataset

    train = TokenSequenceDataset(
        DATA_ROOT / "train.txt",
        seq_len=SEQ_LEN,
    )

    val = TokenSequenceDataset(
        DATA_ROOT / "val.txt",
        seq_len=SEQ_LEN,
    )

    return train, val


def make_batch(data, batch_size, device):
    indices = torch.randint(
        0,
        len(data),
        (batch_size,),
    )

    samples = [data[int(i)] for i in indices]

    x = torch.stack([sample[0] for sample in samples])
    y = torch.stack([sample[1] for sample in samples])

    return x.to(device), y.to(device)


def evaluate(model, val, device):
    model.eval()

    losses = []
    limit = min(len(val), 64)

    with torch.no_grad():
        for start in range(0, limit, BATCH_SIZE):
            samples = [
                val[i]
                for i in range(start, min(start + BATCH_SIZE, limit))
            ]

            if not samples:
                continue

            x = torch.stack([sample[0] for sample in samples]).to(device)
            y = torch.stack([sample[1] for sample in samples]).to(device)

            output = model(x)

            logits = output[0] if isinstance(output, tuple) else output

            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)

def run_candidate(candidate, train, val, device):
    seed_all(SEED)

    cfg = candidate["config"]

    config = NovaConfig(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        d_state=cfg["d_state"],
        num_layers=cfg["num_layers"],
        conv_kernel=cfg["conv_kernel"],
        forget_bias=cfg["forget_bias"],
        learnable_initial_state=cfg["learnable_initial_state"],
    )

    model = NovaModel(config).to(device)

    params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    model.train()

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for step in range(STEPS):
        x, y = make_batch(
            train,
            BATCH_SIZE,
            device,
        )
        optimizer.zero_grad(set_to_none=True)

        output = model(x)
        logits = output[0] if isinstance(output, tuple) else output

        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    val_loss = evaluate(model, val, device)

    if device == "cuda":
        vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        vram = 0.0

    speed = STEPS / elapsed

    result = {
        "candidate_id": candidate["candidate_id"],
        "parent": candidate["parent"],
        "mutation": candidate["mutation"],
        "seed": SEED,
        "steps": STEPS,
        "val_loss": val_loss,
        "perplexity": math.exp(val_loss),
        "params": params,
        "speed_steps_per_sec": speed,
        "time_sec": elapsed,
        "vram_gb": vram,
    }

    del model
    del optimizer

    if device == "cuda":
        torch.cuda.empty_cache()

    return result


with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
    candidates = json.load(f)

train, val = load_data()

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 78)
print("NOVA-EVO GEN 2 — 300 STEP SCREENING")
print("=" * 78)
print(f"Device       : {device}")
print(f"Candidates   : {len(candidates)}")
print(f"Seed         : {SEED}")
print(f"Steps        : {STEPS}")
print(f"Batch        : {BATCH_SIZE}")
print(f"Seq length   : {SEQ_LEN}")
print()

results = []

for i, candidate in enumerate(candidates, 1):
    result = run_candidate(
        candidate,
        train,
        val,
        device,
    )

    results.append(result)

    print(
        f"[{i:02d}/{len(candidates)}] "
        f"{result['candidate_id']:10s} "
        f"parent={result['parent']:10s} "
        f"loss={result['val_loss']:.6f} "
        f"ppl={result['perplexity']:.2f} "
        f"params={result['params']:,} "
        f"speed={result['speed_steps_per_sec']:.3f} "
        f"VRAM={result['vram_gb']:.3f}"
    )

results.sort(key=lambda r: r["val_loss"])

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print()
print("=" * 78)
print("GEN2 SCREENING — TOP 15 BY VALIDATION LOSS")
print("=" * 78)

for i, r in enumerate(results[:15], 1):
    print(
        f"{i:02d}. "
        f"{r['candidate_id']:10s} "
        f"loss={r['val_loss']:.6f} "
        f"params={r['params']:,} "
        f"speed={r['speed_steps_per_sec']:.3f} "
        f"VRAM={r['vram_gb']:.3f}"
    )

print()
print(f"Saved: {OUT_FILE}")
