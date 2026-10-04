import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from nova.model_scan import NovaModel
from nova.data import build_datasets

ROOT = "/opt/ai/work/nova-evo"
DATA_ROOT = os.path.join(ROOT, "data/gen0_unseen2")
CANDIDATES_FILE = os.path.join(ROOT, "evo/gen3/candidates.json")
OUT_FILE = os.path.join(ROOT, "evo/gen3/robust_results.json")

DEVICE = "cuda"
STEPS = 1000
BATCH_SIZE = 8
SEQ_LEN = 128
LR = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

SEEDS = [1001, 2002, 3003]

TARGET_CANDIDATES = [
    "GEN3-007",
    "GEN3-005",
    "GEN3-008",
    "GEN3-016",
    "GEN3-031",
    "GEN3-024",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_batch(dataset, batch_size):
    indices = torch.randint(0, len(dataset), (batch_size,))
    xs = []
    ys = []

    for idx in indices.tolist():
        x, y = dataset[idx]
        xs.append(x)
        ys.append(y)

    return torch.stack(xs).to(DEVICE), torch.stack(ys).to(DEVICE)


@torch.no_grad()
def evaluate(model, dataset):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    batch_size = BATCH_SIZE
    n_batches = max(1, len(dataset) // batch_size)

    for _ in range(n_batches):
        x, y = make_batch(dataset, batch_size)

        result = model(x)
        logits = result[0] if isinstance(result, tuple) else result

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        tokens = y.numel()
        total_loss += loss.item() * tokens
        total_tokens += tokens

    return total_loss / total_tokens


def train_candidate(candidate, train_dataset, val_dataset, seed):
    set_seed(seed)

    cfg = type("Config", (), {})()

    for key, value in candidate["config"].items():
        setattr(cfg, key, value)

    model = NovaModel(cfg).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    model.train()

    start = time.perf_counter()

    for step in range(1, STEPS + 1):
        x, y = make_batch(train_dataset, BATCH_SIZE)

        optimizer.zero_grad(set_to_none=True)

        result = model(x)
        logits = result[0] if isinstance(result, tuple) else result

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    val_loss = evaluate(model, val_dataset)

    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

    result = {
        "candidate": candidate["candidate_id"],
        "mutation": candidate["mutation"],
        "seed": seed,
        "steps": STEPS,
        "params": sum(p.numel() for p in model.parameters()),
        "val_loss": val_loss,
        "perplexity": math.exp(val_loss),
        "elapsed_sec": elapsed,
        "steps_per_sec": STEPS / elapsed,
        "peak_vram_gb": peak_vram,
    }

    del model
    del optimizer
    torch.cuda.empty_cache()

    return result


def main():
    print("=" * 78)
    print("NOVA-EVO GEN 3 — ROBUST VALIDATION")
    print("=" * 78)
    print(f"Steps      : {STEPS}")
    print(f"Batch      : {BATCH_SIZE}")
    print(f"Seeds      : {SEEDS}")
    print(f"Candidates : {len(TARGET_CANDIDATES)}")
    print(f"Dataset    : {DATA_ROOT}")
    print()
    print("TEST SET WILL NOT BE USED.")
    print()

    with open(CANDIDATES_FILE, "r") as f:
        screening_results = json.load(f)

    candidate_map = {
        item["candidate_id"]: item
        for item in screening_results
    }

    selected = []

    for cid in TARGET_CANDIDATES:
        if cid not in candidate_map:
            raise RuntimeError(f"Candidate not found: {cid}")

        item = candidate_map[cid]

        if "config" not in item:
            raise RuntimeError(
                f"{cid} does not contain config information"
            )

        selected.append(item)

    train_dataset, val_dataset, _ = build_datasets(
        DATA_ROOT,
        seq_len=SEQ_LEN,
    )

    all_results = []

    for candidate in selected:
        cid = candidate["candidate_id"]
        mutation = candidate["mutation"]

        print("-" * 78)
        print(f"{cid} {mutation}")
        print("-" * 78)

        for seed in SEEDS:
            torch.cuda.reset_peak_memory_stats()

            result = train_candidate(
                candidate,
                train_dataset,
                val_dataset,
                seed,
            )

            all_results.append(result)

            print(
                f"seed={seed} "
                f"val_loss={result['val_loss']:.6f} "
                f"ppl={result['perplexity']:.2f} "
                f"speed={result['steps_per_sec']:.3f} step/s "
                f"VRAM={result['peak_vram_gb']:.3f} GB"
            )

    summary = {}

    for cid in TARGET_CANDIDATES:
        rows = [
            r for r in all_results
            if r["candidate"] == cid
        ]

        losses = np.array([r["val_loss"] for r in rows])
        speeds = np.array([r["steps_per_sec"] for r in rows])
        vrams = np.array([r["peak_vram_gb"] for r in rows])

        summary[cid] = {
            "mutation": rows[0]["mutation"],
            "params": rows[0]["params"],
            "val_loss_mean": float(losses.mean()),
            "val_loss_std": float(losses.std()),
            "val_loss_min": float(losses.min()),
            "val_loss_max": float(losses.max()),
            "perplexity_mean": float(np.exp(losses).mean()),
            "speed_mean": float(speeds.mean()),
            "speed_std": float(speeds.std()),
            "vram_mean_gb": float(vrams.mean()),
            "vram_max_gb": float(vrams.max()),
            "seeds": SEEDS,
        }

    output = {
        "config": {
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "seeds": SEEDS,
            "dataset": DATA_ROOT,
            "test_used": False,
        },
        "runs": all_results,
        "summary": summary,
    }

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print("=" * 78)
    print("GEN 3 ROBUST VALIDATION SUMMARY")
    print("=" * 78)

    for cid, row in summary.items():
        print(
            f"{cid:10s} "
            f"loss={row['val_loss_mean']:.6f} "
            f"std={row['val_loss_std']:.6f} "
            f"params={row['params']:,} "
            f"speed={row['speed_mean']:.3f} "
            f"VRAM={row['vram_mean_gb']:.3f} GB"
        )

    print()
    print(f"Results saved: {OUT_FILE}")
    print("TEST SET WAS NOT USED.")


if __name__ == "__main__":
    main()
