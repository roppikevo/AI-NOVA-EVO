import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from nova.model import NovaModel as ReferenceNovaModel
from nova.model_scan import NovaModel as ScanNovaModel
from nova.transformer_baseline import TransformerBaseline
from nova.config import NovaConfig
from nova.data import build_datasets


ROOT = "/opt/ai/work/nova-evo"

DATA_ROOT = os.path.join(
    ROOT,
    "data/gen_final_holdout",
)

MANIFEST_FILE = os.path.join(
    ROOT,
    "evo/final/manifest.json",
)

RESULTS_FILE = os.path.join(
    ROOT,
    "evo/final/results.json",
)

CHECKPOINT_ROOT = os.path.join(
    ROOT,
    "evo/final/checkpoints",
)

DEVICE = "cuda"

STEPS = 1000
BATCH_SIZE = 8
SEQ_LEN = 128

LR = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

SEEDS = [1001, 2002, 3003]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_batch(dataset, batch_size):
    indices = torch.randint(
        0,
        len(dataset),
        (batch_size,),
    )

    xs = []
    ys = []

    for idx in indices.tolist():
        x, y = dataset[idx]

        xs.append(x)
        ys.append(y)

    return (
        torch.stack(xs).to(DEVICE),
        torch.stack(ys).to(DEVICE),
    )


def get_logits(model, x):
    result = model(x)

    if isinstance(result, tuple):
        return result[0]

    return result


@torch.no_grad()
def evaluate(model, dataset):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    n_batches = max(
        1,
        len(dataset) // BATCH_SIZE,
    )

    for _ in range(n_batches):

        x, y = make_batch(
            dataset,
            BATCH_SIZE,
        )

        logits = get_logits(
            model,
            x,
        )

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                logits.size(-1),
            ),
            y.reshape(-1),
        )

        tokens = y.numel()

        total_loss += (
            loss.item() * tokens
        )

        total_tokens += tokens

    return total_loss / total_tokens


def build_config(config_dict):
    cfg = type(
        "Config",
        (),
        {},
    )()

    for key, value in config_dict.items():
        setattr(
            cfg,
            key,
            value,
        )

    return cfg


def create_model(candidate):
    cid = candidate["candidate_id"]

    if cid == "GEN0-NOVA":
        cfg = NovaConfig()

        model = ReferenceNovaModel(
            cfg
        )

        model_type = (
            "NOVA-GEN0-REFERENCE"
        )

    elif cid == "TRANSFORMER-BASELINE":
        model = TransformerBaseline()

        model_type = (
            "TRANSFORMER-BASELINE"
        )

    else:
        if "config" not in candidate:
            raise RuntimeError(
                f"Missing config for {cid}"
            )

        cfg = build_config(
            candidate["config"]
        )

        model = ScanNovaModel(
            cfg
        )

        model_type = (
            "NOVA-EVOLVED-SCAN"
        )

    return model, model_type


def train_candidate(
    candidate,
    train_dataset,
    val_dataset,
    seed,
):
    set_seed(seed)

    cid = candidate["candidate_id"]

    model, model_type = (
        create_model(candidate)
    )

    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    torch.cuda.reset_peak_memory_stats()

    model.train()

    start = time.perf_counter()

    for step in range(
        1,
        STEPS + 1,
    ):

        x, y = make_batch(
            train_dataset,
            BATCH_SIZE,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = get_logits(
            model,
            x,
        )

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                logits.size(-1),
            ),
            y.reshape(-1),
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    val_loss = evaluate(
        model,
        val_dataset,
    )

    peak_vram = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    checkpoint_dir = os.path.join(
        CHECKPOINT_ROOT,
        cid,
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    checkpoint_file = os.path.join(
        checkpoint_dir,
        f"seed_{seed}.pt",
    )

    checkpoint = {
        "candidate": cid,
        "model_type": model_type,
        "seed": seed,
        "steps": STEPS,
        "params": params,
        "val_loss": val_loss,
        "config": candidate.get(
            "config"
        ),
        "model_state_dict":
            model.state_dict(),
    }

    torch.save(
        checkpoint,
        checkpoint_file,
    )

    result = {
        "candidate": cid,
        "model_type": model_type,
        "seed": seed,
        "steps": STEPS,
        "params": params,
        "val_loss": val_loss,
        "perplexity": math.exp(
            val_loss
        ),
        "elapsed_sec": elapsed,
        "steps_per_sec": (
            STEPS / elapsed
        ),
        "peak_vram_gb": peak_vram,
        "checkpoint": checkpoint_file,
    }

    del model
    del optimizer

    torch.cuda.empty_cache()

    return result


def main():

    print("=" * 78)
    print("NOVA-EVO FINAL BENCHMARK")
    print("=" * 78)

    print("Candidates : 13")
    print(f"Runs       : {13 * len(SEEDS)}")
    print(f"Steps      : {STEPS}")
    print(f"Batch      : {BATCH_SIZE}")
    print(f"Seq len    : {SEQ_LEN}")
    print(f"LR         : {LR}")
    print(f"Weight dec : {WEIGHT_DECAY}")
    print(f"Grad clip  : {GRAD_CLIP}")
    print(f"Seeds      : {SEEDS}")
    print(f"Dataset    : {DATA_ROOT}")
    print()

    print(
        "TEST SET: NOT USED DURING THIS PHASE."
    )

    print(
        "Only train.txt + val.txt are loaded."
    )

    print()

    os.makedirs(
        CHECKPOINT_ROOT,
        exist_ok=True,
    )

    with open(
        MANIFEST_FILE,
        "r",
    ) as f:
        manifest = json.load(f)

    candidates = manifest[
        "candidates"
    ]

    if len(candidates) != 13:
        raise RuntimeError(
            f"Expected 13 candidates, "
            f"got {len(candidates)}"
        )

    expected_ids = [
        "GEN0-NOVA",
        "TRANSFORMER-BASELINE",
        "GEN1-010",
        "GEN1-001",
        "GEN1-006",
        "GEN2-011",
        "GEN2-004",
        "GEN2-012",
        "GEN2-024",
        "GEN3-007",
        "GEN3-005",
        "GEN3-031",
        "GEN3-024",
    ]

    actual_ids = [
        x["candidate_id"]
        for x in candidates
    ]

    if actual_ids != expected_ids:
        raise RuntimeError(
            "Manifest candidate order "
            "does not match expected order.\n"
            f"Expected: {expected_ids}\n"
            f"Actual:   {actual_ids}"
        )

    print(
        "Manifest: 13 candidates verified."
    )

    train_dataset, val_dataset, _ = (
        build_datasets(
            DATA_ROOT,
            seq_len=SEQ_LEN,
        )
    )

    print(
        f"Train samples : "
        f"{len(train_dataset)}"
    )

    print(
        f"Val samples   : "
        f"{len(val_dataset)}"
    )

    print()

    all_results = []

    total_runs = (
        len(candidates)
        * len(SEEDS)
    )

    run_number = 0

    for candidate in candidates:

        cid = candidate[
            "candidate_id"
        ]

        print("=" * 78)
        print(cid)
        print("=" * 78)

        for seed in SEEDS:

            run_number += 1

            print(
                f"[{run_number}/{total_runs}] "
                f"Starting seed {seed}..."
            )

            result = train_candidate(
                candidate,
                train_dataset,
                val_dataset,
                seed,
            )

            all_results.append(
                result
            )

            print(
                f"seed={seed} "
                f"val_loss="
                f"{result['val_loss']:.6f} "
                f"ppl="
                f"{result['perplexity']:.2f} "
                f"speed="
                f"{result['steps_per_sec']:.3f} "
                f"step/s "
                f"VRAM="
                f"{result['peak_vram_gb']:.3f} GB"
            )

            print(
                "checkpoint="
                f"{result['checkpoint']}"
            )

    if len(all_results) != 39:
        raise RuntimeError(
            f"Expected 39 runs, "
            f"got {len(all_results)}"
        )

    summary = {}

    for candidate in candidates:

        cid = candidate[
            "candidate_id"
        ]

        rows = [
            r
            for r in all_results
            if r["candidate"] == cid
        ]

        if len(rows) != len(SEEDS):
            raise RuntimeError(
                f"{cid}: expected "
                f"{len(SEEDS)} runs, "
                f"got {len(rows)}"
            )

        losses = np.array(
            [
                r["val_loss"]
                for r in rows
            ]
        )

        speeds = np.array(
            [
                r["steps_per_sec"]
                for r in rows
            ]
        )

        vrams = np.array(
            [
                r["peak_vram_gb"]
                for r in rows
            ]
        )

        summary[cid] = {
            "params": rows[0]["params"],
            "model_type":
                rows[0]["model_type"],
            "val_loss_mean":
                float(losses.mean()),
            "val_loss_std":
                float(losses.std()),
            "val_loss_min":
                float(losses.min()),
            "val_loss_max":
                float(losses.max()),
            "perplexity_mean":
                float(
                    np.exp(losses).mean()
                ),
            "speed_mean":
                float(speeds.mean()),
            "speed_std":
                float(speeds.std()),
            "vram_mean_gb":
                float(vrams.mean()),
            "vram_max_gb":
                float(vrams.max()),
            "seeds": SEEDS,
        }

    output = {
        "benchmark":
            "NOVA-EVO-FINAL",
        "dataset":
            DATA_ROOT,
        "test_used":
            False,
        "test_file_loaded":
            False,
        "candidates":
            len(candidates),
        "runs":
            len(all_results),
        "config": {
            "steps": STEPS,
            "batch_size":
                BATCH_SIZE,
            "seq_len":
                SEQ_LEN,
            "lr": LR,
            "weight_decay":
                WEIGHT_DECAY,
            "grad_clip":
                GRAD_CLIP,
            "seeds": SEEDS,
        },
        "runs_detail":
            all_results,
        "summary":
            summary,
    }

    with open(
        RESULTS_FILE,
        "w",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print("=" * 78)
    print("FINAL BENCHMARK COMPLETE")
    print("=" * 78)

    print(
        f"Candidates : "
        f"{len(candidates)}"
    )

    print(
        f"Runs       : "
        f"{len(all_results)}"
    )

    print(
        f"Results    : "
        f"{RESULTS_FILE}"
    )

    print(
        f"Checkpoints: "
        f"{CHECKPOINT_ROOT}"
    )

    print(
        "TEST SET   : NOT USED"
    )


if __name__ == "__main__":
    main()
