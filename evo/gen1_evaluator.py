from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from nova.data import build_datasets
from nova.model_scan import NovaModel


DEVICE = "cuda"

DATASET = "/opt/ai/work/nova-evo/data/gen0_unseen2"

STEPS = 300
BATCH_SIZE = 8
SEQ_LEN = 128

LR = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

SEED = 1001


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_batch(dataset, indices):
    xs = []
    ys = []

    for idx in indices:
        x, y = dataset[idx]
        xs.append(x)
        ys.append(y)

    return (
        torch.stack(xs).to(DEVICE),
        torch.stack(ys).to(DEVICE),
    )


@torch.no_grad()
def evaluate(model, dataset):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for start in range(0, len(dataset), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(dataset))

        x, y = make_batch(
            dataset,
            range(start, end),
        )

        result = model(x)
        logits = result[0] if isinstance(result, tuple) else result

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )

        total_loss += loss.item()
        total_tokens += y.numel()

    model.train()

    return total_loss / total_tokens


def build_config(data):
    class CandidateConfig:
        pass

    cfg = CandidateConfig()

    for key, value in data.items():
        setattr(cfg, key, value)

    return cfg


def evaluate_candidate(candidate, train_dataset, val_dataset):
    candidate_id = candidate["candidate_id"]

    cfg = build_config(candidate["config"])

    set_seed(SEED)

    model = NovaModel(cfg).to(DEVICE)

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    batches = [
        torch.randint(
            0,
            len(train_dataset),
            (BATCH_SIZE,),
            generator=generator,
        ).tolist()
        for _ in range(STEPS)
    ]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()

    for step in range(STEPS):
        x, y = make_batch(
            train_dataset,
            batches[step],
        )

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

    val_loss = evaluate(
        model,
        val_dataset,
    )

    peak_vram = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    steps_per_sec = STEPS / elapsed

    result = {
        "candidate_id": candidate_id,
        "parent": candidate["parent"],
        "mutation": candidate["mutation"],
        "params": params,
        "val_loss": val_loss,
        "perplexity": torch.exp(
            torch.tensor(val_loss)
        ).item(),
        "elapsed_sec": elapsed,
        "steps_per_sec": steps_per_sec,
        "peak_vram_gb": peak_vram,
        "seed": SEED,
        "steps": STEPS,
    }

    del model
    torch.cuda.empty_cache()

    return result


def main():
    candidates = json.loads(
        Path("evo/gen1/valid_candidates.json")
        .read_text()
    )

    print("=" * 78)
    print("NOVA-EVO GEN 1 — AUTOMATIC EVALUATOR")
    print("=" * 78)
    print(f"Candidates : {len(candidates)}")
    print(f"Steps      : {STEPS}")
    print(f"Batch      : {BATCH_SIZE}")
    print(f"Seed       : {SEED}")
    print(f"Dataset    : {DATASET}")
    print()

    train_dataset, val_dataset, test_dataset = build_datasets(
        DATASET,
        seq_len=SEQ_LEN,
    )

    print(
        f"Train={len(train_dataset)} "
        f"Val={len(val_dataset)} "
        f"Test={len(test_dataset)}"
    )
    print()
    print("TEST SET WILL NOT BE USED.")
    print()

    results = []

    for candidate in candidates:
        print("-" * 78)
        print(
            f"{candidate['candidate_id']} "
            f"{candidate['mutation']}"
        )

        result = evaluate_candidate(
            candidate,
            train_dataset,
            val_dataset,
        )

        results.append(result)

        print(
            f"params={result['params']:,} "
            f"val_loss={result['val_loss']:.6f} "
            f"ppl={result['perplexity']:.2f} "
            f"speed={result['steps_per_sec']:.3f} step/s "
            f"VRAM={result['peak_vram_gb']:.3f} GB"
        )

    output = "evo/gen1/results.json"

    Path(output).write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("GEN 1 EVALUATION COMPLETE")
    print("=" * 78)
    print(f"Results saved: {output}")
    print("TEST SET WAS NOT USED.")


if __name__ == "__main__":
    main()
