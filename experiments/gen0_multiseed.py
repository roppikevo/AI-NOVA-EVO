import time
import random
import torch
import torch.nn.functional as F

from nova.config import CONFIG
from nova.data import build_datasets
from nova.model_scan import NovaModel as NovaScanModel
from nova.transformer_baseline import TransformerBaseline

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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_batch(dataset, indices):
    xs = []
    ys = []
    for idx in indices:
        x, y = dataset[idx]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs).to(DEVICE), torch.stack(ys).to(DEVICE)


@torch.no_grad()
def evaluate(model, dataset):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for start in range(0, len(dataset), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(dataset))
        x, y = make_batch(dataset, range(start, end))

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


def train_one(model, train_dataset, val_dataset, seed, name):
    set_seed(seed)

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    batches = [
        torch.randint(
            0,
            len(train_dataset),
            (BATCH_SIZE,),
            generator=generator,
        ).tolist()
        for _ in range(STEPS)
    ]

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start_time = time.perf_counter()

    for step in range(STEPS):
        indices = batches[step]

        x, y = make_batch(train_dataset, indices)

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

    elapsed = time.perf_counter() - start_time
    val_loss = evaluate(model, val_dataset)
    ppl = torch.exp(torch.tensor(val_loss)).item()

    print(
        f"{name:15s} seed={seed} "
        f"time={elapsed:7.2f}s "
        f"val_loss={val_loss:.6f} "
        f"ppl={ppl:.3f}"
    )

    return elapsed, val_loss


def main():
    print("=" * 78)
    print("NOVA GEN 0 — 3-SEED ROBUSTNESS BENCHMARK")
    print("=" * 78)
    print(f"Steps       : {STEPS}")
    print(f"Batch       : {BATCH_SIZE}")
    print(f"Seq len     : {SEQ_LEN}")
    print(f"Seeds       : {SEEDS}")
    print(f"Device      : {DEVICE}")
    print()

    print("Loading dataset...")

    train_dataset, val_dataset, test_dataset = build_datasets(
        "/opt/ai/work/nova-evo/data/gen0",
        seq_len=SEQ_LEN,
    )

    print(f"Train       : {len(train_dataset)}")
    print(f"Validation  : {len(val_dataset)}")
    print(f"Test        : {len(test_dataset)}")
    print()

    results = {
        "NOVA Scan": [],
        "Transformer": [],
    }

    for seed in SEEDS:
        print("-" * 78)
        print(f"SEED {seed}")
        print("-" * 78)

        set_seed(seed)
        nova = NovaScanModel(CONFIG).to(DEVICE)

        set_seed(seed)
        transformer = TransformerBaseline(
            vocab_size=CONFIG.vocab_size,
            d_model=CONFIG.d_model,
            num_layers=4,
            num_heads=8,
            ffn_dim=256,
            pad_token_id=CONFIG.pad_token_id,
        ).to(DEVICE)

        _, nova_loss = train_one(
            nova,
            train_dataset,
            val_dataset,
            seed,
            "NOVA Scan",
        )

        _, transformer_loss = train_one(
            transformer,
            train_dataset,
            val_dataset,
            seed,
            "Transformer",
        )

        results["NOVA Scan"].append(nova_loss)
        results["Transformer"].append(transformer_loss)

        del nova
        del transformer
        torch.cuda.empty_cache()

    print()
    print("=" * 78)
    print("3-SEED SUMMARY")
    print("=" * 78)

    for name, losses in results.items():
        mean = sum(losses) / len(losses)
        variance = sum((x - mean) ** 2 for x in losses) / len(losses)
        std = variance ** 0.5

        print(
            f"{name:15s} "
            f"mean={mean:.6f} "
            f"std={std:.6f}"
        )

    improvements = [
        results["Transformer"][i] - results["NOVA Scan"][i]
        for i in range(len(SEEDS))
    ]

    mean_improvement = sum(improvements) / len(improvements)

    print()
    print("Per-seed Transformer - NOVA loss:")
    for seed, value in zip(SEEDS, improvements):
        print(f"seed={seed}: {value:+.6f}")

    print(f"Mean improvement : {mean_improvement:+.6f}")
    print()
    print("TEST SET WAS NOT USED.")


if __name__ == "__main__":
    main()
