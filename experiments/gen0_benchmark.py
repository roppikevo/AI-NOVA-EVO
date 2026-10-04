import time
import random

import torch
import torch.nn.functional as F

from nova.config import CONFIG
from nova.data import build_datasets
from nova.model import NovaModel
from nova.model_scan import NovaModel as NovaScanModel
from nova.transformer_baseline import TransformerBaseline


DEVICE = "cuda"

STEPS = 1000
BATCH_SIZE = 8
SEQ_LEN = 128

LR = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

EVAL_EVERY = 100

SEED = 20260925


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_batch(dataset, indices):
    xs = []

    for idx in indices:
        x, y = dataset[idx]
        xs.append(x)

    return torch.stack(xs).to(DEVICE)


@torch.no_grad()
def evaluate(model, dataset):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    # Deterministic validation subset/full validation.
    for start in range(0, len(dataset), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(dataset))

        xs = []
        ys = []

        for idx in range(start, end):
            x, y = dataset[idx]
            xs.append(x)
            ys.append(y)

        x = torch.stack(xs).to(DEVICE)
        y = torch.stack(ys).to(DEVICE)

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


def train_model(model, train_dataset, val_dataset, name):
    set_seed(SEED)

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # Fixed deterministic batch sequence.
    generator = torch.Generator()
    generator.manual_seed(SEED)

    batch_indices = []

    for _ in range(STEPS):
        indices = torch.randint(
            0,
            len(train_dataset),
            (BATCH_SIZE,),
            generator=generator,
        ).tolist()

        batch_indices.append(indices)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start_time = time.perf_counter()

    for step in range(1, STEPS + 1):
        indices = batch_indices[step - 1]

        x = make_batch(train_dataset, indices)
        y = torch.stack(
            [train_dataset[i][1] for i in indices]
        ).to(DEVICE)

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

        if step % EVAL_EVERY == 0 or step == 1:
            torch.cuda.synchronize()

            val_loss = evaluate(
                model,
                val_dataset,
            )

            elapsed = time.perf_counter() - start_time

            print(
                f"{name:18s} "
                f"step={step:4d} "
                f"train={loss.item():.6f} "
                f"val={val_loss:.6f} "
                f"ppl={torch.exp(torch.tensor(val_loss)).item():.3f} "
                f"time={elapsed:.1f}s"
            )

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    final_val = evaluate(
        model,
        val_dataset,
    )

    return elapsed, final_val


def main():
    print("=" * 78)
    print("NOVA GEN 0 — ARCHITECTURE BENCHMARK")
    print("=" * 78)

    print()
    print("Configuration")
    print(f"Steps       : {STEPS}")
    print(f"Batch       : {BATCH_SIZE}")
    print(f"Seq len     : {SEQ_LEN}")
    print(f"LR          : {LR}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Seed        : {SEED}")
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
    print("Parameter counts")

    set_seed(SEED)
    nova_reference = NovaModel(CONFIG)

    set_seed(SEED)
    nova_scan = NovaScanModel(CONFIG)

    set_seed(SEED)
    transformer = TransformerBaseline(
        vocab_size=CONFIG.vocab_size,
        d_model=CONFIG.d_model,
        num_layers=4,
        num_heads=8,
        ffn_dim=256,
        pad_token_id=CONFIG.pad_token_id,
    )

    models = [
        ("NOVA Reference", nova_reference),
        ("NOVA Scan", nova_scan),
        ("Transformer", transformer),
    ]

    for name, model in models:
        print(
            f"{name:18s}: "
            f"{sum(p.numel() for p in model.parameters()):,}"
        )

    print()
    print("=" * 78)

    results = []

    for name, model in models:
        set_seed(SEED)

        model = model.to(DEVICE)

        elapsed, val_loss = train_model(
            model,
            train_dataset,
            val_dataset,
            name,
        )

        results.append(
            (
                name,
                elapsed,
                val_loss,
            )
        )

        del model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print()

    print("=" * 78)
    print("FINAL GEN 0 VALIDATION RESULTS")
    print("=" * 78)

    for name, elapsed, val_loss in results:
        ppl = torch.exp(
            torch.tensor(val_loss)
        ).item()

        print(
            f"{name:18s} "
            f"time={elapsed:8.2f}s "
            f"val_loss={val_loss:.6f} "
            f"ppl={ppl:.3f}"
        )

    print("=" * 78)
    print("TEST SET WAS NOT USED.")
    print("=" * 78)


if __name__ == "__main__":
    main()
