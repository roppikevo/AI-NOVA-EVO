import torch
import torch.nn as nn
import torch.nn.functional as F

from nova.blocks import NovaBlock


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D = 256
NUM_CLASSES = 8
SIGNAL_SCALE = 3.0

DISTANCES = [64, 72, 80, 88, 96, 104, 112, 120, 128]

STEPS = 1000
BATCH_SIZE = 32
LR = 2e-3

SEEDS = [1001, 2001, 3001]


class MemoryModel(nn.Module):
    def __init__(self, forget_bias=1.5):
        super().__init__()

        self.block = NovaBlock(
            d_model=D,
            d_state=D,
            conv_kernel=5,
            forget_bias=forget_bias,
        )

        self.classifier = nn.Linear(D, NUM_CLASSES)

        self.signal_embeddings = nn.Parameter(
            torch.randn(NUM_CLASSES, D) * SIGNAL_SCALE
        )

    def forward(self, labels, seq_len):
        batch = labels.shape[0]

        x = torch.zeros(
            batch,
            seq_len,
            D,
            device=labels.device,
        )

        x[:, 0, :] = self.signal_embeddings[labels]

        y, _ = self.block(x)

        logits = self.classifier(y[:, -1, :])

        return logits


def train_once(distance, seed, forget_bias=1.5):
    torch.manual_seed(seed)

    model = MemoryModel(
        forget_bias=forget_bias
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
    )

    model.train()

    for step in range(STEPS):
        labels = torch.randint(
            0,
            NUM_CLASSES,
            (BATCH_SIZE,),
            device=DEVICE,
        )

        logits = model(
            labels,
            seq_len=distance + 1,
        )

        loss = F.cross_entropy(
            logits,
            labels,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

    model.eval()

    with torch.no_grad():
        labels = torch.arange(
            NUM_CLASSES,
            device=DEVICE,
        )

        logits = model(
            labels,
            seq_len=distance + 1,
        )

        predictions = logits.argmax(dim=-1)

        accuracy = (
            predictions == labels
        ).float().mean().item()

        loss = F.cross_entropy(
            logits,
            labels,
        ).item()

    return accuracy, loss


def main():
    print("=" * 78)
    print("NOVA-0 MEMORY BOUNDARY BENCHMARK")
    print("=" * 78)

    print(f"Device       : {DEVICE}")
    print(f"Steps        : {STEPS}")
    print(f"Batch        : {BATCH_SIZE}")
    print(f"Learning     : {LR}")
    print(f"Seeds        : {SEEDS}")
    print()

    print(
        "distance".ljust(12),
        "seed".rjust(8),
        "accuracy".rjust(12),
        "loss".rjust(14),
    )

    print("-" * 78)

    results = {}

    for distance in DISTANCES:
        results[distance] = []

        for seed in SEEDS:
            accuracy, loss = train_once(
                distance=distance,
                seed=seed,
                forget_bias=1.5,
            )

            results[distance].append(accuracy)

            print(
                f"{distance:<12}"
                f"{seed:>8}"
                f"{accuracy:>12.4f}"
                f"{loss:>14.6f}"
            )

        mean_accuracy = sum(results[distance]) / len(results[distance])

        print(
            f"{'MEAN':<12}"
            f"{'':>8}"
            f"{mean_accuracy:>12.4f}"
        )

        print()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    for distance in DISTANCES:
        values = results[distance]

        mean = sum(values) / len(values)

        print(
            f"{distance:>4} tokens : "
            f"{mean * 100:6.2f}% mean accuracy "
            f"{values}"
        )

    print("=" * 78)
    print("BENCHMARK COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
