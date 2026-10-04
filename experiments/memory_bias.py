import torch
import torch.nn as nn
import torch.nn.functional as F

from nova.blocks import NovaBlock


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D = 256
NUM_CLASSES = 8
SIGNAL_SCALE = 3.0

DISTANCES = [64, 80, 96, 112, 128, 160, 192]
FORGET_BIASES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

STEPS = 1000
BATCH_SIZE = 32
LR = 2e-3

SEEDS = [1001, 2001, 3001]


class MemoryModel(nn.Module):
    def __init__(self, forget_bias):
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


def train_once(distance, forget_bias, seed):
    torch.manual_seed(seed)

    model = MemoryModel(
        forget_bias=forget_bias
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
    )

    model.train()

    for _ in range(STEPS):
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
    print("=" * 80)
    print("NOVA-0 MEMORY CAPACITY vs FORGET BIAS")
    print("=" * 80)

    print(f"Device : {DEVICE}")
    print(f"Steps  : {STEPS}")
    print(f"Batch  : {BATCH_SIZE}")
    print(f"LR     : {LR}")
    print(f"Seeds  : {SEEDS}")
    print()

    results = {}

    for bias in FORGET_BIASES:

        results[bias] = {}

        print("=" * 80)
        print(f"FORGET BIAS = {bias}")
        print("=" * 80)

        for distance in DISTANCES:

            accuracies = []

            for seed in SEEDS:

                accuracy, loss = train_once(
                    distance=distance,
                    forget_bias=bias,
                    seed=seed,
                )

                accuracies.append(accuracy)

                print(
                    f"bias={bias:<3} "
                    f"distance={distance:<3} "
                    f"seed={seed} "
                    f"accuracy={accuracy:.4f} "
                    f"loss={loss:.6f}"
                )

            mean_accuracy = sum(accuracies) / len(accuracies)

            results[bias][distance] = mean_accuracy

            print(
                f"  MEAN -> "
                f"{mean_accuracy * 100:.2f}% "
                f"{accuracies}"
            )

    print()
    print("=" * 80)
    print("SUMMARY — MEAN ACCURACY")
    print("=" * 80)

    print(
        "bias".ljust(10),
        *[str(d).rjust(8) for d in DISTANCES]
    )

    print("-" * 80)

    for bias in FORGET_BIASES:

        values = [
            results[bias][d] * 100
            for d in DISTANCES
        ]

        print(
            f"{bias:<10}",
            *[f"{v:7.2f}%" for v in values]
        )

    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
