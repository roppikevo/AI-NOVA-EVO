import torch
import torch.nn as nn
import torch.nn.functional as F

from nova.blocks import NovaBlock


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D = 256

# 8 rôznych signálov.
NUM_CLASSES = 8

# Každá trieda dostane vlastný jednoduchý vektor.
SIGNAL_SCALE = 3.0

# Vzdialenosti, ktoré budeme postupne testovať.
DISTANCES = [8, 16, 32, 64, 128, 256]

# Tréning.
STEPS = 1000
BATCH_SIZE = 32
LR = 2e-3


class MemoryModel(nn.Module):
    """
    Malý wrapper okolo NOVA bloku.

    Úloha:
        prvý token = signál
        ďalšie tokeny = šum
        posledný token = query

    Na query pozícii klasifikujeme pôvodný signál.
    """

    def __init__(self, forget_bias=1.5):
        super().__init__()

        self.block = NovaBlock(
            d_model=D,
            d_state=D,
            conv_kernel=5,
            forget_bias=forget_bias,
        )

        self.classifier = nn.Linear(D, NUM_CLASSES)

        # Naučiteľné reprezentácie signálov.
        self.signal_embeddings = nn.Parameter(
            torch.randn(NUM_CLASSES, D) * SIGNAL_SCALE
        )

    def forward(self, labels, seq_len):
        batch = labels.shape[0]

        # Prázdna sekvencia.
        x = torch.zeros(
            batch,
            seq_len,
            D,
            device=labels.device,
        )

        # Prvý token obsahuje signál.
        x[:, 0, :] = self.signal_embeddings[labels]

        # Posledný token je query.
        # Dostane nulový vstup; rozhoduje stav.
        y, state = self.block(x)

        logits = self.classifier(y[:, -1, :])

        return logits


def train_distance(distance, forget_bias=1.5):
    torch.manual_seed(1000 + distance)

    model = MemoryModel(
        forget_bias=forget_bias
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
    )

    model.train()

    for step in range(1, STEPS + 1):
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

    # Evaluation.
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
    print("=" * 72)
    print("NOVA-0 LEARNED MEMORY BENCHMARK")
    print("=" * 72)
    print(f"Device      : {DEVICE}")
    print(f"Steps       : {STEPS}")
    print(f"Batch       : {BATCH_SIZE}")
    print(f"Learning    : {LR}")
    print()

    print(
        "distance".ljust(12),
        "accuracy".rjust(12),
        "loss".rjust(14),
    )

    print("-" * 72)

    for distance in DISTANCES:
        accuracy, loss = train_distance(
            distance=distance,
            forget_bias=1.5,
        )

        print(
            f"{distance:<12}"
            f"{accuracy:>12.4f}"
            f"{loss:>14.6f}"
        )

    print()
    print("=" * 72)
    print("BENCHMARK COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
