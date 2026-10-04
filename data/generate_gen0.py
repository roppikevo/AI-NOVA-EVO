from pathlib import Path
import random

SEED = 20260924

VOCAB_SIZE = 16384
SEQ_LEN = 128

TRAIN_SAMPLES = 12000
VAL_SAMPLES = 2000
TEST_SAMPLES = 2000

OUT_DIR = Path("/opt/ai/work/nova-evo/data/gen0")


def generate_sample(rng: random.Random) -> list[int]:
    """
    Deterministic synthetic language.

    The sequence contains several interacting local patterns:
    - repeated motifs
    - short-range transitions
    - periodic structure
    - long-range token dependencies
    """

    x = []

    a = rng.randrange(100, 1000)
    b = rng.randrange(1000, 2000)
    c = rng.randrange(2000, 3000)

    period = rng.choice([5, 7, 11, 13])

    for t in range(SEQ_LEN):
        local = (
            a
            + 3 * b
            + 5 * c
            + 17 * t
            + 31 * (t % period)
        )

        if t >= 1:
            local += 7 * x[t - 1]

        if t >= 4:
            local += 11 * x[t - 4]

        if t >= 16:
            local += 13 * x[t - 16]

        token = local % VOCAB_SIZE
        x.append(token)

    return x


def write_split(
    path: Path,
    samples: int,
    seed: int,
):
    rng = random.Random(seed)

    with path.open("w", encoding="utf-8") as f:
        for _ in range(samples):
            sample = generate_sample(rng)
            f.write(" ".join(map(str, sample)))
            f.write("\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    write_split(
        OUT_DIR / "train.txt",
        TRAIN_SAMPLES,
        SEED,
    )

    write_split(
        OUT_DIR / "val.txt",
        VAL_SAMPLES,
        SEED + 1,
    )

    write_split(
        OUT_DIR / "test.txt",
        TEST_SAMPLES,
        SEED + 2,
    )

    print("=" * 60)
    print("NOVA GEN 0 DATASET")
    print("=" * 60)
    print(f"Vocabulary : {VOCAB_SIZE}")
    print(f"Sequence   : {SEQ_LEN}")
    print(f"Train      : {TRAIN_SAMPLES}")
    print(f"Validation : {VAL_SAMPLES}")
    print(f"Test       : {TEST_SAMPLES}")
    print(f"Output     : {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
