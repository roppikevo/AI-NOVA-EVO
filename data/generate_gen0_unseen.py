from pathlib import Path
import random

SEED = 20300117

VOCAB_SIZE = 16384
SEQ_LEN = 128

TRAIN_SAMPLES = 12000
VAL_SAMPLES = 2000
TEST_SAMPLES = 2000

OUT_DIR = Path("/opt/ai/work/nova-evo/data/gen0_unseen")


def generate_sample(rng: random.Random) -> list[int]:
    """
    GEN0-UNSEEN.

    Deliberately different synthetic sequence generator from GEN0.

    Dependencies:
    - lag 1
    - lag 3
    - lag 8
    - lag 32
    - periodic regime changes
    - nonlinear interaction between distant tokens
    - evolving latent state
    """

    x = []

    a = rng.randrange(500, 4000)
    b = rng.randrange(4000, 8000)
    c = rng.randrange(8000, 12000)
    d = rng.randrange(12000, VOCAB_SIZE)

    period = rng.choice([6, 9, 14, 17])

    latent = (a + 3 * b + 5 * c + 7 * d) % VOCAB_SIZE

    for t in range(SEQ_LEN):
        regime = (t // period) % 4

        local = (
            latent
            + 19 * t
            + 23 * (t % period)
            + 29 * regime
        )

        if t >= 1:
            local += 7 * x[t - 1]

        if t >= 3:
            local += 13 * x[t - 3]

        if t >= 8:
            local += 17 * x[t - 8]

        if t >= 32:
            local += 31 * x[t - 32]

        if t >= 8:
            nonlinear = (
                (x[t - 1] * x[t - 8])
                + 3 * (x[t - 3] ^ x[t - 8])
            ) % VOCAB_SIZE
            local += nonlinear

        token = local % VOCAB_SIZE
        x.append(token)

        latent = (
            11 * latent
            + 7 * token
            + 13 * regime
            + t
        ) % VOCAB_SIZE

    return x


def write_split(path: Path, samples: int, seed: int):
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
    print("NOVA GEN 0 — UNSEEN DATASET")
    print("=" * 60)
    print(f"Vocabulary : {VOCAB_SIZE}")
    print(f"Sequence   : {SEQ_LEN}")
    print(f"Train      : {TRAIN_SAMPLES}")
    print(f"Validation : {VAL_SAMPLES}")
    print(f"Test       : {TEST_SAMPLES}")
    print(f"Seed       : {SEED}")
    print(f"Output     : {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
