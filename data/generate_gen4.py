from pathlib import Path
import random

SEED = 20260926
VOCAB_SIZE = 16384
SEQ_LEN = 128

TRAIN_SAMPLES = 12000
VAL_SAMPLES = 2000
TEST_SAMPLES = 2000

OUT_DIR = Path("/opt/ai/work/nova-evo/data/gen4")


def generate_sample(rng: random.Random) -> list[int]:
    x = []

    # Three dependency regimes matching the established benchmark
    # while using a completely new random dataset seed.
    regime = rng.randrange(3)

    a = rng.randrange(1000, 15000)
    b = rng.randrange(1000, 15000)
    c = rng.randrange(1000, 15000)

    base = (a + 3 * b + 5 * c) % VOCAB_SIZE

    for t in range(SEQ_LEN):

        # LOCAL
        if regime == 0:
            if t == 0:
                token = base
            else:
                token = (
                    x[t - 1]
                    + 7
                    + 3 * (t % 5)
                    + (x[t - 1] % 11)
                ) % VOCAB_SIZE

        # MEDIUM RANGE
        elif regime == 1:
            if t == 0:
                token = base
            elif t < 4:
                token = (
                    base
                    + 5 * x[t - 1]
                    + 11 * t
                ) % VOCAB_SIZE
            elif t < 8:
                token = (
                    3 * x[t - 1]
                    + 7 * x[t - 4]
                    + 13 * t
                    + base
                ) % VOCAB_SIZE
            else:
                token = (
                    3 * x[t - 1]
                    + 7 * x[t - 4]
                    + 11 * x[t - 8]
                    + 13 * t
                    + base
                ) % VOCAB_SIZE

        # LONG RANGE / STATE
        else:
            if t == 0:
                state = base
                token = state
            else:
                state = (
                    5 * state
                    + 3 * x[t - 1]
                    + 7 * (x[t - 1] % 17)
                    + 11 * (t % 13)
                ) % VOCAB_SIZE

                if t >= 16:
                    state = (
                        state
                        + 13 * x[t - 16]
                    ) % VOCAB_SIZE

                token = state

        x.append(token)

    return x


def write_split(path: Path, count: int, rng: random.Random):
    with path.open("w", encoding="utf-8") as f:
        for _ in range(count):
            sample = generate_sample(rng)
            f.write(" ".join(map(str, sample)) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)

    write_split(OUT_DIR / "train.txt", TRAIN_SAMPLES, rng)
    write_split(OUT_DIR / "val.txt", VAL_SAMPLES, rng)
    write_split(OUT_DIR / "test.txt", TEST_SAMPLES, rng)

    print("=" * 60)
    print("NOVA GEN 4 — DEDICATED EVOLUTION DATASET")
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
