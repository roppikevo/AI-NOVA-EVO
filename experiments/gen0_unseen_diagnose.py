from pathlib import Path
from collections import Counter
import math

ROOT = Path("/opt/ai/work/nova-evo/data/gen0_unseen")
VOCAB_SIZE = 16384
SEQ_LEN = 128


def load_file(path):
    sequences = []
    with open(path) as f:
        for line in f:
            x = list(map(int, line.split()))
            sequences.append(x)
    return sequences


def entropy(counter, total):
    h = 0.0
    for n in counter.values():
        p = n / total
        h -= p * math.log2(p)
    return h


def lag_accuracy(sequences, lag):
    correct = 0
    total = 0
    for x in sequences:
        for t in range(lag, len(x)):
            if x[t] == x[t - lag]:
                correct += 1
            total += 1
    return correct / total


print("=" * 70)
print("NOVA GEN 0 — UNSEEN DATASET DIAGNOSTIC")
print("=" * 70)

train = load_file(ROOT / "train.txt")
val = load_file(ROOT / "val.txt")

print(f"Train sequences : {len(train)}")
print(f"Val sequences   : {len(val)}")
print(f"Seq len         : {SEQ_LEN}")
print(f"Vocabulary      : {VOCAB_SIZE}")

# ------------------------------------------------------------
# Basic token statistics
# ------------------------------------------------------------

train_counter = Counter(v for x in train for v in x)
val_counter = Counter(v for x in val for v in x)

train_total = sum(train_counter.values())
val_total = sum(val_counter.values())

print("\nTOKEN STATISTICS")
print("-" * 70)
print(f"Train tokens    : {train_total:,}")
print(f"Val tokens      : {val_total:,}")
print(f"Train unique    : {len(train_counter):,}")
print(f"Val unique      : {len(val_counter):,}")

print(f"Train entropy   : {entropy(train_counter, train_total):.6f} bits")
print(f"Val entropy     : {entropy(val_counter, val_total):.6f} bits")
print(f"Uniform entropy : {math.log2(VOCAB_SIZE):.6f} bits")

# ------------------------------------------------------------
# Most frequent tokens
# ------------------------------------------------------------

print("\nTOP TRAIN TOKENS")
print("-" * 70)

for token, count in train_counter.most_common(20):
    p = count / train_total
    print(f"{token:5d}  count={count:8d}  p={p:.8f}")

# ------------------------------------------------------------
# Exact lag copying
# ------------------------------------------------------------

print("\nEXACT LAG COPY ACCURACY")
print("-" * 70)

for lag in [1, 2, 3, 4, 5, 8, 11, 16, 32, 64]:
    acc = lag_accuracy(val, lag)
    print(f"lag={lag:2d}  accuracy={acc:.8f}")

# ------------------------------------------------------------
# Train/validation vocabulary overlap
# ------------------------------------------------------------

train_vocab = set(train_counter)
val_vocab = set(val_counter)

intersection = train_vocab & val_vocab

print("\nVOCABULARY OVERLAP")
print("-" * 70)
print(f"Train vocab     : {len(train_vocab):,}")
print(f"Val vocab       : {len(val_vocab):,}")
print(f"Intersection    : {len(intersection):,}")
print(f"Val unseen rate : {1.0 - len(intersection)/len(val_vocab):.8f}")

# ------------------------------------------------------------
# First sequences
# ------------------------------------------------------------

print("\nFIRST TRAIN SEQUENCE")
print("-" * 70)
print(train[0][:32])

print("\nFIRST VAL SEQUENCE")
print("-" * 70)
print(val[0][:32])

print("\nDIAGNOSTIC COMPLETE")
