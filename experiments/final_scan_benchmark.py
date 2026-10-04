import time
import torch
import torch.nn.functional as F

from nova.config import CONFIG
from nova.model import NovaModel
from nova.model_scan import NovaModel as NovaScanModel


DEVICE = "cuda"
STEPS = 100
BATCH_SIZE = 8
SEQ_LEN = 128
LR = 3e-4
WEIGHT_DECAY = 0.01


def make_batches():
    torch.manual_seed(20260925)
    torch.cuda.manual_seed_all(20260925)

    batches = []

    for _ in range(STEPS):
        x = torch.randint(
            1,
            CONFIG.vocab_size,
            (BATCH_SIZE, SEQ_LEN),
            device=DEVICE,
        )
        batches.append(x)

    return batches


def train_model(model, batches, name):
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    torch.cuda.synchronize()
    start = time.perf_counter()

    final_loss = None

    for step, batch in enumerate(batches, 1):
        optimizer.zero_grad(set_to_none=True)

        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits, _ = model(inputs)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        final_loss = loss.item()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print()
    print(name)
    print("-" * 40)
    print(f"elapsed    : {elapsed:.3f} s")
    print(f"steps/sec  : {STEPS / elapsed:.3f}")
    print(f"final loss : {final_loss:.9f}")

    return elapsed, final_loss


def main():
    print("=" * 70)
    print("NOVA-0 FINAL BACKEND BENCHMARK")
    print("=" * 70)

    print()
    print("Configuration")
    print(f"Parameters : {sum(p.numel() for p in NovaModel(CONFIG).parameters()):,}")
    print(f"Steps      : {STEPS}")
    print(f"Batch      : {BATCH_SIZE}")
    print(f"Seq len    : {SEQ_LEN}")
    print(f"Device     : {DEVICE}")

    batches = make_batches()

    # Same initial seed and therefore same initial weights.
    torch.manual_seed(123456)
    torch.cuda.manual_seed_all(123456)

    reference = NovaModel(CONFIG).to(DEVICE)

    torch.manual_seed(123456)
    torch.cuda.manual_seed_all(123456)

    scan = NovaScanModel(CONFIG).to(DEVICE)

    # Guarantee identical weights.
    scan.load_state_dict(reference.state_dict())

    ref_time, ref_loss = train_model(
        reference,
        batches,
        "NOVA REFERENCE",
    )

    scan_time, scan_loss = train_model(
        scan,
        batches,
        "NOVA SCAN",
    )

    speedup = ref_time / scan_time

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Reference : {ref_time:.3f} s")
    print(f"Scan      : {scan_time:.3f} s")
    print(f"Speedup   : {speedup:.3f}x")
    print(f"Loss diff : {abs(ref_loss - scan_loss):.9e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
