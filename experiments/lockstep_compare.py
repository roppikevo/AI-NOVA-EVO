import copy
import torch
import torch.nn.functional as F

from nova.config import CONFIG
from nova.model import NovaModel
from experiments.custom_scan_refback import CustomRefBackwardModel


DEVICE = "cuda"
STEPS = 1000
LR = 3e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 8
SEQ_LEN = 128


def max_param_diff(a, b):
    max_diff = 0.0
    mean_diff_sum = 0.0
    count = 0

    for pa, pb in zip(a.parameters(), b.parameters()):
        d = (pa.detach() - pb.detach()).abs()
        max_diff = max(max_diff, d.max().item())
        mean_diff_sum += d.sum().item()
        count += d.numel()

    return max_diff, mean_diff_sum / count


def max_grad_diff(a, b):
    max_diff = 0.0

    for pa, pb in zip(a.parameters(), b.parameters()):
        if pa.grad is None or pb.grad is None:
            continue

        d = (pa.grad.detach() - pb.grad.detach()).abs()
        max_diff = max(max_diff, d.max().item())

    return max_diff


def main():
    print("=" * 70)
    print("NOVA-0 LOCKSTEP REFERENCE vs SCAN+REFERENCE-BACKWARD")
    print("=" * 70)

    torch.manual_seed(20260925)
    torch.cuda.manual_seed_all(20260925)

    reference = NovaModel(CONFIG).to(DEVICE)
    scan = CustomRefBackwardModel(CONFIG).to(DEVICE)

    # EXACT SAME INITIAL WEIGHTS
    scan.load_state_dict(copy.deepcopy(reference.state_dict()))

    reference.train()
    scan.train()

    opt_ref = torch.optim.AdamW(
        reference.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    opt_scan = torch.optim.AdamW(
        scan.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    torch.manual_seed(424242)
    torch.cuda.manual_seed_all(424242)

    # Fixed batches — both models receive EXACTLY the same data.
    batches = []

    for step in range(STEPS):
        x = torch.randint(
            1,
            CONFIG.vocab_size,
            (BATCH_SIZE, SEQ_LEN),
            device=DEVICE,
        )
        batches.append(x)

    checkpoints = {1, 2, 5, 10, 25, 50, 75, 100}

    for step in range(1, STEPS + 1):
        x = batches[step - 1]

        opt_ref.zero_grad(set_to_none=True)
        opt_scan.zero_grad(set_to_none=True)

        input_ids = x[:, :-1]
        targets = x[:, 1:]

        logits_ref, _ = reference(input_ids)
        logits_scan, _ = scan(input_ids)

        loss_ref = F.cross_entropy(
            logits_ref.reshape(-1, logits_ref.size(-1)),
            targets.reshape(-1),
        )

        loss_scan = F.cross_entropy(
            logits_scan.reshape(-1, logits_scan.size(-1)),
            targets.reshape(-1),
        )

        loss_ref.backward()
        loss_scan.backward()

        grad_diff = max_grad_diff(reference, scan)

        torch.nn.utils.clip_grad_norm_(
            reference.parameters(),
            1.0,
        )

        torch.nn.utils.clip_grad_norm_(
            scan.parameters(),
            1.0,
        )

        opt_ref.step()
        opt_scan.step()

        torch.cuda.synchronize()

        param_max, param_mean = max_param_diff(reference, scan)
        loss_diff = abs(loss_ref.item() - loss_scan.item())

        if step in checkpoints:
            print(
                f"step={step:3d} "
                f"loss_ref={loss_ref.item():.9f} "
                f"loss_scan={loss_scan.item():.9f} "
                f"loss_diff={loss_diff:.3e} "
                f"grad_diff={grad_diff:.3e} "
                f"param_max={param_max:.3e} "
                f"param_mean={param_mean:.3e}"
            )

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    param_max, param_mean = max_param_diff(reference, scan)

    print(f"Final max parameter difference  : {param_max:.12e}")
    print(f"Final mean parameter difference : {param_mean:.12e}")

    print()
    if param_max < 1e-5:
        print("STATUS: NUMERICALLY VERY CLOSE")
    elif param_max < 1e-4:
        print("STATUS: SMALL PARAMETER DRIFT")
    else:
        print("STATUS: SIGNIFICANT PARAMETER DRIFT")

    print("=" * 70)


if __name__ == "__main__":
    main()
