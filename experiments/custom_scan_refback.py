import torch
import torch.nn as nn

from nova.config import CONFIG
from nova.model import NovaModel
from nova.blocks_reference import NovaBlock
from custom_model_benchmark import CustomNovaModel


DEVICE = "cuda"


class ScanRefBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, f, b, initial_state=None):

        # Forward: parallel associative scan
        from torch._higher_order_ops.associative_scan import associative_scan

        def combine(left, right):
            a1, b1 = left
            a2, b2 = right
            return (
                a2 * a1,
                a2 * b1 + b2,
            )

        f_scan, b_scan = associative_scan(
            combine,
            (f, b),
            dim=1,
        )

        if initial_state is None:
            states = b_scan
        else:
            states = b_scan + f_scan * initial_state

        if initial_state is None:
            ctx.save_for_backward(f, b)
        else:
            ctx.save_for_backward(f, b, initial_state)

        ctx.has_initial = initial_state is not None

        return states

    @staticmethod
    def backward(ctx, grad_states):

        saved = ctx.saved_tensors

        if ctx.has_initial:
            f, b, initial_state = saved
        else:
            f, b = saved
            initial_state = None

        B, T, D = f.shape

        grad_f = torch.zeros_like(f)
        grad_b = torch.zeros_like(b)

        # Reconstruct the exact sequential forward states.
        states = torch.empty_like(f)

        if initial_state is None:
            state = torch.zeros(
                B,
                D,
                device=f.device,
                dtype=f.dtype,
            )
        else:
            state = initial_state

        with torch.no_grad():
            for t in range(T):
                state = (
                    f[:, t] * state
                    + b[:, t]
                )
                states[:, t] = state

        # Exact sequential reverse-mode recurrence.
        grad_state = torch.zeros_like(
            states[:, 0]
        )

        for t in range(T - 1, -1, -1):

            total_grad = (
                grad_states[:, t]
                + grad_state
            )

            if t == 0:
                prev_state = (
                    initial_state
                    if initial_state is not None
                    else torch.zeros_like(
                        total_grad
                    )
                )
            else:
                prev_state = states[:, t - 1]

            grad_f[:, t] = (
                total_grad * prev_state
            )

            grad_b[:, t] = total_grad

            grad_state = (
                total_grad * f[:, t]
            )

        grad_initial = (
            grad_state
            if ctx.has_initial
            else None
        )

        return (
            grad_f,
            grad_b,
            grad_initial,
        )


class CustomRefBackwardBlock(NovaBlock):

    def forward(self, x, state=None):

        if x.ndim != 3:
            raise ValueError(
                f"Expected x with shape [batch, seq, dim], got {tuple(x.shape)}"
            )

        batch, _, dim = x.shape

        if dim != self.d_model:
            raise ValueError(
                f"Expected feature dimension {self.d_model}, got {dim}"
            )

        u = self.norm(x)

        c = self.causal_conv(u)

        f = torch.sigmoid(
            self.forget_proj(u)
        )

        i = torch.sigmoid(
            self.input_proj(u)
        )

        g = torch.sigmoid(
            self.fusion_proj(u)
        )

        b = i * u

        state_seq = ScanRefBackward.apply(
            f,
            b,
            state,
        )

        fused = (
            g * state_seq
            + (1.0 - g) * c
        )

        out = self.output_proj(fused)

        y = x + out

        final_state = state_seq[:, -1]

        return y, final_state


class CustomRefBackwardModel(NovaModel):

    def __init__(self, config):
        super().__init__(config)

        self.blocks = nn.ModuleList([
            CustomRefBackwardBlock(
                d_model=config.d_model,
                d_state=config.d_state,
                conv_kernel=config.conv_kernel,
                forget_bias=config.forget_bias,
            )
            for _ in range(config.num_layers)
        ])

        # Copy exact initialization/weights later.
        self.lm_head.weight = self.embedding.weight


def loss_fn(logits, targets):
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


def compare_gradients():

    print("=" * 70)
    print("SCAN FORWARD + REFERENCE BACKWARD")
    print("=" * 70)

    torch.manual_seed(20260925)
    torch.cuda.manual_seed_all(20260925)

    reference = NovaModel(CONFIG).to(DEVICE)
    custom = CustomRefBackwardModel(CONFIG).to(DEVICE)

    custom.load_state_dict(
        reference.state_dict()
    )

    input_ids = torch.randint(
        0,
        CONFIG.vocab_size,
        (2, 64),
        device=DEVICE,
    )

    targets = torch.randint(
        0,
        CONFIG.vocab_size,
        (2, 64),
        device=DEVICE,
    )

    reference.zero_grad(set_to_none=True)
    custom.zero_grad(set_to_none=True)

    ref_logits, _ = reference(input_ids)
    custom_logits, _ = custom(input_ids)

    ref_loss = loss_fn(
        ref_logits,
        targets,
    )

    custom_loss = loss_fn(
        custom_logits,
        targets,
    )

    ref_loss.backward()
    custom_loss.backward()

    print()
    print("FORWARD")
    print(
        "logits max diff:",
        (
            ref_logits.detach()
            - custom_logits.detach()
        ).abs().max().item()
    )

    print()
    print("LOSS")
    print("reference:", ref_loss.item())
    print("custom   :", custom_loss.item())
    print(
        "difference:",
        abs(
            ref_loss.item()
            - custom_loss.item()
        )
    )

    print()
    print("GRADIENTS")

    max_diff = 0.0

    for (
        (name_r, param_r),
        (name_c, param_c),
    ) in zip(
        reference.named_parameters(),
        custom.named_parameters(),
    ):

        if param_r.grad is None:
            continue

        diff = (
            param_r.grad
            - param_c.grad
        ).abs()

        current = diff.max().item()
        max_diff = max(max_diff, current)

        print(
            f"{name_r:35s} "
            f"max={current:.10e} "
            f"mean={diff.mean().item():.10e}"
        )

    print()
    print(
        "MAX GRADIENT DIFFERENCE:",
        max_diff,
    )


def benchmark(model, name, train_dataset):

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01,
    )

    steps = 100

    torch.cuda.synchronize()

    start = torch.cuda.Event(
        enable_timing=True
    )
    end = torch.cuda.Event(
        enable_timing=True
    )

    start.record()

    final_loss = None

    for step in range(steps):

        input_ids, targets = train_dataset[
            step % len(train_dataset)
        ]

        input_ids = input_ids.unsqueeze(0).to(
            DEVICE
        )
        targets = targets.unsqueeze(0).to(
            DEVICE
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits, _ = model(input_ids)

        loss = loss_fn(
            logits,
            targets,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        final_loss = loss.item()

    end.record()

    torch.cuda.synchronize()

    elapsed = start.elapsed_time(end) / 1000.0

    print()
    print(name)
    print(
        f"elapsed    : {elapsed:.3f} s"
    )
    print(
        f"steps/sec  : {steps / elapsed:.3f}"
    )
    print(
        f"final loss : {final_loss:.6f}"
    )

    return elapsed


def main():

    from nova.data import build_datasets

    train_dataset, _, _ = build_datasets(
        seq_len=128
    )

    compare_gradients()

    print()
    print("=" * 70)
    print("SPEED BENCHMARK")
    print("=" * 70)

    torch.manual_seed(20260925)

    reference = NovaModel(CONFIG).to(DEVICE)

    custom = CustomRefBackwardModel(
        CONFIG
    ).to(DEVICE)

    custom.load_state_dict(
        reference.state_dict()
    )

    ref_time = benchmark(
        reference,
        "REFERENCE NOVA",
        train_dataset,
    )

    del reference
    torch.cuda.empty_cache()

    custom_time = benchmark(
        custom,
        "SCAN + REFERENCE BACKWARD",
        train_dataset,
    )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"Reference : {ref_time:.3f} s"
    )

    print(
        f"Custom    : {custom_time:.3f} s"
    )

    print(
        f"Speedup   : {ref_time / custom_time:.3f}x"
    )


if __name__ == "__main__":
    main()
