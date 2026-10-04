import copy
import time
import torch
import torch.nn as nn

from nova.config import CONFIG
from nova.data import build_datasets
from nova.model import NovaModel
from nova.blocks_reference import NovaBlock as ReferenceNovaBlock


torch.manual_seed(20260925)
torch.cuda.manual_seed_all(20260925)

DEVICE = "cuda"
STEPS = 100


# ============================================================
# CUSTOM AUTOGRAD ASSOCIATIVE SCAN
# ============================================================

from torch._higher_order_ops.associative_scan import associative_scan


class ExactScan(torch.autograd.Function):

    @staticmethod
    def forward(ctx, f, b, initial_state=None):

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

        states = b_scan

        if initial_state is not None:
            states = states + f_scan * initial_state

        if initial_state is None:
            ctx.save_for_backward(f, b, states)
        else:
            ctx.save_for_backward(f, b, states, initial_state)

        ctx.has_initial = initial_state is not None

        return states

    @staticmethod
    def backward(ctx, grad_states):

        saved = ctx.saved_tensors

        if ctx.has_initial:
            f, b, states, initial_state = saved
        else:
            f, b, states = saved
            initial_state = None

        B, T, D = f.shape

        grad_f = torch.zeros_like(f)
        grad_b = torch.zeros_like(b)

        grad_state = grad_states[:, -1]

        for t in range(T - 1, -1, -1):

            grad_b[:, t] += grad_state

            if t == 0:
                prev_state = (
                    initial_state
                    if initial_state is not None
                    else torch.zeros_like(grad_state)
                )
            else:
                prev_state = states[:, t - 1]

            grad_f[:, t] += grad_state * prev_state

            grad_state = grad_state * f[:, t]

        grad_initial = grad_state if ctx.has_initial else None

        return grad_f, grad_b, grad_initial


# ============================================================
# CUSTOM NOVA BLOCK
# ============================================================

class CustomNovaBlock(nn.Module):

    def __init__(
        self,
        d_model,
        d_state,
        conv_kernel=5,
        forget_bias=1.5,
    ):
        super().__init__()

        if d_model != d_state:
            raise ValueError("Custom block requires d_model == d_state")

        self.d_model = d_model
        self.d_state = d_state
        self.conv_kernel = conv_kernel

        # Use the EXACT reference normalization implementation.
        from nova.blocks_reference import RMSNorm

        self.norm = RMSNorm(d_model)

        self.local_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=conv_kernel,
            groups=d_model,
            bias=True,
            padding=0,
        )

        self.forget_proj = nn.Linear(d_model, d_state)
        self.input_proj = nn.Linear(d_model, d_state)
        self.fusion_proj = nn.Linear(d_model, d_state)
        self.output_proj = nn.Linear(d_state, d_model)

        nn.init.constant_(
            self.forget_proj.bias,
            forget_bias,
        )

    def causal_conv(self, x):

        pad = self.conv_kernel - 1

        x_conv = x.transpose(1, 2)

        x_conv = nn.functional.pad(
            x_conv,
            (pad, 0),
        )

        y = self.local_conv(x_conv)

        return y.transpose(1, 2)

    def forward(self, x, state=None):

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

        if state is None:

            b = i * u

            state_seq = ExactScan.apply(
                f,
                b,
                None,
            )

        else:

            b = i * u

            state_seq = ExactScan.apply(
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


# ============================================================
# CUSTOM MODEL
# ============================================================

class CustomNovaModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )

        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.02,
        )

        with torch.no_grad():
            self.embedding.weight[
                config.pad_token_id
            ].zero_()

        self.blocks = nn.ModuleList([
            CustomNovaBlock(
                d_model=config.d_model,
                d_state=config.d_state,
                conv_kernel=config.conv_kernel,
                forget_bias=config.forget_bias,
            )
            for _ in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(
            config.d_model
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):

        x = self.embedding(input_ids)

        states = []

        for layer in self.blocks:

            x, state = layer(
                x,
                state=None,
            )

            states.append(state)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits, states

    def num_parameters(self):

        return sum(
            p.numel()
            for p in self.parameters()
        )


# ============================================================
# COPY REFERENCE WEIGHTS
# ============================================================

def make_models():

    torch.manual_seed(20260925)

    reference = NovaModel(CONFIG).to(DEVICE)

    custom = CustomNovaModel(CONFIG).to(DEVICE)

    custom.load_state_dict(
        reference.state_dict()
    )

    return reference, custom


# ============================================================
# LOSS
# ============================================================

def loss_fn(logits, targets):

    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


# ============================================================
# BENCHMARK
# ============================================================

def benchmark(model, train_dataset, name):

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01,
    )

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    start = time.perf_counter()

    losses = []

    for step in range(1, STEPS + 1):

        input_ids, targets = train_dataset[
            (step - 1) % len(train_dataset)
        ]

        input_ids = input_ids.unsqueeze(0).to(DEVICE)
        targets = targets.unsqueeze(0).to(DEVICE)

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

        losses.append(loss.item())

        if step == 1 or step % 25 == 0:

            print(
                f"step={step:3d} "
                f"loss={loss.item():.6f}"
            )

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print()
    print(
        f"{name} elapsed: "
        f"{elapsed:.3f} s"
    )

    print(
        f"{name} steps/sec: "
        f"{STEPS / elapsed:.3f}"
    )

    print(
        f"{name} final loss: "
        f"{losses[-1]:.6f}"
    )

    return elapsed, losses


# ============================================================
# FULL MODEL NUMERICAL COMPARISON
# ============================================================

def compare_models():

    print()
    print("=" * 70)
    print("FULL MODEL NUMERICAL COMPARISON")
    print("=" * 70)

    reference, custom = make_models()

    torch.manual_seed(12345)

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

    reference.train()
    custom.train()

    reference.zero_grad(set_to_none=True)
    custom.zero_grad(set_to_none=True)

    ref_logits, ref_states = reference(
        input_ids
    )

    custom_logits, custom_states = custom(
        input_ids
    )

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

    print(
        "logits max diff :",
        (
            ref_logits.detach()
            - custom_logits.detach()
        ).abs().max().item()
    )

    print(
        "loss reference  :",
        ref_loss.item()
    )

    print(
        "loss custom     :",
        custom_loss.item()
    )

    print(
        "loss difference :",
        abs(
            ref_loss.item()
            - custom_loss.item()
        )
    )

    print()

    max_grad_diff = 0.0

    for (n1, p1), (n2, p2) in zip(
        reference.named_parameters(),
        custom.named_parameters(),
    ):

        if p1.grad is None or p2.grad is None:
            continue

        diff = (
            p1.grad
            - p2.grad
        ).abs().max().item()

        max_grad_diff = max(
            max_grad_diff,
            diff,
        )

    print(
        "MAX FULL MODEL GRADIENT DIFF:",
        max_grad_diff,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("NOVA CUSTOM SCAN — FULL MODEL BENCHMARK")
    print("=" * 70)

    print(
        "Parameters:",
        NovaModel(CONFIG).num_parameters(),
    )

    train_dataset, _, _ = build_datasets(
        seq_len=128
    )

    reference, custom = make_models()

    print(
        "Reference parameters:",
        reference.num_parameters(),
    )

    print(
        "Custom parameters:",
        custom.num_parameters(),
    )

    compare_models()

    torch.cuda.empty_cache()

    ref_model, custom_model = make_models()

    ref_time, _ = benchmark(
        ref_model,
        train_dataset,
        "REFERENCE NOVA",
    )

    del ref_model

    torch.cuda.empty_cache()

    custom_time, _ = benchmark(
        custom_model,
        train_dataset,
        "CUSTOM-SCAN NOVA",
    )

    del custom_model

    print()
    print("=" * 70)
    print("FINAL RESULT")
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
