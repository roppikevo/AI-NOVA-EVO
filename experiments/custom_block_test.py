import torch
import torch.nn as nn
from torch.autograd import Function
from torch._higher_order_ops.associative_scan import associative_scan

from nova.blocks_reference import NovaBlock as ReferenceNovaBlock


def combine(left, right):
    a1, b1 = left
    a2, b2 = right
    return a2 * a1, a2 * b1 + b2


class ExactScan(Function):

    @staticmethod
    def forward(ctx, f, b, initial_state):

        f_scan, b_scan = associative_scan(
            combine,
            (f, b),
            dim=1,
        )

        if initial_state is None:
            states = b_scan
        else:
            states = (
                f_scan * initial_state.unsqueeze(1)
                + b_scan
            )

        ctx.save_for_backward(
            f,
            b,
            states,
        )

        ctx.has_initial_state = initial_state is not None

        if initial_state is not None:
            ctx.save_for_backward(
                f,
                b,
                states,
                initial_state,
            )

        return states

    @staticmethod
    def backward(ctx, grad_states):

        saved = ctx.saved_tensors

        if ctx.has_initial_state:
            f, b, states, initial_state = saved
        else:
            f, b, states = saved
            initial_state = None

        B, T, D = f.shape

        grad_f = torch.zeros_like(f)
        grad_b = torch.zeros_like(b)

        carry = torch.zeros_like(states[:, 0])

        for t in range(T - 1, -1, -1):

            ds = grad_states[:, t] + carry

            grad_b[:, t] = ds

            if t == 0:
                if initial_state is None:
                    previous_state = torch.zeros_like(ds)
                else:
                    previous_state = initial_state
            else:
                previous_state = states[:, t - 1]

            grad_f[:, t] = ds * previous_state

            carry = ds * f[:, t]

        grad_initial = carry if initial_state is not None else None

        return grad_f, grad_b, grad_initial


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
            raise ValueError(
                "Custom test currently requires d_model == d_state"
            )

        self.d_model = d_model
        self.d_state = d_state
        self.conv_kernel = conv_kernel

        self.norm = nn.RMSNorm(
            d_model,
            eps=1e-6,
        )

        self.local_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=conv_kernel,
            groups=d_model,
            bias=True,
            padding=0,
        )

        self.forget_proj = nn.Linear(
            d_model,
            d_state,
        )

        self.input_proj = nn.Linear(
            d_model,
            d_state,
        )

        self.fusion_proj = nn.Linear(
            d_model,
            d_state,
        )

        self.output_proj = nn.Linear(
            d_state,
            d_model,
        )

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

        states = ExactScan.apply(
            f,
            i * u,
            state,
        )

        fused = (
            g * states
            + (1.0 - g) * c
        )

        out = self.output_proj(fused)

        y = x + out

        return y, states[:, -1]


def compare(state_mode):

    torch.manual_seed(20260925)

    device = "cuda"

    B = 2
    T = 64
    D = 256

    reference = ReferenceNovaBlock(
        d_model=D,
        d_state=D,
        conv_kernel=5,
        forget_bias=1.5,
    ).to(device)

    custom = CustomNovaBlock(
        d_model=D,
        d_state=D,
        conv_kernel=5,
        forget_bias=1.5,
    ).to(device)

    custom.load_state_dict(
        reference.state_dict()
    )

    x1 = torch.randn(
        B,
        T,
        D,
        device=device,
        requires_grad=True,
    )

    x2 = x1.detach().clone().requires_grad_(True)

    if state_mode:

        s0 = torch.randn(
            B,
            D,
            device=device,
        )

        s1 = s0.detach().clone().requires_grad_(True)
        s2 = s0.detach().clone().requires_grad_(True)

    else:
        s1 = None
        s2 = None

    y_ref, final_ref = reference(
        x1,
        state=s1,
    )

    y_custom, final_custom = custom(
        x2,
        state=s2,
    )

    loss_ref = (
        y_ref.pow(2).mean()
        + final_ref.pow(2).mean()
    )

    loss_custom = (
        y_custom.pow(2).mean()
        + final_custom.pow(2).mean()
    )

    loss_ref.backward()
    loss_custom.backward()

    print()
    print("=" * 70)
    print(
        "CUSTOM NOVA BLOCK TEST — "
        + ("INITIAL STATE" if state_mode else "STATE NONE")
    )
    print("=" * 70)

    print()
    print("FORWARD")
    print(
        "Y max diff     :",
        (y_ref - y_custom).abs().max().item(),
    )
    print(
        "Y mean diff    :",
        (y_ref - y_custom).abs().mean().item(),
    )

    print(
        "STATE max diff :",
        (final_ref - final_custom).abs().max().item(),
    )

    print()
    print("LOSS")
    print("reference:", loss_ref.item())
    print("custom   :", loss_custom.item())
    print(
        "difference:",
        abs(loss_ref.item() - loss_custom.item()),
    )

    print()
    print("PARAMETER GRADIENTS")

    max_param_diff = 0.0

    for (name1, p1), (name2, p2) in zip(
        reference.named_parameters(),
        custom.named_parameters(),
    ):

        diff = (
            p1.grad - p2.grad
        ).abs()

        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        max_param_diff = max(
            max_param_diff,
            max_diff,
        )

        print(
            f"{name1:28s} "
            f"max={max_diff:.10f} "
            f"mean={mean_diff:.10f}"
        )

    print()
    print("INPUT GRADIENT")

    input_diff = (
        x1.grad - x2.grad
    ).abs()

    print(
        "max :",
        input_diff.max().item(),
    )

    print(
        "mean:",
        input_diff.mean().item(),
    )

    print()
    print("FINITE")

    print(
        "custom output:",
        torch.isfinite(y_custom).all().item(),
    )

    print(
        "custom gradients:",
        all(
            p.grad is None
            or torch.isfinite(p.grad).all().item()
            for p in custom.parameters()
        ),
    )

    print()
    print(
        "MAX PARAMETER GRADIENT DIFFERENCE:",
        max_param_diff,
    )


def main():

    compare(False)
    compare(True)


if __name__ == "__main__":
    main()
