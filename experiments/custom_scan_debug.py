import torch
from torch.autograd import Function
from torch._higher_order_ops.associative_scan import associative_scan


def combine(left, right):
    a1, b1 = left
    a2, b2 = right
    return a2 * a1, a2 * b1 + b2


def reference(f, b, s0):
    s = s0
    states = []

    for t in range(f.shape[1]):
        s = f[:, t] * s + b[:, t]
        states.append(s)

    return torch.stack(states, dim=1)


class DebugScan(Function):

    @staticmethod
    def forward(ctx, f, b, s0):
        _, b_scan = associative_scan(
            combine,
            (f, b),
            dim=1,
        )

        f_scan, _ = associative_scan(
            combine,
            (f, torch.zeros_like(b)),
            dim=1,
        )

        states = f_scan * s0.unsqueeze(1) + b_scan

        ctx.save_for_backward(f, b, states, s0)

        return states

    @staticmethod
    def backward(ctx, grad_states):

        f, b, states, s0 = ctx.saved_tensors

        B, T, D = f.shape

        grad_f = torch.zeros_like(f)
        grad_b = torch.zeros_like(b)

        carry = torch.zeros(
            B,
            D,
            device=f.device,
            dtype=f.dtype,
        )

        for t in range(T - 1, -1, -1):

            ds = grad_states[:, t] + carry

            grad_b[:, t] = ds

            if t == 0:
                previous_state = s0
            else:
                previous_state = states[:, t - 1]

            grad_f[:, t] = ds * previous_state

            carry = ds * f[:, t]

        grad_s0 = carry

        return grad_f, grad_b, grad_s0


def main():

    torch.manual_seed(1234)

    device = "cuda"

    B = 1
    T = 8
    D = 4

    f0 = torch.sigmoid(
        torch.randn(B, T, D, device=device)
    )

    b0 = torch.randn(
        B, T, D,
        device=device,
    )

    s00 = torch.randn(
        B, D,
        device=device,
    )

    # ------------------------------------------------------------
    # Reference
    # ------------------------------------------------------------

    f1 = f0.detach().clone().requires_grad_(True)
    b1 = b0.detach().clone().requires_grad_(True)
    s1 = s00.detach().clone().requires_grad_(True)

    y1 = reference(f1, b1, s1)

    # Intentional asymmetric loss so every timestep contributes
    weights = torch.arange(
        1,
        T + 1,
        device=device,
        dtype=torch.float32,
    ).view(1, T, 1)

    loss1 = (y1 * weights).sum()
    loss1.backward()

    # ------------------------------------------------------------
    # Custom
    # ------------------------------------------------------------

    f2 = f0.detach().clone().requires_grad_(True)
    b2 = b0.detach().clone().requires_grad_(True)
    s2 = s00.detach().clone().requires_grad_(True)

    y2 = DebugScan.apply(f2, b2, s2)

    loss2 = (y2 * weights).sum()
    loss2.backward()

    # ------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------

    df = (f1.grad - f2.grad).abs()

    print("=" * 70)
    print("CUSTOM SCAN — PER-TIMESTEP GRADIENT DEBUG")
    print("=" * 70)

    print()
    print("FORWARD MAX DIFF:")
    print((y1 - y2).abs().max().item())

    print()
    print("LOSS:")
    print("reference:", loss1.item())
    print("custom   :", loss2.item())

    print()
    print("GRADIENT f PER TIMESTEP")
    print("-" * 70)

    for t in range(T):

        ref = f1.grad[0, t]
        fast = f2.grad[0, t]
        diff = (ref - fast).abs()

        print(
            f"t={t}: "
            f"ref={ref.detach().cpu().numpy()} "
            f"custom={fast.detach().cpu().numpy()} "
            f"max_diff={diff.max().item():.10f}"
        )

    print()
    print("OTHER GRADIENTS")
    print(
        "b max diff :",
        (b1.grad - b2.grad).abs().max().item(),
    )
    print(
        "s0 max diff:",
        (s1.grad - s2.grad).abs().max().item(),
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
