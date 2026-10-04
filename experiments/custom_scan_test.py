import torch
from torch.autograd import Function

from torch._higher_order_ops.associative_scan import associative_scan


def combine(left, right):
    """
    Composition of:

        s_t = f_t * s_{t-1} + b_t

    Each element represents:

        s_out = a * s_in + b

    Composition:

        (a2, b2) o (a1, b1)
        = (a2*a1, a2*b1 + b2)
    """
    a1, b1 = left
    a2, b2 = right

    return (
        a2 * a1,
        a2 * b1 + b2,
    )


def parallel_forward(f, b):
    """
    Parallel prefix recurrence.

    f: [B,T,D]
    b: [B,T,D]

    returns:
        states: [B,T,D]
    """
    _, b_scan = associative_scan(
        combine,
        (f, b),
        dim=1,
    )

    return b_scan


class ExactScan(Function):

    @staticmethod
    def forward(ctx, f, b, initial_state):
        """
        Forward recurrence:

            s_t = f_t * s_{t-1} + b_t
        """

        states = parallel_forward(f, b)

        if initial_state is not None:
            # Prefix products of f.
            f_scan, _ = associative_scan(
                combine,
                (f, torch.zeros_like(b)),
                dim=1,
            )

            states = states + f_scan * initial_state.unsqueeze(1)

        ctx.save_for_backward(f, b, states)

        return states

    @staticmethod
    def backward(ctx, grad_states):
        """
        Reverse-mode differentiation of:

            s_t = f_t * s_{t-1} + b_t

        For each t:

            db_t += ds_t
            df_t += ds_t * s_{t-1}
            ds_{t-1} += ds_t * f_t
        """

        f, b, states = ctx.saved_tensors

        B, T, D = f.shape

        grad_f = torch.zeros_like(f)
        grad_b = torch.zeros_like(b)
        grad_initial = torch.zeros(
            B,
            D,
            device=f.device,
            dtype=f.dtype,
        )

        upstream = torch.zeros_like(grad_states)

        # Gradient flowing into s_t.
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
                previous_state = torch.zeros_like(states[:, 0])
            else:
                previous_state = states[:, t - 1]

            grad_f[:, t] = ds * previous_state

            carry = ds * f[:, t]

        grad_initial = carry

        return grad_f, grad_b, grad_initial


def reference_recurrence(f, b, initial_state):
    state = initial_state
    states = []

    for t in range(f.shape[1]):
        state = f[:, t] * state + b[:, t]
        states.append(state)

    return torch.stack(states, dim=1)


def main():

    torch.manual_seed(1234)

    device = "cuda"

    B = 2
    T = 32
    D = 64

    f_ref = torch.sigmoid(
        torch.randn(B, T, D, device=device)
    )

    b_ref = torch.randn(
        B, T, D,
        device=device,
    )

    s0_ref = torch.randn(
        B, D,
        device=device,
    )

    # ------------------------------------------------------------
    # Forward comparison
    # ------------------------------------------------------------

    f1 = f_ref.detach().clone().requires_grad_(True)
    b1 = b_ref.detach().clone().requires_grad_(True)
    s1 = s0_ref.detach().clone().requires_grad_(True)

    y_ref = reference_recurrence(
        f1,
        b1,
        s1,
    )

    loss_ref = (
        y_ref.pow(2).mean()
        + y_ref[:, -1].mean()
    )

    loss_ref.backward()

    grads_ref = (
        f1.grad.detach().clone(),
        b1.grad.detach().clone(),
        s1.grad.detach().clone(),
    )

    # ------------------------------------------------------------
    # Custom scan
    # ------------------------------------------------------------

    f2 = f_ref.detach().clone().requires_grad_(True)
    b2 = b_ref.detach().clone().requires_grad_(True)
    s2 = s0_ref.detach().clone().requires_grad_(True)

    y_fast = ExactScan.apply(
        f2,
        b2,
        s2,
    )

    loss_fast = (
        y_fast.pow(2).mean()
        + y_fast[:, -1].mean()
    )

    loss_fast.backward()

    grads_fast = (
        f2.grad.detach().clone(),
        b2.grad.detach().clone(),
        s2.grad.detach().clone(),
    )

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    print("=" * 70)
    print("CUSTOM AUTOGRAD SCAN — REFERENCE COMPARISON")
    print("=" * 70)

    print()
    print("FORWARD")
    print(
        "max difference :",
        (y_ref - y_fast).abs().max().item(),
    )
    print(
        "mean difference:",
        (y_ref - y_fast).abs().mean().item(),
    )

    print()
    print("LOSS")
    print("reference:", loss_ref.item())
    print("custom   :", loss_fast.item())
    print(
        "difference:",
        abs(loss_ref.item() - loss_fast.item()),
    )

    names = [
        "f gradient",
        "b gradient",
        "initial-state gradient",
    ]

    print()
    print("GRADIENTS")

    max_difference = 0.0

    for name, a, b in zip(names, grads_ref, grads_fast):

        diff = (a - b).abs()

        current_max = diff.max().item()
        current_mean = diff.mean().item()

        max_difference = max(
            max_difference,
            current_max,
        )

        print(
            f"{name:28s} "
            f"max={current_max:.10f} "
            f"mean={current_mean:.10f}"
        )

    print()
    print("FINITE")
    print(
        "forward:",
        torch.isfinite(y_fast).all().item(),
    )

    print(
        "gradients:",
        all(
            torch.isfinite(g).all().item()
            for g in grads_fast
        ),
    )

    print()
    print("=" * 70)

    if max_difference < 1e-5:
        print("PASS: CUSTOM AUTOGRAD GRADIENT JE NUMERICKY SPRÁVNY.")
    else:
        print("FAIL: CUSTOM AUTOGRAD GRADIENT SA LÍŠI PRÍLIŠ VEĽA.")

    print("=" * 70)


if __name__ == "__main__":
    main()
