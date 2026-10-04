import torch
import torch.nn as nn

from nova.blocks import RMSNorm


class ScanRefBackward(torch.autograd.Function):
    """
    Associative-scan forward + exact sequential reference backward.

    Forward:
        s_t = f_t * s_{t-1} + i_t * u_t

    Backward:
        Exact reverse-time recurrence matching the reference
        timestep implementation.

    The scan changes only floating-point operation ordering in
    the forward pass. The backward reconstructs the exact
    sequential states to preserve gradient fidelity.
    """

    @staticmethod
    def forward(ctx, f, i, u, initial_state):
        # Each timestep represents:
        #
        #   s_t = A_t * s_{t-1} + B_t
        #
        # with:
        #
        #   A_t = f_t
        #   B_t = i_t * u_t

        b = i * u

        def combine(left, right):
            a1, b1 = left
            a2, b2 = right

            return (
                a2 * a1,
                a2 * b1 + b2,
            )

        # Import here because associative_scan is an internal
        # PyTorch operator and should remain isolated to this backend.
        from torch._higher_order_ops.associative_scan import associative_scan

        a = f
        identity = torch.zeros_like(b)

        if initial_state is not None:
            identity = initial_state.unsqueeze(1)

        # Prefix composition.
        A, B = associative_scan(
            combine,
            (a, b),
            dim=1,
            reverse=False,
            combine_mode="pointwise",
        )

        states = A * identity + B

        ctx.save_for_backward(f, i, u, initial_state)

        return states

    @staticmethod
    def backward(ctx, grad_states):
        f, i, u, initial_state = ctx.saved_tensors

        with torch.no_grad():
            batch, seq_len, state_dim = u.shape

            # Reconstruct the exact sequential states used by
            # the reference implementation.
            states = torch.empty_like(u)

            if initial_state is None:
                prev = torch.zeros(
                    batch,
                    state_dim,
                    device=u.device,
                    dtype=u.dtype,
                )
            else:
                prev = initial_state

            for t in range(seq_len):
                prev = (
                    f[:, t] * prev
                    + i[:, t] * u[:, t]
                )
                states[:, t] = prev

        grad_f = torch.zeros_like(f)
        grad_i = torch.zeros_like(i)
        grad_u = torch.zeros_like(u)

        grad_state = torch.zeros_like(
            states[:, 0]
        )

        for t in range(seq_len - 1, -1, -1):
            ds = grad_states[:, t] + grad_state

            prev_state = (
                torch.zeros_like(ds)
                if t == 0
                else states[:, t - 1]
            )

            grad_f[:, t] = ds * prev_state
            grad_i[:, t] = ds * u[:, t]

            grad_u[:, t] = (
                ds * i[:, t]
            )

            grad_state = ds * f[:, t]

        grad_initial = None

        if initial_state is not None:
            grad_initial = grad_state

        return (
            grad_f,
            grad_i,
            grad_u,
            grad_initial,
        )


class NovaScanBlock(nn.Module):
    """
    Optimized NOVA-0 block.

    Architecture is identical to nova.blocks.NovaBlock.

    Difference:
        The recurrent state update uses an associative scan
        for the forward pass, while backward uses the exact
        sequential reference recurrence.
    """

    def __init__(
        self,
        d_model,
        d_state,
        conv_kernel=5,
        forget_bias=1.5,
    ):
        super().__init__()


        if conv_kernel < 1 or conv_kernel % 2 == 0:
            raise ValueError(
                "conv_kernel must be a positive odd number"
            )

        self.d_model = d_model
        self.d_state = d_state
        self.conv_kernel = conv_kernel

        self.norm = RMSNorm(d_model)

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
        if x.ndim != 3:
            raise ValueError(
                "x must have shape [batch, seq, dim]"
            )

        batch, _, dim = x.shape

        if dim != self.d_model:
            raise ValueError(
                f"Expected dim={self.d_model}, got {dim}"
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

        states = ScanRefBackward.apply(
            f,
            i,
            u,
            state,
        )

        fused = (
            g * states
            + (1.0 - g) * c
        )

        out = self.output_proj(fused)

        y = x + out

        final_state = states[:, -1]

        return y, final_state
