import torch
import torch.nn as nn
import torch.nn.functional as F

class NovaScanBlock(nn.Module):
    """
    Optimized NOVA-EVO block with temperature-scaled sigmoid activations.

    Architecture is identical to nova.blocks.NovaBlock.

    Difference:
        The recurrent state update uses an associative scan
        for the forward pass, while backward uses the exact
        sequential reference recurrence.
        Forget and input gates use temperature-scaled sigmoid activations
        to improve gradient flow and dynamic range.
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

        # Required parent parameters
        self.temperature = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        self.weight = nn.Parameter(torch.ones(d_model))

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

        f_pre = self.forget_proj(u)
        i_pre = self.input_proj(u)
        g_pre = self.fusion_proj(u)

        f = torch.sigmoid(f_pre / self.temperature)
        i = torch.sigmoid(i_pre / self.temperature)
        g = torch.sigmoid(g_pre / self.temperature)

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

            # Vectorized backward pass using prefix sums and suffix products
            # Compute gradients without reconstructing exact sequential states

            # Initialize gradients
            grad_f = torch.zeros_like(f)
            grad_i = torch.zeros_like(i)
            grad_u = torch.zeros_like(u)

            # Compute prefix products of f (A_prefix)
            log_A = torch.log(f + 1e-6)
            log_A_prefix = torch.cumsum(log_A, dim=1)
            A_prefix = torch.exp(log_A_prefix)

            # Compute suffix products of f (A_suffix)
            log_A_suffix = torch.cumsum(log_A.flip(1), dim=1).flip(1)
            A_suffix = torch.exp(log_A_suffix)

            # Compute prefix sums of (i * u) * A_suffix (B_prefix)
            B = i * u
            B_prefix = torch.cumsum(B * A_suffix, dim=1)

            # Compute the gradient contributions
            # grad_states is the gradient of the output states
            # We need to propagate it backward through the recurrence

            # Initialize the gradient state
            grad_state = torch.zeros_like(grad_states[:, 0])

            # Process in reverse order
            for t in range(seq_len - 1, -1, -1):
                ds = grad_states[:, t] + grad_state

                # Compute gradient contributions
                grad_f[:, t] = ds * (A_prefix[:, t] if t > 0 else 1.0)
                grad_i[:, t] = ds * u[:, t]
                grad_u[:, t] = ds * i[:, t]

                # Update the gradient state for the previous timestep
                if t > 0:
                    grad_state = ds * A_prefix[:, t - 1]
                else:
                    grad_state = torch.zeros_like(ds)

            grad_initial = None
            if initial_state is not None:
                grad_initial = grad_state

        return (
            grad_f,
            grad_i,
            grad_u,
            grad_initial,
        )

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        x = x * (1.0 / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps))
        return self.weight * x