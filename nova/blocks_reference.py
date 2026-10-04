import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class NovaBlock(nn.Module):
    """
    NOVA-0 PRETEST block.

    Flow:

        x
        │
        ├── RMSNorm
        │
        ├── causal depthwise convolution ── c_t
        │
        ├── forget gate ─────────────────── f_t
        ├── input gate ──────────────────── i_t
        │
        └── recurrent state ─────────────── s_t
                         │
                         └── fusion gate ── g_t
                                  │
                         g*s + (1-g)*c
                                  │
                              output
                                  │
                              residual
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        conv_kernel: int = 5,
        forget_bias: float = 1.5,
    ):
        super().__init__()

        if d_model != d_state:
            raise ValueError(
                "NOVA-0 PRETEST currently requires d_model == d_state"
            )

        if conv_kernel < 1 or conv_kernel % 2 == 0:
            raise ValueError("conv_kernel must be a positive odd number")

        self.d_model = d_model
        self.d_state = d_state
        self.conv_kernel = conv_kernel

        self.norm = RMSNorm(d_model)

        # Local interaction branch.
        # Padding is handled manually to guarantee causality.
        self.local_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_kernel,
            groups=d_model,
            bias=True,
            padding=0,
        )

        # State gates.
        self.forget_proj = nn.Linear(d_model, d_state)
        self.input_proj = nn.Linear(d_model, d_state)

        # State/local fusion gate.
        self.fusion_proj = nn.Linear(d_model, d_state)

        # Output projection.
        self.output_proj = nn.Linear(d_state, d_model)

        # Forget gate starts biased toward retention.
        nn.init.constant_(self.forget_proj.bias, forget_bias)

    def causal_conv(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq, dim]
        returns: [batch, seq, dim]

        Only current and previous tokens may influence each output.
        """

        pad = self.conv_kernel - 1

        # [B, T, D] -> [B, D, T]
        x_conv = x.transpose(1, 2)

        # Left-only padding = causal convolution.
        x_conv = nn.functional.pad(x_conv, (pad, 0))

        y = self.local_conv(x_conv)

        # [B, D, T] -> [B, T, D]
        return y.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

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

        # Local branch.
        c = self.causal_conv(u)

        # Recurrent state gates.
        f = torch.sigmoid(self.forget_proj(u))
        i = torch.sigmoid(self.input_proj(u))

        # Fusion gate.
        g = torch.sigmoid(self.fusion_proj(u))

        if state is None:
            state = torch.zeros(
                batch,
                self.d_state,
                device=x.device,
                dtype=x.dtype,
            )

        outputs = []

        # Sequential state update.
        for t in range(u.shape[1]):
            state = f[:, t] * state + i[:, t] * u[:, t]

            fused = (
                g[:, t] * state
                + (1.0 - g[:, t]) * c[:, t]
            )

            out = self.output_proj(fused)

            # Residual connection.
            outputs.append(x[:, t] + out)

        y = torch.stack(outputs, dim=1)

        return y, state
