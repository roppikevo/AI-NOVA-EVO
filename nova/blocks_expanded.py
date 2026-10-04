import torch
import torch.nn as nn

from nova.blocks import RMSNorm
from nova.blocks_scan import ScanRefBackward


class ExpandedStateScanBlock(nn.Module):
    """
    NOVA expanded-state block.

    d_model and d_state may differ.

    Flow:

        x
        │
        ├── RMSNorm ── u_model [d_model]
        │
        ├── causal depthwise convolution ── c [d_model]
        │
        ├── state input projection ── u_state [d_state]
        │
        ├── recurrent gates ── f, i, g [d_state]
        │
        └── recurrent state ── s [d_state]
                         │
                         └── fusion
                                  │
                         g*s + (1-g)*c_state
                                  │
                              output
                                  │
                              residual
    """

    def __init__(
        self,
        d_model,
        d_state,
        conv_kernel=5,
        forget_bias=1.5,
    ):
        super().__init__()

        if d_state <= d_model:
            raise ValueError(
                "Expanded state requires d_state > d_model"
            )

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

        self.state_input_proj = nn.Linear(
            d_model,
            d_state,
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

        self.conv_state_proj = nn.Linear(
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

        u_state = self.state_input_proj(u)

        f = torch.sigmoid(
            self.forget_proj(u)
        )

        i = torch.sigmoid(
            self.input_proj(u)
        )

        g = torch.sigmoid(
            self.fusion_proj(u)
        )

        c_state = self.conv_state_proj(c)

        states = ScanRefBackward.apply(
            f,
            i,
            u_state,
            state,
        )

        fused = (
            g * states
            + (1.0 - g) * c_state
        )

        out = self.output_proj(fused)

        y = x + out

        final_state = states[:, -1]

        return y, final_state
