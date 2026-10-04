import torch
import torch.nn as nn

from nova.config import NovaConfig
from nova.blocks_expanded import ExpandedStateScanBlock


class NovaExpandedModel(nn.Module):
    """
    NOVA model with expanded recurrent state.

    d_model and d_state may differ.

    The standard NOVA model remains unchanged.
    """

    def __init__(self, config: NovaConfig):
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

        if config.pad_token_id is not None:
            with torch.no_grad():
                self.embedding.weight[config.pad_token_id].zero_()

        self.blocks = nn.ModuleList(
            [
                ExpandedStateScanBlock(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    conv_kernel=config.conv_kernel,
                    forget_bias=config.forget_bias,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, states=None):
        x = self.embedding(input_ids)

        new_states = []

        for index, block in enumerate(self.blocks):
            state = None

            if states is not None:
                state = states[index]

            x, final_state = block(
                x,
                state=state,
            )

            new_states.append(final_state)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits, new_states

    @property
    def num_parameters(self):
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


def build_model(config):
    return NovaExpandedModel(config)
