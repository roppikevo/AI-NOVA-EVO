import torch
import torch.nn as nn

from .config import NovaConfig
from .blocks_scan import NovaScanBlock


class NovaModel(nn.Module):
    """
    NOVA-0 PRETEST language model.

    Architecture:
        token embedding
        -> NOVA recurrent/local blocks
        -> RMSNorm
        -> tied LM head

    Causal language modeling:
        input_ids[:, :-1] -> predict input_ids[:, 1:]
    """

    def __init__(self, config: NovaConfig):
        super().__init__()

        self.config = config

        d_embed = getattr(config, "d_embed", None) or config.d_model
        self.d_embed = d_embed

        self.embedding = nn.Embedding(
            config.vocab_size,
            d_embed,
            padding_idx=config.pad_token_id,
        )
        # factorized embedding: small token table + projections in/out of the core
        self.embed_in = nn.Linear(d_embed, config.d_model, bias=False) if d_embed != config.d_model else None
        self.embed_out = nn.Linear(config.d_model, d_embed, bias=False) if d_embed != config.d_model else None

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
                NovaScanBlock(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    conv_kernel=config.conv_kernel,
                    forget_bias=config.forget_bias,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(config.d_model)

        # Tied embedding / LM head.
        self.lm_head = nn.Linear(
            d_embed,
            config.vocab_size,
            bias=False,
        )

        self.lm_head.weight = self.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        states: list[torch.Tensor] | None = None,
    ):
        if input_ids.ndim != 2:
            raise ValueError(
                f"Expected input_ids [batch, seq], got {tuple(input_ids.shape)}"
            )

        x = self.embedding(input_ids)
        if self.embed_in is not None:
            x = self.embed_in(x)

        new_states = []

        for layer_idx, block in enumerate(self.blocks):
            state = None if states is None else states[layer_idx]

            x, state = block(
                x,
                state=state,
            )

            new_states.append(state)

        x = self.final_norm(x)
        if self.embed_out is not None:
            x = self.embed_out(x)
        logits = self.lm_head(x)

        return logits, new_states

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(
                p.numel()
                for p in self.parameters()
                if p.requires_grad
            )

        return sum(
            p.numel()
            for p in self.parameters()
        )


def build_model(config: NovaConfig | None = None) -> NovaModel:
    if config is None:
        config = NovaConfig()

    return NovaModel(config)
