import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(
            batch,
            seq,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch,
            seq,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch,
            seq,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(
            batch,
            seq,
            dim,
        )

        return self.out_proj(y)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)

        self.attention = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerBaseline(nn.Module):
    """
    Parameter-matched causal Transformer baseline
    for NOVA-0 GEN 0.

    Configuration:
        vocab_size = 16384
        d_model    = 256
        layers     = 4
        heads      = 8
        ffn_dim    = 256

    Embedding and LM head are tied.
    """

    def __init__(
        self,
        vocab_size: int = 16384,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 256,
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_token_id,
        )

        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.02,
        )

        if pad_token_id is not None:
            with torch.no_grad():
                self.embedding.weight[pad_token_id].zero_()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

        self.lm_head.weight = self.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:

        if input_ids.ndim != 2:
            raise ValueError(
                f"Expected input_ids [batch, seq], "
                f"got {tuple(input_ids.shape)}"
            )

        x = self.embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        return self.lm_head(x)

    def num_parameters(self) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )
