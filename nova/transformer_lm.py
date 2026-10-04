"""
A standard transformer language model, used ONLY as a yardstick for the NOVA core.

Decoder-only, pre-norm, rotary positions, GELU MLP (x4), tied token embedding / LM head - the usual
small-GPT recipe. It speaks the same interface as NovaModel:

    logits, states = model(input_ids)             # training / scoring
    logits, states = model(next_token, states)    # token by token; states = key/value cache per layer

so training (long_train), scoring, generation and the speed measurements run unchanged on both.
(The old nova/transformer_baseline.py from GEN 0 has no position information and a tiny MLP; it is kept
only for the old benchmark scripts.)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope(x: torch.Tensor, offset: int) -> torch.Tensor:
    """Rotary position embedding on [batch, heads, time, head_dim]; `offset` = tokens already in the cache."""
    t, d = x.shape[-2], x.shape[-1]
    half = d // 2
    freq = 1.0 / (10000 ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
    ang = torch.arange(offset, offset + t, device=x.device, dtype=torch.float32)[:, None] * freq[None, :]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    a, b = x[..., :half], x[..., half:2 * half]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ff_mult * d_model)
        self.fc2 = nn.Linear(ff_mult * d_model, d_model)

    def forward(self, x: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor] | None = None):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        past = 0 if cache is None else cache[0].shape[2]
        q, k = rope(q, past), rope(k, past)
        if cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            k, v = torch.cat([cache[0], k], dim=2), torch.cat([cache[1], v], dim=2)
            mask = None
            if t > 1:  # new tokens see the whole past and, among themselves, only what came before
                mask = torch.ones(t, past + t, dtype=torch.bool, device=x.device).tril(diagonal=past)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, d))
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x, (k, v)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, n_heads: int, ff_mult: int = 4,
                 pad_token_id: int = 0) -> None:
        super().__init__()
        if d_model % n_heads or (d_model // n_heads) % 2:
            raise ValueError(f"d_model {d_model} must split into {n_heads} heads of even size")
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, ff_mult) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # tied, as in the NOVA core
        self.apply(self._init)
        for blk in self.blocks:  # residual branches start small (GPT-2 style)
            nn.init.normal_(blk.proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))
            nn.init.normal_(blk.fc2.weight, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))
        with torch.no_grad():
            self.embedding.weight[pad_token_id].zero_()

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor, states: list | None = None):
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids [batch, seq], got {tuple(input_ids.shape)}")
        x = self.embedding(input_ids)
        new_states = []
        for i, blk in enumerate(self.blocks):
            x, st = blk(x, None if states is None else states[i])
            new_states.append(st)
        return self.lm_head(self.final_norm(x)), new_states

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)


def build_transformer(config: dict) -> TransformerLM:
    d = int(config["d_model"])
    return TransformerLM(vocab_size=int(config["vocab_size"]), d_model=d, num_layers=int(config["num_layers"]),
                         n_heads=int(config.get("n_heads") or max(1, d // 64)), ff_mult=int(config.get("ff_mult", 4)))
