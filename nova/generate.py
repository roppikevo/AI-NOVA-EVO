"""Greedy / temperature sampling from a NOVA checkpoint (CPU by default)."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from nova.tokenizer import END_THINK_ID, EOS_ID, THINK_ID, NovaTokenizer


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tok: NovaTokenizer,
    prompt: str,
    lang: str,
    max_new_tokens: int = 60,
    temperature: float = 0.0,
    device: str = "cpu",
    think: bool = False,
    max_think_tokens: int = 64,
) -> str:
    """
    think=True starts the answer with <think>; thinking is capped at
    max_think_tokens (then </think> is forced), so compute stays bounded.
    """
    model = model.to(device).eval()
    ids = [tok.lang_id(lang)] + tok.encode(prompt)
    thinking = False
    if think and tok.has_think:
        ids.append(THINK_ID)
        thinking = True
    think_used = 0
    from nova.stepper import Writer

    writer = Writer(model, ids, device)   # NOVA core: constant cost per token; other models: 128-token window
    out = []
    for _ in range(max_new_tokens + (max_think_tokens if thinking else 0)):
        last = writer.logits
        if temperature > 0:
            nxt = int(torch.multinomial(torch.softmax(last / temperature, -1), 1))
        else:
            nxt = int(last.argmax())
        if thinking:
            think_used += 1
            if nxt == END_THINK_ID:
                thinking = False
            elif think_used >= max_think_tokens:
                nxt, thinking = END_THINK_ID, False
        elif nxt == EOS_ID:
            break
        out.append(nxt)
        writer.push(nxt)
    return prompt + tok.decode(out, skip_special=not think)


def load_checkpoint_model(path: str | Path):
    from evo.engine.architecture_factory import build_model

    try:  # released files hold only tensors and plain values: load them without executing anything
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        if os.environ.get("NOVA_SAFE_LOAD"):
            raise
        ckpt = torch.load(path, map_location="cpu", weights_only=False)   # our own training checkpoints
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


@torch.no_grad()
def continuation_logprob(
    model: torch.nn.Module,
    tok: NovaTokenizer,
    prompt: str,
    continuation: str,
    lang: str,
) -> float:
    """Mean log-probability per token of `continuation` after `prompt`."""
    model = model.to("cpu").eval()
    p = [tok.lang_id(lang)] + tok.encode(prompt)
    c = tok.encode(continuation)
    x = torch.tensor([p + c])
    logits = model(x)
    logits = logits[0] if isinstance(logits, (tuple, list)) else logits
    logp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = torch.tensor(p + c)[1:]
    picked = logp[torch.arange(len(targets)), targets][len(p) - 1:]
    return float(picked.mean())
