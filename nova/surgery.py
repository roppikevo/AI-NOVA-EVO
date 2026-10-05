"""
Changing the structure of a trained NOVA core in place - without throwing its knowledge away.

A new structure normally means training from scratch (many hours). These operations edit a checkpoint so
that the edited model computes EXACTLY what the old one did; it then learns to use the new part in a
short training run, and if that does not help, the old checkpoint is still there untouched.

    add_layer(ckpt)        one more block; its output projection starts at zero, so it passes its input through
    set_kernel(ckpt, k)    local view of k tokens: wider = new taps start at zero (exact);
                           narrower = the oldest taps are dropped (not exact - the judge decides if it hurts)

The result is an ordinary checkpoint (config + weights) that long_train --init continues from.
"""

from __future__ import annotations

import copy
from typing import Any

import torch


def parameters(ckpt: dict) -> int:
    seen, n = set(), 0
    for k, v in ckpt["model_state_dict"].items():
        if k == "lm_head.weight" and "embedding.weight" in ckpt["model_state_dict"]:
            continue                                    # tied with the embedding
        if id(v) not in seen:
            seen.add(id(v))
            n += v.numel()
    return n


def is_gen8(config: dict) -> bool:
    return config.get("arch") == "nova8"


def genome(ckpt: dict) -> dict[str, Any]:
    c = ckpt["config"]
    if is_gen8(c):
        pattern = str(c.get("pattern") or "L" * int(c.get("num_layers", 6)))
        return {"layers": len(pattern), "width": int(c["d_model"]), "kernel": int(c.get("lru_kernel", 4)),
                "pattern": pattern, "parameters": parameters(ckpt)}
    return {"layers": int(c["num_layers"]), "width": int(c["d_model"]), "kernel": int(c["conv_kernel"]),
            "parameters": parameters(ckpt)}


def _fresh(config: dict) -> dict:
    from evo.engine.architecture_factory import build_model

    return build_model(config).state_dict()


def _note(ckpt: dict, what: str) -> dict:
    out = {k: v for k, v in ckpt.items() if k != "optimizer"}
    out["surgery"] = list(ckpt.get("surgery", [])) + [what]
    out["kind"] = "surgery"
    return out


def add_layer(ckpt: dict, position: int | None = None, seed: int = 0, kind: str | None = None) -> dict:
    """One more block at `position` (default: on top). The edited model gives the same output as before."""
    config = dict(ckpt["config"])
    gen8 = is_gen8(config)
    if gen8:                                                     # generation 8: the pattern of mixers grows by one
        pattern = str(config.get("pattern") or "L" * int(config.get("num_layers", 6)))
        n = len(pattern)
        position = n if position is None else max(0, min(n, int(position)))
        kind = kind or pattern[-1]
        config["pattern"] = pattern[:position] + kind + pattern[position:]
        silent = ("mixer.out.", "fc_out.")                       # both residual branches of the new block start at zero
    else:
        n = int(config["num_layers"])
        position = n if position is None else max(0, min(n, int(position)))
        silent = ("output_proj.",)
    config["num_layers"] = n + 1
    torch.manual_seed(seed)
    new = _fresh(config)
    old = ckpt["model_state_dict"]
    for key in new:
        if key.startswith("blocks."):
            _, idx, rest = key.split(".", 2)
            i = int(idx)
            if i == position:
                if rest.startswith(silent):
                    new[key] = torch.zeros_like(new[key])        # the new block adds nothing at first
                continue
            src = f"blocks.{i if i < position else i - 1}.{rest}"
            new[key] = old[src].clone().to(new[key].dtype)
        else:
            new[key] = old[key].clone().to(new[key].dtype)
    out = _note(ckpt, f"add_layer at {position}: {n} -> {n + 1} layers")
    out.update({"config": config, "model_state_dict": new})
    return out


def set_kernel(ckpt: dict, kernel: int) -> dict:
    """Local view of `kernel` tokens (odd). Wider is exact; narrower drops the oldest taps."""
    kernel = int(kernel)
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("kernel must be a positive odd number")
    config = dict(ckpt["config"])
    old_k = int(config["conv_kernel"])
    sd = {k: v.clone() for k, v in ckpt["model_state_dict"].items()}
    for key, w in list(sd.items()):
        if key.endswith("local_conv.weight"):                    # [channels, 1, k]; the last tap is the current token
            if kernel > old_k:
                pad = torch.zeros(w.shape[0], w.shape[1], kernel - old_k, dtype=w.dtype)
                sd[key] = torch.cat([pad, w], dim=2)
            else:
                sd[key] = w[:, :, old_k - kernel:].clone()
    config["conv_kernel"] = kernel
    out = _note(ckpt, f"kernel {old_k} -> {kernel}")
    out.update({"config": config, "model_state_dict": sd})
    return out


def apply(ckpt: dict, op: dict, max_parameters: int | None = None, seed: int = 0) -> dict | None:
    """Apply one operation ({"op": "add_layer"} or {"op": "kernel", "delta": +2}); None if it is not possible."""
    kind = op.get("op")
    if kind == "add_layer":
        new = add_layer(ckpt, op.get("position"), seed, op.get("kind"))
    elif kind == "kernel":
        if is_gen8(ckpt["config"]):
            return None                                          # not defined for generation 8 yet
        k = int(op["k"]) if "k" in op else int(ckpt["config"]["conv_kernel"]) + int(op.get("delta", 2))
        if k < 1 or k > 9 or k == int(ckpt["config"]["conv_kernel"]):
            return None
        new = set_kernel(ckpt, k)
    else:
        raise ValueError(f"unknown operation {op}")
    if max_parameters and parameters(new) > max_parameters:
        return None
    return copy.copy(new)
