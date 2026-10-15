"""
Domain packages: what NOVA learns about one field (mathematics, medicine, a programming language, ...) is kept in a
small separate file, while the base core stays exactly as it was.

A package is a set of low-rank additions to some of the core's linear layers: W x  becomes  W x + s * B (A x), with
A (r x in) and B (out x r), r small (8-32). Only A and B are trained; the base weights are frozen, so the base cannot
forget anything while a package learns, packages can be swapped, and one base serves many fields. B starts at zero:
a fresh package changes nothing until it has learned.

    inject(model, rank=16)             add trainable packages to the target layers, freeze everything else
    package_state(model)               the package's own tensors (a few MB) - what is stored and shipped
    merged_state_dict(model)           base + package as one ordinary state dict (loads into build_model; used for
                                       the topic exam and for running the core with the package switched on)
    load_package(model, state)         put a stored package into a model that was injected with the same layout
    save_package(path, ...), read_package(path)

The base core a package was trained on is identified by a fingerprint of its weights; a package refuses a base it
was not made for.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import torch
import torch.nn as nn

TARGETS = r"\.(fc_a|fc_b|fc_out|mixer\.out)$"      # the per-token layers and the output of every mixer


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scale = (alpha if alpha is not None else rank) / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    @property
    def weight(self) -> torch.Tensor:                 # the base weight only - code that fuses layers must call forward
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def delta(self) -> torch.Tensor:
        return (self.lora_b @ self.lora_a) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_a.t().to(x.dtype)) @ self.lora_b.t().to(x.dtype) * self.scale


def inject(model: nn.Module, rank: int = 16, alpha: float | None = None, targets: str = TARGETS) -> list[str]:
    """Wrap the target linear layers; every other parameter is frozen. Returns the names of the wrapped layers."""
    for p in model.parameters():
        p.requires_grad_(False)
    rx, done = re.compile(targets), []
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and rx.search("." + name):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, LoRALinear(mod, rank, alpha))
            done.append(name)
    if not done:
        raise ValueError(f"no layer matches {targets!r}")
    return done


def has_packages(model: nn.Module) -> bool:
    return any(isinstance(m, LoRALinear) for m in model.modules())


def package_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if ".lora_" in k}


def load_package(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    own = dict(model.named_parameters())
    missing = [k for k in state if k not in own]
    if missing:
        raise ValueError(f"the package does not fit this core: {missing[:3]}")
    with torch.no_grad():
        for k, v in state.items():
            own[k].copy_(v.to(own[k].dtype))


def merged_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """An ordinary state dict (the base's key names) with every package added into its layer's weight."""
    out = {}
    wrapped = {name: m for name, m in model.named_modules() if isinstance(m, LoRALinear)}
    for k, v in model.state_dict().items():
        if ".lora_" in k:
            continue
        name = k.rsplit(".", 2)[0] if ".base." in k else None
        if name in wrapped and k.endswith(".base.weight"):
            out[f"{name}.weight"] = (v.float() + wrapped[name].delta().detach().float().cpu().to(v.device)).to(v.dtype).detach().cpu()
        elif name in wrapped and k.endswith(".base.bias"):
            out[f"{name}.bias"] = v.detach().cpu()
        else:
            out[k] = v.detach().cpu()
    return out


def base_fingerprint(state_dict: dict[str, torch.Tensor]) -> str:
    """Identifies the base core: a hash over the names, shapes and a sample of the values of its tensors."""
    h = hashlib.sha256()
    for k in sorted(state_dict):
        if ".lora_" in k:
            continue
        v = state_dict[k]
        k2 = k.replace(".base.", ".")
        h.update(k2.encode() + str(tuple(v.shape)).encode())
        flat = v.detach().float().reshape(-1).cpu()
        h.update(flat[:: max(1, flat.numel() // 64)][:64].numpy().tobytes())
    return h.hexdigest()[:16]


def save_package(path: str | Path, state: dict[str, torch.Tensor], base_fp: str, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "nova-package", "base": base_fp, "meta": meta, "state": state}, path)
    return path


def read_package(path: str | Path) -> dict:
    p = torch.load(path, map_location="cpu", weights_only=True)
    if p.get("kind") != "nova-package":
        raise ValueError(f"{path}: not a NOVA package")
    return p


def apply_package(model: nn.Module, package: dict, base_state: dict[str, torch.Tensor] | None = None) -> nn.Module:
    """Base core + a stored package, ready to run (the layout - rank, targets - comes from the package)."""
    if base_state is not None and package["base"] != base_fingerprint(base_state):
        raise ValueError("this package was trained on another base core")
    meta = package.get("meta") or {}
    inject(model, int(meta.get("rank", 16)), meta.get("alpha"), meta.get("targets", TARGETS))
    load_package(model, package["state"])
    return model


def describe(package: dict) -> str:
    n = sum(v.numel() for v in package["state"].values())
    return json.dumps({**(package.get("meta") or {}), "base": package["base"], "parameters": n}, ensure_ascii=False)
