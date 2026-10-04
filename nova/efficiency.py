"""
Efficiency measurements for NOVA candidates.

NOVA-EVO targets CPU + RAM inference with minimal VRAM, so every trained
candidate is measured on:

    parameters / param_mb        model size (fp32)
    cpu_tokens_per_sec           prompt processing on CPU (batch 1)
    cpu_gen_tokens_per_sec       token-by-token generation on CPU using
                                 the recurrent state (None if unsupported)
    state_kb                     recurrent state size carried per sequence
    peak_vram_mb                 peak CUDA memory during training
                                 (filled in by the training runner)
"""

from __future__ import annotations

import copy
import time
from typing import Any

import torch


def _numel_bytes(obj: Any) -> int:
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(_numel_bytes(o) for o in obj)
    if isinstance(obj, dict):
        return sum(_numel_bytes(o) for o in obj.values())
    return 0


def _logits(out: Any) -> torch.Tensor:
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def measure_cpu(
    model: torch.nn.Module,
    vocab_size: int,
    seq_len: int = 128,
    gen_tokens: int = 64,
    repeats: int = 3,
    threads: int = 8,
) -> dict[str, Any]:
    cpu_model = copy.deepcopy(model).to("cpu").float().eval()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        params = sum(p.numel() for p in cpu_model.parameters())
        x = torch.randint(4, vocab_size, (1, seq_len))

        cpu_model(x)  # warm-up
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            cpu_model(x)
            best = min(best, time.perf_counter() - t0)
        prompt_tps = seq_len / best

        gen_tps = None
        state_kb = None
        try:
            out = cpu_model(x[:, :1])
            if isinstance(out, (tuple, list)) and len(out) > 1:
                states = out[1]
                state_kb = round(_numel_bytes(states) / 1024, 1)
                token = x[:, 1:2]
                t0 = time.perf_counter()
                for _ in range(gen_tokens):
                    logits, states = cpu_model(token, states)[:2]
                    token = logits[:, -1:].argmax(-1)
                gen_tps = gen_tokens / (time.perf_counter() - t0)
        except TypeError:
            pass  # model has no state interface

        return {
            "parameters": params,
            "param_mb": round(params * 4 / 1e6, 2),
            "cpu_threads": threads,
            "cpu_tokens_per_sec": round(prompt_tps, 1),
            "cpu_gen_tokens_per_sec": round(gen_tps, 1) if gen_tps else None,
            "state_kb": state_kb,
        }
    finally:
        torch.set_num_threads(previous_threads)
        del cpu_model


def efficiency_check(
    candidate: dict[str, Any],
    parent: dict[str, Any] | None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compare candidate efficiency with the parent.

    Returns {"ok": bool, "gain": bool, "violations": [...], "ratios": {...}}.
    `gain` = clearly more efficient (smaller or faster) than the parent.
    Missing parent metrics never block a candidate (first measurement).
    """
    policy = {
        "max_param_ratio": 1.25,
        "min_cpu_speed_ratio": 0.80,
        "max_vram_ratio": 1.30,
        "gain_param_ratio": 0.80,
        "gain_cpu_speed_ratio": 1.25,
        **(policy or {}),
    }
    parent = parent or {}
    ratios: dict[str, float] = {}

    def ratio(key: str) -> float | None:
        a, b = candidate.get(key), parent.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b > 0:
            ratios[key] = round(a / b, 3)
            return ratios[key]
        return None

    violations = []
    p = ratio("parameters")
    s = ratio("cpu_tokens_per_sec")
    v = ratio("peak_vram_mb")

    if p is not None and p > policy["max_param_ratio"]:
        violations.append(f"parameters x{p} > x{policy['max_param_ratio']}")
    if s is not None and s < policy["min_cpu_speed_ratio"]:
        violations.append(f"cpu speed x{s} < x{policy['min_cpu_speed_ratio']}")
    if v is not None and v > policy["max_vram_ratio"]:
        violations.append(f"peak vram x{v} > x{policy['max_vram_ratio']}")

    gain = (
        (p is not None and p <= policy["gain_param_ratio"])
        or (s is not None and s >= policy["gain_cpu_speed_ratio"])
    ) and not violations

    return {"ok": not violations, "gain": gain, "violations": violations,
            "ratios": ratios, "policy": policy}
