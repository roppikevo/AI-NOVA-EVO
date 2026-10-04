from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict

from nova.config import NovaConfig


def config_dict(config: NovaConfig) -> dict:
    return asdict(config)


def make_candidate(parent: NovaConfig, candidate_id: str, mutation: str) -> dict:
    cfg = copy.deepcopy(parent)

    if mutation == "layers_minus":
        if cfg.num_layers <= 1:
            raise ValueError("num_layers cannot go below 1")
        cfg.num_layers -= 1

    elif mutation == "layers_plus":
        cfg.num_layers += 1

    elif mutation == "kernel_3":
        cfg.conv_kernel = 3

    elif mutation == "kernel_7":
        cfg.conv_kernel = 7

    elif mutation == "d_model_minus":
        if cfg.d_model <= 64:
            raise ValueError("d_model too small")
        cfg.d_model -= 64
        cfg.d_state = cfg.d_model

    elif mutation == "d_model_plus":
        cfg.d_model += 64
        cfg.d_state = cfg.d_model

    elif mutation == "state_minus":
        if cfg.d_state <= 64:
            raise ValueError("d_state too small")
        cfg.d_state -= 64

    elif mutation == "state_plus":
        cfg.d_state += 64

    elif mutation == "forget_bias_down":
        cfg.forget_bias -= 0.25

    elif mutation == "forget_bias_up":
        cfg.forget_bias += 0.25

    else:
        raise ValueError(f"Unknown mutation: {mutation}")

    return {
        "candidate_id": candidate_id,
        "parent": "NOVA-GEN0",
        "mutation": mutation,
        "config": config_dict(cfg),
    }


def generate_gen1(seed: int = 20260925) -> list[dict]:
    rng = random.Random(seed)

    base = NovaConfig()

    mutations = [
        "layers_minus",
        "layers_plus",
        "kernel_3",
        "kernel_7",
        "d_model_minus",
        "d_model_plus",
        "state_minus",
        "state_plus",
        "forget_bias_down",
        "forget_bias_up",
    ]

    rng.shuffle(mutations)

    candidates = []

    for index, mutation in enumerate(mutations, start=1):
        candidate_id = f"GEN1-{index:03d}"

        try:
            candidate = make_candidate(
                base,
                candidate_id,
                mutation,
            )
            candidates.append(candidate)
        except ValueError:
            continue

    return candidates


def main():
    candidates = generate_gen1()

    output = "evo/gen1/candidates.json"

    with open(output, "w", encoding="utf-8") as f:
        json.dump(
            candidates,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 70)
    print("NOVA-EVO GEN 1 — MUTATION ENGINE")
    print("=" * 70)
    print(f"Parent     : NOVA-GEN0")
    print(f"Candidates : {len(candidates)}")
    print()

    for candidate in candidates:
        cfg = candidate["config"]

        print(
            f"{candidate['candidate_id']:10s} "
            f"{candidate['mutation']:18s} "
            f"layers={cfg['num_layers']} "
            f"d_model={cfg['d_model']} "
            f"d_state={cfg['d_state']} "
            f"kernel={cfg['conv_kernel']} "
            f"forget={cfg['forget_bias']:.2f}"
        )

    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
