from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = ROOT / "evo" / "engine" / "evo_state.json"
SEARCH_FILE = ROOT / "evo" / "engine" / "search_space.json"
REGISTRY_FILE = ROOT / "evo" / "engine" / "mutation_registry.json"
OUTPUT_DIR = ROOT / "evo" / "gen4" / "candidates"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def architecture_signature(config: dict) -> str:
    keys = [
        "vocab_size",
        "d_model",
        "d_state",
        "num_layers",
        "conv_kernel",
        "forget_bias",
        "learnable_initial_state",
        "state_architecture",
        "fusion_architecture",
        "local_context",
    ]

    canonical = {
        key: config.get(key)
        for key in keys
    }

    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()[:16]


def validate_candidate(config: dict, search_space: dict) -> tuple[bool, list[str]]:
    errors = []

    constraints = search_space["constraints"]

    if config["d_model"] < constraints["d_model_min"]:
        errors.append("d_model below minimum")

    if config["d_model"] > constraints["d_model_max"]:
        errors.append("d_model above maximum")

    if config["d_state"] < constraints["d_state_min"]:
        errors.append("d_state below minimum")

    if config["d_state"] > constraints["d_state_max"]:
        errors.append("d_state above maximum")

    if config["num_layers"] < constraints["layers_min"]:
        errors.append("num_layers below minimum")

    if config["num_layers"] > constraints["layers_max"]:
        errors.append("num_layers above maximum")

    if (
        constraints["kernel_must_be_odd"]
        and config["conv_kernel"] % 2 == 0
    ):
        errors.append("conv_kernel must be odd")

    if config["conv_kernel"] <= 0:
        errors.append("conv_kernel must be positive")

    allowed = search_space["mutation_dimensions"]

    for key in [
        "d_model",
        "d_state",
        "num_layers",
        "conv_kernel",
        "forget_bias",
        "state_architecture",
        "fusion_architecture",
        "local_context",
    ]:
        if key in allowed:
            values = allowed[key]["values"]
            if config[key] not in values:
                errors.append(
                    f"{key}={config[key]!r} outside search space"
                )

    # GEN0-GEN3 implementation constraint.
    # Architectural state expansion/compression is registered,
    # but cannot yet be executed by the existing scan block.
    if config["state_architecture"] == "equal":
        if config["d_state"] != config["d_model"]:
            errors.append(
                "equal state architecture requires d_state == d_model"
            )

    return len(errors) == 0, errors


def mutation_label(parent: dict, child: dict) -> str:
    changed = []

    keys = [
        "d_model",
        "d_state",
        "num_layers",
        "conv_kernel",
        "forget_bias",
        "state_architecture",
        "fusion_architecture",
        "local_context",
    ]

    for key in keys:
        if parent.get(key) != child.get(key):
            changed.append(
                f"{key}:{parent.get(key)}->{child.get(key)}"
            )

    return "+".join(changed) if changed else "identity"


def generate_candidates(
    parent: dict,
    search_space: dict,
    registry: dict,
    count: int,
    seed: int,
) -> list[dict]:

    rng = random.Random(seed)

    existing_signatures = {
        item["signature"]
        for item in registry.get("architecture_signatures", [])
    }

    candidates = []
    dimensions = search_space["mutation_dimensions"]

    keys = [
        key
        for key, value in dimensions.items()
        if value.get("enabled", False)
    ]

    attempts = 0
    max_attempts = count * 100

    while len(candidates) < count and attempts < max_attempts:
        attempts += 1

        child = deepcopy(parent)

        mutation_count = rng.randint(
            1,
            search_space["generation_policy"]["max_mutations_per_candidate"],
        )

        selected_keys = rng.sample(
            keys,
            min(mutation_count, len(keys)),
        )

        for key in selected_keys:
            values = dimensions[key]["values"]

            if (
                key == "d_model"
                and child.get("state_architecture") == "equal"
            ):
                alternatives = [
                    value
                    for value in values
                    if value != child.get(key)
                ]

                if alternatives:
                    child[key] = rng.choice(alternatives)
                    child["d_state"] = child["d_model"]

                continue

            if key == "state_architecture":
                value = rng.choice(values)

                if value == "equal":
                    child["d_state"] = child["d_model"]

                elif value == "expanded":
                    possible = [
                        v for v in dimensions["d_state"]["values"]
                        if v > child["d_model"]
                    ]
                    if not possible:
                        continue
                    child["d_state"] = rng.choice(possible)

                elif value == "compressed":
                    possible = [
                        v for v in dimensions["d_state"]["values"]
                        if v < child["d_model"]
                    ]
                    if not possible:
                        continue
                    child["d_state"] = rng.choice(possible)

                child[key] = value

            else:
                alternatives = [
                    value
                    for value in values
                    if value != child.get(key)
                ]

                if alternatives:
                    child[key] = rng.choice(alternatives)

        state_architecture = child.get(
            "state_architecture",
            "equal",
        )

        if state_architecture == "equal":
            child["d_state"] = child["d_model"]

        elif state_architecture == "expanded":
            possible = [
                value
                for value in dimensions["d_state"]["values"]
                if value > child["d_model"]
            ]

            if not possible:
                continue

            if child["d_state"] <= child["d_model"]:
                child["d_state"] = rng.choice(possible)

        elif state_architecture == "compressed":
            possible = [
                value
                for value in dimensions["d_state"]["values"]
                if value < child["d_model"]
            ]

            if not possible:
                continue

            if child["d_state"] >= child["d_model"]:
                child["d_state"] = rng.choice(possible)

        signature = architecture_signature(child)

        parent_signature = architecture_signature(parent)

        if signature == parent_signature:
            continue

        if signature in existing_signatures:
            continue
        valid, errors = validate_candidate(
            child,
            search_space,
        )

        candidate_id = f"GEN4-{len(candidates) + 1:03d}"

        candidate = {
            "candidate": candidate_id,
            "generation": 4,
            "parent": "GEN3-007",
            "mutation": mutation_label(parent, child),
            "signature": signature,
            "config": child,
            "technical_validation": {
                "valid": valid,
                "errors": errors,
            },
            "status": "generated" if valid else "invalid",
        }

        candidates.append(candidate)
        existing_signatures.add(signature)

    return candidates


def main() -> None:
    state = load_json(STATE_FILE)
    search_space = load_json(SEARCH_FILE)
    registry = load_json(REGISTRY_FILE)

    parent = deepcopy(
        state["primary_parent_config"]
    )

    parent.update(
        {
            "state_architecture": "equal",
            "fusion_architecture": "standard",
            "local_context": "single_depthwise",
        }
    )

    count = search_space["generation_policy"][
        "max_candidates_per_generation"
    ]

    candidates = generate_candidates(
        parent=parent,
        search_space=search_space,
        registry=registry,
        count=count,
        seed=20260926,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for candidate in candidates:
        path = OUTPUT_DIR / f"{candidate['candidate']}.json"
        save_json(path, candidate)

    valid = [
        c for c in candidates
        if c["technical_validation"]["valid"]
    ]

    invalid = [
        c for c in candidates
        if not c["technical_validation"]["valid"]
    ]

    manifest = {
        "generation": 4,
        "parent": "GEN3-007",
        "seed": 20260926,
        "requested": count,
        "generated": len(candidates),
        "valid": len(valid),
        "invalid": len(invalid),
        "candidates": [
            {
                "candidate": c["candidate"],
                "mutation": c["mutation"],
                "signature": c["signature"],
                "status": c["status"],
            }
            for c in candidates
        ],
    }

    save_json(
        OUTPUT_DIR / "manifest.json",
        manifest,
    )

    print("=" * 70)
    print("NOVA-EVO GEN4 MUTATION ENGINE")
    print("=" * 70)
    print(f"Parent       : GEN3-007")
    print(f"Requested    : {count}")
    print(f"Generated    : {len(candidates)}")
    print(f"Valid        : {len(valid)}")
    print(f"Invalid      : {len(invalid)}")
    print()

    for candidate in candidates:
        status = "VALID" if candidate["technical_validation"]["valid"] else "INVALID"
        print(
            f"{candidate['candidate']}  "
            f"{status:<7}  "
            f"{candidate['mutation']}"
        )

    print()
    print(f"Manifest     : {OUTPUT_DIR / 'manifest.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
