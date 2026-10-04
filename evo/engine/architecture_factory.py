from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

# Make the NOVA project root importable even when this file
# is executed directly as a script.
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REGISTRY_FILE = (
    ROOT / "evo" / "engine" / "architecture_registry.json"
)


def load_registry() -> dict:
    with REGISTRY_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def architecture_key(config: dict) -> str:
    state = config.get("state_architecture", "equal")
    fusion = config.get("fusion_architecture", "standard")
    local = config.get("local_context", "single_depthwise")

    if (
        state == "equal"
        and fusion == "standard"
        and local == "single_depthwise"
    ):
        return "standard"

    if state == "expanded":
        return "expanded_state"

    if state == "compressed":
        return "compressed_state"

    if fusion == "dual_gate":
        return "dual_gate"

    if fusion == "separate_state_gate":
        return "separate_state_gate"

    if fusion == "separate_conv_gate":
        return "separate_conv_gate"

    if local == "multi_scale":
        return "multi_scale"

    return "unknown"


def get_architecture_info(config: dict) -> dict:
    registry = load_registry()
    key = architecture_key(config)

    if key not in registry["architectures"]:
        return {
            "key": key,
            "supported": False,
            "reason": "Architecture not registered",
        }

    info = dict(registry["architectures"][key])
    info["key"] = key

    return info


def build_model(config: dict):
    if config.get("arch") == "transformer":
        # the yardstick: a standard transformer with the same interface (see nova/transformer_lm.py)
        from nova.transformer_lm import build_transformer

        return build_transformer(config)
    info = get_architecture_info(config)

    if not info.get("supported", False):
        raise RuntimeError(
            f"Architecture '{info['key']}' is not supported: "
            f"{info.get('reason', 'unknown reason')}"
        )

    implementation = info.get("implementation")

    if not implementation:
        raise RuntimeError(
            f"Architecture '{info['key']}' has no implementation"
        )

    module_name, class_name = implementation.rsplit(".", 1)

    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)

    # Existing NOVA implementation expects a NovaConfig object,
    # not a raw dictionary.
    from nova.config import NovaConfig

    nova_config = NovaConfig(
        vocab_size=int(config["vocab_size"]),
        d_model=int(config["d_model"]),
        d_state=int(config["d_state"]),
        num_layers=int(config["num_layers"]),
        conv_kernel=int(config["conv_kernel"]),
        forget_bias=float(config["forget_bias"]),
        learnable_initial_state=bool(
            config.get("learnable_initial_state", False)
        ),
        d_embed=(int(config["d_embed"]) if config.get("d_embed") else None),
    )

    return model_class(nova_config)


def inspect(config: dict) -> dict[str, Any]:
    info = get_architecture_info(config)

    result = {
        "architecture": info["key"],
        "supported": bool(info.get("supported", False)),
        "implementation": info.get("implementation"),
        "reason": info.get("reason"),
    }

    return result


if __name__ == "__main__":
    from pprint import pprint

    standard = {
        "vocab_size": 16384,
        "d_model": 384,
        "d_state": 384,
        "num_layers": 6,
        "conv_kernel": 5,
        "forget_bias": 1.125,
        "learnable_initial_state": False,
        "state_architecture": "equal",
        "fusion_architecture": "standard",
        "local_context": "single_depthwise",
    }

    expanded = dict(standard)
    expanded["d_state"] = 512
    expanded["state_architecture"] = "expanded"

    print("STANDARD:")
    pprint(inspect(standard))

    print()
    print("EXPANDED:")
    pprint(inspect(expanded))

    print()
    print("BUILD TEST:")

    try:
        model = build_model(standard)
        print("STANDARD BUILD: OK")
        print(
            f"PARAMETERS: "
            f"{sum(p.numel() for p in model.parameters()):,}"
        )
    except Exception as exc:
        print(f"STANDARD BUILD: FAILED: {exc}")

    try:
        build_model(expanded)
        print("EXPANDED BUILD: UNEXPECTEDLY SUPPORTED")
    except Exception as exc:
        print(f"EXPANDED BUILD: CORRECTLY BLOCKED: {exc}")
