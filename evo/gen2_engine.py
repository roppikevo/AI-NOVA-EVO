from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/opt/ai/work/nova-evo")

PARETO_FILE = ROOT / "evo/gen1/pareto_front.json"
CANDIDATES_FILE = ROOT / "evo/gen1/candidates.json"

GEN2_DIR = ROOT / "evo/gen2"
GEN2_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = GEN2_DIR / "candidates.json"
PARENT_FILE = GEN2_DIR / "parent_manifest.json"


TARGET_PARENTS = [
    "GEN1-010",
    "GEN1-001",
    "GEN1-006",
]


def config_key(cfg):
    return (
        cfg["vocab_size"],
        cfg["d_model"],
        cfg["d_state"],
        cfg["num_layers"],
        cfg["conv_kernel"],
        cfg["forget_bias"],
        cfg["learnable_initial_state"],
    )


def valid_config(cfg):
    if cfg["d_model"] != cfg["d_state"]:
        return False

    if cfg["d_model"] < 128:
        return False

    if cfg["num_layers"] < 1:
        return False

    if cfg["conv_kernel"] not in (3, 5, 7):
        return False

    if cfg["forget_bias"] < 0.5 or cfg["forget_bias"] > 2.5:
        return False

    return True


def mutate(base, changes):
    cfg = dict(base)

    mutation_names = []

    for name, delta in changes:
        mutation_names.append(name)

        if name == "layers_minus":
            cfg["num_layers"] -= 1

        elif name == "layers_plus":
            cfg["num_layers"] += 1

        elif name == "d_model_minus":
            cfg["d_model"] -= 64
            cfg["d_state"] = cfg["d_model"]

        elif name == "d_model_plus":
            cfg["d_model"] += 64
            cfg["d_state"] = cfg["d_model"]

        elif name == "kernel_3":
            cfg["conv_kernel"] = 3

        elif name == "kernel_5":
            cfg["conv_kernel"] = 5

        elif name == "kernel_7":
            cfg["conv_kernel"] = 7

        elif name == "forget_bias_down":
            cfg["forget_bias"] -= 0.25

        elif name == "forget_bias_up":
            cfg["forget_bias"] += 0.25

        else:
            raise ValueError(f"Unknown mutation: {name}")

    return cfg, "+".join(mutation_names)


with open(PARETO_FILE, "r", encoding="utf-8") as f:
    pareto = json.load(f)

with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
    gen1_candidates = json.load(f)


candidate_map = {
    c["candidate_id"]: c
    for c in gen1_candidates
}


parents = []

for parent_id in TARGET_PARENTS:
    if parent_id not in candidate_map:
        raise RuntimeError(f"Missing GEN1 parent: {parent_id}")

    parent = candidate_map[parent_id]

    parents.append({
        "candidate_id": parent_id,
        "parent": parent["parent"],
        "mutation": parent["mutation"],
        "config": parent["config"],
    })


# ---------------------------------------------------------------------------
# GEN2 mutation strategy
#
# Each Pareto parent gets:
#
#   local structural mutations
#   local width mutations
#   local kernel mutations
#   local memory-gate mutations
#   selected combinations
#
# This is deliberate local search around proven Pareto regions.
# ---------------------------------------------------------------------------

mutation_sets = [
    [("layers_minus", -1)],
    [("layers_plus", +1)],

    [("d_model_minus", -64)],
    [("d_model_plus", +64)],

    [("kernel_3", 3)],
    [("kernel_5", 5)],
    [("kernel_7", 7)],

    [("forget_bias_down", -0.25)],
    [("forget_bias_up", +0.25)],

    # combinations
    [("d_model_plus", +64), ("layers_minus", -1)],
    [("d_model_plus", +64), ("kernel_3", 3)],
    [("d_model_plus", +64), ("forget_bias_down", -0.25)],

    [("layers_minus", -1), ("kernel_3", 3)],
    [("layers_minus", -1), ("forget_bias_down", -0.25)],

    [("d_model_minus", -64), ("kernel_3", 3)],
    [("d_model_minus", -64), ("forget_bias_down", -0.25)],
]


children = []
seen = set()

for parent in parents:
    base = parent["config"]

    for changes in mutation_sets:
        try:
            cfg, mutation_name = mutate(base, changes)
        except ValueError:
            continue

        if not valid_config(cfg):
            continue

        key = config_key(cfg)

        # Do not generate the parent itself.
        if key == config_key(base):
            continue

        # Avoid duplicate architecture configurations.
        if key in seen:
            continue

        seen.add(key)

        child_id = f"GEN2-{len(children)+1:03d}"

        children.append({
            "candidate_id": child_id,
            "parent": parent["candidate_id"],
            "parent_mutation": parent["mutation"],
            "mutation": mutation_name,
            "config": cfg,
        })


# ---------------------------------------------------------------------------
# Save exact parent genealogy
# ---------------------------------------------------------------------------

parent_manifest = {
    "generation": 2,
    "selection_source": str(PARETO_FILE),
    "parents": parents,
    "parent_selection": "GEN1 Pareto front",
}

with open(PARENT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        parent_manifest,
        f,
        indent=2,
        ensure_ascii=False,
    )


with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        children,
        f,
        indent=2,
        ensure_ascii=False,
    )


print("=" * 78)
print("NOVA-EVO GEN 2 — EVOLUTION ENGINE")
print("=" * 78)
print()
print("Parents:")
for p in parents:
    cfg = p["config"]
    print(
        f"  {p['candidate_id']:10s} "
        f"layers={cfg['num_layers']} "
        f"d_model={cfg['d_model']} "
        f"d_state={cfg['d_state']} "
        f"kernel={cfg['conv_kernel']} "
        f"forget={cfg['forget_bias']:.2f}"
    )

print()
print(f"GEN2 candidates : {len(children)}")
print()

for c in children:
    cfg = c["config"]

    print(
        f"{c['candidate_id']:10s} "
        f"parent={c['parent']:10s} "
        f"mutation={c['mutation']:55s} "
        f"layers={cfg['num_layers']} "
        f"d_model={cfg['d_model']} "
        f"kernel={cfg['conv_kernel']} "
        f"forget={cfg['forget_bias']:.2f}"
    )

print()
print(f"Saved candidates : {OUT_FILE}")
print(f"Saved genealogy  : {PARENT_FILE}")
