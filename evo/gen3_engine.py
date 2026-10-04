import json
import os
from itertools import product

ROOT = "/opt/ai/work/nova-evo"
GEN2_DIR = os.path.join(ROOT, "evo/gen2")
GEN3_DIR = os.path.join(ROOT, "evo/gen3")

os.makedirs(GEN3_DIR, exist_ok=True)

with open(os.path.join(GEN2_DIR, "pareto_front.json")) as f:
    pareto = json.load(f)

with open(os.path.join(GEN2_DIR, "candidates.json")) as f:
    all_gen2 = {x["candidate_id"]: x for x in json.load(f)}

# Targeted GEN3 mutations.
# We deliberately stay close to successful GEN2 regions.
mutations = [
    "layers_minus",
    "layers_plus",
    "kernel_3",
    "kernel_7",
    "forget_bias_down",
    "forget_bias_up",
    "forget_bias_1.125",
    "forget_bias_1.375",
    "forget_bias_1.625",
    "layers_minus+forget_bias_down",
    "layers_minus+forget_bias_up",
    "layers_plus+forget_bias_down",
    "layers_plus+forget_bias_up",
    "kernel_3+forget_bias_down",
    "kernel_3+forget_bias_up",
    "kernel_7+forget_bias_down",
    "kernel_7+forget_bias_up",
]

def apply_mutation(cfg, mutation):
    c = dict(cfg)

    for m in mutation.split("+"):
        if m == "layers_minus":
            c["num_layers"] -= 1
        elif m == "layers_plus":
            c["num_layers"] += 1
        elif m == "kernel_3":
            c["conv_kernel"] = 3
        elif m == "kernel_7":
            c["conv_kernel"] = 7
        elif m == "forget_bias_down":
            c["forget_bias"] = 1.25
        elif m == "forget_bias_up":
            c["forget_bias"] = 1.75
        elif m == "forget_bias_1.125":
            c["forget_bias"] = 1.125
        elif m == "forget_bias_1.375":
            c["forget_bias"] = 1.375
        elif m == "forget_bias_1.625":
            c["forget_bias"] = 1.625

    return c

def valid(cfg):
    return (
        cfg["num_layers"] >= 2
        and cfg["d_model"] >= 128
        and cfg["d_state"] == cfg["d_model"]
        and cfg["conv_kernel"] in (3, 5, 7)
        and 0.5 <= cfg["forget_bias"] <= 2.5
    )

candidates = []
seen = set()

# Always include the four Pareto parents as the evolutionary source.
for parent in pareto:
    pid = parent["candidate"]
    base = all_gen2[pid]
    base_cfg = base["config"]

    for mutation in mutations:
        cfg = apply_mutation(base_cfg, mutation)

        if not valid(cfg):
            continue

        key = (
            cfg["num_layers"],
            cfg["d_model"],
            cfg["d_state"],
            cfg["conv_kernel"],
            cfg["forget_bias"],
        )

        if key in seen:
            continue

        seen.add(key)

        candidates.append({
            "candidate_id": f"GEN3-{len(candidates)+1:03d}",
            "parent": pid,
            "parent_mutation": base["mutation"],
            "mutation": mutation,
            "config": cfg,
        })

out = os.path.join(GEN3_DIR, "candidates.json")
with open(out, "w") as f:
    json.dump(candidates, f, indent=2)

manifest = {
    "generation": "GEN3",
    "parents": [x["candidate"] for x in pareto],
    "parent_count": len(pareto),
    "candidate_count": len(candidates),
    "mutation_count": len(mutations),
}

with open(os.path.join(GEN3_DIR, "parent_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print("=" * 70)
print("NOVA-EVO GEN 3 MUTATION ENGINE")
print("=" * 70)
print(f"Parents    : {len(pareto)}")
print(f"Mutations  : {len(mutations)}")
print(f"Candidates : {len(candidates)}")
print(f"Saved      : {out}")
print("=" * 70)

for c in candidates:
    cfg = c["config"]
    print(
        f'{c["candidate_id"]:10s} '
        f'parent={c["parent"]:10s} '
        f'mutation={c["mutation"]:28s} '
        f'layers={cfg["num_layers"]} '
        f'd_model={cfg["d_model"]} '
        f'kernel={cfg["conv_kernel"]} '
        f'forget={cfg["forget_bias"]}'
    )
