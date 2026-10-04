import json
from pathlib import Path

ROOT = Path("/opt/ai/work/nova-evo")
INPUT = ROOT / "evo/gen1/robust_results.json"
OUTPUT = ROOT / "evo/gen1/pareto_front.json"


def dominates(a, b):
    """
    Všetky ciele minimalizujeme:
      loss  ↓
      params ↓
      time/step ↓  <=> speed ↑
      vram  ↓
    """
    a_values = (
        a["val_loss_mean"],
        a["params"],
        1.0 / a["speed_mean"],
        a["vram_mean_gb"],
    )

    b_values = (
        b["val_loss_mean"],
        b["params"],
        1.0 / b["speed_mean"],
        b["vram_mean_gb"],
    )

    no_worse = all(x <= y for x, y in zip(a_values, b_values))
    strictly_better = any(x < y for x, y in zip(a_values, b_values))

    return no_worse and strictly_better


with open(INPUT) as f:
    data = json.load(f)

items = []

for candidate_id, result in data["summary"].items():
    row = dict(result)
    row["candidate"] = candidate_id
    items.append(row)

pareto = []

for candidate in items:
    dominated = False

    for other in items:
        if other["candidate"] == candidate["candidate"]:
            continue

        if dominates(other, candidate):
            dominated = True
            break

    if not dominated:
        pareto.append(candidate)

pareto.sort(key=lambda x: x["val_loss_mean"])

output = {
    "objectives": {
        "val_loss": "minimize",
        "params": "minimize",
        "step_time": "minimize",
        "vram": "minimize",
    },
    "source": str(INPUT),
    "candidates": len(items),
    "pareto_count": len(pareto),
    "pareto_front": pareto,
}

with open(OUTPUT, "w") as f:
    json.dump(output, f, indent=2)

print("=" * 78)
print("NOVA-EVO GEN 1 — PARETO FRONT")
print("=" * 78)
print()
print(f"Candidates : {len(items)}")
print(f"Pareto     : {len(pareto)}")
print()

for r in pareto:
    print(
        f"{r['candidate']:10s} "
        f"loss={r['val_loss_mean']:.6f} "
        f"params={r['params']:,} "
        f"speed={r['speed_mean']:.3f} "
        f"VRAM={r['vram_mean_gb']:.3f} GB"
    )

print()
print(f"Saved: {OUTPUT}")
