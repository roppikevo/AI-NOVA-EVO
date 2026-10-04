"""
Export NOVA's current deployable model (weights only, no optimizer state)
to evo/core_evolution/active/DEPLOY/ for GitHub and local backups.

    python -m evo.engine.export_model
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from nova.weights import current_weights

OUT = Path("evo/core_evolution/active/DEPLOY")


def main() -> int:
    state = json.loads(Path("evo/engine/evo_state.json").read_text(encoding="utf-8"))
    best = state["best_known"]
    src = current_weights(best)
    ck = torch.load(src, map_location="cpu", weights_only=False)
    OUT.mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in ck.items() if k not in ("optimizer",)}
    keep["model_state_dict"] = {k: v.half() if v.is_floating_point() else v
                                for k, v in ck["model_state_dict"].items()}
    keep["exported_from"] = src
    keep["exported_at"] = time.time()
    keep["dtype"] = "float16 (load with .float() for training)"
    torch.save(keep, OUT / "nova_model.pt")
    tok_src = Path(best["dataset"]) / "tokenizer.json"
    if tok_src.exists():
        (OUT / "tokenizer.json").write_bytes(tok_src.read_bytes())
    info = {"source_checkpoint": src, "core": best.get("candidate"), "dataset": best.get("dataset"),
            "config": state.get("primary_parent_config"), "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "size_mb": round((OUT / "nova_model.pt").stat().st_size / 1e6, 1)}
    (OUT / "MODEL.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
