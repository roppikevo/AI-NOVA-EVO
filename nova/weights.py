"""Which weights are NOVA's current deployable model (active_weights.json)."""

from __future__ import annotations

import json
import time
from pathlib import Path

ACTIVE_WEIGHTS = Path("evo/learning/active_weights.json")


def current_weights(best_known: dict, path: Path | None = None) -> str:
    """
    Active weights if they belong to the same core AND dataset as the
    evolution baseline; otherwise the baseline checkpoint (a promoted core
    with a new architecture cannot load older weights).
    """
    path = path or ACTIVE_WEIGHTS
    fallback = best_known.get("rebaseline_checkpoint") or best_known.get("checkpoint")
    if path.exists():
        try:
            aw = json.loads(path.read_text())
        except json.JSONDecodeError:
            return fallback
        if (aw.get("dataset") == best_known.get("dataset")
                and aw.get("candidate") == best_known.get("candidate")
                and Path(aw.get("checkpoint", "")).exists()):
            return aw["checkpoint"]
    return fallback


def set_active(checkpoint: str, best_known: dict, source: str, path: Path | None = None) -> None:
    path = path or ACTIVE_WEIGHTS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "checkpoint": checkpoint,
        "dataset": best_known.get("dataset"),
        "candidate": best_known.get("candidate"),
        "source": source,
        "updated": time.time(),
    }, indent=2))
