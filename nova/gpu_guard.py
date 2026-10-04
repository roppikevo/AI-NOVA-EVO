"""
GPU guard: make room in VRAM before NOVA trains.

The llama.cpp router keeps the last teacher model resident. A 14-17 GB
teacher occupies most of the RTX 4060's 8 GB, and NOVA's training then
fails with CUDA out-of-memory. Before training we ask the router to
unload its models (POST /models/unload) and wait until enough VRAM is
free. Teachers are reloaded automatically on their next request.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request

ROUTER = "http://127.0.0.1:8081"


def free_vram_mb() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        return int(out[0]) if out else None
    except Exception:
        return None


def loaded_models(router: str = ROUTER) -> list[str]:
    try:
        with urllib.request.urlopen(f"{router}/v1/models", timeout=10) as r:
            data = json.load(r)
        return [m["id"] for m in data.get("data", [])
                if (m.get("status") or {}).get("value") == "loaded"]
    except Exception:
        return []


def unload(model: str, router: str = ROUTER) -> bool:
    req = urllib.request.Request(
        f"{router}/models/unload",
        data=json.dumps({"model": model}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def ensure_vram(need_mb: int = 2500, wait_s: int = 90, router: str = ROUTER, log=print) -> dict:
    before = free_vram_mb()
    report = {"free_before_mb": before, "unloaded": [], "free_after_mb": before}
    if before is None or before >= need_mb:
        return report
    for m in loaded_models(router):
        if unload(m, router):
            report["unloaded"].append(m)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        free = free_vram_mb()
        report["free_after_mb"] = free
        if free is not None and free >= need_mb:
            break
        time.sleep(3)
    log(f"[gpu_guard] free {before} -> {report['free_after_mb']} MB, unloaded {report['unloaded']}")
    return report
