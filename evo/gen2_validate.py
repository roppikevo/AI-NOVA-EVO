from __future__ import annotations

import json
import sys
import torch
from pathlib import Path

ROOT = Path("/opt/ai/work/nova-evo")
CANDIDATES = ROOT / "evo/gen2/candidates.json"
OUT = ROOT / "evo/gen2/validation.json"

sys.path.insert(0, str(ROOT))

from nova.config import NovaConfig
from nova.model import NovaModel


with open(CANDIDATES, "r", encoding="utf-8") as f:
    candidates = json.load(f)


device = "cuda" if torch.cuda.is_available() else "cpu"

results = []

print("=" * 78)
print("NOVA-EVO GEN 2 — TECHNICAL VALIDATION")
print("=" * 78)
print(f"Device: {device}")
print(f"Candidates: {len(candidates)}")
print()

for c in candidates:
    cid = c["candidate_id"]
    cfg = c["config"]

    try:
        torch.cuda.empty_cache()

        config = NovaConfig(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            d_state=cfg["d_state"],
            num_layers=cfg["num_layers"],
            conv_kernel=cfg["conv_kernel"],
            forget_bias=cfg["forget_bias"],
            learnable_initial_state=cfg["learnable_initial_state"],
        )

        model = NovaModel(config).to(device)
        model.train()

        params = sum(p.numel() for p in model.parameters())

        x = torch.randint(
            0,
            cfg["vocab_size"],
            (2, 32),
            dtype=torch.long,
            device=device,
        )

        targets = torch.randint(
            0,
            cfg["vocab_size"],
            (2, 32),
            dtype=torch.long,
            device=device,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            weight_decay=0.01,
        )

        optimizer.zero_grad(set_to_none=True)

        output = model(x)

        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        if logits.ndim != 3:
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss: {loss.item()}")

        loss.backward()

        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()
            vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            torch.cuda.reset_peak_memory_stats()
        else:
            vram_gb = 0.0

        results.append({
            "candidate_id": cid,
            "parent": c["parent"],
            "mutation": c["mutation"],
            "valid": True,
            "params": params,
            "vram_gb": round(vram_gb, 4),
            "loss": float(loss.item()),
            "error": None,
        })

        print(
            f"{cid:10s} "
            f"OK "
            f"params={params:,} "
            f"loss={loss.item():.6f} "
            f"VRAM={vram_gb:.3f} GB"
        )

        del model, optimizer, x, targets, output, logits, loss

    except Exception as e:
        results.append({
            "candidate_id": cid,
            "parent": c["parent"],
            "mutation": c["mutation"],
            "valid": False,
            "params": None,
            "vram_gb": None,
            "loss": None,
            "error": repr(e),
        })

        print(
            f"{cid:10s} "
            f"INVALID "
            f"{repr(e)}"
        )

        if "model" in locals():
            del model

        torch.cuda.empty_cache()


with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)


valid = [r for r in results if r["valid"]]
invalid = [r for r in results if not r["valid"]]

print()
print("=" * 78)
print("VALIDATION SUMMARY")
print("=" * 78)
print(f"Valid   : {len(valid)}")
print(f"Invalid : {len(invalid)}")
print()
print(f"Saved: {OUT}")
