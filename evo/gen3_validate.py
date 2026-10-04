import json
import os
import sys
import torch

ROOT = "/opt/ai/work/nova-evo"
sys.path.insert(0, ROOT)

from nova.config import NovaConfig
from nova.model import NovaModel

with open(os.path.join(ROOT, "evo/gen3/candidates.json")) as f:
    candidates = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("NOVA-EVO GEN 3 TECHNICAL VALIDATION")
print("=" * 80)
print(f"Device     : {device}")
print(f"Candidates : {len(candidates)}")
print("=" * 80)

results = []

for i, candidate in enumerate(candidates, 1):
    cid = candidate["candidate_id"]
    cfg_data = candidate["config"]

    try:
        cfg = NovaConfig(
            vocab_size=cfg_data["vocab_size"],
            max_seq_len=cfg_data["max_seq_len"],
            d_model=cfg_data["d_model"],
            d_state=cfg_data["d_state"],
            num_layers=cfg_data["num_layers"],
            conv_kernel=cfg_data["conv_kernel"],
            forget_bias=cfg_data["forget_bias"],
            learnable_initial_state=cfg_data["learnable_initial_state"],
            pad_token_id=cfg_data["pad_token_id"],
            dtype=cfg_data["dtype"],
        )

        model = NovaModel(cfg).to(device)

        params = sum(p.numel() for p in model.parameters())

        x = torch.randint(
            0, cfg.vocab_size,
            (2, 64),
            device=device,
            dtype=torch.long,
        )
        targets = torch.randint(
            0, cfg.vocab_size,
            (2, 64),
            device=device,
            dtype=torch.long,
        )

        model.train()
        logits, _ = model(x)

        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            targets.reshape(-1),
        )

        if loss.ndim != 0:
            raise RuntimeError(f"Unexpected loss shape: {loss.shape}")

        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise RuntimeError(f"Non-finite gradient: {name}")

        results.append({
            "candidate_id": cid,
            "valid": True,
            "params": params,
            "loss": float(loss.detach().cpu()),
            "error": None,
        })

        print(
            f"[{i:02d}/{len(candidates)}] "
            f"{cid:10s} OK "
            f"params={params:,} "
            f"loss={float(loss.detach().cpu()):.6f}"
        )

        del model, x, targets, logits, loss
        if device == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        results.append({
            "candidate_id": cid,
            "valid": False,
            "params": None,
            "loss": None,
            "error": repr(e),
        })

        print(
            f"[{i:02d}/{len(candidates)}] "
            f"{cid:10s} INVALID "
            f"{repr(e)}"
        )

out_path = os.path.join(ROOT, "evo/gen3/validation.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

valid = sum(x["valid"] for x in results)
invalid = len(results) - valid

print("=" * 80)
print(f"VALID   : {valid}")
print(f"INVALID : {invalid}")
print(f"SAVED   : {out_path}")
print("=" * 80)
