"""
Calibration: does NOVA know how sure it is?

For every next-token prediction we take the model's confidence (max
probability) and whether that top token was correct. A well calibrated
model that says 0.9 is right ~90 % of the time.

    ECE  (expected calibration error)  lower is better, 0 = perfect
    accuracy                            top-1 next-token accuracy
    mean_confidence                     average max probability

Cheap: runs on a limited number of validation batches, no training.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def calibration_stats(
    confidences: torch.Tensor,
    correct: torch.Tensor,
    bins: int = 15,
) -> dict[str, Any]:
    confidences = confidences.float().flatten()
    correct = correct.float().flatten()
    n = confidences.numel()
    if n == 0:
        return {}
    edges = torch.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        k = int(mask.sum())
        if k:
            gap = (confidences[mask].mean() - correct[mask].mean()).abs()
            ece += (k / n) * float(gap)
    return {
        "ece": round(ece, 4),
        "accuracy": round(float(correct.mean()), 4),
        "mean_confidence": round(float(confidences.mean()), 4),
        "tokens": n,
    }


@torch.no_grad()
def measure_calibration(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device | str,
    max_batches: int = 40,
    batch_size: int = 32,
    pad_id: int | None = None,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    confs, hits = [], []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        probs = torch.softmax(logits.float(), dim=-1)
        conf, pred = probs.max(dim=-1)
        keep = torch.ones_like(y, dtype=torch.bool) if pad_id is None else y != pad_id
        confs.append(conf[keep].cpu())
        hits.append((pred == y)[keep].cpu())
    if not confs:
        return {}
    return calibration_stats(torch.cat(confs), torch.cat(hits))
