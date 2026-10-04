"""
Collective answer and collective learning.

    mixture_nll()   loss of the collective prediction  p = sum_i w_i * p_i
    fit_weights()   the leader decides how much each node counts (EM on calibration data)
    average_states() merge the weights of several nodes into one model (FedAvg)
    solve_together() nodes propose code, tests verify, the collective keeps what passes
"""

from __future__ import annotations

from typing import Any

import numpy as np


def mixture_nll(logps: np.ndarray, weights: np.ndarray | None = None) -> float:
    """logps [nodes, tokens] = log p_i(true token). Returns mean NLL of the weighted mixture."""
    logps = np.asarray(logps, dtype=np.float64)
    n = logps.shape[0]
    w = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, dtype=np.float64)
    a = logps + np.log(np.maximum(w, 1e-12))[:, None]
    m = a.max(axis=0)
    return float(-(m + np.log(np.exp(a - m).sum(axis=0))).mean())


def fit_weights(logps: np.ndarray, iters: int = 50) -> np.ndarray:
    """Mixture weights that minimise NLL on calibration tokens (EM)."""
    logps = np.asarray(logps, dtype=np.float64)
    n = logps.shape[0]
    w = np.full(n, 1.0 / n)
    for _ in range(iters):
        a = logps + np.log(np.maximum(w, 1e-12))[:, None]
        a -= a.max(axis=0)
        r = np.exp(a)
        r /= r.sum(axis=0)
        w = r.mean(axis=1)
    return w


def gated_nll(logps: np.ndarray, groups: np.ndarray, weights_by_group: dict[int, np.ndarray],
              default: np.ndarray) -> float:
    """Mixture where the weights depend on the token's group (e.g. its language)."""
    logps = np.asarray(logps, dtype=np.float64)
    total, count = 0.0, 0
    for g in np.unique(groups):
        m = groups == g
        total += mixture_nll(logps[:, m], weights_by_group.get(int(g), default)) * int(m.sum())
        count += int(m.sum())
    return total / max(count, 1)


def average_states(states: list[dict], weights: list[float] | None = None) -> dict:
    """Weighted average of model state dicts (integer buffers are taken from the first)."""
    import torch

    w = [1.0 / len(states)] * len(states) if weights is None else [x / sum(weights) for x in weights]
    out = {}
    for k, v in states[0].items():
        if torch.is_floating_point(v):
            out[k] = sum(wi * s[k].float() for wi, s in zip(w, states))
        else:
            out[k] = v.clone()
    return out


def solve_together(nodes: list, task_keys: list[str], samples: int = 0) -> dict[str, Any]:
    """Every node proposes bodies; unit tests verify; a task is solved if any proposal passes."""
    from evo.learning.code_school import run_tests
    from evo.learning.code_tasks import task_bank

    tasks = {t.key: t for t in task_bank()}
    proposals = {n.node_id: n.solve(task_keys, samples) for n in nodes}
    solved_by: dict[str, list[str]] = {}
    per_node = {nid: 0 for nid in proposals}
    for k in task_keys:
        winners = []
        for nid, bodies in proposals.items():
            ok = any(run_tests(tasks[k], b)[0] for b in bodies.get(k, []))
            if ok:
                winners.append(nid)
                per_node[nid] += 1
        solved_by[k] = winners
    solved = sum(1 for v in solved_by.values() if v)
    return {"solved": solved, "tasks": len(task_keys), "per_node": per_node,
            "only_one_node": sum(1 for v in solved_by.values() if len(v) == 1),
            "by_level": _by_level(solved_by, tasks)}


def _by_level(solved_by: dict[str, list[str]], tasks: dict) -> dict[str, str]:
    lv: dict[int, list[bool]] = {}
    for k, w in solved_by.items():
        lv.setdefault(tasks[k].level, []).append(bool(w))
    return {str(l): f"{sum(v)}/{len(v)}" for l, v in sorted(lv.items())}
