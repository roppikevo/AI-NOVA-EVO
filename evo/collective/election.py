"""
Leader election by verified results.

A node cannot simply claim a score. Every node writes challenges from its OWN
held-out data (a context and four continuations, only one is real) and keeps
the answers. Every other node answers them; the coordinator counts what was
right. The leader is the node with the best accuracy on the challenges of the
OTHERS (plus, optionally, verified code-exam results). The election is repeated
after every evolution round, so a better core replaces the leader.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def make_challenges(seqs: np.ndarray, n: int, rng: np.random.Generator, ctx: int = 64,
                    opt_len: int = 16, n_opts: int = 4) -> list[dict]:
    """Challenges from held-out token sequences; item["answer"] stays with the challenger."""
    items = []
    if len(seqs) < n_opts or seqs.shape[1] < ctx + opt_len:
        return items
    for _ in range(n):
        rows = rng.choice(len(seqs), size=n_opts, replace=False)
        true_row = seqs[rows[0]]
        options = [true_row[ctx:ctx + opt_len].tolist()] + [seqs[r][ctx:ctx + opt_len].tolist() for r in rows[1:]]
        order = rng.permutation(n_opts)
        items.append({"context": true_row[:ctx].tolist(), "options": [options[i] for i in order],
                      "answer": int(np.where(order == 0)[0][0])})
    return items


def public(items: list[dict]) -> list[dict]:
    """What is sent to other nodes: without the answer."""
    return [{"context": it["context"], "options": it["options"]} for it in items]


def elect(nodes: list, challenges: dict[str, list[dict]], code_scores: dict[str, float] | None = None,
          code_weight: float = 0.25, eligible: set[str] | None = None) -> dict[str, Any]:
    """nodes: objects with .node_id and .answer(); challenges: node_id -> its challenge items.

    eligible: only these nodes may become leader (e.g. nodes that know the Creator)."""
    matrix: dict[str, dict[str, float]] = {}
    for node in nodes:
        row = {}
        for owner, items in challenges.items():
            if not items:
                continue
            got = node.answer(public(items))
            row[owner] = round(sum(int(a == it["answer"]) for a, it in zip(got, items)) / len(items), 4)
        matrix[node.node_id] = row
    scores = {}
    for nid, row in matrix.items():
        others = [v for owner, v in row.items() if owner != nid] or list(row.values())
        s = sum(others) / max(1, len(others))
        if code_scores and nid in code_scores:
            s = (1 - code_weight) * s + code_weight * code_scores[nid]
        scores[nid] = round(s, 4)
    ranking = sorted(scores, key=lambda k: (-scores[k], k))
    allowed = [k for k in ranking if eligible is None or k in eligible] or ranking
    return {"leader": allowed[0], "scores": scores, "ranking": ranking, "matrix": matrix,
            "not_eligible": [k for k in ranking if k not in allowed]}
