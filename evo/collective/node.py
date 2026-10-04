"""
One NOVA node of the collective: a model + tokenizer that can

    info()      say who it is
    score()     give log p(next token) for token sequences       -> collective answer (mixture)
    answer()    answer multiple-choice challenges                 -> verified election score
    solve()     write code for code-school tasks                  -> collective task solving
    next()      top-k next tokens for a context                   -> collective generation

Everything is plain lists/dicts so it can travel through the HTTP API unchanged.
"""

from __future__ import annotations

from typing import Any

import torch


class Node:
    def __init__(self, node_id: str, model: torch.nn.Module, tok, specialty: str = "",
                 weights: str = "", device: str = "cpu") -> None:
        self.node_id, self.tok, self.specialty, self.weights, self.device = node_id, tok, specialty, weights, device
        self.model = model.to(device).float().eval()

    def info(self) -> dict[str, Any]:
        return {"node": self.node_id, "specialty": self.specialty, "weights": self.weights,
                "parameters": int(sum(p.numel() for p in self.model.parameters())), "device": self.device}

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return (out[0] if isinstance(out, (tuple, list)) else out).float()

    @torch.no_grad()
    def score(self, seqs: list[list[int]]) -> list[list[float]]:
        """log p(token[t+1] | tokens[..t]) for every position of every sequence (equal lengths)."""
        b = torch.tensor(seqs, dtype=torch.long, device=self.device)
        logp = torch.log_softmax(self._logits(b[:, :-1]), dim=-1)
        picked = logp.gather(-1, b[:, 1:].unsqueeze(-1)).squeeze(-1)
        return [[round(v, 4) for v in row] for row in picked.cpu().tolist()]

    @torch.no_grad()
    def answer(self, items: list[dict]) -> list[int]:
        """Multiple choice: which of `options` (token lists) really continues `context`?"""
        out = []
        for it in items:
            ctx, best, best_i = it["context"], -1e30, 0
            for i, opt in enumerate(it["options"]):
                b = torch.tensor([ctx + opt], dtype=torch.long, device=self.device)
                logp = torch.log_softmax(self._logits(b[:, :-1]), dim=-1)
                tgt = b[0, 1:]
                s = float(logp[0, torch.arange(len(tgt)), tgt][len(ctx) - 1:].mean())
                if s > best:
                    best, best_i = s, i
            out.append(best_i)
        return out

    def solve(self, task_keys: list[str], samples: int = 0, temperature: float = 0.7) -> dict[str, list[str]]:
        """Function bodies for code-school tasks: the greedy one first, then `samples` sampled ones."""
        from evo.learning.code_school import write_body
        from evo.learning.code_tasks import task_bank

        tasks = {t.key: t for t in task_bank()}
        cpu_model = self.model if self.device == "cpu" else self.model.to("cpu")
        res = {}
        for k in task_keys:
            if k not in tasks:
                continue
            bodies = [write_body(cpu_model, self.tok, tasks[k])]
            bodies += [write_body(cpu_model, self.tok, tasks[k], temperature=temperature) for _ in range(samples)]
            res[k] = bodies
        if self.device != "cpu":
            self.model.to(self.device)
        return res

    @torch.no_grad()
    def next(self, context: list[int], k: int = 40) -> list[list[float]]:
        """Top-k next tokens as [token_id, log_prob]."""
        b = torch.tensor([context[-128:]], dtype=torch.long, device=self.device)
        logp = torch.log_softmax(self._logits(b)[0, -1], dim=-1)
        top = torch.topk(logp, k)
        return [[int(i), round(float(v), 4)] for v, i in zip(top.values, top.indices)]


def load_node(node_id: str, weights: str, tokenizer: str, specialty: str = "", device: str = "cpu") -> Node:
    from nova.generate import load_checkpoint_model
    from nova.tokenizer import NovaTokenizer

    model, _ = load_checkpoint_model(weights)
    return Node(node_id, model, NovaTokenizer.load(tokenizer), specialty, weights, device)
