"""
choose(): NOVA picks between given options instead of writing text.

Inspired by "decision" models: scoring options only *reads* them
(~30x cheaper on CPU than generating), and even a small model can
choose between a few options long before it can write well.

    r = choose(model, tok, "Ktorého učiteľa použiť na Rust?",
               ["Devstral", "Qwen", "OxCoder"], lang="sk")
    r -> {"choice": "Devstral", "probabilities": {...}, "confident": True}

Scores are length-normalised log-probabilities turned into a
distribution with softmax. If the best probability is below
`min_confidence`, choice is None ("neviem") and the caller should ask
a teacher or the Creator.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from nova.generate import continuation_logprob
from nova.tokenizer import NovaTokenizer


@torch.no_grad()
def choose(
    model: torch.nn.Module,
    tok: NovaTokenizer,
    question: str,
    options: list[str],
    lang: str = "sk",
    min_confidence: float = 0.5,
    temperature: float = 1.0,
) -> dict[str, Any]:
    if len(options) < 2:
        raise ValueError("choose() needs at least two options")
    prompt = question.rstrip() + "\n"
    scores = [
        continuation_logprob(model, tok, prompt, " " + opt, lang) / temperature
        for opt in options
    ]
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    probs = {opt: e / total for opt, e in zip(options, exps)}
    best = max(probs, key=probs.get)
    confident = probs[best] >= min_confidence
    return {
        "choice": best if confident else None,
        "best": best,
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
        "confident": confident,
        "scores": [round(s, 4) for s in scores],
    }
