"""
Recall at a distance: does a core still know a fact it read N tokens ago?

A short fact with a random four-digit number is read ("Tajný kód je 4831."), then N tokens of ordinary held-out
text, then the question ("Tajný kód je") - and we look at how likely the core finds the right number. Every sample
is read twice, once with code A and once with code B, the same filler after both:

    recall = ( log p(A | read A) - log p(A | read B) + log p(B | read B) - log p(B | read A) ) / 2      (nats)

0 means the core no longer tells the two apart (it guesses the same either way); more is better. It measures what
survived in the state, not the general quality of the language: a core can be good at both or at only one.

The filler is read once per distance step with the carried state, so a 32 000-token distance costs as much as
reading 32 000 tokens, and the state keeps its size. Only cores that carry a state (generation 8, NOVA-Q) can be
measured this way.

    python -m nova.recall --models a.pt,b.pt [--distances 0,128,512,2048,8192,32768] [--samples 8]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

DISTANCES = (0, 128, 512, 2048, 8192, 32768)
FACTS = {"sk": ("Tajný kód je {}.", "Tajný kód je"), "en": ("The secret code is {}.", "The secret code is"),
         "cs": ("Tajný kód je {}.", "Tajný kód je"), "pl": ("Tajny kod to {}.", "Tajny kod to")}


def _logits(out) -> torch.Tensor:
    return out[0] if isinstance(out, (tuple, list)) else out


def answer_ids(tok, question: str, code: str) -> tuple[list[int], list[int]]:
    """Token ids of the question and of the answer that follows it (the code), as the tokenizer splits them."""
    q = tok.encode(question)
    full = tok.encode(f"{question} {code}")
    if full[:len(q)] != q:              # the tokenizer merged across the boundary: ask without the last question token
        q = q[:-1]
        if full[:len(q)] != q:
            raise ValueError(f"cannot split '{question} {code}' into question and answer")
    return q, full[len(q):]


@torch.no_grad()
def answer_logprob(model, states, question: list[int], answer: list[int], device: str) -> torch.Tensor:
    """log p(answer | everything read so far + question), for every row of the carried states. [rows]"""
    rows = _rows(states)
    ids = torch.tensor([question + answer[:-1]] * rows, device=device)
    logits = _logits(model(ids, states)).float()[:, len(question) - 1:]
    target = torch.tensor(answer, device=device).expand(rows, -1)
    return -F.cross_entropy(logits.transpose(1, 2), target, reduction="none").sum(dim=1)


def _rows(states) -> int:
    if torch.is_tensor(states):
        return states.shape[0] if states.ndim else 1
    for s in states:
        n = _rows(s)
        if n:
            return n
    return 0


@torch.no_grad()
def recall(model, tok, filler: np.ndarray, distances=DISTANCES, samples: int = 8, lang: str = "sk",
           device: str = "cpu", seed: int = 0) -> dict[str, Any]:
    if not getattr(model, "carries_state", False):
        raise ValueError("this core does not carry a state - recall at a distance is measured on recurrent cores")
    distances = sorted(set(int(d) for d in distances))
    if len(filler) < distances[-1] + 1:
        raise ValueError(f"filler has {len(filler)} tokens, {distances[-1]} are needed")
    fact, question = FACTS[lang]
    rng = random.Random(seed)
    per: dict[int, list[float]] = {d: [] for d in distances}
    model.eval()
    tries = 0
    while len(per[distances[0]]) < samples and tries < samples * 20:
        tries += 1
        a, b = rng.sample(range(1000, 10000), 2)
        rows = [[tok.lang_id(lang)] + tok.encode(fact.format(c)) for c in (a, b)]
        if len(rows[0]) != len(rows[1]):           # both facts must have the same number of tokens (one batch)
            continue
        _, states = model(torch.tensor(rows, device=device))
        qa, ans_a = answer_ids(tok, question, str(a))
        qb, ans_b = answer_ids(tok, question, str(b))
        start = rng.randrange(0, max(1, len(filler) - distances[-1]))
        done = 0
        for d in distances:
            if d > done:
                chunk = torch.tensor(filler[start + done:start + d], dtype=torch.long, device=device).expand(2, -1)
                _, states = model(chunk, states)
                done = d
            la = answer_logprob(model, states, qa, ans_a, device)      # [read A, read B]
            lb = answer_logprob(model, states, qb, ans_b, device)
            per[d].append(float((la[0] - la[1] + lb[1] - lb[0]) / 2))
    return {"lang": lang, "samples": len(per[distances[0]]),
            "recall_nats": {str(d): round(float(np.mean(v)), 4) for d, v in per.items() if v},
            "spread": {str(d): round(float(np.std(v)), 4) for d, v in per.items() if v}}


def filler_tokens(directory: Path, lang: str = "sk", need: int = 40000, start_row: int = 6000) -> np.ndarray:
    """Held-out running text of one language: consecutive 128-token rows of the web validation shards, joined.
    The first rows (used by the judge and the long-context test) are skipped."""
    files = sorted(directory.glob(f"{lang}-*.npy"))
    parts, total = [], 0
    for n, f in enumerate(files):
        a = np.load(f, mmap_mode="r")
        a = np.asarray(a[start_row if n == 0 else 0:]).reshape(-1)
        parts.append(a[: need - total])
        total += len(parts[-1])
        if total >= need:
            break
    if total < need:
        raise SystemExit(f"{directory}: only {total} tokens of {lang}, {need} needed")
    return np.concatenate(parts).astype(np.int64)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="comma list of checkpoints or release files")
    ap.add_argument("--tokenizer", default="", help="tokenizer.json (default: the one next to the first model, else the newest release)")
    ap.add_argument("--web-dir", default="data/bulk_val_v1")
    ap.add_argument("--lang", default="sk")
    ap.add_argument("--distances", default=",".join(map(str, DISTANCES)))
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="evo/learning/recall.json")
    args = ap.parse_args(argv)
    from nova import demo
    from nova.generate import load_checkpoint_model
    from nova.tokenizer import NovaTokenizer

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    paths = [Path(p) for p in args.models.split(",") if p]
    tpath = Path(args.tokenizer) if args.tokenizer else next((p.parent / "tokenizer.json" for p in paths if (p.parent / "tokenizer.json").exists()),
                                                           demo.releases()[-1] / "tokenizer.json")
    tok = NovaTokenizer.load(tpath)
    distances = [int(x) for x in args.distances.split(",")]
    filler = filler_tokens(Path(args.web_dir), args.lang, distances[-1] + 5000)
    results = {}
    for p in paths:
        model, _ = load_checkpoint_model(p)
        model = model.float().to(device).eval()
        r = recall(model, tok, filler, distances, args.samples, args.lang, device)
        results[str(p)] = r
        print(p, json.dumps(r["recall_nats"]), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update(results)
    out.write_text(json.dumps(old, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
