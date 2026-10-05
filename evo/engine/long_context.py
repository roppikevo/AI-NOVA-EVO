"""
Running text: what does a model get from context beyond the 128 tokens it was trained on, and what does it cost?

Held-out web text is stored as consecutive 128-token rows. For a sample of positions in every row (after the
first one) the loss of predicting that token is measured in several ways of reading:

    row        only the tokens of the same row before it (how all losses in this project are measured)
    window     the 127 tokens before it, whatever row they are in (a sliding window: the most a model trained on
               128-token rows was ever shown)
    two_rows   the whole row before + the row so far, in one pass (twice the training length)
    carried    recurrent cores only: the state carried through the whole text, row after row

"carried" costs a recurrent core nothing: its state has a fixed size. A transformer that wants the "window"
quality on running text has to keep the last 127 keys and values of every layer.

    python -m evo.engine.long_context --models a.pt,b.pt [--rows 120] [--every 8]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT = Path("evo/learning/long_context.json")


def is_transformer(model: torch.nn.Module) -> bool:
    return bool(getattr(model, "blocks", None)) and hasattr(model.blocks[0], "qkv") and not getattr(model, "carries_state", False)


def _logits(out) -> torch.Tensor:
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def measure(model: torch.nn.Module, rows: np.ndarray, every: int = 8, batch: int = 64) -> dict[str, np.ndarray]:
    """Per-position losses [rows - 1, positions] for every way of reading. `rows` are consecutive pieces of text."""
    model = model.float().eval()
    seq = rows.shape[1]
    pos = np.arange(every // 2, seq, every)                       # positions inside a row whose token is predicted
    x = torch.from_numpy(np.ascontiguousarray(rows)).long()
    n = len(x) - 1
    out: dict[str, np.ndarray] = {}

    def pick(logits: torch.Tensor, target: torch.Tensor, offset: int = 0) -> np.ndarray:
        # logits at index (offset + p - 1) predict the token at position p of the row
        idx = torch.from_numpy(pos - 1 + offset)
        return F.cross_entropy(logits[:, idx].reshape(-1, logits.shape[-1]).float(), target[:, torch.from_numpy(pos)].reshape(-1),
                               reduction="none").reshape(len(target), len(pos)).numpy()

    row = []
    for i in range(1, len(x), batch):
        b = x[i:i + batch]
        row.append(pick(_logits(model(b[:, :-1])), b))
    out["row"] = np.concatenate(row)

    two = []
    for i in range(1, len(x), batch):
        both = torch.cat([x[i - 1:i - 1 + batch][:len(x[i:i + batch])], x[i:i + batch]], dim=1)
        two.append(pick(_logits(model(both[:, :-1])), x[i:i + batch], offset=seq))
    out["two_rows"] = np.concatenate(two)

    flat = x.reshape(-1)
    ends = (torch.arange(1, len(x))[:, None] * seq + torch.from_numpy(pos)[None, :]).reshape(-1)     # index of the predicted token
    win = []
    for i in range(0, len(ends), batch):
        e = ends[i:i + batch]
        ctx = torch.stack([flat[j - (seq - 1):j] for j in e.tolist()])
        logits = _logits(model(ctx))[:, -1]
        win.append(F.cross_entropy(logits.float(), flat[e], reduction="none").numpy())
    out["window"] = np.concatenate(win).reshape(n, len(pos))

    if not is_transformer(model):
        carried, states = [], None
        _, states = model(x[0:1])
        for i in range(1, len(x)):
            logits, states = model(x[i:i + 1], states)
            carried.append(pick(logits, x[i:i + 1]))
        out["carried"] = np.concatenate(carried)
    return out


def summary(losses: dict[str, np.ndarray]) -> dict[str, float]:
    base = float(losses["row"].mean())
    s = {"row": round(base, 4)}
    for k, v in losses.items():
        if k != "row":
            s[k] = round(float(v.mean()), 4)
            s[f"{k}_vs_row_percent"] = round(100 * (float(v.mean()) - base) / base, 2)
    return s


def web_rows(directory: Path, per_lang: int, start: int = 3000) -> np.ndarray:
    parts = []
    for lang in ("sk", "cs", "pl", "en"):
        files = sorted(glob.glob(str(directory / f"{lang}-*.npy")))
        if files:
            parts.append(np.asarray(np.load(files[0], mmap_mode="r")[start:start + per_lang]))
    return np.concatenate(parts)


def text(results: dict[str, dict]) -> str:
    L = ["RUNNING TEXT: loss of predicting the same tokens with different amounts of context (lower = better)"]
    for name, s in results.items():
        line = f"  {name:<22} own row {s['row']:.4f} | sliding window of 127 tokens {s['window']:.4f} ({s['window_vs_row_percent']:+.2f} %)"
        line += f" | two rows in one pass {s['two_rows']:.4f} ({s['two_rows_vs_row_percent']:+.2f} %)"
        if "carried" in s:
            line += f" | state carried through the text {s['carried']:.4f} ({s['carried_vs_row_percent']:+.2f} %)"
        L.append(line)
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="comma list of checkpoints")
    ap.add_argument("--web-dir", default="data/bulk_val_v1")
    ap.add_argument("--rows", type=int, default=120, help="rows per language")
    ap.add_argument("--every", type=int, default=8, help="measure every n-th position of a row")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)

    from nova.generate import load_checkpoint_model

    torch.set_num_threads(args.threads)
    rows = web_rows(Path(args.web_dir), args.rows)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for path in [p for p in args.models.split(",") if p]:
        model, _ = load_checkpoint_model(path)
        results[Path(path).stem] = summary(measure(model, rows, args.every))
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, indent=1))
        print(text({Path(path).stem: results[Path(path).stem]}), flush=True)
    print(text(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
