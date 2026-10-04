"""
Experiment with the core on your own text - self-contained, no project state needed.

    # continue a released core on your text (CPU works, a GPU is used when there is one)
    python examples/train_on_text.py --text my_notes.txt --lang en --steps 300

    # an untrained core of any size, same tokenizer
    python examples/train_on_text.py --text my_notes.txt --lang en --steps 2000 \
        --fresh '{"d_model": 256, "d_state": 256, "num_layers": 4}'

The text is cut into 128-token sequences, the last tenth is held out and never trained on;
the held-out loss before and after is printed. The result (--out, default my_core.pt) loads with
nova.generate.load_checkpoint_model and writes with nova.generate.generate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nova import demo  # noqa: E402
from nova.tokenizer import EOS_ID, NovaTokenizer, pack_sequences  # noqa: E402

BASE = {"vocab_size": 16384, "d_model": 384, "d_state": 384, "num_layers": 6, "conv_kernel": 5, "forget_bias": 1.125,
        "learnable_initial_state": False}


def sequences(text: str, tok: NovaTokenizer, lang: str, seq_len: int = 128) -> torch.Tensor:
    """Paragraphs -> documents (<lang> ... <eos>) -> fixed-length rows."""
    docs = [[tok.lang_id(lang)] + tok.encode(p.strip()) + [EOS_ID] for p in text.split("\n\n") if p.strip()]
    rows = list(pack_sequences(docs, seq_len))
    if len(rows) < 10:
        raise SystemExit(f"too little text: {len(rows)} sequences of {seq_len} tokens (at least 10 are needed)")
    return torch.tensor(rows, dtype=torch.long)


def loss_on(model: torch.nn.Module, rows: torch.Tensor, device: str, batch: int = 16) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            x = rows[i:i + batch].to(device)
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            total += float(F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), x[:, 1:].reshape(-1), reduction="sum"))
            count += x[:, 1:].numel()
    return total / count


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", required=True, help="a UTF-8 text file; blank lines separate documents")
    ap.add_argument("--lang", default="en", choices=["sk", "cs", "pl", "en", "py", "rs"])
    ap.add_argument("--release", default="", help="core to continue from (default: the newest in evo/releases/)")
    ap.add_argument("--fresh", default="", help="JSON with core settings: start from an untrained core instead")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.0, help="default 5e-5 when continuing, 5e-4 for a fresh core")
    ap.add_argument("--out", default="my_core.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from evo.engine.architecture_factory import build_model

    torch.manual_seed(args.seed)
    found = demo.releases()
    release = demo.RELEASES / args.release if args.release else (found[-1] if found else None)
    if release is None or not (release / "tokenizer.json").exists():
        raise SystemExit("no release in evo/releases/ - its tokenizer is needed even for a fresh core")
    if args.fresh:
        config = {**BASE, **json.loads(args.fresh)}
        model, tok = build_model(config), NovaTokenizer.load(release / "tokenizer.json")
        start = "untrained core"
    else:
        model, tok, ckpt = demo.load(release)
        config, start = ckpt["config"], release.name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.float().to(device)
    rows = sequences(Path(args.text).read_text(encoding="utf-8"), tok, args.lang)
    rows = rows[torch.randperm(len(rows), generator=torch.Generator().manual_seed(args.seed))]
    held = max(2, len(rows) // 10)
    train, val = rows[held:], rows[:held]
    params = sum(p.numel() for p in model.parameters())
    before = loss_on(model, val, device)
    print(f"{start}: {params / 1e6:.1f} M parameters on {device}; {len(train)} training and {len(val)} held-out sequences")
    print(f"held-out loss before: {before:.4f}")

    lr = args.lr or (5e-4 if args.fresh else 5e-5)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    began = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        x = train[torch.randint(len(train), (min(args.batch, len(train)),))].to(device)
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 10) == 0 or step == args.steps:
            print(f"step {step:>6}  train loss {float(loss):.4f}  {time.time() - began:.0f} s")
    after = loss_on(model, val, device)
    print(f"held-out loss after:  {after:.4f}  ({(after - before) / before * 100:+.1f} %)")
    torch.save({"config": config, "model_state_dict": model.state_dict(), "steps": args.steps, "started_from": start,
                "held_out_loss": after}, args.out)
    tok.save(Path(args.out).with_suffix(".tokenizer.json"))
    print(f"saved {args.out} (+ tokenizer); try:\n  python -c \"from nova.generate import generate, load_checkpoint_model; "
          f"from nova.tokenizer import NovaTokenizer; m, _ = load_checkpoint_model('{args.out}'); "
          f"print(generate(m, NovaTokenizer.load('{Path(args.out).with_suffix('.tokenizer.json')}'), 'The', '{args.lang}'))\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
