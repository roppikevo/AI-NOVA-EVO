"""
Train a NEW model line (another size of the NOVA core) from scratch, in resumable segments.

    python -m evo.engine.train_line --name NOVA-24M \
        --config-override '{"d_model": 640, "d_state": 640, "num_layers": 8}' \
        --total-steps 300000 --max-hours 5.4

  * the line lives in evo/lines/<name>/ (checkpoints, progress, state.json); the deployed
    model, the active weights and the releases are never touched
  * same tokenizer, data mix and validation data as the current core, so numbers are comparable
  * one schedule (warm-up + cosine) over --total-steps; every call continues where the last
    one stopped (time limit, STOP_TRAINING file or a restart of the server)
  * state.json says where the line stands: steps done, best validation loss, finished or not

When a line is finished it can be frozen with evo.engine.release (--only-candidates) and
cloned for the collective.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

LINES = Path("evo/lines")


def line_state(name: str) -> dict:
    p = LINES / name / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def web_validation(directory: Path, per_lang: int = 800) -> np.ndarray | None:
    """Held-out web text: the first rows of every language part in `directory` (equal share per language)."""
    rows = []
    for lang in ("sk", "cs", "pl", "en"):
        files = sorted(directory.glob(f"{lang}-*.npy"))
        if files:
            rows.append(np.asarray(np.load(files[0], mmap_mode="r")[:per_lang]))
    return np.concatenate(rows) if rows else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config-override", default="{}")
    ap.add_argument("--total-steps", type=int, required=True)
    ap.add_argument("--max-hours", type=float, default=5.4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--bulk-dir", default="data/bulk_v1")
    ap.add_argument("--bulk-frac", type=float, default=0.8)
    ap.add_argument("--code-frac", type=float, default=0.05)
    ap.add_argument("--extra-dirs", default="data/teacher_v1,data/web_v1,data/self_v2")
    ap.add_argument("--val-extra-dir", default="", help="held-out web validation (parts never used for training)")
    ap.add_argument("--compile", action="store_true", help="compile the training forward pass (faster for the generation-8 core)")
    ap.add_argument("--carry", type=int, default=0,
                    help="read the web corpus as running text: this many consecutive rows per stream with the state carried over")
    ap.add_argument("--carry-share", type=float, default=0.0,
                    help="with --carry: this share of every batch is running text, the rest is read with a fresh state")
    args = ap.parse_args(argv)

    from evo.corpus.bulk_web import load_bulk
    from evo.engine import long_train as lt
    from evo.engine.architecture_factory import build_model
    from nova.gpu_guard import ensure_vram

    out = LINES / args.name
    out.mkdir(parents=True, exist_ok=True)
    state = json.loads(Path("evo/engine/evo_state.json").read_text(encoding="utf-8"))
    dataset = Path(state["best_known"]["dataset"])
    st = line_state(args.name)
    config = st.get("config") or {**state["primary_parent_config"], **json.loads(args.config_override)}
    if st.get("finished"):
        print(f"line {args.name} is already finished: {json.dumps(st)}")
        return 0

    ensure_vram()
    model = build_model(config)
    params = int(sum(p.numel() for p in model.parameters()))
    last = out / f"{args.name}-last.pt"
    resume = torch.load(last, map_location="cpu", weights_only=False) if last.exists() else None
    print(f"line {args.name}: {params / 1e6:.1f} M parameters, config {config}, "
          f"{'resuming at step ' + str(resume['step']) if resume else 'starting from scratch'}")

    t0 = time.time()
    train = lt.load_tokens(dataset / "train.txt")
    val = lt.load_tokens(dataset / "val.txt")
    dirs = [Path(d) for d in args.extra_dirs.split(",") if d]
    extra = lt.extra_sequences(dataset, dirs) if dirs else np.zeros((0, train.shape[1]), dtype=np.int32)
    if len(extra):
        train = np.concatenate([train, extra])
    bulk = load_bulk(Path(args.bulk_dir)) if Path(args.bulk_dir).exists() else None
    code = lt.code_practice_sequences(dataset, train.shape[1]) if args.code_frac > 0 else None
    print(f"data: dataset {train.shape}, bulk {0 if bulk is None else len(bulk)} sequences "
          f"({0 if bulk is None else len(bulk) * 128 / 1e6:.0f} M tokens), code {0 if code is None else len(code)} "
          f"({time.time() - t0:.0f}s)")

    extra_val = web_validation(Path(args.val_extra_dir)) if args.val_extra_dir else None
    if extra_val is not None:
        print(f"held-out web validation: {extra_val.shape}")
    meta = {"tag": args.name, "candidate": args.name, "config": config, "dataset": str(dataset), "kind": "long_train",
            "line": args.name}
    rep = lt.long_train(model, train, val, steps=args.total_steps, out_dir=out, meta=meta,
                        batch_size=args.batch_size, lr=args.lr, warmup=args.warmup, eval_every=args.eval_every,
                        save_every=args.save_every, patience=10 ** 6, max_hours=args.max_hours, resume=resume,
                        bulk=bulk, bulk_frac=args.bulk_frac, code=code, code_frac=args.code_frac,
                        extra_val=extra_val, progress=out / "progress.jsonl", stop_file=lt.STOP_FILE,
                        compile_model=args.compile, carry=args.carry, carry_share=args.carry_share)
    finished = rep["steps_done"] >= args.total_steps
    st = {"name": args.name, "config": config, "params": params, "total_steps": args.total_steps,
          "steps_done": rep["steps_done"], "best_val": rep["best_val"], "finished": finished,
          "select_metric": rep.get("select_metric"), "best_parts": rep.get("best_parts"),
          "stop_reason": rep["stop_reason"], "best_checkpoint": rep["best_checkpoint"],
          "last_checkpoint": rep["last_checkpoint"], "tokens_seen": rep["steps_done"] * args.batch_size * (train.shape[1] - 1),
          "hours_total": round(st.get("hours_total", 0.0) + rep["hours"], 2), "updated": time.strftime("%Y-%m-%d %H:%M")}
    (out / "state.json").write_text(json.dumps(st, indent=1))
    print("=== LINE STATE ===")
    print(json.dumps(st, indent=1))
    if rep["best_checkpoint"]:
        from nova.generate import continuation_logprob, generate, load_checkpoint_model
        from nova.tokenizer import NovaTokenizer

        m, _ = load_checkpoint_model(rep["best_checkpoint"])
        tok = NovaTokenizer.load(dataset / "tokenizer.json")
        torch.manual_seed(0)
        print("=== SAMPLES (CPU, temperature 0.7) ===")
        for lang, prompt in [("sk", "Môj tvorca je"), ("sk", "Bratislava je"), ("en", "The capital of"), ("py", "def main(")]:
            print(f"[{lang}] {generate(m, tok, prompt, lang, temperature=0.7)!r}")
        print(f"creator recall log-prob: {continuation_logprob(m, tok, 'Môj tvorca je', ' roppik', 'sk'):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
