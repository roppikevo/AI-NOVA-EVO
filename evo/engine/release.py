"""
Freeze NOVA as a named, self-contained release (a backup that never changes).

    python -m evo.engine.release --name NOVA-10M-v1 [--candidates a.pt,b.pt]

  * measures every candidate checkpoint on the scoreboard (language, code,
    creator) and picks the one with the lowest validation loss that still
    knows its Creator
  * writes evo/releases/<name>/:
        nova_model.pt        fp16 weights (inference; this one goes to GitHub)
        nova_model_fp32.pt   full-precision weights (to continue training / clone)
        tokenizer.json, blocks_scan.py, model_scan.py, config.py
        MODEL.json           config, scores, source checkpoint, git commit
        SHA256SUMS
  * copies the folder to ~/nova-evo-backups/releases/<name>/ (+ .tar.gz),
    read-only
  * makes the chosen checkpoint the active weights, so "deployed" == "released"

A release is never overwritten: an existing name is an error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import torch

RELEASES = Path("evo/releases")
BACKUPS = Path.home() / "nova-evo-backups" / "releases"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick(rows: list[dict], min_creator: float = -0.5) -> dict:
    """Lowest validation loss among candidates that still know the Creator.

    With a held-out web validation ("web_val") the mean of both losses decides."""
    ok = [r for r in rows if r.get("creator", -99) >= min_creator] or rows
    return min(ok, key=lambda r: (r["val"] + r["web_val"]) / 2 if r.get("web_val") is not None else r["val"])


def _parameters(config: dict | None, sd: dict) -> int:
    """Parameters of the model: the table of tokens is saved under two names (reading and writing) but is one table."""
    try:
        from evo.engine.architecture_factory import build_model

        model = build_model(config)
        model.load_state_dict(sd)                       # the count is trusted only for the model these weights fit
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:
        return int(sum(v.numel() for v in sd.values()))


def _core(ckpt: dict) -> str | None:
    """Name of the core in MODEL.json. A generation-8 model is named by its pattern: the checkpoint's own label
    is inherited from the weights training started with and may name an older core."""
    cfg = ckpt.get("config") or {}
    if cfg.get("arch") == "nova8":
        return f"nova8 {cfg.get('pattern', '')}".strip()
    return ckpt.get("candidate")


def write_release(out: Path, ckpt: dict, source: str, row: dict, tokenizer: Path, extra: dict) -> None:
    out.mkdir(parents=True, exist_ok=False)
    meta = {k: v for k, v in ckpt.items() if k not in ("optimizer", "model_state_dict")}
    sd = ckpt["model_state_dict"]
    torch.save({**meta, "model_state_dict": {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()},
                "dtype": "float16 (load with .float() for training)"}, out / "nova_model.pt")
    torch.save({**meta, "model_state_dict": {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}},
               out / "nova_model_fp32.pt")
    shutil.copy2(tokenizer, out / "tokenizer.json")
    gen8 = (ckpt.get("config") or {}).get("arch") == "nova8"
    for src in (("nova/core8.py",) if gen8 else ("nova/blocks_scan.py", "nova/model_scan.py", "nova/config.py")):
        if Path(src).exists():
            shutil.copy2(src, out / Path(src).name)
    info = {"name": out.name, "frozen": time.strftime("%Y-%m-%d %H:%M:%S"), "source_checkpoint": source,
            "config": ckpt.get("config"), "core": _core(ckpt),
            "parameters": _parameters(ckpt.get("config"), sd), "scores": row, **extra}
    (out / "MODEL.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    sums = [f"{sha256(p)}  {p.name}" for p in sorted(out.iterdir()) if p.is_file()]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--candidates", default="", help="comma list of checkpoints; default: the active weights")
    ap.add_argument("--web-val-dir", default="", help="held-out web validation parts; the mean of both losses decides")
    ap.add_argument("--only-candidates", action="store_true",
                    help="release of another model line (different size): do not compare with or change the active weights")
    args = ap.parse_args(argv)

    from evo.engine.long_train import load_tokens
    from evo.engine.scoreboard import measure
    from nova.generate import load_checkpoint_model
    from nova.gpu_guard import ensure_vram
    from nova.tokenizer import NovaTokenizer
    from nova.weights import current_weights, set_active

    out = RELEASES / args.name
    if out.exists():
        print(f"release {args.name} already exists - a release is never overwritten")
        return 2
    best = json.loads(Path("evo/engine/evo_state.json").read_text(encoding="utf-8"))["best_known"]
    dataset = Path(best["dataset"])
    tok = NovaTokenizer.load(dataset / "tokenizer.json")
    val = load_tokens(dataset / "val.txt")
    cands = [c for c in args.candidates.split(",") if c and Path(c).exists()] or []
    active = current_weights(best)
    if active not in cands and not args.only_candidates:
        cands.append(active)
    if not cands:
        print("no candidate checkpoint found")
        return 2
    ensure_vram()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    web = None
    if args.web_val_dir:
        from evo.engine.train_line import web_validation

        web = web_validation(Path(args.web_val_dir))
    rows = []
    for c in cands:
        model, _ = load_checkpoint_model(c)
        row = measure(model, tok, val, device)
        if web is not None:
            from evo.engine.long_train import evaluate

            with torch.no_grad():
                row["web_val"] = round(evaluate(model.to(device), web, device), 4)
        row["checkpoint"] = c
        rows.append(row)
        print(f"candidate {c}\n   val {row['val']}  web {row.get('web_val')}  code {row['code']['solved']}/{row['code']['tasks']}  "
              f"creator {row['creator']}  languages {row['lang']}")
    chosen = pick(rows)
    print(f"CHOSEN: {chosen['checkpoint']} (val {chosen['val']})")
    ckpt = torch.load(chosen["checkpoint"], map_location="cpu", weights_only=False)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    write_release(out, ckpt, chosen["checkpoint"], chosen, dataset / "tokenizer.json",
                  {"dataset": str(dataset), "git_commit": commit,
                   "other_candidates": [{k: r[k] for k in ("checkpoint", "val", "creator")} for r in rows]})
    if chosen["checkpoint"] != active and not args.only_candidates:
        set_active(chosen["checkpoint"], best, f"release:{args.name}")
        print("active weights now point to the released checkpoint")
    BACKUPS.mkdir(parents=True, exist_ok=True)
    dst = BACKUPS / args.name
    if not dst.exists():
        shutil.copytree(out, dst)
        with tarfile.open(BACKUPS / f"{args.name}.tar.gz", "w:gz") as tar:
            tar.add(out, arcname=args.name)
        for p in list(dst.iterdir()) + [BACKUPS / f"{args.name}.tar.gz"]:
            p.chmod(0o444)
    print(f"release written: {out}  backup: {dst}  archive: {BACKUPS / (args.name + '.tar.gz')}")
    print((out / "SHA256SUMS").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
