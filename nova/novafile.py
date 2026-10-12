"""
.nova - NOVA's own model file: everything in one file (weights, tokenizer, configuration, what it has learned and how
it did on its exams), read straight from the disk (memory-mapped), with optional smaller weights (f16 / int8 / int4).
It is our counterpart of GGUF, made for NOVA's own core (llama.cpp does not know its recurrent blocks).

    python -m nova.novafile pack evo/releases/NOVA8-53M-v3 --dtype int8          # -> NOVA8-53M-v3-int8.nova
    python -m nova.novafile info NOVA8-53M-v3-int8.nova                         # what is inside, what it learned
    python -m nova.novafile check NOVA8-53M-v3-int8.nova evo/releases/NOVA8-53M-v3   # same answers as the original?
    python -m nova.novafile ask NOVA8-53M-v3-int8.nova "Bratislava je" --lang sk
    python -m nova.novafile learn NOVA8-53M-v3-int8.nova "fotosyntéza" --langs sk,en   # "uč sa <téma>"

NOVA is not a chat bot: `learn` hands the topic to the director, which collects material (teachers, the web, the
invisible browser), keeps an exam part aside, trains, has the judge compare the result with the champion and releases
a new version only when it is better - the new version is then packed into a new .nova whose history shows the topic.

File layout (little endian):
    b"NOVA" + u16 format version + u16 0 + u64 header length + header (UTF-8 JSON) + zero padding to 64 bytes
    tensor data, every tensor starts on a 64-byte boundary (offsets in the header are from the start of the data)
Header: {"format": "nova", "version", "config", "tokenizer" (its JSON), "model" (name, parameters, dtype, sha256 of
the source), "learned" (history: versions, topics, exam scores), "tensors": [{name, shape, dtype, offset, nbytes,
scale?}]}. Types: f32, f16, int8 (one f16 scale per row), int4 (two values per byte, one f16 scale per 32 values).
Small tensors and 1-D tensors (norms, biases, time scales) always stay f32.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

MAGIC = b"NOVA"
FORMAT_VERSION = 1
ALIGN = 64
GROUP = 32                 # int4: values per scale
MIN_QUANT = 4096           # tensors smaller than this stay f32
DTYPES = ("f32", "f16", "int8", "int4")


# ------------------------------------------------------------------ packing tensors

def _quantise(t: torch.Tensor, dtype: str) -> tuple[str, list[np.ndarray]]:
    """-> (stored dtype, arrays to write: data[, scales])."""
    w = t.detach().float().cpu()
    if dtype == "f32" or w.ndim < 2 or w.numel() < MIN_QUANT or not torch.is_floating_point(t):
        return "f32", [w.numpy().astype("<f4")]
    if dtype == "f16":
        return "f16", [w.numpy().astype("<f2")]
    rows = w.reshape(w.shape[0], -1)
    if dtype == "int8":
        scale = rows.abs().amax(1).clamp_min(1e-12) / 127.0
        q = torch.round(rows / scale[:, None]).clamp(-127, 127).to(torch.int8)
        return "int8", [q.numpy(), scale.numpy().astype("<f2")]
    if dtype == "int4":
        cols = rows.shape[1]
        pad = (-cols) % GROUP
        g = torch.nn.functional.pad(rows, (0, pad)).reshape(rows.shape[0], -1, GROUP)
        scale = g.abs().amax(2).clamp_min(1e-12) / 7.0
        q = (torch.round(g / scale[..., None]).clamp(-8, 7) + 8).to(torch.uint8).reshape(rows.shape[0], -1)
        packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
        return "int4", [packed.numpy(), scale.numpy().astype("<f2")]
    raise ValueError(f"unknown type {dtype} (use {', '.join(DTYPES)})")


def _restore(info: dict, raw: list[np.ndarray]) -> torch.Tensor:
    shape = info["shape"]
    if info["dtype"] in ("f32", "f16"):
        return torch.from_numpy(np.array(raw[0], dtype=np.float32)).reshape(shape)
    rows = shape[0]
    cols = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    if info["dtype"] == "int8":
        q = torch.from_numpy(np.array(raw[0], dtype=np.int8)).float().reshape(rows, cols)
        return (q * torch.from_numpy(np.array(raw[1], dtype=np.float32))[:, None]).reshape(shape)
    packed = torch.from_numpy(np.array(raw[0], dtype=np.uint8)).reshape(rows, -1)
    q = torch.empty(rows, packed.shape[1] * 2, dtype=torch.float32)
    q[:, 0::2] = (packed & 0xF).float() - 8
    q[:, 1::2] = (packed >> 4).float() - 8
    scale = torch.from_numpy(np.array(raw[1], dtype=np.float32)).reshape(rows, -1)
    w = (q.reshape(rows, -1, GROUP) * scale[..., None]).reshape(rows, -1)[:, :cols]
    return w.reshape(shape)


# ------------------------------------------------------------------ the file

def write(path: str | Path, state_dict: dict[str, torch.Tensor], config: dict, tokenizer_json: str,
          model: dict | None = None, learned: list | None = None, dtype: str = "int8") -> dict:
    if dtype not in DTYPES:
        raise ValueError(f"unknown type {dtype} (use {', '.join(DTYPES)})")
    tensors, blobs, offset = [], [], 0
    for name, t in state_dict.items():
        stored, arrays = _quantise(t, dtype)
        info: dict[str, Any] = {"name": name, "shape": list(t.shape), "dtype": stored, "parts": []}
        for a in arrays:
            b = np.ascontiguousarray(a).tobytes()
            offset += (-offset) % ALIGN
            info["parts"].append({"offset": offset, "nbytes": len(b), "np": a.dtype.str, "shape": list(a.shape)})
            blobs.append((offset, b))
            offset += len(b)
        tensors.append(info)
    header = {"format": "nova", "version": FORMAT_VERSION, "config": config, "tokenizer": tokenizer_json,
              "model": {**(model or {}), "dtype": dtype, "packed": time.strftime("%Y-%m-%d %H:%M", time.gmtime())},
              "learned": learned or [], "tensors": tensors}
    hb = json.dumps(header, ensure_ascii=False).encode("utf-8")
    pre = MAGIC + struct.pack("<HHQ", FORMAT_VERSION, 0, len(hb)) + hb
    pre += b"\0" * ((-len(pre)) % ALIGN)
    path = Path(path)
    with path.open("wb") as f:
        f.write(pre)
        pos = 0
        for off, b in blobs:
            f.write(b"\0" * (off - pos))
            f.write(b)
            pos = off + len(b)
    return {"path": str(path), "bytes": path.stat().st_size, "tensors": len(tensors)}


def read_header(path: str | Path) -> tuple[dict, int]:
    with Path(path).open("rb") as f:
        head = f.read(16)
        if head[:4] != MAGIC:
            raise ValueError(f"{path}: not a .nova file")
        version, _, n = struct.unpack("<HHQ", head[4:16])
        if version > FORMAT_VERSION:
            raise ValueError(f"{path}: format {version} is newer than this program ({FORMAT_VERSION}) - update NOVA")
        header = json.loads(f.read(n).decode("utf-8"))
    start = 16 + n
    return header, start + (-start) % ALIGN


def read_tensors(path: str | Path) -> tuple[dict, dict[str, torch.Tensor]]:
    """Header and full-precision tensors (the file is memory-mapped; smaller types are expanded on loading)."""
    header, data = read_header(path)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for info in header["tensors"]:
        raw = [np.frombuffer(mm[data + p["offset"]: data + p["offset"] + p["nbytes"]], dtype=np.dtype(p["np"]))
               .reshape(p["shape"]) for p in info["parts"]]
        out[info["name"]] = _restore(info, raw)
    return header, out


def load(path: str | Path):
    """-> (model on the CPU in eval mode, tokenizer, header)."""
    from evo.engine.architecture_factory import build_model
    from nova.tokenizer import NovaTokenizer

    header, sd = read_tensors(path)
    model = build_model(header["config"])
    model.load_state_dict(sd)
    tok_file = Path(path).with_suffix(".tokenizer.tmp.json")
    try:
        tok_file.write_text(header["tokenizer"], encoding="utf-8")
        tok = NovaTokenizer.load(tok_file)
    finally:
        tok_file.unlink(missing_ok=True)
    return model.float().eval(), tok, header


# ------------------------------------------------------------------ from a release

def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def learned_history(release: Path, topics_file: Path | None = None) -> list[dict]:
    """What this core knows it has learned: its release record and the topics the director finished for it."""
    meta = json.loads((release / "MODEL.json").read_text(encoding="utf-8")) if (release / "MODEL.json").exists() else {}
    hist = [{"version": meta.get("name", release.name), "date": meta.get("frozen"), "parent": meta.get("parent"),
             "scores": meta.get("scores") or meta.get("metrics")}]
    if topics_file and topics_file.exists():
        try:
            rows = json.loads(topics_file.read_text(encoding="utf-8"))
            rows = rows.values() if isinstance(rows, dict) else rows
            for t in rows:
                if t.get("status") == "accepted":
                    hist.append({"topic": t.get("text"), "langs": t.get("langs"), "kind": t.get("kind"),
                                 "exam": t.get("exam"), "date": t.get("finished") or t.get("updated")})
        except (ValueError, AttributeError):
            pass
    return hist


def pack_release(release: str | Path, out: str | Path | None = None, dtype: str = "int8",
                 topics_file: Path | None = Path("evo/director/topics.json")) -> dict:
    release = Path(release)
    if release.suffix == ".pt":
        ckpt_path, release = release, release.parent
    else:
        from nova.parts import join
        ckpt_path = release / "nova_model.pt"
        join(ckpt_path)
        if not ckpt_path.exists() and (release / "nova_model_fp32.pt").exists():
            ckpt_path = release / "nova_model_fp32.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = {k: v for k, v in ckpt["model_state_dict"].items()}
    meta = json.loads((release / "MODEL.json").read_text(encoding="utf-8")) if (release / "MODEL.json").exists() else {}
    name = meta.get("name", release.name)
    model = {"name": name, "params": sum(v.numel() for v in sd.values() if torch.is_floating_point(v)),
             "source_sha256": _sha256(ckpt_path), "author": "roppik", "project": "NOVA-EVO"}
    out = Path(out) if out else Path(f"{name}-{dtype}.nova")
    r = write(out, sd, ckpt["config"], (release / "tokenizer.json").read_text(encoding="utf-8"), model,
              learned_history(release, topics_file), dtype)
    return {**r, "name": name}


# ------------------------------------------------------------------ checks

@torch.no_grad()
def compare(nova_path: str | Path, release: str | Path, texts: list[str] | None = None) -> dict:
    """The packed core against the original: largest logit difference, agreement of the next-token choice and the loss
    on a few texts (so a smaller type is released only when it keeps the quality)."""
    from nova.demo import load as load_release

    a, tok, _ = load(nova_path)
    b, _, _ = load_release(Path(release))
    texts = texts or ["Bratislava je hlavné mesto Slovenska a leží na rieke Dunaj.",
                      "Praha je hlavní město České republiky.", "Warszawa jest stolicą Polski.",
                      "The river flows through the old town towards the sea.",
                      "def times_4(x):\n    return x * 4\n"]
    langs = ["sk", "cs", "pl", "en", "py"]
    diff, agree, n, la, lb = 0.0, 0, 0, 0.0, 0.0
    for t, lang in zip(texts, langs + ["en"] * len(texts)):
        ids = torch.tensor([[tok.lang_id(lang)] + tok.encode(t)])
        xa, xb = a(ids)[0], b(ids)[0]
        diff = max(diff, float((xa - xb).abs().max()))
        agree += int((xa[0, :-1].argmax(-1) == xb[0, :-1].argmax(-1)).sum())
        n += ids.shape[1] - 1
        tgt = ids[0, 1:]
        la += float(torch.nn.functional.cross_entropy(xa[0, :-1], tgt, reduction="sum"))
        lb += float(torch.nn.functional.cross_entropy(xb[0, :-1], tgt, reduction="sum"))
    return {"max_logit_diff": round(diff, 5), "same_next_token": round(agree / max(n, 1), 4),
            "loss_packed": round(la / max(n, 1), 4), "loss_original": round(lb / max(n, 1), 4)}


# ------------------------------------------------------------------ CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("release", help="release folder (evo/releases/NAME) or a .pt checkpoint")
    p.add_argument("--dtype", default="int8", choices=DTYPES)
    p.add_argument("--out")
    i = sub.add_parser("info")
    i.add_argument("file")
    c = sub.add_parser("check")
    c.add_argument("file")
    c.add_argument("release")
    a = sub.add_parser("ask")
    a.add_argument("file")
    a.add_argument("prompt")
    a.add_argument("--lang", default="sk")
    a.add_argument("--tokens", type=int, default=60)
    a.add_argument("--temperature", type=float, default=0.7)
    le = sub.add_parser("learn")
    le.add_argument("file")
    le.add_argument("topic")
    le.add_argument("--langs", default="sk,en")
    le.add_argument("--kind", default="explain", choices=["explain", "code"])
    args = ap.parse_args(argv)

    if args.cmd == "pack":
        r = pack_release(args.release, args.out, args.dtype)
        print(f"{r['path']}: {r['bytes'] / 1e6:.1f} MB, {r['tensors']} tensors ({r['name']}, {args.dtype})")
    elif args.cmd == "info":
        h, _ = read_header(args.file)
        m = h["model"]
        kinds: dict[str, int] = {}
        for t in h["tensors"]:
            kinds[t["dtype"]] = kinds.get(t["dtype"], 0) + 1
        print(f"{m.get('name')} - {m.get('params', 0) / 1e6:.1f} M parameters, type {m.get('dtype')}, packed {m.get('packed')}")
        print(f"  core: {h['config'].get('arch')} {h['config'].get('pattern', '')} width {h['config'].get('d_model')}"
              f" | tensors {kinds} | author {m.get('author')}")
        print("  learned:")
        for e in h["learned"]:
            print("   -", ", ".join(f"{k}: {v}" for k, v in e.items() if v not in (None, "", [], {})))
    elif args.cmd == "check":
        print(json.dumps(compare(args.file, args.release), ensure_ascii=False))
    elif args.cmd == "ask":
        from nova.generate import generate
        model, tok, _ = load(args.file)
        print(generate(model, tok, args.prompt.replace("\\n", "\n"), args.lang, args.tokens, args.temperature))
    else:
        from evo.learning import topics
        h, _ = read_header(args.file)
        langs = [x for x in args.langs.split(",") if x]
        t = topics.add(args.topic, langs, args.kind, path=Path("evo/director/topics.json"))
        print(f"'{t['text']}' is in the learning queue as {t['slug']} ({t['status']}) for {h['model'].get('name')}."
              f"\nThe director takes it when the card is free: material -> exam part aside -> training -> judge;"
              f"\na better core is released and packed into a new .nova that lists this topic in its history."
              f"\nprogress: python -m evo.learning.topics list")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
