"""
Build the first real NOVA text corpus: data/text_v1.

    python -m evo.corpus.build_text_v1 --mb-per-source 10

Output (compatible with nova.data.TokenSequenceDataset, seq_len 128):

    data/text_v1/train.txt  val.txt  test.txt   one packed sequence per line
    data/text_v1/tokenizer.json                 byte-level BPE, 16384 tokens
    data/text_v1/manifest.json                  sources, licenses, sizes, stats

Splits are made per document (hash-based, deterministic), so no document
appears in two splits. The tokenizer is trained on the train split only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Iterator

from evo.corpus import sources
from evo.corpus.sources import Document
from nova.tokenizer import DEFAULT_VOCAB_SIZE, NovaTokenizer, pack_sequences

WIKI_LANGS = ("sk", "cs", "pl", "en")


def encode_doc(tok: NovaTokenizer, d: Document) -> list[int]:
    if d.reasoning and tok.has_think:
        return tok.encode_reasoned(d.prompt, d.reasoning, d.text, d.lang)
    if d.prompt:
        return tok.encode_document(d.prompt + "\n" + d.text, d.lang)
    return tok.encode_document(d.text, d.lang)


def split_of(doc: Document) -> str:
    bucket = int(doc.key[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "val"
    return "test"


def interleave(groups: dict[str, list[Document]]) -> Iterator[Document]:
    """Round-robin across sources so the tokenizer sees a balanced mix."""
    iters = {k: iter(v) for k, v in groups.items()}
    while iters:
        for k in list(iters):
            try:
                yield next(iters[k])
            except StopIteration:
                del iters[k]


def collect(
    fetchers: dict[str, Callable[[], Iterable[Document]]],
    log: Callable[[str], None] = print,
) -> dict[str, list[Document]]:
    seen: set[str] = set()
    groups: dict[str, list[Document]] = {}
    for name, fetch in fetchers.items():
        start = time.time()
        docs = []
        try:
            for doc in fetch():
                if doc.key in seen:
                    continue
                seen.add(doc.key)
                docs.append(doc)
        except Exception as exc:  # a failed source must not kill the build
            log(f"[{name}] FAILED: {type(exc).__name__}: {exc}")
        size = sum(len(d.text.encode("utf-8")) for d in docs)
        log(f"[{name}] {len(docs)} docs, {size / 1e6:.1f} MB, {time.time() - start:.0f}s")
        if docs:
            groups[name] = docs
    return groups


def build(
    groups: dict[str, list[Document]],
    out_dir: Path,
    seq_len: int = 128,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    seed: int = 20260930,
    log: Callable[[str], None] = print,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, dict[str, list[Document]]] = {
        s: defaultdict(list) for s in ("train", "val", "test")
    }
    for name, docs in groups.items():
        for doc in docs:
            splits[split_of(doc)][name].append(doc)

    log("training tokenizer ...")
    tok = NovaTokenizer.train(
        (" ".join(filter(None, (d.prompt, d.reasoning, d.text)))
         for d in interleave(splits["train"])),
        vocab_size=vocab_size,
    )
    tok.save(out_dir / "tokenizer.json")

    rng = random.Random(seed)
    stats: dict = {"splits": {}, "bytes_per_token": {}}

    for split, by_source in splits.items():
        docs = [d for ds in by_source.values() for d in ds]
        rng.shuffle(docs)
        encoded = (encode_doc(tok, d) for d in docs)

        path = out_dir / f"{split}.txt"
        n_seq = 0
        with path.open("w", encoding="utf-8") as f:
            for seq in pack_sequences(encoded, seq_len):
                f.write(" ".join(map(str, seq)))
                f.write("\n")
                n_seq += 1

        stats["splits"][split] = {
            "documents": len(docs),
            "sequences": n_seq,
            "tokens": n_seq * seq_len,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        log(f"{split}: {len(docs)} docs -> {n_seq} sequences")

    # tokenizer efficiency per source, measured on validation documents
    for name, docs in splits["val"].items():
        sample = docs[:200]
        n_bytes = sum(len(d.text.encode("utf-8")) for d in sample)
        n_tok = sum(len(tok.encode(d.text)) for d in sample)
        if n_tok:
            stats["bytes_per_token"][name] = round(n_bytes / n_tok, 3)

    manifest = {
        "name": out_dir.name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seq_len": seq_len,
        "vocab_size": tok.vocab_size,
        "tokenizer": "tokenizer.json (byte-level BPE)",
        "special_tokens": "0 <pad> 1 <unk> 2 <bos> 3 <eos> 4 <sk> 5 <cs> 6 <pl> 7 <en> 8 <py> 9 <rs>",
        "document_format": "<lang> text <eos>, packed without padding",
        "split_rule": "sha256(text) bucket: 90 train / 5 val / 5 test",
        "sources": {
            name: {
                "documents": len(docs),
                "bytes": sum(len(d.text.encode("utf-8")) for d in docs),
                "license": sources.LICENSES[docs[0].source],
                "origin": docs[0].source,
            }
            for name, docs in groups.items()
        },
        **stats,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def default_fetchers(
    mb: float,
    cache: Path,
    identity_docs: int = 0,
    teacher_dir: Path | None = None,
    self_model_state: Path | None = None,
    web_dir: Path | None = None,
) -> dict[str, Callable[[], Iterable[Document]]]:
    max_bytes = int(mb * 1_000_000)
    fetchers: dict[str, Callable[[], Iterable[Document]]] = {
        f"wiki-{lang}": (lambda lang=lang: sources.wikipedia(lang, max_bytes, cache / "hf"))
        for lang in WIKI_LANGS
    }
    fetchers["python"] = lambda: sources.python_stdlib(max_bytes)
    fetchers["rust"] = lambda: sources.rust_src(max_bytes, cache / "rust")
    if identity_docs:
        from evo.corpus.identity import identity_documents

        fetchers["identity"] = lambda: identity_documents(identity_docs)
    if teacher_dir and Path(teacher_dir).exists():
        fetchers["teacher"] = lambda: sources.teacher_jsonl(Path(teacher_dir), max_bytes)
    if web_dir and Path(web_dir).exists():
        fetchers["web"] = lambda: sources.web_jsonl(Path(web_dir), max_bytes)
    if self_model_state and Path(self_model_state).exists():
        from evo.corpus.self_model import self_model_documents

        fetchers["self-model"] = lambda: self_model_documents(Path(self_model_state))
    return fetchers


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/text_v1")
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "nova-corpus"))
    ap.add_argument("--mb-per-source", type=float, default=10.0)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    ap.add_argument("--identity-docs", type=int, default=0,
                    help="add N identity documents (NOVA + Creator name)")
    ap.add_argument("--teacher-dir", default=None, help="e.g. data/teacher_v1")
    ap.add_argument("--self-model-state", default=None,
                    help="evo/engine/evo_state.json -> documents about NOVA itself")
    ap.add_argument("--web-dir", default=None, help="e.g. data/web_v1 (web learner output)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if (out / "manifest.json").exists():
        print(f"{out} already exists - refusing to overwrite", file=sys.stderr)
        return 2

    groups = collect(default_fetchers(
        args.mb_per_source, Path(args.cache), args.identity_docs,
        args.teacher_dir, args.self_model_state, args.web_dir,
    ))
    if not groups:
        print("no sources available", file=sys.stderr)
        return 1

    manifest = build(groups, out, args.seq_len, args.vocab_size)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
