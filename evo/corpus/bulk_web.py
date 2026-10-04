"""
Bulk web corpus for NOVA: cleaned, free, openly licensed web text from
Hugging Face, fetched in resumable batches.

    FineWeb-2   (ODC-By)  sk / cs / pl    HuggingFaceFW/fineweb-2
    FineWeb-Edu (ODC-By)  en, edu score>=3 HuggingFaceFW/fineweb-edu (sample/10BT)

Parquet files are read lazily over HTTPS (row group by row group), so only
what is needed is downloaded. Documents are filtered (length, letter ratio,
language score, edu score), de-duplicated, encoded with the tokenizer of the
CURRENT dataset (so the existing weights keep working) and packed into
int32 arrays:

    data/bulk_v1/<source>-<part>.npy     [n_sequences, 128]
    data/bulk_v1/state.json              where each source continues next time
    data/bulk_v1/manifest.json           sizes, licences, filters

long_train mixes these arrays into training (--bulk-dir); validation stays
on the dataset's own val split, so "better" keeps meaning the same thing.

    python -m evo.corpus.bulk_web --mchars sk=220,cs=150,pl=150,en=250
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

OUT = Path("data/bulk_v1")
MAX_DISK_GB = 8.0

SOURCES: dict[str, dict] = {
    "sk": {"repo": "HuggingFaceFW/fineweb-2", "glob": "data/slk_Latn/train/*.parquet", "lang": "sk",
           "license": "ODC-By 1.0 (FineWeb-2, Common Crawl ToU)", "min_lang_score": 0.7},
    "cs": {"repo": "HuggingFaceFW/fineweb-2", "glob": "data/ces_Latn/train/*.parquet", "lang": "cs",
           "license": "ODC-By 1.0 (FineWeb-2, Common Crawl ToU)", "min_lang_score": 0.7},
    "pl": {"repo": "HuggingFaceFW/fineweb-2", "glob": "data/pol_Latn/train/*.parquet", "lang": "pl",
           "license": "ODC-By 1.0 (FineWeb-2, Common Crawl ToU)", "min_lang_score": 0.7},
    "en": {"repo": "HuggingFaceFW/fineweb-edu", "glob": "sample/10BT/*.parquet", "lang": "en",
           "license": "ODC-By 1.0 (FineWeb-Edu, Common Crawl ToU)", "min_edu_score": 3},
}

MIN_CHARS, MAX_CHARS = 200, 20_000
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
_WS = re.compile(r"\s+")


# ------------------------------------------------------------------ filters

def clean(text: str) -> str:
    from evo.corpus.sources import normalize_spaces

    text = normalize_spaces(text).replace("\r", "")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:MAX_CHARS]


def keep_doc(text: str, row: dict, spec: dict) -> bool:
    if len(text) < MIN_CHARS:
        return False
    if spec.get("min_lang_score") is not None and row.get("language_score") is not None:
        if float(row["language_score"]) < spec["min_lang_score"]:
            return False
    if spec.get("min_edu_score") is not None and row.get("int_score") is not None:
        if int(row["int_score"]) < spec["min_edu_score"]:
            return False
    sample = text[:3000]
    letters = len(_LETTER.findall(sample))
    if letters / max(1, len(sample)) < 0.6:
        return False
    lines = [l for l in text.splitlines() if l.strip()]
    if lines and len(set(lines)) / len(lines) < 0.7:  # boilerplate / repeated lines
        return False
    return True


def fingerprint(text: str) -> str:
    return hashlib.sha1(_WS.sub(" ", text[:2000].lower()).encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------ reading

def _open_parquet(fs, path: str):
    """Lazy HTTPS range reads; if that fails, download the shard (proven wikipedia path)."""
    import pyarrow.parquet as pq

    try:
        fh = fs.open(path, "rb", block_size=8 * 1024 * 1024)
        return pq.ParquetFile(fh), fh
    except Exception as exc:  # pragma: no cover - network dependent
        from huggingface_hub import hf_hub_download

        _, _, repo_a, repo_b, *rest = path.split("/")
        print(f"  lazy read failed ({type(exc).__name__}) - downloading {rest[-1]}")
        local = hf_hub_download(f"{repo_a}/{repo_b}", "/".join(rest), repo_type="dataset",
                                cache_dir=str(Path.home() / ".cache" / "nova-corpus" / "hf"))
        return pq.ParquetFile(local), None


def iter_rows(spec: dict, pos: dict, log: Callable = print) -> Iterator[tuple[dict, dict]]:
    """Yield (row, position) from the parquet files of a source, resuming at pos."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{spec['repo']}/{spec['glob']}"))
    if not files:
        raise RuntimeError(f"no parquet files for {spec['repo']}/{spec['glob']}")
    fi, rg = pos.get("file", 0), pos.get("row_group", 0)
    cols = ["text", "language_score", "int_score"]
    while fi < len(files):
        pf, fh = _open_parquet(fs, files[fi])
        try:
            names = set(pf.schema_arrow.names)
            use = [c for c in cols if c in names]
            log(f"  {files[fi].split('/')[-1]}: row groups {rg}..{pf.num_row_groups - 1}")
            while rg < pf.num_row_groups:
                table = pf.read_row_group(rg, columns=use)
                rg += 1
                nxt = {"file": fi, "row_group": rg}
                for row in table.to_pylist():
                    yield row, nxt
        finally:
            if fh is not None:
                fh.close()
        fi, rg = fi + 1, 0


def fetch_source(name: str, mchars: float, tok, out: Path, state: dict, seen: set[str],
                 seq_len: int = 128, log: Callable = print) -> dict:
    from nova.tokenizer import pack_sequences

    spec = SOURCES[name]
    pos = state.setdefault("positions", {}).get(name, {})
    target = int(mchars * 1_000_000)
    chars = docs = dropped = 0
    encoded: list[list[int]] = []
    chunks: list[np.ndarray] = []
    n_tok = 0

    def flush() -> None:
        seqs = list(pack_sequences(encoded, seq_len))
        if seqs:
            chunks.append(np.array(seqs, dtype=np.int32))
        encoded.clear()

    last_pos = pos
    t0 = time.time()
    for row, nxt in iter_rows(spec, pos, log):
        text = clean(row.get("text") or "")
        if not keep_doc(text, row, spec):
            dropped += 1
            continue
        fp = fingerprint(text)
        if fp in seen:
            dropped += 1
            continue
        seen.add(fp)
        ids = tok.encode_document(text, spec["lang"])
        encoded.append(ids)
        n_tok += len(ids)
        if len(encoded) >= 20_000:
            flush()
        chars += len(text)
        docs += 1
        if chars >= target:
            last_pos = nxt  # the rest of this row group is skipped: simple and resumable
            break
        last_pos = nxt
    flush()
    n_seq = sum(len(c) for c in chunks)
    part = len(list(out.glob(f"{name}-*.npy")))
    path = out / f"{name}-{part:03d}.npy"
    if n_seq:
        np.save(path, np.concatenate(chunks))
    state["positions"][name] = last_pos
    rep = {"source": name, "docs": docs, "dropped": dropped, "mchars": round(chars / 1e6, 1),
           "sequences": n_seq, "tokens": n_seq * seq_len, "file": path.name if n_seq else None,
           "chars_per_token": round(chars / max(1, n_tok), 2),
           "minutes": round((time.time() - t0) / 60, 1)}
    log(f"[{name}] {rep}")
    return rep


def disk_gb(out: Path) -> float:
    return sum(p.stat().st_size for p in out.glob("*.npy")) / 1e9


class BulkParts:
    """Several memory-mapped [n, L] arrays that behave like one (len, shape, fancy indexing).

    The web corpus can be larger than RAM; training only ever asks for a few rows at a time."""

    def __init__(self, parts: list[np.ndarray]) -> None:
        self.parts = parts
        self.offsets = np.cumsum([0] + [len(p) for p in parts])
        self.shape = (int(self.offsets[-1]), parts[0].shape[1])
        self.dtype = parts[0].dtype

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx) -> np.ndarray:
        idx = np.atleast_1d(np.asarray(idx))
        which = np.searchsorted(self.offsets, idx, side="right") - 1
        out = np.empty((len(idx), self.shape[1]), dtype=self.parts[0].dtype)
        for k in np.unique(which):
            m = which == k
            out[m] = self.parts[k][idx[m] - self.offsets[k]]
        return out


def _parts(out: Path, pattern: str) -> BulkParts | None:
    files = sorted(out.glob(pattern))
    return BulkParts([np.load(p, mmap_mode="r") for p in files]) if files else None


def load_bulk(out: Path = OUT) -> BulkParts | None:
    """All bulk sequences as one array-like object (memory-mapped, nothing is copied to RAM)."""
    return _parts(out, "*.npy")


def load_bulk_lang(out: Path, lang: str) -> BulkParts | None:
    """Only the parts of one source (sk / cs / pl / en)."""
    return _parts(out, f"{lang}-*.npy")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mchars", default="sk=30,cs=20,pl=20,en=30",
                    help="million characters to add per source, e.g. sk=220,cs=150,pl=150,en=250")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--max-gb", type=float, default=MAX_DISK_GB, help="stop fetching when the corpus reaches this size")
    args = ap.parse_args(argv)

    from nova.tokenizer import NovaTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if disk_gb(out) >= args.max_gb:
        print(f"bulk corpus already {disk_gb(out):.1f} GB >= {args.max_gb} GB - nothing to do")
        return 0
    evo = json.loads(Path("evo/engine/evo_state.json").read_text(encoding="utf-8"))
    dataset = Path(evo["best_known"]["dataset"])
    tok = NovaTokenizer.load(dataset / "tokenizer.json")
    state_path = out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("tokenizer_dataset") not in (None, str(dataset)):
        print(f"tokenizer changed ({state['tokenizer_dataset']} -> {dataset}): bulk must be rebuilt - stopping")
        return 2
    state["tokenizer_dataset"] = str(dataset)
    seen_path = out / "seen.txt"
    seen = set(seen_path.read_text().split()) if seen_path.exists() else set()

    reports = []
    for item in args.mchars.split(","):
        name, mc = item.split("=")
        try:
            reports.append(fetch_source(name.strip(), float(mc), tok, out, state, seen))
        except Exception as exc:  # one failing source must not stop the others
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}")
        state_path.write_text(json.dumps(state, indent=2))
        seen_path.write_text("\n".join(sorted(seen)))

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"batches": []}
    manifest.update({
        "tokenizer_dataset": str(dataset), "seq_len": 128,
        "sources": {k: {"repo": v["repo"], "files": v["glob"], "license": v["license"]} for k, v in SOURCES.items()},
        "filters": f"{MIN_CHARS}-{MAX_CHARS} chars, letters>=60%, unique lines>=70%, "
                   "language_score>=0.7 (FineWeb-2), edu int_score>=3 (FineWeb-Edu), near-dup fingerprint",
        "disk_gb": round(disk_gb(out), 2),
    })
    manifest["batches"].append({"time": time.strftime("%Y-%m-%d %H:%M"), "reports": reports})
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    total = sum(np.load(p, mmap_mode="r").shape[0] for p in out.glob("*.npy")) * 128
    print(f"bulk corpus: {total / 1e6:.0f} M tokens, {disk_gb(out):.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
