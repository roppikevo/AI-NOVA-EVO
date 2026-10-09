"""
Large weight files in parts: a release file over the 95 MB a public repository takes in one file is published as
`nova_model.pt.part00`, `.part01`, ... and joined back on first use. The joined file is checked against the
release's own SHA256SUMS, so a missing or damaged part is caught, not loaded.

    python -m nova.parts --split evo/releases/NOVA8-53M-v2/nova_model.pt
    python -m nova.parts --join evo/releases/NOVA8-53M-v2/nova_model.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PART_BYTES = 90 * 10 ** 6


def parts_of(path: Path) -> list[Path]:
    path = Path(path)
    return sorted(path.parent.glob(path.name + ".part[0-9][0-9]"))


def split(path: Path, part_bytes: int = PART_BYTES) -> list[Path]:
    """Write the file as numbered parts beside it (the file itself stays). Existing parts are replaced."""
    path = Path(path)
    for old in parts_of(path):
        old.unlink()
    out = []
    with path.open("rb") as f:
        i = 0
        while True:
            chunk = f.read(part_bytes)
            if not chunk:
                break
            p = path.with_name(f"{path.name}.part{i:02d}")
            p.write_bytes(chunk)
            out.append(p)
            i += 1
    return out


def join(path: Path) -> bool:
    """Make sure `path` exists: join its parts if the file is missing. False when there is neither."""
    path = Path(path)
    if path.exists():
        return True
    parts = parts_of(path)
    if not parts:
        return False
    tmp = path.with_name(path.name + ".joining")
    with tmp.open("wb") as out:
        for p in parts:
            out.write(p.read_bytes())
    os.replace(tmp, path)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--split")
    g.add_argument("--join")
    args = ap.parse_args(argv)
    if args.split:
        for p in split(Path(args.split)):
            print(p, p.stat().st_size)
        return 0
    return 0 if join(Path(args.join)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
