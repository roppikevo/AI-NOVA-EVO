"""
Try a released NOVA core: check its files, let it write, measure its speed on this machine.

    python -m nova.demo                                   # newest release: check, two short texts, speed
    python -m nova.demo --list                            # releases in evo/releases/
    python -m nova.demo --release NOVA-24M-v1 --lang sk --prompt "Bratislava je" --tokens 80
    python -m nova.demo --lang py --temperature 0 --prompt "def times_4(x):"       # \n in a prompt is a new line
    python -m nova.demo --speed                           # tokens per second and the size of the state

Languages: sk, cs, pl, en, py, rs. Everything runs on the CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

RELEASES = Path("evo/releases")
# (language, prompt, temperature); code is written greedily after a signature and a docstring
SAMPLES = [("sk", "Bratislava je", 0.7), ("en", "The river", 0.7), ("py", 'def times_4(x):\n    """Return x multiplied by 4."""', 0.0)]


def releases(root: Path | None = None) -> list[Path]:
    """Release folders that hold weights, oldest first."""
    root = RELEASES if root is None else root
    from nova.parts import parts_of

    found = [p for p in root.iterdir() if ((p / "nova_model.pt").exists() or parts_of(p / "nova_model.pt")) and (p / "MODEL.json").exists()] \
        if root.exists() else []
    return sorted(found, key=lambda p: (json.loads((p / "MODEL.json").read_text(encoding="utf-8")).get("frozen", ""), p.name))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(release: Path) -> dict:
    """Compare the files with SHA256SUMS. Files that are not shipped (full-precision weights) are 'absent'."""
    out = {"ok": [], "bad": [], "absent": []}
    sums = release / "SHA256SUMS"
    if not sums.exists():
        return out
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        want, name = line.split(None, 1)
        f = release / name.strip()
        out["absent" if not f.exists() else "ok" if sha256(f) == want else "bad"].append(name.strip())
    return out


def load(release: Path):
    """Model and tokenizer of a release. Refuses a file whose checksum does not match."""
    from nova.generate import load_checkpoint_model
    from nova.parts import join
    from nova.tokenizer import NovaTokenizer

    join(release / "nova_model.pt")                   # a large release comes in parts: joined once, then checked below
    check = verify(release)
    if check["bad"]:
        raise SystemExit(f"{release.name}: checksum mismatch for {', '.join(check['bad'])} - download the release again")
    if "nova_model.pt" not in check["ok"]:          # no checksum to trust: read the file as plain tensors only
        os.environ.setdefault("NOVA_SAFE_LOAD", "1")
    model, ckpt = load_checkpoint_model(release / "nova_model.pt")
    return model.float().eval(), NovaTokenizer.load(release / "tokenizer.json"), ckpt


def speed(model, tok, new_tokens: int = 200, threads: int | None = None) -> dict:
    """Tokens per second when writing on the CPU, and the size of the state the core carries."""
    import torch

    from nova.stepper import Stepper, supports

    if threads:
        torch.set_num_threads(threads)
    ids = [tok.lang_id("en")] + tok.encode("The river runs through the old town and")
    with torch.no_grad():
        if getattr(model, "carries_state", False):          # generation 8: the plain call continues from its state
            from nova.core8 import state_bytes

            from nova.stepper8 import Stepper8

            st8 = Stepper8(model)
            logits = st8.prime(torch.tensor([ids]))
            start = time.perf_counter()
            for _ in range(new_tokens):
                logits = st8.step(logits.argmax(dim=-1))
            seconds = time.perf_counter() - start
            return {"tokens_per_second": round(new_tokens / seconds, 1), "state_bytes": int(state_bytes(st8.states)),
                    "threads": torch.get_num_threads()}
        if not supports(model):
            return {"tokens_per_second": None, "state_bytes": None}
        st = Stepper(model)
        logits = st.prime(torch.tensor([ids]))
        start = time.perf_counter()
        for _ in range(new_tokens):
            logits = st.step(logits[0].argmax().view(1))
        seconds = time.perf_counter() - start
    return {"tokens_per_second": round(new_tokens / seconds, 1), "state_bytes": int(st.state_bytes()),
            "threads": torch.get_num_threads()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", default="", help="name in evo/releases/ (default: the newest)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--lang", default="sk", choices=["sk", "cs", "pl", "en", "py", "rs"])
    ap.add_argument("--tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--speed", action="store_true", help="only measure the speed")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args(argv)

    found = releases()
    if args.list or not found:
        if not found:
            print(f"no release with weights in {RELEASES}/ - see INSTALL.md")
            return 0 if args.list else 2
        for p in found:
            info = json.loads((p / "MODEL.json").read_text(encoding="utf-8"))
            s = info.get("scores", {})
            print(f"{p.name:<16} frozen {info.get('frozen', '?'):<19}  loss {s.get('val', '?')}  web {s.get('web_val', '-')}  "
                  f"code exam {s.get('code', {}).get('solved', '?')}/{s.get('code', {}).get('tasks', '?')}")
        return 0
    release = RELEASES / args.release if args.release else found[-1]
    from nova.parts import join

    if not join(release / "nova_model.pt"):
        print(f"no such release: {release} (try --list)")
        return 2

    import torch

    from nova.generate import generate

    check = verify(release)
    model, tok, ckpt = load(release)
    params = sum(p.numel() for p in model.parameters())
    print(f"{release.name}: {params / 1e6:.1f} M parameters, {len(check['ok'])} files verified"
          + (f", not shipped here: {', '.join(check['absent'])}" if check["absent"] else ""))
    if args.threads:
        torch.set_num_threads(args.threads)
    if not args.speed:
        torch.manual_seed(0)
        for lang, prompt, temperature in ([(args.lang, args.prompt, args.temperature)] if args.prompt else SAMPLES):
            # a prompt must not end on a bare newline: in training text the newline and the indentation are one token
            text = generate(model, tok, prompt.replace("\\n", "\n").rstrip("\n"), lang, max_new_tokens=args.tokens, temperature=temperature)
            print(f"\n[{lang}] {text}")
        print()
    if args.speed or not args.prompt:
        s = speed(model, tok, threads=args.threads or None)
        if s["tokens_per_second"]:
            print(f"writing on this CPU: {s['tokens_per_second']} tokens/s with {s['threads']} threads; "
                  f"the state the core carries: {s['state_bytes'] / 1000:.0f} kB, the same after any length of text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
