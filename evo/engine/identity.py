"""
The identity test: is this core still NOVA?

Hard condition (a core that fails it is not NOVA, however good its loss):

    the state the core carries from token to token has the same size after any length of text

Recorded with it, for the ledger and for later analysis:

    state_kb                       size of the carried state
    growth_bytes_per_token         between the two longest texts read (0 for a NOVA core)
    attends_over_stored_tokens     the state holds past tokens that are read by attention (a window)
    causal                         a later token never changes an earlier prediction
    stepping_exact                 reading token by token gives the same predictions as one pass
    class                          NOVA (recurrent state only), NOVA-HYBRID (bounded state with a window of
                                   stored tokens) or NOT-NOVA (the state grows with the text)

The size of the state is not judged here: it is a cost, weighed with speed after quality.

    python -m evo.engine.identity --models a.pt,b.pt
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

LENGTHS = (128, 1024, 4096)
CHUNK = 128


def _nbytes(state) -> int:
    import torch

    if torch.is_tensor(state):
        return state.numel() * state.element_size()
    if isinstance(state, dict):
        return sum(_nbytes(v) for v in state.values())
    if isinstance(state, (list, tuple)):
        return sum(_nbytes(v) for v in state)
    return 0


def _read(model, ids, single_from: int | None = None):
    """Logits and the carried state after reading `ids` piece by piece (the last tokens one at a time)."""
    import torch

    from nova.stepper import Stepper, supports

    if supports(model):                                        # generation 7: its stepper holds the state
        st = Stepper(model)
        head = ids if single_from is None else ids[:, :single_from]
        parts = [st.prime(head)[:, None]]
        for t in range(head.shape[1], ids.shape[1]):
            parts.append(st.step(ids[:, t])[:, None])
        return torch.cat(parts, dim=1), st.state_bytes()
    state, parts, t = None, [], 0
    while t < ids.shape[1]:
        size = 1 if single_from is not None and t >= single_from else min(CHUNK, ids.shape[1] - t, (single_from or 10 ** 9) - t)
        out = model(ids[:, t:t + size], state) if state is not None else model(ids[:, t:t + size])
        logits, state = out[0], out[1]
        parts.append(logits[:, -1:])
        t += size
    return torch.cat(parts, dim=1), _nbytes(state)


def _keeps_tokens(model) -> bool:
    """Does the state hold past tokens that attention reads (a window in a generation-8 core, a transformer's cache)?"""
    from nova.core8 import WinMixer

    if getattr(model, "carries_state", False):
        return any(isinstance(mod, WinMixer) for mod in model.modules())
    return any(hasattr(mod, "qkv") for mod in model.modules())


def check(model, vocab_size: int, lengths: tuple[int, ...] = LENGTHS, seed: int = 0) -> dict:
    import torch

    m = copy.deepcopy(model).to("cpu").float().eval()
    g = torch.Generator().manual_seed(seed)
    out: dict = {}
    with torch.no_grad():
        sizes = {}
        for n in lengths:
            _, sizes[n] = _read(m, torch.randint(12, vocab_size, (1, n), generator=g))
        long_a, long_b = sorted(lengths)[-2:]
        growth = (sizes[long_b] - sizes[long_a]) / (long_b - long_a)
        out["state_bytes_after"] = {str(n): int(v) for n, v in sizes.items()}
        out["state_kb"] = round(sizes[long_b] / 1024, 1)
        out["growth_bytes_per_token"] = round(growth, 3)
        out["constant_state"] = sizes[long_a] == sizes[long_b]
        out["attends_over_stored_tokens"] = _keeps_tokens(m)

        x = torch.randint(12, vocab_size, (1, 48), generator=g)
        y = x.clone()
        y[:, 40:] = torch.randint(12, vocab_size, (1, 8), generator=g)
        full = m(x)
        full = full[0] if isinstance(full, (tuple, list)) else full
        other = m(y)
        other = other[0] if isinstance(other, (tuple, list)) else other
        scale = max(1.0, float(full.abs().max()))
        out["causal"] = bool((full[:, :40] - other[:, :40]).abs().max() < 1e-4 * scale)
        stepped, _ = _read(m, x, single_from=40)                       # 40 tokens, then 8 single ones
        out["stepping_exact"] = bool((stepped[:, -8:] - full[:, -8:]).abs().max() < 2e-3 * scale)
    out["class"] = "NOT-NOVA" if not out["constant_state"] else "NOVA-HYBRID" if out["attends_over_stored_tokens"] else "NOVA"
    out["pass"] = bool(out["constant_state"] and out["causal"])
    return out


def text(name: str, r: dict) -> str:
    sizes = ", ".join(f"{n} tokens: {v / 1024:.1f} kB" for n, v in r["state_bytes_after"].items())
    return (f"{name}: {'PASS' if r['pass'] else 'FAIL'}  class {r['class']}  state {r['state_kb']} kB  "
            f"growth {r['growth_bytes_per_token']} bytes/token  ({sizes})  "
            f"window of stored tokens: {'yes' if r['attends_over_stored_tokens'] else 'no'}  causal: {'yes' if r['causal'] else 'NO'}  "
            f"token by token = one pass: {'yes' if r['stepping_exact'] else 'NO'}")


def check_checkpoint(path: str | Path) -> dict:
    from nova.generate import load_checkpoint_model

    model, ck = load_checkpoint_model(path)
    return check(model, int((ck.get("config") or {}).get("vocab_size") or model.lm_head.weight.shape[0]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="comma list of checkpoints")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    import torch

    torch.set_num_threads(args.threads)
    ok = True
    for path in [p for p in args.models.split(",") if p]:
        r = check_checkpoint(path)
        ok = ok and r["pass"]
        print(text(Path(path).stem, r), flush=True)
        print(json.dumps(r))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
