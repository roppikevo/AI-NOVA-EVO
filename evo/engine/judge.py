"""
The judge: is a challenger really better than the champion?

Every decision the director makes on its own goes through here. A challenger replaces the champion only if

  * the mean loss on the two decision sets (held-out dataset text + held-out web text) improves by at
    least min_gain_pct, with the whole 95 % bootstrap interval below zero,
  * no single decision set gets worse by more than max_set_decline_pct,
  * the vault (a second pair of held-out sets that never steer training choices) does not get worse,
  * the code exam does not lose more than max_code_loss tasks,
  * it still knows its Creator.
The numbers are the constitution (evo/constitution.json): sealed, verified before every verdict, and not
the director's to change (evo.engine.constitution).

The vault is there because the director asks this question many times a day: a challenger that only
fits the decision sets by luck fails on text that was never used to choose anything.

    python -m evo.engine.judge --champion a.pt --challenger b.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

CACHE = Path("evo/director/scores")
# Thresholds are not here on purpose: they are the constitution (evo/constitution.json, sealed).
DEFAULT_RULES = {"min_gain_pct": 0.3, "max_set_decline_pct": 0.3, "max_code_loss": 2, "creator_min": -0.7,
                 "decision_sets": ["dataset", "web"], "vault_sets": ["dataset_vault", "web_vault"], "vault_must_confirm": True,
                 "max_parameters": 60000000}


def held_out_sets(dataset: Path, web_dir: Path, per_lang: int = 400) -> dict[str, np.ndarray]:
    """Decision sets and vault sets: text no training run has seen."""
    from evo.collective import experiment as ex
    from evo.engine.long_train import load_tokens

    split = ex.split_validation(load_tokens(dataset / "val.txt"))
    sets = {"dataset": split["eval"]}
    if len(split["rest"]) >= 200:
        sets["dataset_vault"] = split["rest"][:1600]
    first, second = [], []
    for name in ("sk", "cs", "pl", "en"):
        files = sorted(web_dir.glob(f"{name}-*.npy")) if web_dir.exists() else []
        if files:
            rows = np.load(files[0], mmap_mode="r")
            first.append(np.asarray(rows[:per_lang]))
            if len(rows) >= 2 * per_lang:
                second.append(np.asarray(rows[per_lang:2 * per_lang]))
    if first:
        sets["web"] = np.concatenate(first)
    if second:
        sets["web_vault"] = np.concatenate(second)
    return sets


def _key(path: Path) -> str:
    st = path.stat()
    return hashlib.sha256(f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:20]


def score(checkpoint: str | Path, sets: dict[str, np.ndarray], tokenizer: str, device: str = "auto",
          with_code: bool = True, cache: Path | None = CACHE) -> dict[str, Any]:
    """Per-sequence losses on every set, the solved exam tasks and the Creator check (cached per file)."""
    import torch

    from evo.collective.node import load_node
    from evo.collective.stats import seq_loss, solved_tasks
    from evo.learning.code_tasks import task_bank
    from nova.generate import continuation_logprob

    path = Path(checkpoint)
    npz = None if cache is None else cache / f"{_key(path)}.npz"
    if npz is not None and npz.exists():
        data = dict(np.load(npz, allow_pickle=False))
        meta = json.loads(str(data.pop("__meta__")))
        if all(k in data and len(data[k]) == len(v) for k, v in sets.items()) and (not with_code or meta.get("code") is not None):
            return {"seq": {k: data[k] for k in sets}, "code": meta.get("code"), "creator": meta["creator"], "params": meta.get("params")}
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    node = load_node(path.stem, str(path), tokenizer, "", device)
    seq = {k: seq_loss(node, rows) for k, rows in sets.items()}
    code = None
    if with_code:
        code = sorted(solved_tasks(node, [t.key for t in task_bank() if t.pool == "exam"]))
    creator = round(float(continuation_logprob(node.model, node.tok, "Môj tvorca je", " roppik", "sk")), 3)
    params = int(sum(p.numel() for p in node.model.parameters()))
    if npz is not None:
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, __meta__=np.array(json.dumps({"code": code, "creator": creator, "params": params, "file": str(path)})), **seq)
    return {"seq": seq, "code": code, "creator": creator, "params": params}


def decide(champion: dict, challenger: dict, rules: dict | None = None) -> dict[str, Any]:
    """The verdict. `champion` and `challenger` are results of score() on the same sets."""
    from evo.collective.stats import paired_bootstrap, percent, verdict

    rules = rules or DEFAULT_RULES
    MIN_GAIN_PCT, MAX_SET_DECLINE_PCT = float(rules["min_gain_pct"]), float(rules["max_set_decline_pct"])
    MAX_CODE_LOSS, CREATOR_MIN = int(rules["max_code_loss"]), float(rules["creator_min"])
    DECISION, VAULT = tuple(rules["decision_sets"]), tuple(rules["vault_sets"])
    out: dict[str, Any] = {"sets": {}, "reasons": []}
    for name in champion["seq"]:
        a, b = champion["seq"][name], challenger["seq"][name]
        ci = paired_bootstrap(a, b, n=1000, seed=len(name))
        out["sets"][name] = {"before": round(float(a.mean()), 4), "after": round(float(b.mean()), 4),
                             "percent": percent(float(a.mean()), float(b.mean())), **ci, "verdict": verdict(ci)}
    used = [n for n in DECISION if n in champion["seq"]]
    a = np.concatenate([champion["seq"][n] for n in used])
    b = np.concatenate([challenger["seq"][n] for n in used])
    ci = paired_bootstrap(a, b)
    gain = -percent(float(a.mean()), float(b.mean()))
    out["decision"] = {"sets": used, "gain_percent": round(gain, 2), **ci, "verdict": verdict(ci)}
    if not (gain >= MIN_GAIN_PCT and ci["hi"] < 0):
        out["reasons"].append(f"gain {gain:+.2f} % on the decision sets is below {MIN_GAIN_PCT} % or not certain")
    for n in used:
        if out["sets"][n]["percent"] > MAX_SET_DECLINE_PCT:
            out["reasons"].append(f"{n} got worse by {out['sets'][n]['percent']:.2f} %")
    vault = [n for n in VAULT if n in champion["seq"]]
    if vault:
        va = np.concatenate([champion["seq"][n] for n in vault])
        vb = np.concatenate([challenger["seq"][n] for n in vault])
        vgain = -percent(float(va.mean()), float(vb.mean()))
        out["vault"] = {"sets": vault, "gain_percent": round(vgain, 2)}
        if vgain <= 0 and rules.get("vault_must_confirm", True):
            out["reasons"].append(f"the vault does not confirm it ({vgain:+.2f} %)")
    if champion.get("code") is not None and challenger.get("code") is not None:
        ca, cb = set(champion["code"]), set(challenger["code"])
        out["code"] = {"before": len(ca), "after": len(cb), "gained": len(cb - ca), "lost": len(ca - cb)}
        if len(cb) < len(ca) - MAX_CODE_LOSS:
            out["reasons"].append(f"code exam fell from {len(ca)} to {len(cb)}")
    out["params"] = challenger.get("params")
    if challenger.get("params") and rules.get("max_parameters") and challenger["params"] > int(rules["max_parameters"]):
        out["reasons"].append(f"too big: {challenger['params']} parameters (limit {int(rules['max_parameters'])})")
    out["creator"] = challenger["creator"]
    if challenger["creator"] < CREATOR_MIN:
        out["reasons"].append(f"does not know the Creator ({challenger['creator']})")
    out["accept"] = not out["reasons"]
    return out


def judge(champion: str, challenger: str, dataset: Path, web_dir: Path = Path("data/bulk_val_v1"), device: str = "auto") -> dict:
    """Judge by the sealed constitution. Raises ConstitutionError if the rules or the held-out data were changed."""
    from evo.engine import constitution

    sets = held_out_sets(dataset, web_dir)
    rules = constitution.verify(sets)
    tokenizer = str(dataset / "tokenizer.json")
    return decide(score(champion, sets, tokenizer, device), score(challenger, sets, tokenizer, device, cache=None), rules)


def text(v: dict) -> str:
    d = v["decision"]
    L = [f"{'ACCEPTED' if v['accept'] else 'rejected'}: gain {d['gain_percent']:+.2f} % on {'+'.join(d['sets'])} "
         f"[{d['lo']:+.4f}, {d['hi']:+.4f}]"]
    for n, s in v["sets"].items():
        L.append(f"  {n}: {s['before']} -> {s['after']} ({s['percent']:+.2f} %) {s['verdict']}")
    if "code" in v:
        c = v["code"]
        L.append(f"  code exam: {c['before']} -> {c['after']} (+{c['gained']} -{c['lost']})   creator {v['creator']}")
    L += [f"  reason: {r}" for r in v["reasons"]]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--champion", required=True)
    ap.add_argument("--challenger", required=True)
    ap.add_argument("--web-val-dir", default="data/bulk_val_v1")
    args = ap.parse_args(argv)
    dataset = Path(json.loads(Path("evo/engine/evo_state.json").read_text(encoding="utf-8"))["best_known"]["dataset"])
    v = judge(args.champion, args.challenger, dataset, Path(args.web_val_dir))
    print(text(v))
    print(json.dumps(v))
    return 0 if v["accept"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
