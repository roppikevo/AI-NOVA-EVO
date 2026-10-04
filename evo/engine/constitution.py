"""
The constitution: what "better" means and what must never be lost.

The director chooses the STRATEGY (recipes, budgets, order of experiments, model sizes). It may not
touch the CONSTITUTION: the judge's thresholds, the held-out texts the judge measures on, the Creator
check. These live in evo/constitution.json and are sealed: a fingerprint of the rules and of the
held-out data is kept outside the project. Before every decision the director verifies the seal; if
anything differs it stops and releases nothing until the Creator has looked.

    python -m evo.engine.constitution --verify
    python -m evo.engine.constitution --seal        # the Creator's act (after he changed the rules)

Seal locations, first one found wins:
    /etc/ai/nova-constitution.seal                  (root-owned: the strongest; copy it there with sudo)
    ~/nova-evo-backups/constitution.seal            (outside the repository, read-only)
This is a tripwire against a program (or a mistake) changing the rules, not cryptography.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

RULES_FILE = Path("evo/constitution.json")
SEALS = [Path("/etc/ai/nova-constitution.seal"), Path.home() / "nova-evo-backups" / "constitution.seal"]
REQUIRED = ("min_gain_pct", "max_set_decline_pct", "max_code_loss", "creator_min", "decision_sets", "vault_sets")


class ConstitutionError(RuntimeError):
    pass


def load(path: Path | None = None) -> dict:
    path = path or RULES_FILE
    if not path.exists():
        raise ConstitutionError(f"{path} is missing")
    rules = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in rules]
    if missing:
        raise ConstitutionError(f"rules are incomplete: {missing}")
    return rules


def fingerprint(rules: dict, sets: dict[str, np.ndarray]) -> dict:
    """Hash of the rules and of every held-out set the judge uses."""
    out = {"rules": hashlib.sha256(json.dumps(rules, sort_keys=True, ensure_ascii=False).encode()).hexdigest(), "sets": {}}
    for name in sorted(sets):
        a = np.ascontiguousarray(sets[name])
        out["sets"][name] = hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()
    return out


def seal_file(seals: list[Path] | None = None) -> Path | None:
    return next((p for p in (seals or SEALS) if p.exists()), None)


def seal(sets: dict[str, np.ndarray], path: Path | None = None, rules_path: Path | None = None) -> Path:
    path = path or SEALS[-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o644)
    path.write_text(json.dumps(fingerprint(load(rules_path), sets), indent=1))
    path.chmod(0o444)
    return path


def verify(sets: dict[str, np.ndarray], seals: list[Path] | None = None, rules_path: Path | None = None) -> dict:
    """The rules, if they and the held-out data are exactly what was sealed. Raises ConstitutionError otherwise."""
    rules = load(rules_path)
    f = seal_file(seals)
    if f is None:
        raise ConstitutionError("no seal found: the Creator has to seal the constitution first")
    sealed = json.loads(f.read_text())
    now = fingerprint(rules, sets)
    if sealed.get("rules") != now["rules"]:
        raise ConstitutionError("the rules differ from the sealed ones")
    for name in set(rules["decision_sets"]) | set(rules["vault_sets"]):
        if name in sealed.get("sets", {}) and sealed["sets"][name] != now["sets"].get(name):
            raise ConstitutionError(f"held-out set '{name}' differs from the sealed one")
    if not all(n in now["sets"] for n in rules["decision_sets"]):
        raise ConstitutionError("a decision set is missing")
    return rules


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seal", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--web-val-dir", default="data/bulk_val_v1")
    args = ap.parse_args(argv)
    from evo.engine.judge import held_out_sets

    dataset = Path(json.loads(Path("evo/engine/evo_state.json").read_text(encoding="utf-8"))["best_known"]["dataset"])
    sets = held_out_sets(dataset, Path(args.web_val_dir))
    if args.seal:
        p = seal(sets)
        print(f"sealed: {p}  (stronger: sudo cp {p} {SEALS[0]} && sudo chmod 444 {SEALS[0]})")
        return 0
    try:
        rules = verify(sets)
    except ConstitutionError as e:
        print(f"CONSTITUTION BROKEN: {e}")
        return 1
    print(f"constitution intact (seal {seal_file()}): " + json.dumps({k: rules[k] for k in REQUIRED}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
