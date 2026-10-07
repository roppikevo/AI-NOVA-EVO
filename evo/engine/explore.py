"""
The director's own tournaments: when learning stalls, it does what was done by hand for generation 8.

    1. noise       the champion's architecture is trained again in short runs with other seeds, until the spread
                   between runs of the same experiment is known (three runs)
    2. candidate   the next architecture from the registry that the ledger has no record of gets one short run,
                   the same way as every candidate before it
    3. verdict     identity first (a state that grows is out), then quality (the gain must be well above the
                   noise), then cost (equal quality with a clearly smaller state or faster training also wins;
                   a much larger state has to earn itself with a larger gain)
    4. a winner becomes the next generation in the ladder: trained in full, judged by the constitution

Hypotheses (the registry of candidates) still come from outside. With no untried candidate left the director
is "waiting for a new hypothesis": alive, measuring nothing in vain.

Everything here is decision logic on ledger records; running the training stays with the director.
"""

from __future__ import annotations

from typing import Any

SEEDS = (1001, 2001, 3001)          # the first is the seed every candidate was measured with
DEFAULT_NOISE_PCT = 0.5             # assumed spread between seeds until it is measured
GAIN_FACTOR = 2.0                   # a gain counts when it is this many times the noise ...
MIN_GAIN_PCT = 0.3                  # ... and at least this much
BIG_STATE = 1.5                     # a state this many times larger ...
BIG_STATE_GAIN_PCT = 1.0            # ... needs at least this gain
SMALL_STATE = 0.85                  # equal quality wins with a state this much smaller ...
FASTER = 1.15                       # ... or with training this much faster


def quality(entry: dict) -> float | None:
    m = entry.get("metrics") or {}
    return (m["dataset"] + m["web"]) / 2 if "dataset" in m and "web" in m else None


def candidate_of(override: dict | None, registry: dict[str, dict]) -> str | None:
    """Name of the registry entry this architecture is: every setting of the entry is in `override` with the same value
    (a full model config holds more keys than the entry); among several, the most specific entry."""
    if not override:
        return None
    fits = [n for n, cfg in registry.items() if all(override.get(k) == v for k, v in cfg.items())]
    return max(fits, key=lambda n: len(registry[n])) if fits else None


def run_name(candidate: str, group: str, seed: int = SEEDS[0]) -> str:
    return f"n8-{candidate}-{group}" + ("" if seed == SEEDS[0] else f"-s{seed}")


def base_runs(rows: list[dict], candidate: str, group: str) -> list[dict]:
    """Measured plain short runs of the champion's architecture, one per seed."""
    names = {run_name(candidate, group, s) for s in SEEDS}
    return [r for r in rows if r.get("name") in names and r.get("verdict") == "measured" and quality(r) is not None]


def noise_pct(rows: list[dict], candidate: str, group: str) -> float | None:
    """Spread of the quality between seeds (standard deviation, percent of the mean); None until three runs exist."""
    q = [quality(r) for r in base_runs(rows, candidate, group)]
    if len(q) < len(SEEDS):
        return None
    mean = sum(q) / len(q)
    return round(100 * (sum((x - mean) ** 2 for x in q) / len(q)) ** 0.5 / mean, 3)


def next_action(rows: list[dict], candidate: str, group: str, registry: dict[str, dict]) -> dict | None:
    """What the director should measure next: {"what": "noise", "seed"} or {"what": "candidate", "name"}; None = nothing left."""
    done = {r.get("name") for r in rows if r.get("kind") == "tournament"}
    if run_name(candidate, group) not in done:
        return {"what": "candidate", "name": candidate}                  # the champion's own short run is the yardstick
    for seed in SEEDS[1:]:
        if run_name(candidate, group, seed) not in done:
            return {"what": "noise", "name": candidate, "seed": seed}
    for name in registry:
        if run_name(name, group) not in done:
            return {"what": "candidate", "name": name}
    return None


def verdict(entry: dict, base: dict, noise: float | None) -> dict[str, Any]:
    """Is the candidate worth a full generation? Identity, then quality, then cost."""
    noise = DEFAULT_NOISE_PCT if noise is None else noise
    out: dict[str, Any] = {"win": False, "noise_percent": noise}
    if entry.get("verdict") != "measured" or quality(entry) is None:
        return {**out, "reason": "the run failed: " + str(entry.get("reason", ""))[:120]}
    if entry.get("identity") not in ("NOVA", "NOVA-HYBRID"):
        return {**out, "reason": f"identity: {entry.get('identity') or 'not tested'}"}
    gain = 100 * (quality(base) - quality(entry)) / quality(base)
    need = max(MIN_GAIN_PCT, GAIN_FACTOR * noise)
    cb, ce = base.get("cost") or {}, entry.get("cost") or {}
    state = ce["state_kb"] / cb["state_kb"] if ce.get("state_kb") and cb.get("state_kb") else 1.0
    speed = ce["train_tok_s"] / cb["train_tok_s"] if ce.get("train_tok_s") and cb.get("train_tok_s") else 1.0
    if state >= BIG_STATE:
        need = max(need, BIG_STATE_GAIN_PCT)
    out.update({"gain_percent": round(gain, 2), "needed_percent": round(need, 2), "state_ratio": round(state, 2), "speed_ratio": round(speed, 2)})
    if gain >= need:
        return {**out, "win": True, "reason": f"quality {gain:+.2f} % (needed {need:.2f} %, noise {noise:.2f} %), state x{state:.2f}"}
    if gain > -noise and state <= 1.0 and (state <= SMALL_STATE or speed >= FASTER):
        return {**out, "win": True, "reason": f"same quality ({gain:+.2f} %, noise {noise:.2f} %) at lower cost: state x{state:.2f}, training x{speed:.2f}"}
    return {**out, "reason": f"quality {gain:+.2f} % is below the needed {need:.2f} % (noise {noise:.2f} %); state x{state:.2f}, training x{speed:.2f}"}


def generation_row(candidate: str, group: str, registry: dict[str, dict], recipe: dict | None = None) -> dict:
    """The ladder entry that trains a winning candidate in full (the champion's own training recipe unless told otherwise)."""
    recipe = {"steps": 300000, "lr": 0.001, "compile": True, **(recipe or {})}
    return {"line": f"NOVA8-{group}-{candidate}", "override": registry[candidate], "candidate": candidate, "group": group,
            **{k: v for k, v in recipe.items() if k in ("steps", "lr", "compile", "carry", "carry_share") and v}}
