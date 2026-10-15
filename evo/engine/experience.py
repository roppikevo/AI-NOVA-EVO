"""
Experience: the director learns how to learn from every attempt it has made.

`lessons` reads the attempts one setting at a time ("a higher web share came with better results"). This module
reads them together: a small model of its own experience - a Gaussian process over the settings of a learning
attempt (learning rate, web share, code share, length) - predicts what an untried combination would bring, and how
sure it is. The next attempt is the combination with the best chance of beating what was already reached
(expected improvement), so the director spends the card on attempts that teach it the most and does not repeat
what already failed.

It also checks itself. Each proposal is logged with its prediction; when the judge measures the attempt, the
difference between the prediction and the measurement is the model's own error. Those errors are added to its
uncertainty (it becomes more careful when it was wrong) and are reported (`python -m evo.engine.experience`).

Only plain attempts are used (settings only, no surgery, no collective); results of different champions are made
comparable by taking each champion's own mean away. The model never decides anything: what it proposes is an
ordinary recipe that the judge and the constitution decide on like every other.

    python -m evo.engine.experience         # what the experience says now, and how good its past predictions were
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any

from evo.engine import lessons

SETTINGS = ("lr", "web", "code", "steps")
LOG_SCALE = {"lr", "steps"}
MIN_EXPERIMENTS = 8          # fewer than this: the experience says nothing yet
CANDIDATES = 2000            # combinations scored per proposal
LENGTHS = (0.2, 0.3, 0.45, 0.7, 1.0, 3.0)   # length scales tried per setting; the ones the data explains best are used
                                            # (3.0 = "this setting hardly matters")
TRUST = 0.7                  # proposals stay where the model is fairly sure (sd below this share of the prior sd)
NEAR = 0.04                  # a combination this close to one already measured on this champion is not new


def scale(settings: dict[str, float]) -> list[float] | None:
    out = []
    for s in SETTINGS:
        if s not in settings:
            return None
        lo, hi = lessons.LIMITS[s]
        v = min(hi, max(lo, float(settings[s])))
        out.append((math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) if s in LOG_SCALE else (v - lo) / (hi - lo))
    return out


def unscale(x: list[float]) -> dict[str, float]:
    out = {}
    for s, u in zip(SETTINGS, x):
        lo, hi = lessons.LIMITS[s]
        out[s] = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))) if s in LOG_SCALE else lo + u * (hi - lo)
    out["steps"] = float(int(round(out["steps"], -2)))
    out["lr"] = float(f"{out['lr']:.1e}")
    out["web"] = round(out["web"], 2)
    out["code"] = float(f"{out['code']:.3f}")
    return out


def points(exps: list[dict]) -> list[dict]:
    """Plain experiments with every setting known, centred on their champion's mean gain."""
    use = [e for e in exps if not e["other"] and e.get("gain") is not None and scale(e["settings"]) is not None
           and float(e["settings"].get("clones", 1)) == 1]
    by_champ: dict[str, list[float]] = {}
    for e in use:
        by_champ.setdefault(e["champion"], []).append(e["gain"])
    overall = mean(e["gain"] for e in use) if use else 0.0
    offset = {c: (mean(g) if len(g) >= 2 else overall) for c, g in by_champ.items()}
    return [{"x": scale(e["settings"]), "y": e["gain"] - offset[e["champion"]], "runs": e["runs"], "champion": e["champion"],
             "recipe": e["recipe"], "gain": e["gain"], "offset": offset[e["champion"]]} for e in use]


def _k(a: list[float], b: list[float], var: float, length: tuple[float, ...]) -> float:
    return var * math.exp(-sum((p - q) ** 2 / (2 * l * l) for p, q, l in zip(a, b, length)))


def _cholesky(A: list[list[float]]) -> list[list[float]]:
    """Cholesky factor of a small symmetric positive definite matrix (no numpy needed)."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = math.sqrt(max(s, 1e-12)) if i == j else s / L[j][j]
    return L


def _forward(L: list[list[float]], b: list[float]) -> list[float]:
    y = [0.0] * len(b)
    for i in range(len(b)):
        y[i] = (b[i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
    return y


def _backward(L: list[list[float]], y: list[float]) -> list[float]:
    n = len(y)
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
    return x


class Model:
    def __init__(self, pts: list[dict], noise_std: float, length: tuple[float, ...] | None = None) -> None:
        self.pts = pts
        ys = [p["y"] for p in pts]
        self.var = max(0.05, sum(y * y for y in ys) / len(ys))           # signal variance from the data itself
        self.noise = [max(noise_std, 0.02) ** 2 / max(1, p["runs"]) for p in pts]
        if length is None:                    # one length per setting, the ones the experience explains best (2 sweeps)
            length = (0.45,) * len(SETTINGS)
            for _ in range(2):
                for i in range(len(SETTINGS)):
                    length = max((length[:i] + (l,) + length[i + 1:] for l in LENGTHS),
                                 key=lambda ls: Model(pts, noise_std, ls).evidence)
        self.length = length
        n = len(pts)
        self.K = [[_k(pts[i]["x"], pts[j]["x"], self.var, length) + (self.noise[i] if i == j else 0.0) for j in range(n)] for i in range(n)]
        self.L = _cholesky(self.K)
        self.alpha = _backward(self.L, _forward(self.L, ys))
        # log marginal likelihood: how well this model explains what was measured
        self.evidence = -0.5 * sum(y * a for y, a in zip(ys, self.alpha)) - sum(math.log(self.L[i][i]) for i in range(n))

    def predict(self, x: list[float]) -> tuple[float, float]:
        ks = [_k(x, p["x"], self.var, self.length) for p in self.pts]
        mu = sum(k * a for k, a in zip(ks, self.alpha))
        v = _forward(self.L, ks)
        s2 = max(1e-9, self.var - sum(w * w for w in v))
        return mu, math.sqrt(s2)


def _ei(mu: float, sd: float, best: float) -> float:
    z = (mu - best) / sd
    return (mu - best) * 0.5 * (1 + math.erf(z / math.sqrt(2))) + sd * math.exp(-z * z / 2) / math.sqrt(2 * math.pi)


def calibration(rows: list[dict]) -> dict[str, Any]:
    """Past proposals against what the judge then measured: how wrong the experience was."""
    proposed = {(r.get("champion"), r.get("name")): r for r in rows if r.get("event") == "experience"}
    pairs = []
    for o in lessons.observations(rows):
        p = proposed.get((o["champion"], o["recipe"]))
        if p and o["gain"] is not None:
            pairs.append({"recipe": o["recipe"], "predicted": p["predicted"], "sd": p["sd"], "measured": o["gain"]})
    if not pairs:
        return {"checked": 0}
    err = [x["measured"] - x["predicted"] for x in pairs]
    inside = sum(abs(e) <= 2 * x["sd"] for e, x in zip(err, pairs))
    return {"checked": len(pairs), "mean_error": round(mean(err), 3), "rms_error": round(math.sqrt(mean(e * e for e in err)), 3),
            "within_2sd": inside, "last": pairs[-5:]}


def propose(rows: list[dict], recipes: dict[str, dict], champion: str, taken: set[str] | None = None,
            seed: int = 0) -> dict | None:
    """The untried combination of settings with the highest expected improvement for this champion, as a recipe."""
    taken = taken or set()
    exps = lessons.experiments(lessons.observations(rows))
    pts = points(exps)
    if len(pts) < MIN_EXPERIMENTS:
        return None
    noise = lessons.repeat_noise(exps).get("gain_std") or 0.1
    cal = calibration(rows)
    if cal.get("checked", 0) >= 3:
        noise = max(noise, cal["rms_error"] / 2)                    # it was wrong before: it is less sure now
    model = Model(pts, noise)
    here = [p for p in pts if p["champion"] == champion]
    offset = here[0]["offset"] if here else mean(p["gain"] for p in pts) - mean(p["y"] for p in pts)
    best = max((p["y"] for p in here), default=max(p["y"] for p in pts))
    plain = [e for e in exps if e["champion"] == champion and not e["other"] and e["recipe"] in recipes and "flags" in recipes[e["recipe"]]]
    base_name = max(plain, key=lambda e: e["gain"])["recipe"] if plain else "gentle"
    if base_name not in recipes or "flags" not in recipes[base_name]:
        return None
    rng = random.Random(seed)
    measured = [p["x"] for p in here]
    top = sorted(pts, key=lambda p: -p["y"])[:5]
    cands = [[rng.random() for _ in SETTINGS] for _ in range(CANDIDATES)]
    cands += [[min(1, max(0, v + rng.gauss(0, 0.08))) for v in p["x"]] for p in top for _ in range(200)]   # near what worked
    scored = []
    for x in cands:
        if any(math.dist(x, m) < NEAR for m in measured):
            continue
        mu, sd = model.predict(x)
        if sd > TRUST * math.sqrt(model.var):          # too far from anything measured: a guess, not experience
            continue
        scored.append((_ei(mu, sd, best), mu, sd, x))
    scored.sort(key=lambda t: -t[0])
    old = recipes[base_name]
    parent = old.get("parent") or base_name.split("~")[0]
    for ei, mu, sd, x in scored[:50]:
        s = unscale(x)
        name = f"{parent}~xp-lr{s['lr']:.0e}-w{s['web']:.2f}-c{s['code']:g}-s{int(s['steps'])}".replace("e-0", "e-")
        if name in recipes or name in taken:
            continue
        new = {**old, "flags": dict(old["flags"]), "parent": parent, "steps": int(s["steps"]),
               "experience": {"predicted": round(mu + offset, 3), "sd": round(sd, 3)}}
        new.pop("lesson", None)
        new["flags"].update({"--lr": f"{s['lr']:.1e}".replace("e-0", "e-"), "--bulk-frac": f"{s['web']:.2f}",
                             "--code-frac": f"{s['code']:g}"})
        return {"name": name, "recipe": new, "settings": s, "predicted": round(mu + offset, 3), "sd": round(sd, 3),
                "expected_improvement": round(ei, 4), "base": base_name, "experiments": len(pts),
                "reason": f"experience of {len(pts)} experiments: expected {mu + offset:+.2f} % (+-{sd:.2f}), "
                          f"the best chance to beat {best + offset:+.2f} % on this champion"}
    return None


def summary(rows: list[dict]) -> dict[str, Any]:
    exps = lessons.experiments(lessons.observations(rows))
    pts = points(exps)
    out: dict[str, Any] = {"experiments": len(pts), "enough": len(pts) >= MIN_EXPERIMENTS, "calibration": calibration(rows)}
    if len(pts) >= MIN_EXPERIMENTS:
        model = Model(pts, lessons.repeat_noise(exps).get("gain_std") or 0.1)
        out["length_scale"] = dict(zip(SETTINGS, model.length))   # a long one: that setting hardly matters
        # what the experience thinks of each setting alone: the prediction along that axis, the rest at the best point
        bestp = max(pts, key=lambda p: p["y"])["x"]
        out["best_measured"] = unscale(bestp)
        out["along"] = {}
        for i, s in enumerate(SETTINGS):
            row = []
            for u in (0.0, 0.25, 0.5, 0.75, 1.0):
                x = list(bestp)
                x[i] = u
                mu, sd = model.predict(x)
                row.append([round(unscale(x)[s], 6), round(mu, 3), round(sd, 3)])
            out["along"][s] = row
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=str(lessons.LOG))
    ap.add_argument("--champion", default=None, help="also show the next proposal for this champion")
    args = ap.parse_args(argv)
    rows = lessons.load_rows(Path(args.log))
    out = summary(rows)
    if args.champion:
        from evo.engine.director import all_recipes, load_state

        st = load_state() or {}
        p = propose(rows, all_recipes(st) if st else {}, args.champion)
        out["proposal"] = {k: v for k, v in (p or {}).items() if k != "recipe"} or None
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
