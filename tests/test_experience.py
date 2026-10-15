"""evo.engine.experience: a model of the director's own attempts proposes the next one and checks its own errors."""

import json
import math
import random

from evo.engine import director as d
from evo.engine import experience as X


def world_gain(lr, web, code, steps):
    """A hidden world: best near lr 6e-5, web 0.80, little code; length matters little."""
    return 0.6 - 3.0 * (math.log10(lr) - math.log10(6e-5)) ** 2 - 4.0 * (web - 0.80) ** 2 - 2.0 * (code - 0.05) ** 2 + 0.00001 * steps


def attempt_row(champion, recipe, lr, web, code, steps, gain):
    return {"event": "attempt", "recipe": recipe, "champion": champion, "steps": steps, "clones": 1,
            "flags": {"--lr": f"{lr:.1e}", "--bulk-frac": f"{web:.2f}", "--code-frac": f"{code:g}"},
            "verdict": {"accept": gain > 0.3, "decision": {"gain_percent": round(gain, 3)}, "sets": {}}}


def history(n=24, seed=0, champions=("NOVA8-53M-v2", "NOVA8-53M-v3")):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        c = champions[i % len(champions)]
        lr = 10 ** rng.uniform(math.log10(5e-6), math.log10(2e-4))
        web, code, steps = rng.uniform(0.5, 0.98), rng.uniform(0.01, 0.4), rng.choice([2000, 4000, 8000])
        g = world_gain(lr, web, code, steps) - (0.3 if c.endswith("v3") else 0.0) + rng.gauss(0, 0.03)
        rows.append(attempt_row(c, f"gentle~t{i}", lr, web, code, steps, g))
    return rows


def test_too_little_experience_says_nothing():
    assert X.propose(history(5), d.RECIPES, "NOVA8-53M-v3") is None
    assert X.summary(history(5))["enough"] is False


def test_the_proposal_heads_where_the_world_is_better_and_is_an_ordinary_recipe():
    rows = history(30)
    p = X.propose(rows, d.RECIPES, "NOVA8-53M-v3", seed=1)
    assert p is not None and p["name"].startswith("gentle~xp-")
    s = p["settings"]
    measured = [r["verdict"]["decision"]["gain_percent"] for r in rows if r["champion"] == "NOVA8-53M-v3"]
    assert world_gain(s["lr"], s["web"], s["code"], s["steps"]) - 0.3 > sorted(measured)[len(measured) // 2]   # better than a typical try
    rec = p["recipe"]
    assert rec["steps"] == int(s["steps"]) and set(rec["flags"]) == {"--lr", "--bulk-frac", "--code-frac"}
    lo, hi = 5e-6, 2e-4
    assert lo <= float(rec["flags"]["--lr"]) <= hi and 0.5 <= float(rec["flags"]["--bulk-frac"]) <= 0.98
    assert X.propose(rows, d.RECIPES, "NOVA8-53M-v3", taken={p["name"]}, seed=1)["name"] != p["name"]   # never the same twice


def test_it_measures_its_own_errors_and_becomes_more_careful():
    rows = history(30)
    p = X.propose(rows, d.RECIPES, "NOVA8-53M-v3", seed=1)
    sd_before = p["sd"]
    for i in range(4):                                       # the world surprised it four times: much worse than predicted
        name = f"{p['name']}-{i}"
        rows.append({"event": "experience", "champion": "NOVA8-53M-v3", "name": name, "predicted": 0.5, "sd": 0.05})
        rows.append(attempt_row("NOVA8-53M-v3", name, 1.9e-4, 0.51, 0.39, 2000, -1.5))
    cal = X.calibration(rows)
    assert cal["checked"] == 4 and cal["mean_error"] < -1 and cal["within_2sd"] == 0
    p2 = X.propose(rows, d.RECIPES, "NOVA8-53M-v3", seed=1)
    assert p2 is not None and p2["sd"] >= sd_before * 0.5   # still proposes, with an honest uncertainty


def test_the_director_takes_a_proposal_only_a_few_times_per_champion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "evo/director").mkdir(parents=True)
    log = tmp_path / "evo/director/log.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in history(30)) + "\n")
    monkeypatch.setattr(d, "LOG", log)
    st = {"champion": {"name": "NOVA8-53M-v3"}, "seed": 3, "attempts": 40, "recipes": {}, "tried": {}}
    names = [d.experience_recipe(st) for _ in range(d.EXPERIENCE_TESTS + 2)]
    made = [n for n in names if n]
    assert len(made) == d.EXPERIENCE_TESTS and len(set(made)) == len(made)
    for n in made:                                           # the director marks them as tried when it runs them
        st["tried"].setdefault("NOVA8-53M-v3", {})[n] = {"outcome": "rejected", "gain": 0.0, "time": 0}
    events = [json.loads(l) for l in log.read_text().splitlines() if '"experience"' in l]
    assert [e["name"] for e in events] == made and all("predicted" in e and "sd" in e for e in events)
    assert all(n in st["own_recipes"] for n in made)
