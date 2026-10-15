"""Abilities: the judge does not let NOVA get worse at what it can do, and a real new ability is worth a little text;
the director goes after weak and lost abilities with verified practice and does not repeat a way that failed."""

import json

import numpy as np

from evo.engine import abilities as A
from evo.engine import director as d
from evo.engine import judge as j
from evo.learning import drive


def board(acc: dict[str, float], n=200, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for k, a in acc.items():
        c = (rng.random(n) < a).astype(int)
        out[k] = {"accuracy": round(c.mean(), 3), "chance": 0.25, "correct": "".join(map(str, c))}
    return out


def improve(b: dict, key: str, extra: int) -> dict:
    """The same answers, with `extra` more items right (or wrong if negative) - a paired change."""
    out = json.loads(json.dumps(b))
    c = list(out[key]["correct"])
    idx = [i for i, x in enumerate(c) if x == ("0" if extra > 0 else "1")][:abs(extra)]
    for i in idx:
        c[i] = "1" if extra > 0 else "0"
    out[key]["correct"] = "".join(c)
    out[key]["accuracy"] = round(c.count("1") / len(c), 3)
    return out


def pair(deltas, code=(40, 40)):
    rng = np.random.default_rng(0)
    base = {"dataset": 3.15, "web": 3.41, "dataset_vault": 3.20, "web_vault": 3.40}
    hard = {k: rng.normal(0, 0.4, size=800) for k in base}
    champ = {"seq": {k: (m + hard[k]).astype(np.float32) for k, m in base.items()}, "code": [f"t{i}" for i in range(code[0])], "creator": -0.01}
    chall = {"seq": {k: (base[k] + deltas.get(k, 0.0) + hard[k] + rng.normal(0, 0.02, size=800)).astype(np.float32) for k in base},
             "code": [f"t{i}" for i in range(code[1])], "creator": -0.01}
    return champ, chall


RULES = {**j.DEFAULT_RULES, "abilities": {"text_tolerance_pct": 1.0, "vault_tolerance_pct": 1.0, "min_better": 1, "max_worse": 0}}
B0 = board({"skills/ordering/new": 0.3, "skills/compare/new": 0.5, "rulelab/deeper": 0.1})


def test_compare_is_paired_and_corrected():
    up = A.compare(B0, improve(B0, "skills/ordering/new", 80))
    assert up["better"] == ["skills/ordering/new"] and not up["worse"]
    down = A.compare(B0, improve(B0, "skills/compare/new", -40))
    assert down["worse"] == ["skills/compare/new"]
    noise = A.compare(B0, improve(B0, "rulelab/deeper", 3))
    assert not noise["better"] and not noise["worse"]


def test_a_new_ability_may_cost_a_little_text_but_never_another_ability():
    cost = {"dataset": +0.02, "web": +0.01, "dataset_vault": +0.02, "web_vault": +0.01}          # ~0.5 % worse text
    better = A.compare(B0, improve(B0, "skills/ordering/new", 80))
    v = j.decide(*pair(cost), rules=RULES, abilities=better)
    assert v["accept"] and v["path"] == "abilities"
    too_costly = j.decide(*pair({k: x * 4 for k, x in cost.items()}), rules=RULES, abilities=better)
    assert not too_costly["accept"] and any("tolerance" in r for r in too_costly["reasons"])
    traded = A.compare(B0, improve(improve(B0, "skills/ordering/new", 80), "skills/compare/new", -40))
    assert not j.decide(*pair(cost), rules=RULES, abilities=traded)["accept"]
    # a text gain does not buy a lost ability either
    good_text = {"dataset": -0.03, "web": -0.03, "dataset_vault": -0.03, "web_vault": -0.03}
    lost = A.compare(B0, improve(B0, "skills/compare/new", -40))
    v = j.decide(*pair(good_text), rules=RULES, abilities=lost)
    assert not v["accept"] and any("abilities significantly worse" in r for r in v["reasons"])
    assert j.decide(*pair(good_text), rules=RULES, abilities=A.compare(B0, B0))["path"] == "text"
    # without the abilities rules the old judge is unchanged
    assert not j.decide(*pair(cost), abilities=better)["accept"]


def test_deficits_put_losses_first_then_half_learned_skills():
    rows = [{"name": "v1", "role": "champion", "board": board({"skills/compare/new": 0.97, "skills/ordering/new": 0.3, "rulelab/deeper": 0.12})},
            {"name": "v2", "role": "champion", "board": board({"skills/compare/new": 0.80, "skills/ordering/new": 0.6, "rulelab/deeper": 0.11}, seed=1)}]
    gaps = A.deficits(rows, "v2")
    assert gaps[0]["metric"] == "skills/compare/new" and gaps[0]["kind"] == "loss"
    assert [g["metric"] for g in gaps[1:]] == ["skills/ordering/new", "rulelab/deeper"]
    assert gaps[1]["kind"] == "weak" and gaps[2]["kind"] == "frontier"          # an edge has no fixed bar
    assert drive.targets_of(gaps) == ["skills/compare", "skills/ordering", "rulelab"]
    assert any("ZHORŠENIA" in l for l in A.text(rows, "v2"))


def test_a_way_of_practising_that_failed_is_not_repeated_and_winners_go_first(tmp_path):
    st = {"attempts": 0}
    t = ["skills/ordering"]
    seen = []
    while (s := drive.next_try(st, "v1", t)) is not None:
        seen.append(s)
        drive.note_try(st, "v1", t, s)
    assert seen == list(drive.STRATEGIES)
    assert drive.next_try(st, "v2", t) == "practice"                             # a new champion: everything again
    st["drive_wins"] = {"gentle": 2}
    assert drive.order(st)[0] == "gentle"
    m = drive.build_material(t, "practice", tmp_path, seed=1)
    rows = [json.loads(l) for l in (m / "material.jsonl").read_text().splitlines()]
    topics = [r["topic"] for r in rows]
    assert topics.count("ordering") == drive.STRATEGIES["practice"]["n_target"]
    assert "deduction" not in topics and topics.count("compare") == drive.STRATEGIES["practice"]["n_replay"]
    name, rec = drive.recipe(t, "practice", m)
    assert rec["flags"]["--focus"] == f"dir:{m}" and str(m) in rec["flags"]["--extra-dirs"] and name.startswith("drive:practice")


def test_the_text_anchor_stops_a_slow_drift(monkeypatch):
    monkeypatch.setattr(d, "_ability_rules", lambda: RULES["abilities"])
    st = {"text_anchor": 3.0}
    ok = {"accept": True, "path": "abilities", "sets": {"dataset": {"before": 3.0, "after": 3.02}, "web": {"before": 3.0, "after": 3.02}}}
    assert d.text_anchor_check(st, ok)["accept"]
    drift = {"accept": True, "path": "abilities", "sets": {"dataset": {"before": 3.02, "after": 3.04}, "web": {"before": 3.02, "after": 3.04}}}
    assert d.text_anchor_check(st, drift)["accept"] is False
    text = {"accept": True, "path": "text", "sets": {"dataset": {"before": 3.0, "after": 2.9}, "web": {"before": 3.0, "after": 2.9}}}
    d.text_anchor_check(st, text)
    assert st["text_anchor"] == 2.9
