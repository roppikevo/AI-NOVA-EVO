"""The problem register and the core's own diagnosis: what is open, what was tried and why it failed, what comes
back and when - and a problem is solved only when the reserved confirmation sample agrees."""

import json

import numpy as np

from evo.engine import abilities as A
from evo.engine import problems as P
from evo.learning import drive, skills

S = list(drive.STRATEGIES)


def metric(acc, n=200, seed=0, feats=None):
    rng = np.random.default_rng(seed)
    c = (rng.random(n) < acc).astype(int)
    return {"accuracy": round(float(c.mean()), 3), "chance": 0.25, "correct": "".join(map(str, c)), "feat": feats or [], "margin": []}


def test_the_cores_errors_show_a_pattern_and_how_sure_it_was():
    rng = np.random.default_rng(1)
    feats = [{"lang": "sk" if i % 2 else "en", "people": 3 if i % 3 else 4} for i in range(200)]
    correct = "".join("0" if (f["people"] == 4 and rng.random() < 0.8) or rng.random() < 0.1 else "1" for f in feats)
    m = {"accuracy": correct.count("1") / 200, "correct": correct, "feat": feats,
         "margin": [2.0 if c == "0" else 1.0 for c in correct]}
    d = A.diagnose(m)
    assert d["patterns"][0]["feature"] == "people" and d["patterns"][0]["value"] == "4"
    assert not any(p["feature"] == "lang" for p in d["patterns"])
    assert d["confidently_wrong"] == 1.0                                 # wrong and sure: a learned wrong rule
    h = P.hypothesis("focus-errors", "skills/ordering/new", d)
    assert "people=4" in h["cause"] and "klesne" in h["prediction"]


def test_open_try_park_reopen_and_solve_only_with_confirmation():
    reg = {"problems": {}}
    board = {"skills/ordering/new": metric(0.30), "skills/compare/new": metric(0.97)}
    gaps = [{"metric": "skills/ordering/new", "kind": "weak", "now": 0.30}]
    notes = P.sync(reg, "v1", board, None, gaps, S)
    assert notes[0].startswith("opened skills/ordering/new")
    assert "skills/compare/new" not in reg["problems"]
    for s in S:                                                          # every way fails
        assert P.choose(reg, gaps, S, S) == ("skills/ordering/new", s)
        P.record(reg, "skills/ordering/new", {"strategy": s, "before": 0.30, "after": 0.31, "steps": 6000, "hours": 0.4,
                                              "hypothesis": {"cause": "x"}}, S, "v1")
    p = reg["problems"]["skills/ordering/new"]
    assert p["status"] == "parked" and len(p["failed_hypotheses"]) == len(S) and P.choose(reg, gaps, S, S) is None
    P.sync(reg, "v1", board, None, gaps, S)                               # same core, nothing new: stays parked
    assert p["status"] == "parked"
    P.sync(reg, "v2", board, None, gaps, S)                               # the core changed: it comes back, all ways again
    assert p["status"] == "open" and p["tried"] == [] and "core changed" in p["reopened"][-1]["reason"]
    P.record(reg, "skills/ordering/new", {"strategy": "practice", "before": 0.30, "after": 0.93, "steps": 6000, "hours": 0.4}, S, "v2")
    assert p["worked"][-1]["strategy"] == "practice" and p["history"][-1]["supported"]
    good = {"skills/ordering/new": metric(0.93, seed=3)}
    P.sync(reg, "v3", good, None, [], S)
    assert p["status"] == "open"                                          # 0.93 alone is not solved: no confirmation
    P.sync(reg, "v3", good, {"skills/ordering/new": {"accuracy": 0.88}}, [], S)
    assert p["status"] == "open"                                          # the confirmation sample disagrees
    P.sync(reg, "v3", good, {"skills/ordering/new": {"accuracy": 0.92}}, [], S)
    assert p["status"] == "solved"
    P.sync(reg, "v4", {"skills/ordering/new": metric(0.70, seed=4)}, {"skills/ordering/new": {"accuracy": 0.7}}, [], S)
    assert p["status"] == "open" and "regression" in p["reopened"][-1]["reason"]
    eff = P.efficiency(reg)
    assert eff["practice"]["supported"] == 1 and eff["practice"]["tries"] == 2
    assert any("vyriešené" in l or "Register" in l for l in P.text(reg))


def test_a_crash_is_not_a_try_and_errors_drive_the_material(tmp_path):
    reg = {"problems": {}}
    board = {"rulelab/deeper": metric(0.2)}
    gaps = [{"metric": "rulelab/deeper", "kind": "weak", "now": 0.2}]
    P.sync(reg, "v1", board, None, gaps, S)
    P.record(reg, "rulelab/deeper", {"strategy": "practice", "before": 0.2, "after": None, "crashed": True}, S, "v1")
    assert reg["problems"]["rulelab/deeper"]["tried"] == [] and P.choose(reg, gaps, S, S)[1] == "practice"
    pattern = {"feature": "asks", "value": "least", "items": 100}
    m = drive.build_material(["skills/ordering"], "focus-errors", tmp_path, seed=1, pattern=pattern, pattern_target="skills/ordering")
    rows = [json.loads(l) for l in (m / "material.jsonl").read_text().splitlines()]
    ordering = [r for r in rows if r["topic"] == "ordering"]
    least = sum(("najmenej" in (r.get("prompt") or r["text"])) or ("fewest" in (r.get("prompt") or r["text"])) for r in ordering)
    assert len(ordering) == drive.STRATEGIES["focus-errors"]["n_target"] and 0.62 < least / len(ordering) < 0.78


def test_the_confirmation_sample_is_reserved_and_never_the_selection_sample():
    sel = {it["prompt"] for it in skills.exam_items("ordering", 200)}
    conf = {it["prompt"] for it in skills.exam_items("ordering", 200, start=200)}
    assert len(sel & conf) < 10                                           # different items (a few equal by chance)
    assert A.SAMPLES == {"select": 0, "confirm": 200}


def test_only_a_masterable_skill_is_ever_solved_an_edge_stays_open():
    reg = {"problems": {}}
    gaps = [{"metric": "rulelab/deeper", "kind": "frontier", "now": 0.5}]
    P.sync(reg, "v1", {"rulelab/deeper": metric(0.5)}, None, gaps, S)
    P.sync(reg, "v2", {"rulelab/deeper": metric(0.95, seed=2)}, {"rulelab/deeper": {"accuracy": 0.95}}, gaps, S)
    assert reg["problems"]["rulelab/deeper"]["class"] == "frontier" and reg["problems"]["rulelab/deeper"]["status"] == "open"
    assert A.kind_of("skills/ordering/new") == "mastery" and A.kind_of("skills/ordering/form") == "frontier"
    assert A.kind_of("skills/deduction/new") == "frontier" and A.kind_of("rulelab/same") == "mastery"
