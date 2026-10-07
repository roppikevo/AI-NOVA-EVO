"""Lessons from the director's own attempts: repeats are noise, settings are evidence, controlled pairs decide."""

from evo.engine import lessons as L

RECIPES = {
    "gentle": {"steps": 4000, "flags": {"--lr": "2e-5", "--bulk-frac": "0.95", "--code-frac": "0.03"}},
    "fresh-web": {"steps": 5000, "fresh": True, "flags": {"--lr": "5e-5", "--bulk-frac": "0.97", "--code-frac": "0.02"}},
    "code": {"steps": 4000, "flags": {"--lr": "3e-5", "--bulk-frac": "0.8", "--code-frac": "0.25"}},
    "teachers": {"steps": 3000, "flags": {"--lr": "3e-5", "--bulk-frac": "0.6", "--code-frac": "0.03", "--focus": "teacher", "--focus-frac": "0.5"}},
    "one-more-layer": {"surgery": {"op": "add_layer"}, "steps": 6000, "flags": {"--lr": "5e-5", "--warmup": "300", "--bulk-frac": "0.95", "--code-frac": "0.03"}},
    "average-of-5": {"soup": 5, "steps": 3000, "flags": {"--lr": "1e-4", "--bulk-frac": "0.7", "--code-frac": "0.05"}},
    "average-of-5~web0.77": {"soup": 5, "steps": 3000, "parent": "average-of-5", "flags": {"--lr": "1e-4", "--bulk-frac": "0.77", "--code-frac": "0.05"}},
    "collective": {"collective": True, "rounds": 2, "steps": 3000},
}


def row(recipe, gain, champion="NOVA8-24M-v2", dataset=0.0, web=0.0, code=0, released=None):
    r = RECIPES[recipe]
    return {"event": "attempt", "recipe": recipe, "champion": champion, "steps": r["steps"], "flags": r.get("flags"),
            "clones": r.get("soup") or (5 if r.get("collective") else 1), **({"surgery": r["surgery"]} if r.get("surgery") else {}),
            "verdict": {"accept": released is not None, "decision": {"gain_percent": gain}, "code": {"before": 53, "after": 53 + code}},
            "sets": {"dataset": [3.0, round(3.0 * (1 - dataset / 100), 5), 0], "web": [3.2, round(3.2 * (1 - web / 100), 5), 0]},
            **({"released": released} if released else {})}


def log():
    """The shape of the 34 hours that were wasted: the same recipes again and again on one champion, plus an earlier champion."""
    rows = []
    for _ in range(4):
        rows += [row("teachers", -0.87, dataset=-1.35, web=-0.44, code=2), row("fresh-web", -0.20, dataset=-0.6, web=0.2),
                 row("one-more-layer", -0.14, dataset=-0.49, web=0.2, code=-1), row("gentle", -0.08, dataset=-0.3, web=0.14),
                 row("code", -0.04, dataset=0.1, web=-0.18)]
    rows += [row("average-of-5", g, dataset=0.4, web=0.0, code=c) for g, c in ((0.18, 0), (0.19, -2), (0.18, 1), (0.19, 0), (0.18, -1), (0.19, 2))]
    rows += [row("average-of-5~web0.77", 0.16, dataset=0.3, web=0.02) for _ in range(5)]
    rows += [row("collective", 0.29), row("collective", 0.30, released="NOVA8-24M-v3"), row("collective", 0.29), row("collective", 0.30, released="NOVA8-24M-v4")]
    rows += [{"event": "probation", "name": "NOVA8-24M-v3", "reverted": True}, {"event": "probation", "name": "NOVA8-24M-v4", "reverted": True}]
    rows += [row("teachers", -1.4, champion="NOVA-24M-v2"), row("gentle", -0.1, champion="NOVA-24M-v2", dataset=-0.2),
             row("fresh-web", -0.3, champion="NOVA-24M-v2", dataset=-0.5), row("teachers", -0.5, champion="NOVA8-24M-v1")]
    rows += [{"event": "explore", "what": "noise"}, {"event": "attempt", "recipe": "gentle", "rc": 1, "tail": "crashed"}]
    return rows


def test_repeats_of_one_attempt_are_one_experiment_and_measure_the_noise_of_the_judge():
    obs = L.observations(log())
    assert len(obs) == 20 + 6 + 5 + 4 + 4                                  # only judged attempts count
    exps = L.experiments(obs)
    assert len(exps) == 5 + 1 + 1 + 1 + 4
    soup = next(e for e in exps if e["recipe"] == "average-of-5")
    assert soup["runs"] == 6 and abs(soup["gain"] - 0.185) < 1e-6 and soup["settings"] == {"web": 0.7, "lr": 1e-4, "code": 0.05, "steps": 3000.0, "clones": 5.0}
    noise = L.repeat_noise(exps)
    assert noise["experiments"] == 8 and noise["gain_std"] < 0.01 and noise["gain_widest"] == 0.01
    assert 0.5 < noise["code_std"] < 1.0 and noise["code_widest"] == 4          # the code exam moves by itself: 51 .. 55
    assert L.threshold(noise) == L.MIN_EFFECT and L.threshold({"gain_std": 0.2}) == 0.4


def test_a_recipe_is_known_only_after_several_experiments_on_several_champions():
    r = L.by_recipe(L.experiments(L.observations(log())))
    assert r["teachers"]["reading"] == "negative" and r["teachers"]["confidence"] == "high" and r["teachers"]["champions"] == 3
    assert r["gentle"]["reading"] == "negative" and r["gentle"]["confidence"] == "medium"
    assert r["one-more-layer"]["confidence"] == "medium" and r["one-more-layer"]["runs"] == 4     # four repeats are still one experiment
    assert r["code"]["reading"] == "no clear effect"
    assert r["average-of-5"]["reading"] == "small gain" and r["average-of-5"]["experiments"] == 2     # the variation counts with its parent; below the judge's bar
    assert r["collective"]["accepted"] == 2 and r["collective"]["stepped_back"] == 2 and r["collective"]["reading"] == "does not hold"


def test_a_setting_is_evidence_and_stays_confounded_until_a_controlled_pair_exists():
    exps = L.experiments(L.observations(log()))
    s = L.by_setting(exps)
    web = s["web"]
    assert "teachers" not in web["low"]["recipes"] and "one-more-layer" not in web["high"]["recipes"]   # special attempts are left out
    assert web["high"]["values"] == [0.95, 0.97] and web["low"]["values"] == [0.7, 0.8]
    assert web["high_minus_low"]["gain"] < 0 and web["high_minus_low"]["dataset"] < 0 and web["better"] == "low"
    assert len(web["controlled_pairs"]) == 1 and abs(web["controlled_pairs"][0]["gain"] + 0.025) < 1e-6    # average-of-5: 0.70 against 0.77
    assert web["confidence"] == "low" and web["confounded"] and web["pairs_with_a_sign"] == 0     # the one pair is inside the threshold: no sign from it
    assert s["lr"]["confounded"] and s["lr"]["confidence"] in ("low", "medium")
    txt = L.text(L.summary(log()))
    assert "confounded" in txt and "teachers" in txt and "controlled pairs 1 (0 with a clear sign)" in txt
    sk = "\n".join(L.lines_sk(L.summary(log())))
    assert "teachers: škodí (istota vysoká" in sk and "skúška z kódu" in sk


def test_the_next_test_moves_one_setting_of_a_known_recipe_and_nothing_else():
    exps = L.experiments(L.observations(log()))
    made = L.suggest(exps, RECIPES, "NOVA8-24M-v2")
    assert made and made["name"].startswith(("gentle~", "fresh-web~", "code~", "average-of-5~")) and made["name"] not in RECIPES
    base, new = RECIPES[made["base"]], made["recipe"]
    changed = [k for k in base["flags"] if new["flags"][k] != base["flags"][k]] + (["steps"] if new["steps"] != base["steps"] else [])
    assert len(changed) == 1 and new["parent"] == made["base"].split("~")[0] and new["lesson"] == made["setting"]
    assert "confidence" in made["reason"] and made["from"] != made["to"]
    again = L.suggest(exps, RECIPES, "NOVA8-24M-v2", taken={made["name"]})
    assert again is None or again["name"] != made["name"]
    assert L.suggest(exps, RECIPES, "NOVA8-24M-v9") is None                 # nothing judged on that champion: nothing to pair with
    assert L.suggest([], RECIPES, "NOVA8-24M-v2") is None and L.summary([])["experiments"] == 0 and L.lines_sk(L.summary([])) == []


def test_two_agreeing_controlled_pairs_make_a_lesson_firm():
    rows = log() + [row("gentle", -0.08), ]
    extra = {**RECIPES, "gentle~web0.70": {"steps": 4000, "parent": "gentle", "flags": {"--lr": "2e-5", "--bulk-frac": "0.70", "--code-frac": "0.03"}},
             "fresh-web~web0.70": {"steps": 5000, "parent": "fresh-web", "flags": {"--lr": "5e-5", "--bulk-frac": "0.70", "--code-frac": "0.02"}}}
    def r2(name, gain):
        r = extra[name]
        return {"event": "attempt", "recipe": name, "champion": "NOVA8-24M-v2", "steps": r["steps"], "flags": r["flags"], "clones": 1,
                "verdict": {"accept": False, "decision": {"gain_percent": gain}}, "sets": {}}
    rows += [r2("gentle~web0.70", 0.12), r2("fresh-web~web0.70", 0.05)]
    web = L.by_setting(L.experiments(L.observations(rows)))["web"]
    assert web["confidence"] == "high" and web["better"] == "low" and len(web["controlled_pairs"]) == 3 and not web["confounded"]
    assert all(m is None or m["setting"] != "web" for m in [L.suggest(L.experiments(L.observations(rows)), extra, "NOVA8-24M-v2")])   # settled: not tested again
