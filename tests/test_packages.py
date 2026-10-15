"""Packages for topics: the constitution's rule, a failed setting is never repeated, what worked goes first."""

from evo.learning import packages as P

RULES = {"min_topic_gain_pct": 1.0, "max_general_decline_pct": 2.0}


def test_the_rule_needs_a_better_field_a_bearable_cost_and_an_unchanged_base():
    ok = P.decide(RULES, 3.44, 2.93, 2.856, 2.87, True)
    assert ok["accept"] and ok["topic_gain_pct"] > 14 and 0 < ok["general_decline_pct"] < 1
    assert not P.decide(RULES, 3.04, 3.02, 2.856, 2.86, True)["accept"]           # the field barely moved
    assert not P.decide(RULES, 3.44, 2.93, 2.856, 2.95, True)["accept"]           # the rest suffered too much
    bad = P.decide(RULES, 3.44, 2.93, 2.856, 2.86, False)
    assert not bad["accept"] and any("base" in r for r in bad["reasons"])
    assert not P.decide(RULES, None, 2.9, 2.856, 2.86, True)["accept"]


def test_a_failed_setting_is_not_repeated_and_what_worked_goes_first():
    ix = {"packages": {}, "tries": {"chemia": [{"variant": 0}]}, "variant_wins": {"1": 2}}
    assert P.variant_order(ix)[0] == 1                                            # the setting that won twice
    assert P.next_variant(ix, "chemia") == 1 and P.next_variant(ix, "fyzika") == 1
    ix["tries"]["chemia"] += [{"variant": 1}, {"variant": 2}]
    assert P.next_variant(ix, "chemia") is None                                   # every setting tried: collect again


def test_a_crash_does_not_use_up_a_setting():
    ix = {"packages": {}, "tries": {"matematika": [{"variant": 0, "error": True}, {"variant": 1, "error": True}]}, "variant_wins": {}}
    assert P.next_variant(ix, "matematika") == 0


def test_basics_get_one_short_try():
    ix = {"packages": {}, "tries": {}, "variant_wins": {"1": 3}}
    assert P.next_variant(ix, "fyzika", basic=True) == 0                          # always the short one, not the winner
    ix["tries"]["fyzika"] = [{"variant": 0}]
    assert P.next_variant(ix, "fyzika", basic=True) is None                       # one try, then the next step (more material once)
    assert P.next_variant(ix, "fyzika") is not None
    assert P.BASIC["steps"] < min(v["steps"] for v in P.VARIANTS)
    assert P.is_basic("Geometria") and not P.is_basic("genetika")
