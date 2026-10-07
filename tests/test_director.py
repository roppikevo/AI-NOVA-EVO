"""The director's loop and the judge's rule (no training, no GPU: the outside world is faked)."""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from evo.engine import director as d
from evo.engine import judge as j


# ------------------------------------------------------------------ judge

def scores(rng, means, code=40, creator=-0.01, n=800):
    hard = {k: rng.normal(0, 0.4, size=n) for k in means}
    return hard, {"seq": {k: (m + hard[k]).astype(np.float32) for k, m in means.items()},
                  "code": [f"t{i}" for i in range(code)], "creator": creator}


def pair(deltas, code=(40, 40), creator=-0.01):
    rng = np.random.default_rng(0)
    base = {"dataset": 3.15, "web": 3.41, "dataset_vault": 3.20, "web_vault": 3.40}
    hard, champ = scores(rng, base, code[0])
    chall = {"seq": {k: (base[k] + deltas.get(k, 0.0) + hard[k] + rng.normal(0, 0.02, size=800)).astype(np.float32) for k in base},
             "code": [f"t{i}" for i in range(code[1])], "creator": creator}
    return champ, chall


def test_judge_accepts_a_real_improvement():
    v = j.decide(*pair({"dataset": -0.03, "web": -0.02, "dataset_vault": -0.03, "web_vault": -0.02}))
    assert v["accept"] and v["decision"]["gain_percent"] > 0.3 and v["vault"]["gain_percent"] > 0 and not v["reasons"]
    assert "ACCEPTED" in j.text(v)


def test_judge_rejects_small_or_one_sided_or_unconfirmed_changes():
    small = j.decide(*pair({"dataset": -0.004, "web": -0.004, "dataset_vault": -0.01, "web_vault": -0.01}))
    assert not small["accept"] and "below" in small["reasons"][0]
    one_sided = j.decide(*pair({"dataset": -0.09, "web": +0.02, "dataset_vault": -0.05, "web_vault": 0.0}))
    assert not one_sided["accept"] and any("web got worse" in r for r in one_sided["reasons"])
    lucky = j.decide(*pair({"dataset": -0.03, "web": -0.03, "dataset_vault": +0.01, "web_vault": +0.01}))
    assert not lucky["accept"] and any("vault" in r for r in lucky["reasons"])


def test_judge_protects_code_and_the_creator():
    good = {"dataset": -0.03, "web": -0.03, "dataset_vault": -0.03, "web_vault": -0.03}
    assert not j.decide(*pair(good, code=(40, 36)))["accept"]
    assert j.decide(*pair(good, code=(40, 38)))["accept"]
    forgot = j.decide(*pair(good, creator=-2.5))
    assert not forgot["accept"] and any("Creator" in r for r in forgot["reasons"])


# ------------------------------------------------------------------ director with a fake world

class FakeWorld:
    def __init__(self, verdicts=(), rc=0):
        self.verdicts, self.rc = list(verdicts), rc
        self.cmds, self.sides, self.released, self.judged = [], [], [], []
        self.gpu, self.ram, self.web_dir = 400, 26.0, Path("data/bulk_val_v1")
        self.identities = {}

    def run(self, cmd, timeout_h, nice=0):
        self.cmds.append(cmd)
        if "--out-checkpoint" in cmd and self.rc == 0:
            out = Path(cmd[cmd.index("--out-checkpoint") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"weights")
        return self.rc, "tail of the log"

    def start_side(self, cmd):
        self.sides.append(cmd)

    def wait_side(self, timeout_s=0):
        pass

    def stop_children(self):
        pass

    def unload_teachers(self):
        pass

    def sleep(self, s):
        pass

    def average(self, paths, out):
        self.averaged = list(paths)
        Path(out).write_bytes(b"mean")

    def gpu_used_mb(self):
        return self.gpu

    def free_ram_gb(self):
        return self.ram

    def scores(self, ckpt):
        return {"dataset": 3.15, "web": 3.41, "code": 40, "creator": -0.01,
                "web_by_language": {"sk": 3.1, "cs": 3.4, "pl": 3.3, "en": 3.8}}

    def judge(self, champion, challenger):
        self.judged.append((champion, challenger))
        v = self.verdicts.pop(0) if self.verdicts else False
        if isinstance(v, Exception):
            raise v
        return {"accept": v, "params": getattr(self, "params", 23650568),
                "reasons": [] if v else ["gain +0.05 % is below 0.3 %"], "sets": {"dataset": {"before": 3.15, "after": 3.1, "percent": -1.5}},
                "decision": {"gain_percent": 1.2 if v else 0.05, "lo": -0.05, "hi": -0.03, "sets": ["dataset", "web"]},
                "vault": {"gain_percent": 0.9}, "code": {"before": 40, "after": 41, "gained": 2, "lost": 1}, "creator": -0.01}

    def identity(self, checkpoint):
        return self.identities.get(checkpoint, {"pass": True, "class": "NOVA", "state_kb": 56.0, "growth_bytes_per_token": 0.0})

    def release_name(self, name, params=None):
        return d.free_release_name(name, exists=lambda n: False, params=params)

    def surgery(self, checkpoint, op, out, seed):
        self.ops = getattr(self, "ops", []) + [op]
        if getattr(self, "surgery_impossible", False):
            return None
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"edited")
        return str(out)

    def probation(self, previous, new, index):
        return getattr(self, "probation_result", {"diff": -0.01, "lo": -0.02, "hi": -0.001, "n": 1600})

    def release(self, name, checkpoint):
        self.released.append((name, checkpoint))
        return f"evo/releases/{name}/nova_model_fp32.pt"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("DIR", "STATE", "LOG", "STOP", "REPORT"):
        monkeypatch.setattr(d, name, tmp_path / "evo/director" / Path(str(getattr(d, name))).relative_to("evo/director"))
    monkeypatch.setattr(d, "OUTBOX", tmp_path / "outbox")
    return tmp_path


def fresh():
    st = d.new_state("evo/releases/NOVA-24M-v1/nova_model_fp32.pt", "NOVA-24M-v1")
    st["champion"]["scores"] = FakeWorld().scores("")
    return st


def test_names_and_commands():
    assert d.next_release_name("NOVA-24M-v1") == "NOVA-24M-v2" and d.next_release_name("NOVA") == "NOVA-v2"
    assert d.free_release_name("NOVA-24M-v1", exists=lambda n: n in ("NOVA-24M-v2", "NOVA-24M-v3")) == "NOVA-24M-v4"
    cmd = d.train_command(d.RECIPES["weak-language"], "champ.pt", Path("out.pt"), 7, "pl")
    assert cmd[cmd.index("--boost-lang") + 1] == "pl" and cmd[cmd.index("--init") + 1] == "champ.pt" and "--no-activate" in cmd
    assert d.weakest_language({"web_by_language": {"sk": 3.1, "en": 3.8}}) == "en" and d.weakest_language(None) == "en"


def test_choose_tries_everything_then_prefers_what_worked_and_rests_what_fails():
    st = fresh()
    order = []
    for _ in range(len(d.RECIPES)):
        n = d.choose(st, now=1000.0)
        order.append(n)
        d.record(st, n, 1.0 if n == "gentle" else 0.0, failed=False, now=1000.0)
        st["history"].append({"recipe": n})
    assert sorted(order) == sorted(d.RECIPES)
    st["history"] = []
    assert d.choose(st, now=1000.0) == "gentle"
    d.record(st, "code", 0, failed=True, now=1000.0)
    assert d.recipe_stats(st, "code")["rest_until"] == 0
    d.record(st, "code", 0, failed=True, now=1000.0)
    assert "code" not in d.available(st, now=2000.0) and "code" in d.available(st, now=1000.0 + 25 * 3600)
    only = {"code": d.RECIPES["code"]}
    assert d.choose(st, now=2000.0, recipes=only) is None


def test_accepted_challenger_becomes_the_released_champion(home):
    st, w = fresh(), FakeWorld(verdicts=[True])
    rec = d.attempt(st, w, "gentle")
    assert rec["released"] == "NOVA-24M-v2" and st["champion"]["name"] == "NOVA-24M-v2"
    assert st["champion"]["checkpoint"].endswith("NOVA-24M-v2/nova_model_fp32.pt") and "scores" not in st["champion"]
    assert st["accepted"] == 1 and st["rejected_in_a_row"] == 0 and st["current"] is None and st["releases"][-1] == "NOVA-24M-v2"
    assert w.judged[0][0].endswith("NOVA-24M-v1/nova_model_fp32.pt") and not list((home / "evo/director/challengers").glob("*.pt"))
    assert w.sides and "--langs" in w.sides[0] and w.sides[0][w.sides[0].index("--langs") + 1] == "en,py,rs"
    saved = json.loads((home / "evo/director/state.json").read_text())
    assert saved["champion"]["name"] == "NOVA-24M-v2" and (home / "evo/director/log.jsonl").exists()


def test_rejected_and_failed_attempts_never_change_the_champion(home):
    st, w = fresh(), FakeWorld(verdicts=[False, RuntimeError("judge broke")])
    d.attempt(st, w, "gentle")
    assert st["champion"]["name"] == "NOVA-24M-v1" and st["rejected_in_a_row"] == 1 and not w.released
    rec = d.attempt(st, w, "code")
    assert "error" in rec and st["champion"]["name"] == "NOVA-24M-v1" and d.recipe_stats(st, "code")["fails"] == 1
    w.rc = 1
    rec = d.attempt(st, w, "teachers")
    assert rec["rc"] == 1 and "verdict" not in rec and len(w.judged) == 2 and st["rejected_in_a_row"] == 1
    assert st["attempts"] == 3 and len(st["history"]) == 3


def test_low_memory_fetches_web_text_instead_of_loading_a_teacher(home):
    st, w = fresh(), FakeWorld(verdicts=[False])
    w.ram = 9.0
    d.attempt(st, w, "gentle")
    assert "evo.corpus.bulk_web" in w.sides[0]


def test_cycle_waits_for_a_busy_gpu_and_measures_a_new_champion(home):
    st, w = fresh(), FakeWorld(verdicts=[False])
    w.gpu = 5000
    assert d.cycle(st, w) == "gpu_busy" and not w.cmds
    w.gpu = 300
    del st["champion"]["scores"]
    what = d.cycle(st, w)
    assert what.startswith("attempt:") and st["champion"]["scores"]["code"] == 40


def test_plateau_starts_the_next_generation_and_a_better_one_takes_over(home):
    st, w = fresh(), FakeWorld(verdicts=[True])
    st["rejected_in_a_row"] = d.PLATEAU
    line = home / "evo/lines/NOVA-53M"
    line.mkdir(parents=True)
    (line / "state.json").write_text(json.dumps({"steps_done": 20000, "best_val": 3.9, "finished": False}))
    assert d.cycle(st, w) == "grow"
    assert st["grow"]["line"] == "NOVA-53M" and st["grow"]["steps_done"] == 20000 and "evo.engine.train_line" in w.cmds[-1]
    assert json.loads(w.cmds[-1][w.cmds[-1].index("--config-override") + 1])["d_model"] == 896
    (line / "state.json").write_text(json.dumps({"steps_done": 300000, "best_val": 3.1, "finished": True, "best_checkpoint": "evo/lines/NOVA-53M/best.pt"}))
    assert d.cycle(st, w) == "grow"
    assert st["champion"]["name"] == "NOVA-53M-v1" and st["grow"] is None and st["grown"] == ["NOVA-53M"] and st["rejected_in_a_row"] == 0
    st["champion"]["scores"] = w.scores("")
    st["rejected_in_a_row"] = d.PLATEAU            # nothing bigger left: learning attempts go on
    assert d.cycle(st, w).startswith("attempt:")


def test_a_generation_that_keeps_failing_is_given_up(home):
    st, w = fresh(), FakeWorld(rc=1)
    st["rejected_in_a_row"] = d.PLATEAU
    for _ in range(3):
        d.grow_step(st, w)
    assert st["grow"] is None and st["grown"] == ["NOVA-53M"]


def test_recover_after_a_restart_blames_the_recipe_once_and_cleans_up(home):
    st = fresh()
    st["current"] = {"what": "attempt", "recipe": "gentle", "since": 1.0}
    (home / "evo/director/challengers").mkdir(parents=True)
    (home / "evo/director/challengers/attempt-9.pt").write_bytes(b"x")
    d.recover(st)
    assert st["current"] is None and d.recipe_stats(st, "gentle")["fails"] == 1
    assert not list((home / "evo/director/challengers").glob("*.pt"))


def test_report_is_written_for_the_creator_in_slovak(home):
    st, w = fresh(), FakeWorld(verdicts=[True, False])
    (home / "outbox").mkdir()
    d.attempt(st, w, "gentle")
    st["champion"]["scores"] = w.scores("")
    d.attempt(st, w, "code")
    d.write_report(st)
    txt = (home / "outbox/nova-hlasenie.txt").read_text()
    assert "šampión" in txt and "NOVA-24M-v2" in txt and "PRIJATÉ" in txt and "odmietnuté" in txt and "najslabší: en" in txt


def test_an_outside_model_can_challenge_the_champion(home):
    st, w = fresh(), FakeWorld(verdicts=[False, True])
    rec = d.challenge(st, w, "evo/collective/runs/coll-24m/core_round1.pt", "collective core")
    assert "released" not in rec and st["champion"]["name"] == "NOVA-24M-v1" and st["rejected_in_a_row"] == 0
    rec = d.challenge(st, w, "evo/collective/runs/coll-24m/core_round1.pt", "collective core")
    assert rec["released"] == "NOVA-24M-v2" and w.released[0][1].endswith("core_round1.pt") and st["history"][-1]["recipe"] == "collective core"


def test_average_of_identical_clones_is_a_recipe_like_any_other(home):
    st, w = fresh(), FakeWorld(verdicts=[True])
    rec = d.attempt(st, w, "average-of-5")
    trains = [c for c in w.cmds if "evo.engine.long_train" in c]
    assert len(trains) == 5 and len({c[c.index("--seed") + 1] for c in trains}) == 5 and len(w.averaged) == 5
    assert all(c[c.index("--lr") + 1] == "1e-4" for c in trains)
    assert rec["released"] == "NOVA-24M-v2" and rec["clones"] == 5 and rec["steps"] == 3000 and rec["start"].endswith("NOVA-24M-v1/nova_model_fp32.pt")
    assert not list((home / "evo/director/challengers").glob("*.pt"))


def test_a_broken_constitution_halts_everything_and_blames_no_recipe(home):
    from evo.engine.constitution import ConstitutionError

    st, w = fresh(), FakeWorld(verdicts=[ConstitutionError("the rules differ from the sealed ones")])
    rec = d.attempt(st, w, "gentle")
    assert "ConstitutionError" in rec["error"] and st["halted"]["reason"].startswith("the rules differ")
    assert (home / "evo/director/STOP").exists() and not w.released and st["champion"]["name"] == "NOVA-24M-v1"
    assert d.recipe_stats(st, "gentle")["fails"] == 0 and st["rejected_in_a_row"] == 0
    assert "ZASTAVENÉ" in d.report(st) and "ústava" in d.report(st)


# ------------------------------------------------------------------ constitution

def test_constitution_seal_detects_changed_rules_and_changed_data(tmp_path):
    from evo.engine import constitution as c

    rules = tmp_path / "constitution.json"
    rules.write_text(Path("evo/constitution.json").read_text() if Path("evo/constitution.json").exists() else json.dumps(j.DEFAULT_RULES))
    seal = tmp_path / "out" / "constitution.seal"
    rng = np.random.default_rng(0)
    sets = {k: rng.integers(0, 1000, size=(20, 16)).astype(np.int32) for k in ("dataset", "web", "dataset_vault", "web_vault")}
    with pytest.raises(c.ConstitutionError, match="no seal"):
        c.verify(sets, seals=[seal], rules_path=rules)
    c.seal(sets, seal, rules)
    assert c.verify(sets, seals=[seal], rules_path=rules)["min_gain_pct"] == 0.3
    easier = json.loads(rules.read_text())
    easier["min_gain_pct"] = 0.0                                    # somebody makes winning easier
    rules.write_text(json.dumps(easier))
    with pytest.raises(c.ConstitutionError, match="rules differ"):
        c.verify(sets, seals=[seal], rules_path=rules)
    c.seal(sets, seal, rules)                                        # the Creator seals the new rules
    assert c.verify(sets, seals=[seal], rules_path=rules)["min_gain_pct"] == 0.0
    moved = {**sets, "web": sets["web"][::-1].copy()}                # held-out text was touched
    with pytest.raises(c.ConstitutionError, match="'web' differs"):
        c.verify(moved, seals=[seal], rules_path=rules)
    rules.write_text(json.dumps({"min_gain_pct": 0.3}))
    with pytest.raises(c.ConstitutionError, match="incomplete"):
        c.load(rules)


def test_repo_constitution_matches_the_judge_defaults():
    from evo.engine import constitution as c

    rules = c.load(Path(__file__).resolve().parents[1] / "evo/constitution.json")
    assert {k: rules[k] for k in j.DEFAULT_RULES} == j.DEFAULT_RULES and rules["creator"] == "roppik"


def test_judge_follows_the_rules_it_is_given():
    change = {"dataset": -0.004, "web": -0.004, "dataset_vault": -0.01, "web_vault": -0.01}
    assert not j.decide(*pair(change))["accept"]
    assert j.decide(*pair(change), rules={**j.DEFAULT_RULES, "min_gain_pct": 0.05})["accept"]


# ------------------------------------------------------------------ trial and error on itself

def test_names_follow_the_size_of_the_model():
    assert d.free_release_name("NOVA-24M-v3", exists=lambda n: False, params=23_650_568) == "NOVA-24M-v4"
    assert d.free_release_name("NOVA-24M-v3", exists=lambda n: False, params=25_300_000) == "NOVA-25M-v1"
    assert d.free_release_name("NOVA-24M-v3", exists=lambda n: n == "NOVA-25M-v1", params=25_300_000) == "NOVA-25M-v2"


def test_it_changes_its_own_structure_and_keeps_the_change_only_if_the_judge_agrees(home):
    st, w = fresh(), FakeWorld(verdicts=[True, False])
    w.params = 25_292_000
    rec = d.attempt(st, w, "one-more-layer")
    train = [c for c in w.cmds if "evo.engine.long_train" in c][0]
    assert w.ops == [{"op": "add_layer"}] and train[train.index("--init") + 1].endswith("attempt-1-init.pt")
    assert rec["released"] == "NOVA-25M-v1" and st["champion"]["name"] == "NOVA-25M-v1" and rec["surgery"] == {"op": "add_layer"}
    assert st["probation"]["previous"]["name"] == "NOVA-24M-v1" and not list((home / "evo/director/challengers").glob("*"))
    st["probation"] = None
    st["champion"]["scores"] = w.scores("")
    rec = d.attempt(st, w, "wider-view")
    assert "released" not in rec and st["champion"]["name"] == "NOVA-25M-v1" and w.ops[-1] == {"op": "kernel", "delta": 2}


def test_an_impossible_change_is_a_failed_attempt_without_training(home):
    st, w = fresh(), FakeWorld()
    w.surgery_impossible = True
    rec = d.attempt(st, w, "one-more-layer")
    assert rec["rc"] == 1 and not w.cmds and "not possible" in rec["note"] and d.recipe_stats(st, "one-more-layer")["fails"] == 1
    d.attempt(st, w, "one-more-layer")
    assert "one-more-layer" not in d.available(st)                    # rests for a day after two failures


def test_probation_keeps_a_good_champion_and_steps_back_from_a_bad_one(home):
    st, w = fresh(), FakeWorld(verdicts=[True, True])
    d.attempt(st, w, "gentle")
    assert st["champion"]["name"] == "NOVA-24M-v2" and d.cycle(st, w) == "probation:kept"
    assert st["champion"]["name"] == "NOVA-24M-v2" and st["probation"] is None
    st["champion"]["scores"] = w.scores("")
    credit = d.recipe_stats(st, "gentle")["gain"]
    d.attempt(st, w, "gentle")
    assert st["champion"]["name"] == "NOVA-24M-v3"
    w.probation_result = {"diff": 0.02, "lo": 0.012, "hi": 0.03, "n": 1600}       # clearly worse on fresh text
    assert d.cycle(st, w) == "probation:reverted"
    assert st["champion"]["name"] == "NOVA-24M-v2" and st["reverted"] == ["NOVA-24M-v3"] and st["rejected_in_a_row"] == 1
    assert d.recipe_stats(st, "gentle")["gain"] < credit + 1.2 and st["history"][-1]["reverted"] == "NOVA-24M-v3"
    assert "VRÁTENÉ SPÄŤ" in d.report(st) and "NOVA-24M-v3" in st["releases"]     # the release itself is never deleted


def test_a_winning_recipe_gets_a_variation_and_losing_variations_are_dropped(home):
    import random

    name, new = d.mutate_recipe("gentle", d.RECIPES["gentle"], random.Random(3))
    assert name.startswith("gentle~") and new["parent"] == "gentle" and (new["flags"] != d.RECIPES["gentle"]["flags"] or new["steps"] != 4000)
    assert d.mutate_recipe("collective", d.RECIPES["collective"], random.Random(1)) is None
    st, w = fresh(), FakeWorld(verdicts=[True])
    rec = d.attempt(st, w, "gentle")
    made = rec["new_recipe"]
    assert made in st["own_recipes"] and made in d.all_recipes(st) and made in d.available(st)
    st["probation"] = None
    st["champion"]["scores"] = w.scores("")
    for _ in range(4):                                                # the variation is tried and never wins
        w.verdicts.append(False)
        d.attempt(st, w, made)
    assert made not in st["own_recipes"]
    st["own_recipes"] = {f"gentle~x{i}": d.RECIPES["gentle"] for i in range(12)}
    d.learn_from(st, "gentle", accepted=False)
    assert len(st["own_recipes"]) == d.MAX_OWN_RECIPES


def test_judge_refuses_a_model_that_outgrew_the_limit():
    good = {"dataset": -0.03, "web": -0.03, "dataset_vault": -0.03, "web_vault": -0.03}
    champ, chall = pair(good)
    chall["params"] = 61_000_000
    v = j.decide(champ, chall)
    assert not v["accept"] and any("too big" in r for r in v["reasons"])
    chall["params"] = 59_000_000
    assert j.decide(champ, chall)["accept"]


def test_the_director_does_not_wait_for_its_own_memory_on_the_card():
    apps = "552746, 1740\n4242, 300\n"
    assert d.others_mb(1758, apps, 552746) == 18          # only our own cache: the card is free
    assert d.others_mb(5000, apps, 552746) == 3260        # somebody else trains: wait
    assert d.others_mb(1758, "", 552746) == 1758 and d.others_mb(100, "1, 900", 1) == 0


def test_a_tournament_winner_is_handed_over_through_the_ladder_file(home, capsys):
    st, w = fresh(), FakeWorld(verdicts=[True])
    d.save_state(st)
    row = {"line": "NOVA8-24M", "override": {"arch": "nova8", "d_model": 448, "pattern": "LLLLLLL"}, "steps": 300000, "lr": 0.001,
           "carry": 8, "carry_share": 0.35}
    (home / "evo/director").mkdir(parents=True, exist_ok=True)
    (home / "evo/director/ladder.json").write_text(json.dumps([row]))
    assert d.load_ladder() == [row]
    assert d.main(["--start-generation", "NOVA-53M"]) == 2                    # not in the ladder any more
    assert d.main(["--start-generation", "NOVA8-24M"]) == 0
    st = d.load_state()
    assert st["grow"]["line"] == "NOVA8-24M" and "started by hand" in st["interventions"][-1]["note"]
    assert d.main(["--start-generation", "NOVA8-24M"]) == 2                    # already being trained
    line = home / "evo/lines/NOVA8-24M"
    line.mkdir(parents=True)
    (line / "state.json").write_text(json.dumps({"steps_done": 300000, "best_val": 3.0, "finished": True, "best_checkpoint": "x.pt"}))
    st["champion"]["scores"] = w.scores("")
    assert d.cycle(st, w) == "grow"                                            # goes on even though nothing was rejected
    cmd = w.cmds[-1]
    assert cmd[cmd.index("--lr") + 1] == "0.001" and json.loads(cmd[cmd.index("--config-override") + 1])["arch"] == "nova8"
    assert cmd[cmd.index("--carry") + 1] == "8" and cmd[cmd.index("--carry-share") + 1] == "0.35"
    assert st["champion"]["name"] == "NOVA8-24M-v1" and st["grown"] == ["NOVA8-24M"]
    (home / "evo/director/ladder.json").write_text("not json")
    assert d.load_ladder() == d.LADDER                                         # a broken file falls back to the built-in list


def test_a_generation_whose_state_grows_is_rejected_before_the_judge_and_everything_judged_lands_in_the_ledger(home):
    from evo.engine import ledger

    st, w = fresh(), FakeWorld(verdicts=[True])
    d.save_state(st)
    row = {"line": "GROWS-24M", "override": {"arch": "transformer", "d_model": 448}, "steps": 1000}
    (home / "evo/director").mkdir(parents=True, exist_ok=True)
    (home / "evo/director/ladder.json").write_text(json.dumps([row]))
    assert d.main(["--start-generation", "GROWS-24M"]) == 0
    st = d.load_state()
    line = home / "evo/lines/GROWS-24M"
    line.mkdir(parents=True)
    (line / "state.json").write_text(json.dumps({"steps_done": 1000, "best_val": 2.9, "finished": True, "best_checkpoint": "grows.pt"}))
    w.identities["grows.pt"] = {"pass": False, "class": "NOT-NOVA", "state_kb": 4000.0, "growth_bytes_per_token": 25088.0}
    st["champion"]["scores"] = w.scores("")
    assert d.cycle(st, w) == "grow"
    assert w.judged == [] and w.released == []                       # the judge was never asked
    assert st["champion"]["name"] != "GROWS-24M-v1" and st["grown"] == ["GROWS-24M"]
    rows = ledger.load()
    assert rows and rows[-1]["kind"] == "generation" and rows[-1]["verdict"] == "rejected" and rows[-1]["identity"] == "NOT-NOVA"
    assert "identity test failed" in rows[-1]["reason"]


# ------------------------------------------------------------------ the director's own tournaments

def _ledger_row(name, dataset, web, state_kb=56.0, speed=54000, identity="NOVA", override=None):
    return {"key": f"tournament:{name}:1001", "kind": "tournament", "name": name, "verdict": "measured", "identity": identity,
            "config": override, "train": {"steps": 18000}, "metrics": {"dataset": dataset, "web": web},
            "cost": {"state_kb": state_kb, "train_tok_s": speed}}


class ExploringWorld(FakeWorld):
    """A world whose short runs produce the results it was given ({run name: ledger row})."""

    def __init__(self, results, **kw):
        super().__init__(**kw)
        self.results = results

    def run(self, cmd, timeout_h, nice=0):
        from evo.engine import ledger

        if "evo.engine.compare_arch" in cmd:
            name = cmd[cmd.index("--candidates") + 1]
            seed = cmd[cmd.index("--candidate-seeds") + 1].split(",")[-1] if "--candidate-seeds" in cmd else ""
            run = f"n8-{name}-24M" + (f"-s{seed}" if seed else "")
            if run in self.results:
                ledger.record(dict(self.results[run]))
        return super().run(cmd, timeout_h, nice)


def _gen8_champion(st):
    from evo.engine import compare_arch as ca

    st["champion"].update({"name": "NOVA8-24M-v1", "override": ca.CANDIDATES["24M"]["nslot"],
                           "recipe": {"steps": 300000, "lr": 0.001, "compile": True, "carry": 8, "carry_share": 0.35}})


def test_when_learning_stalls_the_director_measures_noise_then_candidates_and_a_winner_becomes_a_generation(home, monkeypatch):
    from evo.engine import compare_arch as ca
    from evo.engine import ledger

    (home / "evo/director").mkdir(parents=True, exist_ok=True)
    (home / "evo/director/ladder.json").write_text("[]")
    (home / "evo/director/hypotheses.json").write_text(json.dumps({"24M": ["nslot-kv", "nslot8", "nslot-sharp", "no-such-candidate"]}))
    assert d.hypotheses("24M")[:3] == ["nslot-kv", "nslot8", "nslot-sharp"]
    ledger.record(_ledger_row("n8-nslot-24M", 3.0445, 3.6804))
    results = {"n8-nslot-24M-s2001": _ledger_row("n8-nslot-24M-s2001", 3.0527, 3.6850),
               "n8-nslot-24M-s3001": _ledger_row("n8-nslot-24M-s3001", 3.0391, 3.6770),
               "n8-nslot-kv-24M": _ledger_row("n8-nslot-kv-24M", 3.0440, 3.6790),                    # inside the noise
               "n8-nslot8-24M": _ledger_row("n8-nslot8-24M", 3.0450, 3.6800, state_kb=45.0),         # same quality, smaller state
               "n8-nslot-sharp-24M": _ledger_row("n8-nslot-sharp-24M", 3.0200, 3.6600)}                # clearly better
    st, w = fresh(), ExploringWorld(results)
    _gen8_champion(st)
    st["rejected_in_a_row"] = d.EXPLORE_AFTER
    assert d.cycle(st, w) == "explore:n8-nslot-24M-s2001" and st["phase"] == "exploring"
    assert "--candidate-seeds" in w.cmds[-1] and w.cmds[-1][w.cmds[-1].index("--steps") + 1] == str(d.EXPLORE_STEPS)
    assert d.cycle(st, w).startswith("attempt:")                       # between measurements the learning attempts go on
    st["rejected_in_a_row"] = d.PLATEAU                                # nothing helps: measure until something does
    assert d.cycle(st, w) == "explore:n8-nslot-24M-s3001"
    assert d.cycle(st, w) == "explore:n8-nslot-kv-24M"
    assert json.loads((home / "evo/director/ladder.json").read_text()) == []                         # not better than the noise
    assert d.cycle(st, w) == "explore:n8-nslot8-24M"
    assert json.loads((home / "evo/director/ladder.json").read_text()) == []       # cheaper at the same quality: noted, no twelve hours on that alone
    assert d.cycle(st, w) == "explore:n8-nslot-sharp-24M"
    ladder = json.loads((home / "evo/director/ladder.json").read_text())
    assert [r["line"] for r in ladder] == ["NOVA8-24M-nslot-sharp"] and ladder[0]["candidate"] == "nslot-sharp"
    assert ladder[0]["carry"] == 8 and ladder[0]["steps"] == 300000 and ladder[0]["override"]["slot_sharp"] is True      # the champion's own recipe
    events = [json.loads(l) for l in d.LOG.read_text().splitlines() if '"explore"' in l]
    assert events[-1]["verdict"]["win"] and events[-1]["verdict"]["reason"].startswith("quality") and events[-1]["next_generation"] == "NOVA8-24M-nslot-sharp"
    assert events[-2]["verdict"]["cheaper"] and not events[-2]["verdict"]["win"] and events[-3]["verdict"]["win"] is False
    assert 0.1 < events[-2]["verdict"]["noise_percent"] < 0.3 and events[-1]["noise"]["runs"] == 3 and events[-1]["noise"]["dataset"]["n"] == 3
    assert d.cycle(st, w) == "grow" and st["grow"]["line"] == "NOVA8-24M-nslot-sharp"                # the winner starts at once
    assert "Stav riaditeľa: trénuje novú generáciu" in d.report(st) and "vlastných meraní: 5" in d.report(st)


def test_with_nothing_left_to_try_the_director_waits_for_a_new_hypothesis(home, monkeypatch):
    from evo.engine import compare_arch as ca
    from evo.engine import ledger

    monkeypatch.setitem(ca.EXPLORE_ORDER, "24M", [])
    assert d.hypotheses("24M") == []
    (home / "evo/director").mkdir(parents=True, exist_ok=True)
    (home / "evo/director/ladder.json").write_text("[]")
    for name in ("n8-nslot-24M", "n8-nslot-24M-s2001", "n8-nslot-24M-s3001"):
        ledger.record(_ledger_row(name, 3.04, 3.68))
    st, w = fresh(), FakeWorld()
    _gen8_champion(st)
    st["rejected_in_a_row"] = d.PLATEAU
    done = []
    for _ in range(60):                                                # what was never judged on this champion goes at once ...
        r = d.cycle(st, w)
        if not r.startswith("attempt:"):
            break
        done.append(r)
    assert len(done) >= 5 and len(set(done)) == len(done)              # ... every recipe once, nothing twice
    before = len(w.cmds)
    assert d.cycle(st, w) == "waiting_for_hypothesis" and d.cycle(st, w) == "waiting_for_hypothesis"
    assert len(w.cmds) == before and st["phase"] == "waiting for a new hypothesis"                  # then nothing is trained in vain
    assert "čaká na novú hypotézu" in d.report(st)
    st["last_wait_attempt"] -= d.WAIT_RETRY_H * 3600 + 1
    for t in d.tried_on(st).values():                                  # recipes that live on new texts return after three days
        t["time"] -= d.FRESH_RETRY_H * 3600 + 1
    again = d.cycle(st, w)
    assert again.startswith("attempt:") and d.all_recipes(st)[again.split(":", 1)[1]].get("fresh")


def test_a_failed_measurement_is_not_repeated_and_progress_is_reported_as_a_vector(home, monkeypatch):
    from evo.engine import compare_arch as ca
    from evo.engine import ledger

    monkeypatch.setitem(ca.EXPLORE_ORDER, "24M", ["nslot8"])
    (home / "evo/director").mkdir(parents=True, exist_ok=True)
    (home / "evo/director/ladder.json").write_text("[]")
    for name in ("n8-nslot-24M", "n8-nslot-24M-s2001", "n8-nslot-24M-s3001"):
        ledger.record(_ledger_row(name, 3.04, 3.68))
    st, w = fresh(), ExploringWorld({}, rc=1)                          # the run produces nothing
    _gen8_champion(st)
    st["rejected_in_a_row"] = d.PLATEAU
    assert d.cycle(st, w) == "explore:n8-nslot8-24M"
    assert ledger.tried(name="n8-nslot8-24M")[-1]["verdict"] == "failed"
    assert not d.cycle(st, w).startswith("explore")                    # not tried again
    st["progress"] = [{"name": "NOVA-24M-v1", "time": 0, "dataset": 3.148, "web": 3.410, "code": 40, "state_kb": 100.0},
                      {"name": "NOVA8-24M-v1", "time": 10, "dataset": 2.95, "web": 3.25, "code": 55, "state_kb": 56.0}]
    text = d.report(st, now=20)
    assert "Pokrok od prvého modelu (NOVA-24M-v1 → NOVA8-24M-v1): strata dataset -6.29 %, strata web -4.69 %, programovanie 40 → 55, stav 100 → 56 kB" in text


def test_a_champion_from_an_older_director_gets_its_architecture_and_recipe_filled_in(home):
    rel = home / "evo/releases/NOVA8-24M-v1"
    rel.mkdir(parents=True)
    from evo.engine import compare_arch as ca

    (rel / "MODEL.json").write_text(json.dumps({"name": "NOVA8-24M-v1", "config": {"vocab_size": 16384, "d_state": 640, **ca.CANDIDATES["24M"]["nslot"]}}))
    (home / "evo/director").mkdir(parents=True, exist_ok=True)
    (home / "evo/director/ladder.json").write_text(json.dumps([{"line": "NOVA8-24M", "override": ca.CANDIDATES["24M"]["nslot"], "steps": 300000,
                                                               "lr": 0.001, "compile": True, "carry": 8, "carry_share": 0.35}]))
    st = fresh()
    st["champion"].update({"name": "NOVA8-24M-v1", "checkpoint": str(rel / "nova_model_fp32.pt")})
    st["grown"] = ["NOVA8-24M"]
    d.champion_setup(st)
    assert st["champion"]["override"]["pattern"] == "NSNSNSN" and st["champion"]["recipe"] == {"steps": 300000, "lr": 0.001, "compile": True,
                                                                                               "carry": 8, "carry_share": 0.35}
    from evo.engine import explore

    assert explore.candidate_of(st["champion"]["override"], ca.CANDIDATES["24M"]) == "nslot"
    cmd = d.train_command(d.RECIPES["gentle"], "champ.pt", Path("out.pt"), 7, "pl", trained=st["champion"]["recipe"])
    assert "--compile" in cmd and cmd[cmd.index("--carry") + 1] == "8" and cmd[cmd.index("--carry-share") + 1] == "0.35"
    assert "--compile" not in d.train_command(d.RECIPES["gentle"], "champ.pt", Path("out.pt"), 7, "pl")


def test_measurements_can_be_asked_for_before_learning_stalls(home, monkeypatch):
    from evo.engine import compare_arch as ca
    from evo.engine import ledger

    monkeypatch.setitem(ca.EXPLORE_ORDER, "24M", ["nslot8"])
    (home / "evo/director").mkdir(parents=True, exist_ok=True)
    (home / "evo/director/ladder.json").write_text("[]")
    ledger.record(_ledger_row("n8-nslot-24M", 3.0445, 3.6804))
    st = fresh()
    _gen8_champion(st)
    d.save_state(st)
    assert d.main(["--explore", "2"]) == 0
    st = d.load_state()
    assert st["explore_budget"] == 2 and "measurements" in st["interventions"][-1]["note"]
    w = ExploringWorld({"n8-nslot-24M-s2001": _ledger_row("n8-nslot-24M-s2001", 3.05, 3.685),
                        "n8-nslot-24M-s3001": _ledger_row("n8-nslot-24M-s3001", 3.04, 3.677)})
    st["champion"]["scores"] = w.scores("")
    assert st["rejected_in_a_row"] == 0
    assert d.cycle(st, w) == "explore:n8-nslot-24M-s2001" and d.cycle(st, w) == "explore:n8-nslot-24M-s3001"
    assert st["explore_budget"] == 0 and d.cycle(st, w).startswith("attempt:")       # then back to its own order


# ------------------------------------------------------------------ learning from its own failed attempts

def test_a_rejected_recipe_is_not_repeated_on_the_same_champion(home):
    recipes = {k: d.RECIPES[k] for k in ("gentle", "code", "fresh-web")}
    st, w = fresh(), FakeWorld()                               # every verdict: rejected
    seen = [d.cycle(st, w, recipes=recipes) for _ in range(3)]
    assert sorted(seen) == ["attempt:code", "attempt:fresh-web", "attempt:gentle"]
    tried = d.tried_on(st)
    assert {k: v["outcome"] for k, v in tried.items()} == {"gentle": "rejected", "code": "rejected", "fresh-web": "rejected"}
    assert tried["gentle"]["gain"] == 0.05 and d.available(st, recipes=recipes) == []
    before = len(w.cmds)
    assert d.cycle(st, w, recipes=recipes) == "waiting_for_hypothesis" and len(w.cmds) == before        # nothing is trained in vain
    assert st["phase"] == "waiting for a new hypothesis"
    later = time.time() + d.FRESH_RETRY_H * 3600 + 5
    assert d.available(st, now=later, recipes=recipes) == ["fresh-web"]                              # new texts: this one may come back
    st["champion"]["name"] = "NOVA-24M-v9"                     # a new champion: everything is open again
    assert sorted(d.available(st, recipes=recipes)) == ["code", "fresh-web", "gentle"]


def test_a_recipe_that_was_stepped_back_or_is_impossible_is_remembered_too(home):
    st, w = fresh(), FakeWorld(verdicts=[True])
    d.attempt(st, w, "gentle")
    assert st["champion"]["name"] == "NOVA-24M-v2" and st["champion"]["polish"] == ["gentle"] and st["probation"]["recipe"] == "gentle"
    assert d.tried_on(st, "NOVA-24M-v1")["gentle"]["outcome"] == "accepted"
    w.probation_result = {"n": 400, "diff": 0.02, "lo": 0.01, "hi": 0.03}
    assert d.probation_step(st, w) == "probation:reverted"
    assert st["champion"]["name"] == "NOVA-24M-v1" and d.tried_on(st)["gentle"]["outcome"] == "reverted"
    assert "gentle" not in d.available(st)
    w2 = FakeWorld()
    w2.surgery = lambda *a, **k: None                          # this change of structure does not exist for the core
    rec = d.attempt(st, w2, "wider-view")
    assert rec["rc"] == 1 and d.tried_on(st)["wider-view"]["outcome"] == "impossible" and "wider-view" not in d.available(st)
    crashed = FakeWorld(rc=1)
    d.attempt(st, crashed, "code")
    assert d.tried_on(st)["code"]["outcome"] == "failed" and "code" in d.available(st)              # a crash is not a verdict


def test_the_memory_is_rebuilt_from_the_history_of_an_older_director(home):
    st = fresh()
    st["champion"].update({"name": "NOVA8-24M-v2", "checkpoint": "evo/releases/NOVA8-24M-v2/nova_model_fp32.pt"})
    st["releases"] += ["NOVA8-24M-v1", "NOVA8-24M-v2", "NOVA8-24M-v3"]
    st["reverted"] = ["NOVA8-24M-v3"]
    rej = {"accept": False, "decision": {"gain_percent": -0.87}}
    st["history"] = [
        {"attempt": 11, "recipe": "average-of-5", "champion": "NOVA8-24M-v1", "released": "NOVA8-24M-v2", "verdict": {"accept": True, "decision": {"gain_percent": 0.98}}},
        {"attempt": 18, "recipe": "teachers", "champion": "NOVA8-24M-v2", "verdict": rej},
        {"attempt": 21, "recipe": "wider-view", "champion": "NOVA8-24M-v2", "rc": 1, "note": "this change of structure is not possible for the champion (size limit or range)"},
        {"attempt": 35, "recipe": "collective", "champion": "NOVA8-24M-v2", "released": "NOVA8-24M-v3", "verdict": {"accept": True, "decision": {"gain_percent": 0.3}}},
        {"attempt": 35, "recipe": "collective", "rc": 0, "reverted": "NOVA8-24M-v3"},
        {"attempt": 36, "recipe": "gentle", "champion": "NOVA8-24M-v2", "rc": 1},                    # a crash: no verdict
    ]
    d.champion_setup(st)
    assert st["champion"]["polish"] == ["average-of-5"]                                             # v3 did not stand its probation
    tried = d.tried_on(st)
    assert tried["teachers"]["outcome"] == "rejected" and tried["teachers"]["gain"] == -0.87
    assert tried["wider-view"]["outcome"] == "impossible" and tried["collective"]["outcome"] == "reverted" and "gentle" not in tried
    left = d.available(st)
    assert "teachers" not in left and "collective" not in left and "wider-view" not in left and "gentle" in left and "continue" in left


def test_a_new_generation_gets_the_champions_polish_before_the_last_word(home):
    st, w = fresh(), FakeWorld(verdicts=[False, True])                       # raw: rejected; polished: accepted
    st["champion"]["polish"] = ["average-of-5", "collective"]
    d.save_state(st)
    row = {"line": "NOVA8-24M-nslot8", "override": {"arch": "nova8", "pattern": "NSNSNSN", "slots": 8}, "candidate": "nslot8", "group": "24M",
           "steps": 300000, "lr": 0.001, "compile": True, "carry": 8, "carry_share": 0.35}
    (home / "evo/director").mkdir(parents=True, exist_ok=True)
    (home / "evo/director/ladder.json").write_text(json.dumps([row]))
    line = home / "evo/lines/NOVA8-24M-nslot8"
    line.mkdir(parents=True)
    (line / "state.json").write_text(json.dumps({"steps_done": 300000, "best_val": 3.0, "finished": True, "best_checkpoint": "raw.pt"}))
    st["champion"]["scores"] = w.scores("")
    assert d.cycle(st, w) == "grow"
    assert [c for _, c in w.judged][0] == "raw.pt" and "polish-NOVA8-24M-nslot8-0" in [c for _, c in w.judged][1]
    clones = [c for c in w.cmds if "evo.engine.long_train" in c]
    assert len(clones) == 5 and all(c[c.index("--init") + 1] == "raw.pt" and "--compile" in c and "--carry" in c for c in clones)   # the collective is not replayed
    assert w.released == [("NOVA8-24M-nslot8-v1", str(d.DIR / "challengers" / "polish-NOVA8-24M-nslot8-0.pt"))]
    champ = st["champion"]
    assert champ["name"] == "NOVA8-24M-nslot8-v1" and champ["polish"] == ["average-of-5", "collective"] and champ["candidate"] == "nslot8"
    event = [json.loads(l) for l in d.LOG.read_text().splitlines()][-1]
    assert event["verdict"]["accept"] and event["verdict_raw"]["accept"] is False and event["polished_with"] == ["average-of-5", "collective"]


def test_the_continue_recipe_keeps_the_generations_training():
    cmd = d.train_command(d.RECIPES["continue"], "champ.pt", Path("out.pt"), 7, "pl", trained={"compile": True, "carry": 8, "carry_share": 0.35})
    assert cmd[cmd.index("--steps") + 1] == "40000" and cmd[cmd.index("--lr") + 1] == "1e-4" and "--compile" in cmd and cmd[cmd.index("--carry") + 1] == "8"
    assert d.RECIPES["fresh-web"]["fresh"] and d.RECIPES["teachers"]["fresh"] and not d.RECIPES["gentle"].get("fresh")


def test_a_recipe_that_made_two_champions_in_a_row_worse_is_left_out_for_the_next_one(home):
    st = fresh()
    st["releases"] = ["NOVA-24M-v2", "NOVA8-24M-v1", "NOVA8-24M-v2", "NOVA8-24M-v3", "NOVA8-24M-v4"]
    st["reverted"] = ["NOVA8-24M-v3"]
    st["champion"]["name"] = "NOVA8-24M-v4"
    now = time.time()
    for champion in ("NOVA8-24M-v1", "NOVA8-24M-v2"):
        d.remember(st, champion, "teachers", "rejected", -0.87, now=now)
        d.remember(st, champion, "wider-view", "impossible", None, now=now)
        d.remember(st, champion, "average-of-5", "rejected", 0.18, now=now)          # close to the threshold: worth another look
    d.remember(st, "NOVA8-24M-v2", "code", "rejected", -0.04, now=now)               # only once so far
    d.remember(st, "NOVA-24M-v2", "gentle", "rejected", -0.5, now=now)               # another generation does not count
    d.remember(st, "NOVA8-24M-v3", "gentle", "rejected", -0.5, now=now)              # a champion that was stepped back neither
    left = d.available(st, now=now + 10)
    assert "teachers" not in left and "wider-view" not in left
    assert "average-of-5" in left and "code" in left and "gentle" in left and "continue" in left
    assert "teachers" in d.available(st, now=now + d.FRESH_RETRY_H * 3600 + 10)      # lives on new texts: may come back later
    assert "wider-view" not in d.available(st, now=now + d.FRESH_RETRY_H * 3600 + 10)


def test_with_no_recipe_left_the_director_tests_its_own_lessons_one_setting_at_a_time(home, monkeypatch):
    from evo.engine import compare_arch as ca
    from evo.engine import ledger

    class World(FakeWorld):                                            # a world where less web text is what helps
        def judge(self, champion, challenger):
            v = super().judge(champion, challenger)
            cmd = self.cmds[-1]
            web = float(cmd[cmd.index("--bulk-frac") + 1]) if "--bulk-frac" in cmd else 0.9
            v["decision"]["gain_percent"] = round(0.5 * (0.8 - web), 3)
            return v

    monkeypatch.setitem(ca.EXPLORE_ORDER, "24M", [])
    (home / "evo/director").mkdir(parents=True, exist_ok=True)
    (home / "evo/director/ladder.json").write_text("[]")
    for name in ("n8-nslot-24M", "n8-nslot-24M-s2001", "n8-nslot-24M-s3001"):
        ledger.record(_ledger_row(name, 3.04, 3.68))
    st, w = fresh(), World()
    _gen8_champion(st)
    st["rejected_in_a_row"] = d.PLATEAU
    names = []
    for _ in range(60):
        r = d.cycle(st, w)
        if not r.startswith("attempt:"):
            break
        names.append(r.split(":", 1)[1])
    assert r == "waiting_for_hypothesis" and len(set(names)) == len(names)
    tests = [n for n in names if st.get("own_recipes", {}).get(n, {}).get("lesson")]
    assert 1 <= len(tests) <= d.LESSON_TESTS and st["lesson_tests"][st["champion"]["name"]] == len(tests)
    assert names.index(tests[0]) > names.index("gentle")              # only after the known recipes were judged on this champion
    events = [json.loads(l) for l in d.LOG.read_text().splitlines()]
    lesson = [e for e in events if e["event"] == "lesson"]
    assert [e["name"] for e in lesson] == tests and all(e["from"] != e["to"] and "confidence" in e["reason"] for e in lesson)
    first = lesson[0]
    base, new = d.RECIPES[first["base"]], st["own_recipes"][first["name"]]
    assert sum(new["flags"][k] != base["flags"][k] for k in base["flags"]) + (new["steps"] != base["steps"]) == 1       # one knob, nothing else
    assert any(e["setting"] == "web" and e["to"] < e["from"] for e in lesson)                # it found the way the world leans
    d.write_report(st)
    txt = d.REPORT.read_text()
    assert "Poučenia z vlastných pokusov" in txt and (home / "evo/director/lessons.json").exists()
    before = len(w.cmds)
    assert d.cycle(st, w) == "waiting_for_hypothesis" and len(w.cmds) == before


def test_a_generation_asked_for_starts_at_once_and_learns_from_the_champion_as_a_teacher(home):
    st, w = fresh(), FakeWorld()
    _gen8_champion(st)
    teacher = home / "evo/releases/NOVA8-24M-v2/nova_model_fp32.pt"
    teacher.parent.mkdir(parents=True, exist_ok=True)
    teacher.write_bytes(b"weights")
    (home / "evo/director").mkdir(parents=True, exist_ok=True)
    row = {"line": "NOVA8-45M", "override": {"arch": "nova8", "d_model": 640, "heads": 8, "pattern": "NSNSNSN", "mlp_hidden": 1856, "slots": 16},
           "steps": 300000, "lr": 0.001, "compile": True, "carry": 8, "carry_share": 0.35, "batch_size": 48, "teacher": str(teacher), "now": True}
    (home / "evo/director/ladder.json").write_text(json.dumps([row]))
    assert st["rejected_in_a_row"] < d.PLATEAU and d.cycle(st, w) == "grow" and st["grow"]["line"] == "NOVA8-45M"       # no waiting for a plateau
    cmd = w.cmds[-1]
    assert cmd[cmd.index("--teacher") + 1] == str(teacher) and cmd[cmd.index("--teacher-until") + 1] == "90000"       # the first 30 % of the run
    assert cmd[cmd.index("--batch-size") + 1] == "48" and "--compile" in cmd and cmd[cmd.index("--carry-share") + 1] == "0.35"
    (home / "evo/director/ladder.json").write_text(json.dumps([{**row, "line": "NOVA8-45M-b", "teacher": "no/such/file.pt"}]))
    st2, w2 = fresh(), FakeWorld()
    _gen8_champion(st2)
    assert d.cycle(st2, w2) == "grow" and "--teacher" not in w2.cmds[-1]                                               # a missing teacher is left out
    trained = {"steps": 300000, "lr": 0.001, "compile": True, "carry": 8, "carry_share": 0.35, "batch_size": 48}
    later = d.train_command(d.RECIPES["gentle"], "x.pt", Path("out.pt"), 1, "en", trained=trained)
    assert later[-2:] == ["--batch-size", "48"]                                                                       # its later attempts keep the batch that fits
