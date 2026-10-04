"""The director's loop and the judge's rule (no training, no GPU: the outside world is faked)."""

import json
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
