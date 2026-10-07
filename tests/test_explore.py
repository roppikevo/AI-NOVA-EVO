"""The decision logic of the director's own tournaments: identity, then quality above the noise, then cost."""

from evo.engine import explore


def row(name, dataset, web, state_kb=56.0, speed=54000, identity="NOVA", verdict="measured"):
    return {"kind": "tournament", "name": name, "verdict": verdict, "identity": identity, "metrics": {"dataset": dataset, "web": web},
            "cost": {"state_kb": state_kb, "train_tok_s": speed}}


BASE = row("n8-nslot-24M", 3.0445, 3.6804)
REG = {"nslot-sharp": {"pattern": "NSNSNSN", "slot_sharp": True}, "nslot8": {"pattern": "NSNSNSN", "slots": 8}}


def test_what_to_measure_next():
    rows = [BASE]
    assert explore.next_action(rows, "nslot", "24M", REG) == {"what": "noise", "name": "nslot", "seed": 2001}
    rows += [row("n8-nslot-24M-s2001", 3.05, 3.685), row("n8-nslot-24M-s3001", 3.04, 3.677)]
    assert explore.next_action(rows, "nslot", "24M", REG) == {"what": "candidate", "name": "nslot-sharp"}
    rows += [row("n8-nslot-sharp-24M", 3.0, 3.6), {"kind": "tournament", "name": "n8-nslot8-24M", "verdict": "failed"}]
    assert explore.next_action(rows, "nslot", "24M", REG) is None                      # a failed run counts as tried
    assert explore.next_action([], "nslot", "24M", REG) == {"what": "candidate", "name": "nslot"}       # the yardstick first
    assert explore.noise_pct(rows[:2], "nslot", "24M") is None and 0.05 < explore.noise_pct(rows, "nslot", "24M") < 0.3
    assert explore.candidate_of({"pattern": "NSNSNSN", "slots": 8}, REG) == "nslot8" and explore.candidate_of({"pattern": "X"}, REG) is None
    full = {"vocab_size": 16384, "d_state": 640, "pattern": "NSNSNSN", "slots": 8, "slot_sharp": True}       # a whole model config
    wide = {**REG, "plain": {"pattern": "NSNSNSN"}}
    assert explore.candidate_of(full, wide) in ("nslot-sharp", "nslot8") and explore.candidate_of({"pattern": "NSNSNSN", "slots": 16}, wide) == "plain"


def test_the_verdict_has_three_layers():
    grows = explore.verdict(row("x", 2.9, 3.5, identity="NOT-NOVA"), BASE, 0.15)
    assert not grows["win"] and "identity" in grows["reason"]                          # however good the loss
    assert not explore.verdict(row("x", 2.9, 3.5, identity=None), BASE, 0.15)["win"]
    assert not explore.verdict(row("x", 0, 0, verdict="failed"), BASE, 0.15)["win"]
    clear = explore.verdict(row("x", 3.01, 3.65), BASE, 0.15)                          # about 1 % better
    assert clear["win"] and clear["gain_percent"] > 0.9 and clear["needed_percent"] == 0.3
    inside = explore.verdict(row("x", 3.040, 3.676), BASE, 0.15)                       # 0.13 %: inside the noise
    assert not inside["win"] and "below the needed" in inside["reason"]
    noisy = explore.verdict(row("x", 3.02, 3.66), BASE, 0.4)                           # 0.67 % but the noise is 0.4 %
    assert not noisy["win"] and noisy["needed_percent"] == 0.8
    unknown = explore.verdict(row("x", 3.02, 3.66), BASE, None)                        # noise not measured: assume 0.5 %
    assert not unknown["win"] and unknown["noise_percent"] == explore.DEFAULT_NOISE_PCT
    window = explore.verdict(row("x", 3.0155, 3.6675, state_kb=257.0, identity="NOVA-HYBRID"), BASE, 0.15)   # nsw: +0.62 %, state x4.6
    assert not window["win"] and window["needed_percent"] == explore.BIG_STATE_GAIN_PCT and window["state_ratio"] > 4
    big_and_good = explore.verdict(row("x", 2.98, 3.63, state_kb=257.0, identity="NOVA-HYBRID"), BASE, 0.15)
    assert big_and_good["win"]
    smaller = explore.verdict(row("x", 3.045, 3.681, state_kb=45.5), BASE, 0.15)       # same quality, state x0.81
    assert not smaller["win"] and smaller["cheaper"] and "lower cost" in smaller["reason"]    # noted, not trained in full: the judge wants a gain
    faster = explore.verdict(row("x", 3.046, 3.681, speed=64000), BASE, 0.15)
    assert not faster["win"] and faster["cheaper"]
    worse_and_small = explore.verdict(row("x", 3.07, 3.70, state_kb=30.0), BASE, 0.15)
    assert not worse_and_small["win"]


def test_a_winner_is_trained_with_the_champions_recipe():
    line = explore.generation_row("nslot8", "24M", REG, {"steps": 300000, "lr": 0.001, "compile": True, "carry": 8, "carry_share": 0.35})
    assert line == {"line": "NOVA8-24M-nslot8", "override": REG["nslot8"], "candidate": "nslot8", "group": "24M",
                    "steps": 300000, "lr": 0.001, "compile": True, "carry": 8, "carry_share": 0.35}
    assert explore.generation_row("nslot8", "24M", REG)["steps"] == 300000 and "carry" not in explore.generation_row("nslot8", "24M", REG)


def test_the_code_exam_is_a_check_measured_against_its_own_spread_between_seeds():
    def scored(name, dataset, web, solved, score, **kw):
        r = row(name, dataset, web, **kw)
        r["metrics"].update({"code": f"{solved}/79", "code_score": score})
        return r

    rows = [row("n8-nslot-24M", 3.0445, 3.6804), scored("n8-nslot-24M-s2001", 3.0476, 3.6814, 57, 80.59), scored("n8-nslot-24M-s3001", 3.0516, 3.6812, 54, 78.82)]
    rows[0]["metrics"]["code"] = "54/79"
    n = explore.noise_stats(rows, "nslot", "24M")
    assert n["runs"] == 3 and n["dataset"]["n"] == 3 and abs(n["dataset"]["mean"] - 3.0479) < 1e-4 and n["web"]["std"] < 0.001
    assert n["code_solved"] == {"mean": 55.0, "std": 1.4142, "min": 54.0, "max": 57.0, "n": 3} and n["code_score"]["n"] == 2
    floor = explore.code_floor(rows, "nslot", "24M")
    assert floor == 77.05 and explore.code_floor(rows[:2], "nslot", "24M") is None          # one scored run: no floor yet
    sharp = explore.verdict(scored("n8-nslot-sharp-24M", 3.039, 3.6814, 49, 74.51), rows[0], 0.048, floor)     # what was measured 2026-10-07
    assert not sharp["win"] and sharp["gain_percent"] == 0.07 and "code exam clearly worse" in sharp["reason"]
    good_loss_bad_code = explore.verdict(scored("x", 3.00, 3.64, 49, 74.5), rows[0], 0.048, floor)
    assert not good_loss_bad_code["win"] and "but the code exam is clearly worse" in good_loss_bad_code["reason"]
    assert explore.verdict(scored("x", 3.00, 3.64, 53, 78.0), rows[0], 0.048, floor)["win"]
    assert not explore.verdict(scored("x", 3.045, 3.681, 49, 74.5, state_kb=45.0), rows[0], 0.048, floor).get("cheaper")     # cheaper but worse at code: nothing
    assert explore.verdict(scored("x", 3.00, 3.64, 49, 74.5), rows[0], 0.048)["win"]        # without a floor the check is silent
