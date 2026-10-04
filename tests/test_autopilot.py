"""Autopilot scheduling (no real activities are run)."""

import subprocess
from types import SimpleNamespace

from evo.engine import autopilot as ap


def _state(**stats):
    base = {n: {"runs": 1, "benefit": 0.0, "hours": 1.0} for n in ap.ACTIVITIES}
    base.update(stats)
    return {"stats": base, "last": [], "seed": 100}


def test_every_activity_tried_first():
    s = {"stats": {n: {"runs": 0, "benefit": 0.0, "hours": 0.0} for n in ap.ACTIVITIES}, "last": []}
    assert ap.choose(s) == next(iter(ap.ACTIVITIES))


def test_prefers_activity_with_best_benefit_per_hour():
    s = _state(self_correct={"runs": 3, "benefit": 3.0, "hours": 0.5})
    assert ap.choose(s, exploration=0.0) == "self_correct"


def test_avoids_three_in_a_row():
    s = _state(long_train={"runs": 3, "benefit": 0.3, "hours": 1.0},
               evolution={"runs": 3, "benefit": 0.25, "hours": 1.0})
    s["last"] = ["long_train", "long_train"]
    assert ap.choose(s, exploration=0.0) == "evolution"


def test_run_activity_records_and_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(ap, "LOG", tmp_path / "log.jsonl")
    calls = []

    def fake_run(cmd, timeout, capture_output, text):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    s = _state()
    rec = ap.run_activity("teachers", s, runner=fake_run, log=lambda *_: None)
    assert rec["rc"] == 0 and abs(s["stats"]["teachers"]["runs"] - (1 * ap.DISCOUNT + 1)) < 1e-9
    assert calls[0][:3] == ["nice", "-n", "19"] and calls[0][-2:] == ["--seed", "101"]


def test_timeout_counts_as_no_benefit(tmp_path, monkeypatch):
    monkeypatch.setattr(ap, "LOG", tmp_path / "log.jsonl")

    def slow(cmd, timeout, capture_output, text):
        raise subprocess.TimeoutExpired(cmd, timeout)

    s = _state()
    rec = ap.run_activity("evolution", s, runner=slow, log=lambda *_: None)
    assert rec["rc"] == "timeout" and rec["benefit"] == 0.0


def test_old_history_fades_so_a_long_ignored_activity_is_retried():
    """97 old long_train runs with zero measured benefit must not block it forever."""
    s = _state(long_train={"runs": 97.0, "benefit": 0.0, "hours": 14.0})
    picks = []
    for _ in range(60):
        name = ap.choose(s)
        picks.append(name)
        ap.discount(s)
        st = s["stats"][name]
        st["runs"] += 1
        st["hours"] += 0.1
        s["last"] = (s["last"] + [name])[-10:]
    assert "long_train" in picks
