"""Statistics of a collective run (pure functions; no model, no server)."""

import json

import numpy as np

from evo.collective import stats as st


def table(rng, n, means):
    """Per-sequence losses: a shared difficulty per sequence + a small model-specific noise."""
    hard = rng.normal(0, 0.5, size=n)
    return {name: (m + hard + rng.normal(0, 0.05, size=n)).astype(np.float32) for name, m in means.items()}


def test_bootstrap_sees_a_real_improvement_and_a_real_decline():
    rng = np.random.default_rng(0)
    t = table(rng, 800, {"a": 3.30, "better": 3.27, "worse": 3.33, "same": 3.30})
    up = st.paired_bootstrap(t["a"], t["better"])
    assert st.verdict(up) == "improvement" and -0.04 < up["diff"] < -0.02 and up["lo"] < up["diff"] < up["hi"]
    assert st.verdict(st.paired_bootstrap(t["a"], t["worse"])) == "decline"
    assert st.verdict(st.paired_bootstrap(t["a"], t["same"])) == "no clear change"
    assert st.verdict(st.paired_bootstrap(np.zeros(0), np.zeros(0))) == "no data"


def test_trend_and_small_helpers():
    assert st.trend([3.30, 3.28, 3.26, 3.24, 3.22])["state"] == "still improving"
    assert st.trend([3.30, 3.25, 3.25, 3.25, 3.25])["state"] == "flat"
    assert st.trend([3.20, 3.21, 3.23, 3.25])["state"] == "getting worse"
    assert st.trend([3.2])["state"] == "too short"
    assert st.exam_change({"a", "b"}, {"b", "c", "d"}) == {"solved": 3, "gained": 2, "lost": 1, "net": 1}
    assert st.percent(3.30, 3.267) == -1.0
    assert st.efficiency(3.30, 3.27, 2e8) == 0.015 and st.efficiency(3.3, 3.2, 0) == 0.0


def fake_run(rng):
    names = ["start", "core_round0", "core_round1", "core_round2", "baseline_single", "clone_sk"]
    means = {"start": 3.30, "core_round0": 3.29, "core_round1": 3.27, "core_round2": 3.26, "baseline_single": 3.28, "clone_sk": 3.31}
    losses = {"dataset": table(rng, 600, means), "web": table(rng, 400, {k: v + 0.1 for k, v in means.items()})}
    langs = {"dataset": np.repeat([4, 5, 6, 7, 8, 9], 100), "web": np.repeat([4, 5, 6, 7], 100)}
    exams = {"start": {"t1", "t2"}, "core_round0": {"t1", "t2"}, "core_round1": {"t1", "t3"}, "core_round2": {"t1", "t2", "t3"},
             "baseline_single": {"t1"}}
    report = {"name": "fake", "steps_per_round": 1500, "focuses": ["sk", "cs"], "baseline_rounds": 2,
              "nodes": {"rounds": [{"round": 0, "led_by": "nova0", "next_leader": "sk", "agreement": True,
                                    "core": {"yes": 2, "of": 3, "accepted": True,
                                             "votes": {"sk": [3.3, 3.29, True], "cs": [3.4, 3.38, True], "nova0": [3.3, 3.31, False]}},
                                    "scores": {"sk": 0.8, "cs": 0.7, "nova0": 0.9},
                                    "core_change": {"candidate": {"conv_kernel": 3}, "yes": 1, "of": 2, "accepted": False}}]}}
    return names, losses, langs, exams, report


def test_analyse_builds_every_table():
    names, losses, langs, exams, report = fake_run(np.random.default_rng(1))
    s = st.analyse(names, losses, langs, exams, report)
    assert s["clones"] == 2 and s["tokens_per_round_M"] == round(1500 * 64 * 127 * 2 / 1e6, 1)
    d = s["sets"]["dataset"]
    assert d["start_to_final_core"]["verdict"] == "improvement" and d["start_to_final_core"]["percent"] < 0
    assert set(d["by_language"]) == {"sk", "cs", "pl", "en", "py", "rs"} and set(s["sets"]["web"]["by_language"]) == {"sk", "cs", "pl", "en"}
    c = d["collective_vs_single_model"]                  # baseline had 2 rounds of tokens -> compared with core_round1
    assert c["rounds"] == 2 and c["verdict"] == "improvement" and c["efficiency_collective"] > c["efficiency_single"] > 0
    assert [r["round"] for r in s["rounds"]] == [0, 1, 2] and s["rounds"][2]["dataset"]["total_verdict"] == "improvement"
    assert s["rounds"][1]["code"] == {"solved": 2, "gained": 1, "lost": 1, "net": 0}
    assert s["code_start_to_final"]["net"] == 1 and s["code"]["baseline_single"] == 1
    n = s["nodes"][0]
    assert n["yes"] == 2 and n["exam_best"] == 0.9 and "conv_kernel" in n["core_change"] and n["mean_change_seen_by_nodes"] < 0


def test_report_text_and_files(tmp_path, monkeypatch):
    names, losses, langs, exams, report = fake_run(np.random.default_rng(2))
    s = st.analyse(names, losses, langs, exams, report)
    a = {"dataset/final_core": losses["dataset"]["core_round2"], "web/final_core": losses["web"]["core_round2"]}
    b = {"dataset/final_core": losses["dataset"]["start"], "web/final_core": losses["web"]["start"]}
    s["compare"] = st.compare("fake", a, "other", b)
    assert s["compare"]["sets"]["dataset"]["winner"] == "fake" and s["compare"]["sets"]["web"]["winner"] == "fake"
    assert st.compare("x", a, "y", a)["sets"]["dataset"]["winner"] == "no clear difference"
    t = st.text(s)
    assert "IMPROVEMENT" in t and "round by round" in t and "fake vs other" in t and "efficiency" in t
    monkeypatch.setattr(st, "RUNS", tmp_path)
    (tmp_path / "fake").mkdir()
    st.write("fake", s)
    assert json.loads((tmp_path / "fake/stats.json").read_text())["run"] == "fake"
    rows = (tmp_path / "fake/stats.csv").read_text().strip().splitlines()
    assert len(rows) == 4 and rows[0].startswith("round,tokens_M,dataset_loss")


def test_runs_without_a_baseline_or_code_exam_still_work():
    names, losses, langs, _, report = fake_run(np.random.default_rng(3))
    names = [n for n in names if not n.startswith("baseline")]
    s = st.analyse(names, losses, langs, {}, {**report, "focuses": None, "node_reports": {"a": {}, "b": {}, "n0": {"frozen": True}}})
    assert s["clones"] == 2 and "collective_vs_single_model" not in s["sets"]["dataset"] and "code" not in s
    assert "STATISTICS" in st.text(s)
