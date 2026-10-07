"""The experiment ledger: one record per experiment, no duplicates, noise between seeds."""

import json

from evo.engine import ledger

ROW = {"name": "n8-nslot-24M", "group": "24M", "arch": "nova8", "override": {"arch": "nova8", "pattern": "NSNSNSN", "slots": 16}, "lr": "1e-3",
       "compile": True, "loss": {"dataset": 3.0445, "web": 3.6804}, "web_carried": 3.5973, "code_solved": 54, "code_tasks": 79,
       "train_tokens_per_s": 53899, "train_hours": 0.754, "cpu": {"write_tokens_per_s": 161.6, "state_kb_after_writing": 56.0},
       "identity": {"class": "NOVA", "pass": True}}


def test_a_measured_run_becomes_one_record(tmp_path):
    path = tmp_path / "ledger.jsonl"
    e = ledger.record(ledger.from_report_row(dict(ROW), "2026-10-05 13:00", 18000), path)
    assert e["id"] == "EXP-00001" and e["kind"] == "tournament" and e["verdict"] == "measured" and e["identity"] == "NOVA"
    assert e["metrics"] == {"dataset": 3.0445, "web": 3.6804, "web_carried": 3.5973, "code": "54/79"}
    assert e["cost"]["state_kb"] == 56.0 and e["train"] == {"steps": 18000, "lr": "1e-3", "seed": 1001, "compiled": True}
    assert ledger.record(ledger.from_report_row(dict(ROW), "2026-10-05 14:00", 18000), path) is None        # not twice
    failed = ledger.record(ledger.from_report_row({"name": "n8-x-24M", "arch": "nova8", "override": {"pattern": "XX"}, "lr": "1e-3",
                                                   "error": "RuntimeError: out of memory"}), path)
    assert failed["id"] == "EXP-00002" and failed["verdict"] == "failed" and "out of memory" in failed["reason"]
    rows = ledger.load(path)
    assert len(rows) == 2 and ledger.tried(config=ROW["override"], rows=rows)[0]["name"] == "n8-nslot-24M"
    assert ledger.tried(name="n8-x-24M", rows=rows)[0]["verdict"] == "failed" and ledger.tried(config={"pattern": "new"}, rows=rows) == []
    assert "EXP-00001" in ledger.text(rows) and "state 56.0 kB" in ledger.text(rows)


def test_director_events_with_a_verdict_are_recorded(tmp_path):
    path = tmp_path / "ledger.jsonl"
    rejected = {"event": "attempt", "date": "2026-10-05 00:25", "attempt": 7, "recipe": "wider-view", "champion": "NOVA-24M-v2", "steps": 4000,
                "hours": 0.4, "rc": 0, "verdict": {"accept": False, "reasons": ["gain -0.16 % is below 0.3 %"], "decision": {"gain_percent": -0.16},
                                                    "code": {"before": 43, "after": 42}},
                "sets": {"dataset": [3.1049, 3.11, 0.16], "web": [3.4149, 3.42, 0.15]}}
    e = ledger.record(ledger.from_director_event(rejected), path)
    assert e["kind"] == "attempt" and e["verdict"] == "rejected" and e["gain_percent"] == -0.16 and e["parent"] == "NOVA-24M-v2"
    assert e["metrics"] == {"dataset": 3.11, "web": 3.42, "code": 42} and "below 0.3" in e["reason"]
    assert ledger.from_director_event({"event": "grow_segment", "line": "NOVA8-24M", "rc": 0, "steps_done": 82000}) is None     # still training
    assert ledger.from_director_event({"event": "stopped_by_signal"}) is None
    done = {"event": "grow_segment", "date": "2026-10-06 02:10", "line": "NOVA8-24M", "rc": 0, "steps_done": 300000, "released": "NOVA8-24M-v1",
            "identity": {"class": "NOVA", "pass": True}, "verdict": {"accept": True, "reasons": [], "decision": {"gain_percent": 4.2}}}
    g = ledger.record(ledger.from_director_event(done), path)
    assert g["kind"] == "generation" and g["verdict"] == "accepted" and g["identity"] == "NOVA" and "NOVA8-24M-v1" in g["reason"]
    crashed = ledger.from_director_event({"event": "attempt", "date": "2026-10-05 03:00", "attempt": 9, "recipe": "collective", "rc": 1})
    assert crashed["verdict"] == "failed"


def test_backfill_and_the_noise_between_seeds(tmp_path):
    path, report, log = tmp_path / "ledger.jsonl", tmp_path / "report.json", tmp_path / "log.jsonl"
    rows = {}
    for seed, (d, w) in {1001: (3.0445, 3.6804), 2001: (3.0527, 3.6850), 3001: (3.0391, 3.6770)}.items():
        name = "n8-nslot-24M" + ("" if seed == 1001 else f"-s{seed}")
        rows[name] = {**ROW, "name": name, "loss": {"dataset": d, "web": w}, **({"seed": seed} if seed != 1001 else {})}
    rows["n8-nsw-24M"] = {**ROW, "name": "n8-nsw-24M", "override": {"arch": "nova8", "pattern": "NSWNSWN"}, "loss": {"dataset": 3.0155, "web": 3.6675}}
    report.write_text(json.dumps({"date": "2026-10-05 15:00", "steps": 18000, "results": rows}))
    log.write_text(json.dumps({"event": "attempt", "date": "2026-10-05 00:25", "attempt": 7, "recipe": "wider-view", "rc": 0,
                               "verdict": {"accept": False, "reasons": ["no gain"]}}) + "\nnot json\n")
    assert ledger.backfill(report, log, path) == 5 and ledger.backfill(report, log, path) == 0
    spread = ledger.noise(ledger.load(path))
    assert list(spread) == ["n8-nslot-24M"] and spread["n8-nslot-24M"]["runs"] == 3
    assert abs(spread["n8-nslot-24M"]["dataset"]["mean"] - 3.0454) < 1e-3 and 0.004 < spread["n8-nslot-24M"]["dataset"]["std"] < 0.007
    pct = ledger.noise_percent(ledger.load(path))
    assert 0.1 < pct < 0.3 and ledger.noise_percent([]) is None
    assert "between seeds" in ledger.text(ledger.load(path))
