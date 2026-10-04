"""Tests for efficiency measurement and efficiency-aware evaluation."""

import torch
import torch.nn as nn

from evo.engine import core_evolution_engine as ce
from evo.engine.training_runner import aggregate_efficiency
from nova.efficiency import efficiency_check, measure_cpu


class TinyStateful(nn.Module):
    def __init__(self, vocab=64, d=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.cell = nn.GRU(d, d, batch_first=True)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids, states=None):
        h, s = self.cell(self.emb(ids), states)
        return self.head(h), s


def test_measure_cpu_reports_speed_and_state():
    m = measure_cpu(TinyStateful(), vocab_size=64, seq_len=16, gen_tokens=8, threads=1)
    assert m["parameters"] > 0
    assert m["cpu_tokens_per_sec"] > 0
    assert m["cpu_gen_tokens_per_sec"] > 0
    assert m["state_kb"] > 0


def test_efficiency_ok_without_parent_metrics():
    r = efficiency_check({"parameters": 10}, None)
    assert r["ok"] and not r["gain"]


def test_efficiency_violations():
    parent = {"parameters": 100, "cpu_tokens_per_sec": 1000, "peak_vram_mb": 500}
    r = efficiency_check({"parameters": 200, "cpu_tokens_per_sec": 500,
                          "peak_vram_mb": 1000}, parent)
    assert not r["ok"] and len(r["violations"]) == 3


def test_efficiency_gain():
    parent = {"parameters": 100, "cpu_tokens_per_sec": 1000}
    assert efficiency_check({"parameters": 70, "cpu_tokens_per_sec": 1000}, parent)["gain"]
    assert efficiency_check({"parameters": 100, "cpu_tokens_per_sec": 1400}, parent)["gain"]


def test_aggregate_efficiency():
    a = aggregate_efficiency([
        {"parameters": 5, "cpu_tokens_per_sec": 100, "peak_vram_mb": 10},
        {"parameters": 5, "cpu_tokens_per_sec": 200, "peak_vram_mb": 30},
        {"error": "x"},
    ])
    assert a == {"parameters": 5, "cpu_tokens_per_sec": 150.0, "peak_vram_mb": 30}


# ---------------------------------------------------------------- evaluate

def _engine(project):
    e = ce.CoreEvolutionEngine.__new__(ce.CoreEvolutionEngine)
    e.load_project_state = lambda: project
    return e


def _training(loss, eff, dataset="data/text_v1"):
    return {"status": "COMPLETED", "robust": {
        "seeds": [1, 2, 3], "dataset": dataset,
        "metrics": {"validation_loss_mean": loss, "efficiency": eff}}}


PARENT_EFF = {"parameters": 100, "cpu_tokens_per_sec": 1000, "peak_vram_mb": 500}


def _project(loss=5.0, dataset="data/text_v1"):
    return {"data_policy": {"training_data": ["data/text_v1"]},
            "best_known": {"validation_loss": loss, "dataset": dataset,
                           "efficiency": PARENT_EFF}}


SMOKE = {"status": "PASS"}


def test_better_loss_within_budget_passes():
    r = _engine(_project()).evaluate({}, SMOKE, _training(4.9, PARENT_EFF))
    assert r["decision"] == "PASS"


def test_better_loss_but_too_big_rejected():
    eff = dict(PARENT_EFF, parameters=300)
    r = _engine(_project()).evaluate({}, SMOKE, _training(4.5, eff))
    assert r["decision"] == "REJECT" and "efficiency" in r["reason"]


def test_equal_loss_with_cpu_gain_passes():
    eff = dict(PARENT_EFF, cpu_tokens_per_sec=1500)
    r = _engine(_project()).evaluate({}, SMOKE, _training(5.01, eff))
    assert r["decision"] == "PASS" and "efficiency gain" in r["reason"]


def test_worse_loss_rejected():
    r = _engine(_project()).evaluate({}, SMOKE, _training(5.5, PARENT_EFF))
    assert r["decision"] == "REJECT"


def test_dataset_mismatch_is_retry_not_reject():
    r = _engine(_project(dataset="data/gen4")).evaluate({}, SMOKE, _training(1.0, PARENT_EFF))
    assert r["decision"] == "RETRY" and r["reason"] == "baseline_dataset_mismatch"


def test_noise_level_improvement_is_rejected():
    proj = _project(5.7698)
    proj["best_known"]["validation_loss_std"] = 0.0033
    t = _training(5.7693, PARENT_EFF)
    t["robust"]["metrics"]["validation_loss_std"] = 0.004
    r = _engine(proj).evaluate({}, SMOKE, t)
    assert r["decision"] == "REJECT" and r["required_improvement"] > 0.01


def test_clear_improvement_passes_despite_noise():
    proj = _project(5.7698)
    proj["best_known"]["validation_loss_std"] = 0.0033
    t = _training(5.70, PARENT_EFF)
    t["robust"]["metrics"]["validation_loss_std"] = 0.004
    assert _engine(proj).evaluate({}, SMOKE, t)["decision"] == "PASS"


def test_promoted_ids_blocks_recovery_of_active_core():
    e = _engine({"primary_parent": "GEN7-CORE-001",
                 "lineage": [{"candidate": "GEN4-CORE-002"}],
                 "generation_history": [{"promoted": "GEN6-CORE-001"}]})
    assert e._promoted_ids() == {"GEN7-CORE-001", "GEN4-CORE-002", "GEN6-CORE-001"}


def test_smoke_oom_is_retry_not_reject():
    r = _engine(_project()).evaluate({}, {"status": "INFRA_ERROR"}, {"status": "COMPLETED"})
    assert r["decision"] == "RETRY" and r["reason"] == "infrastructure_error"
