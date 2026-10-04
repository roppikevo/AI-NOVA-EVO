"""Calibration, choose() and decision log (offline)."""

import json

import pytest
import torch

pytest.importorskip("tokenizers")

from evo.learning import decision_log as dl
from nova.calibration import calibration_stats, measure_calibration
from nova.choose import choose
from nova.tokenizer import NovaTokenizer


def test_perfect_calibration_has_zero_ece():
    conf = torch.tensor([0.9] * 10)
    correct = torch.tensor([1] * 9 + [0])
    assert calibration_stats(conf, correct)["ece"] == pytest.approx(0.0, abs=1e-6)


def test_overconfident_model_has_high_ece():
    s = calibration_stats(torch.tensor([0.99] * 10), torch.tensor([1, 0] * 5))
    assert s["ece"] > 0.4 and s["accuracy"] == 0.5


def test_measure_calibration_runs_on_dataset():
    class DS(torch.utils.data.Dataset):
        def __len__(self): return 8
        def __getitem__(self, i):
            t = torch.arange(9) % 5
            return t[:-1], t[1:]

    class M(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.one_hot((x + 1) % 5, 5).float() * 10, None

    s = measure_calibration(M(), DS(), "cpu", max_batches=2, batch_size=4)
    assert s["accuracy"] == 1.0 and s["ece"] < 0.01


def _tok():
    return NovaTokenizer.train(["Devstral je dobrý na Rust. Qwen je dobrý na jazyky."] * 30,
                               vocab_size=400, min_frequency=1)


class Prefers(torch.nn.Module):
    """Toy model that strongly prefers one token id."""
    def __init__(self, vocab, favourite):
        super().__init__()
        self.vocab, self.fav = vocab, favourite

    def forward(self, x, states=None):
        logits = torch.zeros(1, x.shape[1], self.vocab)
        logits[..., self.fav] = 8.0
        return logits, None


def test_choose_picks_preferred_option_and_can_abstain():
    tok = _tok()
    fav = tok.encode(" Devstral")[0]
    r = choose(Prefers(tok.vocab_size, fav), tok, "Kto?", ["Devstral", "Qwen"])
    assert r["best"] == "Devstral" and r["confident"]
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-3
    flat = choose(Prefers(tok.vocab_size, 0), tok, "Kto?", ["Devstral", "Qwen", "OxCoder"],
                  min_confidence=0.99)
    assert flat["choice"] is None


def test_outcome_labels():
    assert dl.outcome_of({"status": "CYCLE_ERROR", "error": "Teacher response did not contain a core source"})[0] == "NO_SOURCE"
    assert dl.outcome_of({"evaluation": {"decision": "PASS"}})[0] == "PASS"
    assert dl.outcome_of({"status": "REJECT", "stage": "source_contract"})[0] == "CONTRACT_FAILED"
    assert dl.outcome_of({"evaluation": {"decision": "REJECT", "reason": "runtime_smoke_failed"}})[0] == "SMOKE_FAILED"
    assert dl.outcome_of({"evaluation": {"decision": "REJECT", "reason": "Lower loss but efficiency budget violated"}})[0] == "REJECT_EFFICIENCY"


def test_backfill_writes_once(tmp_path):
    res = tmp_path / "results"; res.mkdir()
    (res / "GEN8-CORE-001.json").write_text(json.dumps(
        {"candidate": "GEN8-CORE-001", "stage": "runtime_smoke",
         "evaluation": {"decision": "REJECT", "reason": "runtime_smoke_failed"}}))
    (res / "GEN4-CORE-002.official.json").write_text("{}")
    log = tmp_path / "decisions.jsonl"
    assert dl.backfill(res, log) == {"SMOKE_FAILED": 1}
    assert dl.backfill(res, log) == {}
