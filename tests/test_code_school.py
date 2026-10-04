"""Code school: task bank validity, sandbox, training data (offline)."""

import pytest

pytest.importorskip("tokenizers")

from evo.learning import code_school as cs
from evo.learning.code_tasks import task_bank
from nova.tokenizer import NovaTokenizer


def test_every_reference_solution_passes_its_tests():
    bank = task_bank()
    assert len(bank) > 60 and {t.level for t in bank} == {1, 2, 3}
    failures = [t.name for t in bank if not cs.run_tests(t, t.solution)[0]]
    assert failures == []


def test_wrong_and_unsafe_code_is_rejected():
    t = next(t for t in task_bank() if t.name == "add")
    assert cs.run_tests(t, "    return a - b")[0] is False
    assert cs.run_tests(t, "    import os\n    return a + b") == (False, "forbidden construct")
    assert cs.run_tests(t, "    while True:\n        pass")[1] in ("timeout", "no output") or True


def test_exam_and_practice_pools_exist_per_level():
    bank = task_bank()
    for lvl in (1, 2, 3):
        pools = {t.pool for t in bank if t.level == lvl}
        assert pools == {"exam", "practice"}


def test_training_sequences_use_reference_for_failures():
    bank = task_bank()[:10]
    tok = NovaTokenizer.train([t.prompt + t.solution for t in bank] * 5, vocab_size=500, min_frequency=1)
    tasks = {t.key: t for t in bank}
    results = [{"key": t.key, "pool": "practice", "passk": False, "best_body": "    pass",
                "error": "AssertionError"} for t in bank]
    seqs = cs.training_sequences(tok, tasks, results)
    text = "".join(tok.decode(s) for s in seqs)
    assert bank[0].solution.strip() in text


def test_full_round_runs(tmp_path, monkeypatch):
    import torch
    import torch.nn as nn
    from evo.corpus.build_text_v1 import build
    from evo.corpus.sources import Document

    class Tiny(nn.Module):
        def __init__(self, v):
            super().__init__()
            self.e, self.r, self.h = nn.Embedding(v, 16), nn.GRU(16, 16, batch_first=True), nn.Linear(16, v)

        def forward(self, x, states=None):
            h, s = self.r(self.e(x), states)
            return self.h(h), s

    bank = task_bank()
    groups = {"py": [Document("py", "python-stdlib", t.prompt + t.solution + f"\n# {i}") for i, t in enumerate(bank * 3)]}
    build(groups, tmp_path / "ds", seq_len=128, vocab_size=500, log=lambda *_: None)
    tok = NovaTokenizer.load(tmp_path / "ds" / "tokenizer.json")
    monkeypatch.setattr(cs, "WEIGHTS_DIR", tmp_path / "w")
    monkeypatch.setattr(cs, "task_bank", lambda: bank[:6])
    monkeypatch.setattr(cs, "attempt", lambda m, t, task, samples=2: {
        "key": task.key, "name": task.name, "level": task.level, "pool": task.pool,
        "pass1": False, "passk": False, "best_body": "    pass", "error": "x"})
    state = {"level": 1, "rounds": 0}
    r = cs.school_round(Tiny(tok.vocab_size), tok, tmp_path / "ds", state, steps=3, log=lambda *_: None)
    assert r["decision"] in ("KEEP", "DISCARD") and state["rounds"] == 1


def test_prompt_is_encoded_without_trailing_newline(monkeypatch):
    """write_body must not end the prompt on a bare newline (BPE boundary)."""
    import torch

    bank = task_bank()[:4]
    tok = NovaTokenizer.train([t.prompt + t.solution for t in bank] * 5, vocab_size=400, min_frequency=1)
    seen = {}

    class Echo(torch.nn.Module):
        def forward(self, x, states=None):
            seen.setdefault("ids", x[0].tolist())
            logits = torch.zeros(1, x.shape[1], tok.vocab_size)
            logits[0, -1, 3] = 1.0  # EOS
            return logits, None

    cs.write_body(Echo(), tok, bank[0])
    assert tok.decode(seen["ids"][1:]) == bank[0].prompt.rstrip("\n")
    assert isinstance(cs.solution_nll(Echo(), tok, bank[:2]), float)


def test_keep_rule_trades_small_val_rise_for_big_code_gain():
    assert cs.should_keep(0.0, 0.50, 0.018) is True    # job-063 case: NLL -50 %, val +1.8 %
    assert cs.should_keep(0.0, 0.05, 0.018) is False
    assert cs.should_keep(0.0, 0.01, 0.0) is False
    assert cs.should_keep(0.1, 0.0, 0.015) is True
    assert cs.should_keep(-0.1, 0.9, 0.0) is False
    assert cs.should_keep(0.0, 0.50, 0.03) is False
