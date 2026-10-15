"""Initiative: the core ranks the causes of its own weakness from its own evidence; its proposals and the
programmed ones take turns and are counted apart - help is recorded, never passed off as the core's own."""

import json
from pathlib import Path

import numpy as np
import torch

from evo.engine import director as d
from evo.engine import initiative as I

PROBLEM = {"description": "poradie a vzťahy (nové úlohy)", "last": {"accuracy": 0.3}, "tried": ["practice"],
           "failed_hypotheses": [{"strategy": "practice", "gain": 0.01}], "class": "mastery"}
DIAG = {"patterns": [{"feature": "asks", "value": "least", "error_rate": 0.8, "elsewhere": 0.3, "items": 100, "z": 7.1}],
        "confidently_wrong": 0.7}


def test_the_question_carries_the_cores_own_evidence_and_the_proposal_has_six_parts():
    q = I.question(PROBLEM, DIAG)
    assert "30%" in q and "asks = least" in q and "70%" in q and "practice" in q and q.endswith("Príčina:")
    p = I.proposal(PROBLEM, "skills/ordering/new", "focus-errors", {"cause": "x", "prediction": "y"}, "core-structured",
                   ["focus-errors", "longer"], "štruktúrované otázky")
    assert set(p) >= {"author", "help", "cannot_do", "cause", "approach", "expected", "verification", "if_fails"}
    assert p["author"] == "core-structured" and p["help"] and "longer" in p["if_fails"] and "vybrala jadro" in p["cause"]


def test_a_core_ranks_only_the_causes_it_may_still_test(tmp_path):
    from evo.engine.architecture_factory import build_model
    from nova.tokenizer import NovaTokenizer

    tok = NovaTokenizer.train([I.question(PROBLEM, DIAG)] + [c[0] for c in I.CAUSES.values()] * 5, vocab_size=300)
    cfg = {"vocab_size": 300, "arch": "nova8", "d_model": 32, "pattern": "NSN", "mlp_hidden": 48, "heads": 2, "slots": 4}
    torch.manual_seed(0)
    r = I.core_choice(build_model(cfg).eval(), tok, PROBLEM, DIAG, ["longer", "gentle", "focus-errors"])
    assert sorted(x[0] for x in r) == ["focus-errors", "gentle", "longer"] and r[0][1] >= r[-1][1]


def test_the_core_and_the_rules_take_turns_and_are_counted_apart(tmp_path, monkeypatch):
    from tests.test_director import AbilityWorld, _board, fresh, home  # noqa: F401
    monkeypatch.chdir(tmp_path)
    for name in ("DIR", "STATE", "LOG", "STOP", "REPORT"):
        monkeypatch.setattr(d, name, tmp_path / "evo/director" / Path(str(getattr(d, name))).relative_to("evo/director"))
    monkeypatch.setattr(d, "OUTBOX", tmp_path / "outbox")
    monkeypatch.setattr(d, "gpu_is_free", lambda w: True)
    monkeypatch.setattr("evo.learning.drive.build_material", lambda targets, strategy, root=None, seed=0, **kw: Path(f"data/drive_v1/{strategy}"))

    class CoreWorld(AbilityWorld):
        def core_rank(self, checkpoint, problem, diag, allowed):
            return [(s, float(i)) for i, s in enumerate(reversed(allowed))]     # the core prefers the last allowed way

    champ = _board({"skills/ordering/new": 0.3})
    w = CoreWorld(champ, [champ] * 4, verdicts=[False] * 4)
    st = fresh()
    [d.cycle(st, w) for _ in range(3)]
    events = [json.loads(l) for l in (tmp_path / "evo/director/log.jsonl").read_text().splitlines()]
    props = [e["proposal"] for e in events if e["event"] == "drive"]
    assert [p["author"] for p in props[:2]] == ["core-structured", "programmed"]
    assert props[0]["approach"] != props[1]["approach"] and props[0]["help"] and props[1]["help"] is None
    reg = json.loads((tmp_path / "evo/director/problems.json").read_text())
    summary = I.authors_summary(reg)
    assert set(summary) == {"core-structured", "programmed"} and all(v["tries"] >= 1 for v in summary.values())
