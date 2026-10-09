"""nova.build: the machine, the plan for a size, a trial run and a short training with a teacher - all on the CPU."""

import json

import pytest
import torch

from nova import build as B
from nova import sizes as S
from nova.tokenizer import NovaTokenizer

TINY = {"vocab_size": 300, "arch": "nova8", "d_model": 32, "pattern": "NSN", "mlp_hidden": 48, "heads": 2, "slots": 4}
CARD8 = {"system": "test", "cpu_threads": 8, "ram_gb": 30, "torch": "x",
         "gpu": {"name": "card", "vram_gb": 7.6, "count": 1, "bf16": True, "capability": "8.9"}}
NOCARD = {**CARD8, "gpu": None}
REL = [{"name": "NOVA8-24M-v5", "path": None, "params": 24_167_975, "arch": "nova8", "vocab": 16384, "frozen": "1"},
       {"name": "NOVA8-53M-v2", "path": None, "params": 52_959_271, "arch": "nova8", "vocab": 16384, "frozen": "2"}]


def test_the_machine_is_described():
    m = B.detect()
    assert m["cpu_threads"] >= 1 and "torch" in m and ("gpu" in m)
    assert m["ram_gb"] is None or m["ram_gb"] > 0


def test_sizes_are_read_in_any_usual_form():
    assert B.parse_size("auto") is None
    assert B.parse_size("24M") == 24e6 and B.parse_size("1.5B") == 1.5e9 and B.parse_size("3e8") == 3e8
    assert B.parse_size("100_000_000") == 1e8
    with pytest.raises(ValueError):
        B.parse_size("10K")


def test_a_released_size_is_used_and_another_is_built_with_the_largest_release_as_teacher():
    p = B.plan(53e6, CARD8, REL)
    assert p["action"] == "use" and p["release"] == "NOVA8-53M-v2"
    assert B.plan(25e6, CARD8, REL)["release"] == "NOVA8-24M-v5"
    p = B.plan(100e6, CARD8, REL)
    assert p["action"] == "build" and p["teacher"] == "NOVA8-53M-v2" and 85e6 < p["params"] < 115e6
    assert p["config"]["pattern"].startswith("N") and p["config"]["vocab_size"] == 16384
    assert p["tokens"] == int(20 * p["params"]) and p["batch"] in (16, 24, 32) and p["hours_by_rule"] > 24
    small = B.plan(10e6, CARD8, REL)
    assert small["action"] == "build" and any("larger than the new core" in n for n in small["notes"])
    huge = B.plan(1e9, CARD8, REL)
    assert huge["batch"] is None and any("does not fit" in n for n in huge["notes"])


def test_auto_takes_the_card_then_the_text_and_the_time_into_account():
    by_card = B.plan(None, CARD8, REL)                        # the rules give ~100 M for 8 GB at batch 32
    assert by_card["action"] == "build" and 60e6 < by_card["params"] < 130e6 and "the card" in by_card["notes"][0]
    with pytest.raises(SystemExit):                            # 2 M tokens, 4 passes -> about 0.4 M parameters: too small
        B.plan(None, CARD8, REL, text_tokens=2_000_000)
    by_text = B.plan(None, CARD8, REL, text_tokens=60_000_000)    # 60 M tokens -> about 12 M parameters
    assert by_text["params"] < 20e6 and "the text" in by_text["notes"][0]
    by_time = B.plan(None, CARD8, REL, hours=1.0)
    assert by_time["params"] < by_card["params"] and "the time" in by_time["notes"][0]
    assert abs(S.train_hours(B.largest_for_hours(1.0, 9.0), 20 * B.largest_for_hours(1.0, 9.0)) - 1.0) < 0.01
    assert B.plan(None, NOCARD, REL)["action"] == "use"


def test_a_small_text_for_a_large_core_is_named():
    p = B.plan(100e6, CARD8, REL, text_tokens=10_000_000)
    assert p["tokens"] == 40_000_000 and any("memorise" in n for n in p["notes"])


def test_the_trial_runs_real_steps_and_falls_back_to_a_smaller_batch(monkeypatch):
    t = B.trial(TINY, 4, "cpu", steps=3)
    assert t["batch"] == 4 and t["tokens_per_s"] > 0 and t["peak_mb"] is None and t["params"] > 0
    teacher = __import__("evo.engine.architecture_factory", fromlist=["build_model"]).build_model({**TINY, "max_seq_len": 128}).eval()
    assert B.trial(TINY, 2, "cpu", teacher=teacher, steps=3)["batch"] == 2

    from evo.engine import architecture_factory as af

    real = af.build_model
    calls = []

    def fake(cfg):
        m = real(cfg)
        f = m.forward

        def forward(x, *a, **k):
            if x.shape[0] > 2:
                raise torch.OutOfMemoryError("CUDA out of memory (test)") if hasattr(torch, "OutOfMemoryError") \
                    else RuntimeError("CUDA out of memory (test)")
            return f(x, *a, **k)
        m.forward = forward
        calls.append(cfg)
        return m
    monkeypatch.setattr(af, "build_model", fake)
    t = B.trial(TINY, 8, "cpu", steps=3)
    assert t["batch"] == 2 and "ran out of memory" in t["larger_batches"] and len(calls) == 3   # 8, 4, 2


def test_a_short_training_with_a_teacher_saves_a_core_that_loads_and_writes(tmp_path):
    from evo.engine.architecture_factory import build_model
    from nova.generate import generate, load_checkpoint_model

    text = "\n\n".join(["The river runs through the old town and the bridge crosses it."] * 300)
    (tmp_path / "t.txt").write_text(text, encoding="utf-8")
    tok = NovaTokenizer.train([text[:2000]], vocab_size=300)
    rows = B.text_rows(tmp_path / "t.txt", tok, "en")
    assert rows.shape[1] == 128
    torch.manual_seed(1)
    teacher = build_model({**TINY, "max_seq_len": 128}).eval()
    lines = []
    res = B.train(TINY, rows, "cpu", 4, 30, tmp_path / "core.pt", teacher=teacher, lr=3e-3, log=lines.append)
    assert res["steps"] == 30 and res["held_out_after"] < res["held_out_before"] and lines
    ck = torch.load(tmp_path / "core.pt", weights_only=False)
    assert ck["teacher_until"] == 9 and ck["config"] == TINY
    m, _ = load_checkpoint_model(tmp_path / "core.pt")
    assert isinstance(generate(m.float().eval(), tok, "The river", "en", max_new_tokens=5, temperature=0.0), str)


def test_the_command_plans_uses_a_release_and_reports(tmp_path, monkeypatch, capsys):
    from evo.engine import release
    from evo.engine.architecture_factory import build_model
    from nova import demo

    tok = NovaTokenizer.train(["Bratislava je mesto na Dunaji.", "The river runs through the town."] * 20, vocab_size=300)
    tok.save(tmp_path / "tokenizer.json")
    torch.manual_seed(0)
    root = tmp_path / "rel"
    cfg = S.config(256, 3, vocab=300)                                   # about 2.4 M parameters: a size the command accepts
    release.write_release(root / "NOVA8-T-v1", {"config": cfg, "candidate": "NOVA8-T", "model_state_dict": build_model(cfg).state_dict()},
                          "x.pt", {"val": 5.0}, tmp_path / "tokenizer.json", {})
    monkeypatch.setattr(demo, "RELEASES", root)
    monkeypatch.setenv("NOVA_SAFE_LOAD", "1")
    monkeypatch.setattr(B, "detect", lambda: dict(NOCARD))
    rel = B.released()
    assert [r["name"] for r in rel] == ["NOVA8-T-v1"] and rel[0]["params"] == S.parameters(cfg)
    assert B.main([]) == 0 and "processor only" in capsys.readouterr().out
    size = f"{rel[0]['params']}"
    assert B.main(["--size", size, "--json"]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["plan"]["action"] == "use" and out["ready"] == "NOVA8-T-v1"
