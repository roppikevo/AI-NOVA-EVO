"""The public entry point: python -m nova.demo and the release checks behind it."""

import json

import pytest
import torch

from nova import demo
from nova.tokenizer import NovaTokenizer

CONFIG = {"vocab_size": 300, "d_model": 32, "d_state": 32, "num_layers": 2, "conv_kernel": 5, "forget_bias": 1.125,
          "learnable_initial_state": False}


def _release(root, name="NOVA-T-v1", frozen="2026-01-01 00:00:00"):
    from evo.engine.architecture_factory import build_model

    out = root / name
    out.mkdir(parents=True)
    tok = NovaTokenizer.train(["Bratislava je mesto na Dunaji.", "The river runs through the town.", "def add(a, b): return a + b"] * 20,
                              vocab_size=300)
    tok.save(out / "tokenizer.json")
    torch.manual_seed(0)
    model = build_model(CONFIG)
    torch.save({"config": CONFIG, "model_state_dict": {k: v.half() for k, v in model.state_dict().items()}}, out / "nova_model.pt")
    (out / "MODEL.json").write_text(json.dumps({"name": name, "frozen": frozen, "scores": {"val": 5.0}}), encoding="utf-8")
    sums = [f"{demo.sha256(p)}  {p.name}" for p in sorted(out.iterdir())] + [f"{'0' * 64}  nova_model_fp32.pt"]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return out


def test_verify_reports_ok_absent_and_bad(tmp_path):
    rel = _release(tmp_path)
    check = demo.verify(rel)
    assert set(check["ok"]) == {"MODEL.json", "nova_model.pt", "tokenizer.json"}
    assert check["absent"] == ["nova_model_fp32.pt"] and not check["bad"]
    (rel / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert demo.verify(rel)["bad"] == ["tokenizer.json"]


def test_load_refuses_a_changed_file(tmp_path):
    rel = _release(tmp_path)
    with (rel / "nova_model.pt").open("ab") as f:
        f.write(b"x")
    with pytest.raises(SystemExit):
        demo.load(rel)


def test_releases_are_listed_oldest_first_and_need_weights(tmp_path):
    _release(tmp_path, "B", "2026-02-01 00:00:00")
    _release(tmp_path, "A", "2026-03-01 00:00:00")
    (tmp_path / "empty").mkdir()
    assert [p.name for p in demo.releases(tmp_path)] == ["B", "A"]
    assert demo.releases(tmp_path / "nothing") == []


def test_released_file_loads_as_plain_tensors_and_writes(tmp_path, monkeypatch):
    rel = _release(tmp_path)
    monkeypatch.setenv("NOVA_SAFE_LOAD", "1")   # no fallback to pickle: the file must load in the safe mode
    model, tok, _ = demo.load(rel)
    from nova.generate import generate

    text = generate(model, tok, "The river", "en", max_new_tokens=5)
    assert text.startswith("The river")
    s = demo.speed(model, tok, new_tokens=5)
    assert s["tokens_per_second"] > 0 and s["state_bytes"] > 0


def test_main_runs_on_a_release(tmp_path, monkeypatch, capsys):
    _release(tmp_path)
    monkeypatch.setattr(demo, "RELEASES", tmp_path)
    assert demo.main(["--list"]) == 0
    assert demo.main(["--lang", "en", "--prompt", "The river", "--tokens", "4"]) == 0
    out = capsys.readouterr().out
    assert "NOVA-T-v1" in out and "[en] The river" in out
    monkeypatch.setattr(demo, "RELEASES", tmp_path / "nothing")
    assert demo.main([]) == 2
