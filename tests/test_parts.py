"""Large weight files go public in parts and are joined (and checked) on first use."""

import json

import torch

from evo.engine import publish
from nova import demo, parts


def test_split_and_join_give_back_the_same_file(tmp_path):
    f = tmp_path / "nova_model.pt"
    data = bytes(range(256)) * 1000
    f.write_bytes(data)
    ps = parts.split(f, part_bytes=100_000)
    assert [p.name for p in ps] == ["nova_model.pt.part00", "nova_model.pt.part01", "nova_model.pt.part02"]
    assert parts.parts_of(f) == ps and sum(p.stat().st_size for p in ps) == len(data)
    f.unlink()
    assert parts.join(f) and f.read_bytes() == data and not (tmp_path / "nova_model.pt.joining").exists()
    assert parts.join(f)                                                    # already there: nothing to do
    assert not parts.join(tmp_path / "other.pt")


def test_a_release_in_parts_is_found_joined_checked_and_loaded(tmp_path, monkeypatch):
    from evo.engine import release
    from evo.engine.architecture_factory import build_model

    cfg = {"arch": "nova8", "vocab_size": 64, "d_model": 16, "pattern": "NS", "heads": 2, "mlp_hidden": 32, "slots": 4, "max_seq_len": 16}
    tok = tmp_path / "tokenizer.json"
    tok.write_text("{}")
    out = tmp_path / "releases" / "NOVA8-T-v1"
    release.write_release(out, {"config": cfg, "model_state_dict": build_model(cfg).state_dict()}, "src.pt", {"val": 3.0}, tok, {})
    whole = (out / "nova_model.pt").read_bytes()
    parts.split(out / "nova_model.pt", part_bytes=len(whole) // 3 + 1)
    (out / "nova_model.pt").unlink()
    monkeypatch.setattr(demo, "RELEASES", tmp_path / "releases")
    assert [p.name for p in demo.releases()] == ["NOVA8-T-v1"]
    monkeypatch.setattr("nova.tokenizer.NovaTokenizer.load", staticmethod(lambda p: "tok"))
    model, t, ck = demo.load(out)
    assert (out / "nova_model.pt").read_bytes() == whole and "nova_model.pt" in demo.verify(out)["ok"] and t == "tok"
    (out / "nova_model.pt").unlink()
    p1 = parts.parts_of(out / "nova_model.pt")[1]
    p1.write_bytes(p1.read_bytes()[:-1] + b"x")                               # a damaged part is caught, not loaded
    try:
        demo.load(out)
        raise AssertionError("a damaged part was loaded")
    except SystemExit as exc:
        assert "checksum mismatch" in str(exc)


def test_publishing_takes_all_parts_of_a_release_or_none():
    files = {"evo/releases/A/nova_model.pt.part00": 90_000_000, "evo/releases/A/nova_model.pt.part01": 39_000_000,
             "evo/releases/A/MODEL.json": 100, "evo/releases/B/nova_model.pt": 63_000_000, "README.md": 10}
    chosen = publish.select(files, set(), 0.0, budget_mb=900)
    assert {"evo/releases/A/nova_model.pt.part00", "evo/releases/A/nova_model.pt.part01", "evo/releases/B/nova_model.pt"} <= set(chosen["keep"])
    assert chosen["weights_mb"] == 192.0 and chosen["without_weights"] == []
    tight = publish.select(files, set(), 0.0, budget_mb=100)
    assert not any("A/nova_model" in x for x in tight["keep"]) and "evo/releases/A/MODEL.json" in tight["keep"]
    assert "A" in tight["without_weights"] and "evo/releases/B/nova_model.pt" in tight["keep"]
    too_big = publish.select({"evo/releases/C/nova_model.pt": 129_000_000}, set(), 0.0)
    assert too_big["without_weights"] == ["C"] and too_big["keep"] == []
