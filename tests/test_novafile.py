""".nova: one file with the core, its tokenizer and what it learned; smaller types keep the answers."""

import json

import pytest
import torch

from nova import novafile as NF


def tiny(tmp_path):
    from evo.engine.architecture_factory import build_model
    from nova.tokenizer import NovaTokenizer

    texts = ["Dunaj je najdlhšia rieka, ktorá tečie cez Bratislavu do Čierneho mora."] * 30 + \
            ["The river flows through the old town towards the sea."] * 30
    tok = NovaTokenizer.train(texts, vocab_size=300)
    rel = tmp_path / "NOVA8-TINY-v1"
    rel.mkdir()
    tok.save(rel / "tokenizer.json")
    cfg = {"vocab_size": 300, "arch": "nova8", "d_model": 64, "pattern": "NSN", "mlp_hidden": 128, "heads": 2, "slots": 4}
    torch.manual_seed(0)
    model = build_model(cfg)
    torch.save({"config": cfg, "model_state_dict": model.state_dict()}, rel / "nova_model.pt")
    (rel / "MODEL.json").write_text(json.dumps({"name": "NOVA8-TINY-v1", "frozen": "2026-10-11", "parent": "NOVA8-NONE"}))
    return rel, model.eval(), tok


def test_a_tensor_survives_every_type_within_its_precision():
    torch.manual_seed(0)
    w = torch.randn(70, 100)
    for dtype, tol in (("f32", 0.0), ("f16", 2e-3), ("int8", 0.03), ("int4", 0.6)):
        stored, arrays = NF._quantise(w, dtype)
        back = NF._restore({"shape": list(w.shape), "dtype": stored}, arrays)
        assert back.shape == w.shape and float((back - w).abs().max()) <= tol + 1e-6, dtype
    small = torch.randn(64)
    assert NF._quantise(small, "int4")[0] == "f32"                     # norms and time scales stay exact


def test_pack_read_and_the_core_answers_the_same(tmp_path):
    rel, model, tok = tiny(tmp_path)
    topics = tmp_path / "topics.json"
    topics.write_text(json.dumps([{"slug": "rieky", "text": "rieky Slovenska", "langs": ["sk"], "status": "accepted",
                                   "exam": {"before": 4.1, "after": 3.6}},
                                  {"slug": "x", "text": "nehotové", "status": "queued"}]))
    ids = torch.tensor([[tok.lang_id("sk")] + tok.encode("Dunaj je rieka")])
    with torch.no_grad():
        ref = model(ids)[0]
    sizes = {}
    for dtype, tol in (("f32", 1e-5), ("f16", 0.05), ("int8", 0.5)):
        r = NF.pack_release(rel, tmp_path / f"m-{dtype}.nova", dtype, topics)
        sizes[dtype] = r["bytes"]
        m2, tok2, header = NF.load(r["path"])
        with torch.no_grad():
            got = m2(ids)[0]
        assert float((got - ref).abs().max()) < tol, dtype
        assert tok2.encode("Dunaj je rieka") == tok.encode("Dunaj je rieka")
        assert header["model"]["name"] == "NOVA8-TINY-v1" and header["model"]["dtype"] == dtype
        assert [e.get("topic") for e in header["learned"]] == [None, "rieky Slovenska"]        # only finished topics
    assert sizes["int8"] < sizes["f16"] < sizes["f32"]
    r = NF.compare(tmp_path / "m-int8.nova", rel)
    assert r["same_next_token"] > 0.9 and abs(r["loss_packed"] - r["loss_original"]) < 0.05


def test_the_file_says_what_it_is_and_refuses_what_it_is_not(tmp_path, capsys):
    rel, _, _ = tiny(tmp_path)
    f = NF.pack_release(rel, tmp_path / "m.nova", "int4", None)["path"]
    head = open(f, "rb").read(4)
    assert head == b"NOVA"
    h, data_start = NF.read_header(f)
    assert data_start % NF.ALIGN == 0 and all(p["offset"] % NF.ALIGN == 0 for t in h["tensors"] for p in t["parts"])
    NF.main(["info", f])
    assert "NOVA8-TINY-v1" in capsys.readouterr().out
    bad = tmp_path / "bad.nova"
    bad.write_bytes(b"GGUF" + b"\0" * 60)
    with pytest.raises(ValueError, match="not a .nova"):
        NF.read_header(bad)
    with pytest.raises(ValueError, match="unknown type"):
        NF.write(tmp_path / "x.nova", {}, {}, "{}", dtype="q3")


def test_learn_puts_the_topic_in_the_directors_queue(tmp_path, monkeypatch, capsys):
    rel, _, _ = tiny(tmp_path)
    f = NF.pack_release(rel, tmp_path / "m.nova", "int8", None)["path"]
    monkeypatch.chdir(tmp_path)
    (tmp_path / "evo/director").mkdir(parents=True)
    NF.main(["learn", f, "Fotosyntéza rastlín", "--langs", "sk,en"])
    rows = json.loads((tmp_path / "evo/director/topics.json").read_text())
    rows = list(rows.values()) if isinstance(rows, dict) else rows
    assert rows[0]["slug"] == "fotosynteza-rastlin" and rows[0]["status"] == "queued" and rows[0]["langs"] == ["sk", "en"]
    assert "learning queue" in capsys.readouterr().out
