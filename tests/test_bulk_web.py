"""Bulk web corpus: filters, resumable fetch, mixing into long_train (offline)."""

import json

import numpy as np
import pytest

pytest.importorskip("tokenizers")

from evo.corpus import bulk_web as bw
from nova.tokenizer import NovaTokenizer

GOOD = "Bratislava je hlavné mesto Slovenska a leží na Dunaji. " * 8


def test_filters():
    spec = bw.SOURCES["sk"]
    assert bw.keep_doc(GOOD, {"language_score": 0.95}, spec)
    assert not bw.keep_doc("krátke", {}, spec)
    assert not bw.keep_doc(GOOD, {"language_score": 0.3}, spec)
    assert not bw.keep_doc("1234 5678 ### " * 40, {}, spec)
    assert not bw.keep_doc("\n".join(["Kúpiť teraz zľava!"] * 30), {}, spec)
    assert not bw.keep_doc(GOOD, {"int_score": 2}, bw.SOURCES["en"])
    assert bw.fingerprint(GOOD) == bw.fingerprint(GOOD.upper())


def test_fetch_is_resumable_and_deduplicated(tmp_path, monkeypatch):
    tok = NovaTokenizer.train([GOOD] * 20, vocab_size=300, min_frequency=1)
    rows = [({"text": GOOD + f" Dokument {i}.", "language_score": 0.9}, {"file": 0, "row_group": i + 1})
            for i in range(10)] + [({"text": GOOD + " Dokument 0.", "language_score": 0.9}, {"file": 0, "row_group": 11})]
    calls = []

    def fake_rows(spec, pos, log=print):
        calls.append(dict(pos))
        start = pos.get("row_group", 0)
        yield from rows[start:]

    monkeypatch.setattr(bw, "iter_rows", fake_rows)
    state, seen = {}, set()
    r1 = bw.fetch_source("sk", 0.0015, tok, tmp_path, state, seen, seq_len=32, log=lambda *_: None)
    assert r1["docs"] >= 1 and (tmp_path / "sk-000.npy").exists()
    r2 = bw.fetch_source("sk", 1.0, tok, tmp_path, state, seen, seq_len=32, log=lambda *_: None)
    assert r1["docs"] + r2["docs"] == 10          # duplicate of document 0 dropped
    arr = bw.load_bulk(tmp_path)
    assert arr.dtype == np.int32 and arr.shape[1] == 32


def test_long_train_mixes_bulk(tmp_path):
    import torch
    import torch.nn as nn
    from evo.engine.long_train import long_train

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.e, self.h = nn.Embedding(50, 8), nn.Linear(8, 50)

        def forward(self, x, states=None):
            return self.h(self.e(x)), None

    train = np.random.default_rng(0).integers(0, 50, size=(40, 16)).astype(np.int32)
    bulk = np.random.default_rng(1).integers(0, 50, size=(80, 16)).astype(np.int32)
    rep = long_train(Tiny(), train, train[:8], steps=4, out_dir=tmp_path, meta={"tag": "t"},
                     batch_size=4, warmup=1, eval_every=2, save_every=100, bulk=bulk, bulk_frac=0.7,
                     code=train[:3], code_frac=0.2,
                     progress=tmp_path / "p.jsonl", stop_file=tmp_path / "STOP", log=lambda *_: None)
    assert rep["steps_done"] == 4
    rows = [json.loads(l) for l in (tmp_path / "p.jsonl").read_text().splitlines()]
    assert rows[0]["start"] is True


def test_code_practice_sequences_hold_out_exam(tmp_path):
    from evo.engine.long_train import code_practice_sequences
    from evo.learning.code_tasks import task_bank

    bank = task_bank()
    tok = NovaTokenizer.train([t.prompt + t.solution for t in bank] * 3, vocab_size=600, min_frequency=1)
    tok.save(tmp_path / "tokenizer.json")
    arr = code_practice_sequences(tmp_path, 64)
    assert arr.shape[1] == 64 and len(arr) > 10
    text = "".join(tok.decode(r.tolist()) for r in arr)
    exam = [t for t in bank if t.pool == "exam"]
    assert not any(t.prompt.strip() in text for t in exam)


def test_optimizer_state_is_carried_between_runs(tmp_path):
    import torch
    import torch.nn as nn
    from evo.engine.long_train import long_train

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.e, self.h = nn.Embedding(50, 8), nn.Linear(8, 50)

        def forward(self, x, states=None):
            return self.h(self.e(x)), None

    train = np.random.default_rng(0).integers(0, 50, size=(40, 16)).astype(np.int32)
    kw = dict(steps=4, out_dir=tmp_path, batch_size=4, warmup=1, eval_every=2, save_every=100,
              progress=tmp_path / "p.jsonl", stop_file=tmp_path / "STOP")
    m = Tiny()
    rep = long_train(m, train, train[:8], meta={"tag": "a", "kind": "long_train"}, log=lambda *_: None, **kw)
    ck = torch.load(rep["last_checkpoint"], map_location="cpu", weights_only=False)
    logs = []
    long_train(m, train, train[:8], meta={"tag": "b"}, warm_optimizer=ck["optimizer"], log=logs.append, **kw)
    assert any("carried over" in l for l in logs)
    logs.clear()
    long_train(m, train, train[:8], meta={"tag": "c"}, warm_optimizer={"bad": 1}, log=logs.append, **kw)
    assert any("cold start" in l for l in logs)


def test_bulk_parts_behave_like_one_array(tmp_path):
    a = np.arange(12, dtype=np.int32).reshape(3, 4)
    b = np.arange(12, 32, dtype=np.int32).reshape(5, 4)
    np.save(tmp_path / "sk-000.npy", a)
    np.save(tmp_path / "sk-001.npy", b)
    np.save(tmp_path / "en-000.npy", b)
    bulk = bw.load_bulk(tmp_path)
    full = np.concatenate([b, a, b])          # files are sorted by name: en-000, sk-000, sk-001
    assert len(bulk) == 13 and bulk.shape == (13, 4)
    idx = np.array([0, 4, 5, 7, 8, 12])
    assert (bulk[idx] == full[idx]).all()
    assert len(bw.load_bulk_lang(tmp_path, "sk")) == 8 and bw.load_bulk_lang(tmp_path, "pl") is None


def test_a_training_line_continues_where_it_stopped(tmp_path):
    import torch
    import torch.nn as nn
    from evo.engine.long_train import long_train

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.e, self.h = nn.Embedding(50, 8), nn.Linear(8, 50)

        def forward(self, x, states=None):
            return self.h(self.e(x)), None

    train = np.random.default_rng(0).integers(0, 50, size=(40, 16)).astype(np.int32)
    stop = tmp_path / "STOP"
    kw = dict(steps=6, out_dir=tmp_path, meta={"tag": "LINE", "kind": "long_train"}, batch_size=4, warmup=1,
              eval_every=3, save_every=100, patience=10 ** 6, progress=tmp_path / "p.jsonl", stop_file=stop,
              log=lambda *_: None)
    stop.write_text("x")                                   # first segment is interrupted right away
    r1 = long_train(Tiny(), train, train[:8], **kw)
    assert r1["stop_reason"] == "stop_file" and r1["steps_done"] == 1
    ck = torch.load(tmp_path / "LINE-last.pt", map_location="cpu", weights_only=False)
    r2 = long_train(Tiny(), train, train[:8], resume=ck, **kw)
    assert r2["stop_reason"] == "steps_done" and r2["steps_done"] == 6 and (tmp_path / "LINE-best.pt").exists()


def test_second_validation_set_decides_best_and_metric_change_resets_best(tmp_path):
    import torch
    import torch.nn as nn
    from evo.engine.long_train import long_train
    from evo.engine.release import pick
    from evo.engine.train_line import web_validation

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.e, self.h = nn.Embedding(50, 8), nn.Linear(8, 50)

        def forward(self, x, states=None):
            return self.h(self.e(x)), None

    rng = np.random.default_rng(0)
    train = rng.integers(0, 50, size=(40, 16)).astype(np.int32)
    web = rng.integers(0, 50, size=(8, 16)).astype(np.int32)
    kw = dict(steps=4, out_dir=tmp_path, meta={"tag": "L", "kind": "long_train"}, batch_size=4, warmup=1, eval_every=2,
              save_every=100, patience=10 ** 6, progress=tmp_path / "p.jsonl", stop_file=tmp_path / "STOP", log=lambda *_: None)
    r1 = long_train(Tiny(), train, train[:8], **kw)
    assert r1["select_metric"] == "val" and "web_val" not in r1["best_parts"]
    ck = torch.load(tmp_path / "L-last.pt", map_location="cpu", weights_only=False)
    ck["best_val"] = 0.001                                # an unbeatable old best in the OLD metric ...
    r2 = long_train(Tiny(), train, train[:8], resume=ck, extra_val=web, **{**kw, "steps": 8})
    assert r2["select_metric"] == "mean(val,web_val)" and r2["best_val"] > 1.0   # ... does not block the new metric
    assert "web_val" in r2["best_parts"]
    rows = [json.loads(l) for l in (tmp_path / "p.jsonl").read_text().splitlines()]
    assert any("web_val" in r for r in rows)
    np.save(tmp_path / "sk-003.npy", web)
    np.save(tmp_path / "en-001.npy", web)
    assert web_validation(tmp_path, per_lang=5).shape == (10, 16)
    assert pick([{"val": 3.30, "web_val": 4.0, "creator": 0, "checkpoint": "old"},
                 {"val": 3.34, "web_val": 3.8, "creator": 0, "checkpoint": "new"}])["checkpoint"] == "new"
