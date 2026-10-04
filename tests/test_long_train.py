"""Long training loop on a tiny CPU model (offline)."""

import json

import numpy as np
import torch
import torch.nn as nn

from evo.engine import long_train as lt


class Tiny(nn.Module):
    def __init__(self, vocab=50):
        super().__init__()
        self.emb = nn.Embedding(vocab, 32)
        self.rnn = nn.GRU(32, 32, batch_first=True)
        self.head = nn.Linear(32, vocab)

    def forward(self, x, states=None):
        h, s = self.rnn(self.emb(x), states)
        return self.head(h), s


def _data(n=64, L=17, vocab=50):
    base = (np.arange(L)[None, :] + np.arange(n)[:, None]) % vocab  # learnable pattern
    return base.astype(np.int32)


def test_lr_schedule():
    assert lt.lr_at(0, 1000, 1e-3, 100) < lt.lr_at(99, 1000, 1e-3, 100)
    assert abs(lt.lr_at(100, 1000, 1e-3, 100) - 1e-3) < 1e-9
    assert abs(lt.lr_at(1000, 1000, 1e-3, 100) - 1e-4) < 1e-9


def test_training_improves_and_saves(tmp_path):
    torch.manual_seed(0)
    r = lt.long_train(Tiny(), _data(), _data(16), steps=120, out_dir=tmp_path, meta={"tag": "t"},
                      batch_size=8, lr=3e-3, warmup=10, eval_every=40, save_every=40,
                      progress=tmp_path / "p.jsonl", stop_file=tmp_path / "STOP", log=lambda *_: None)
    assert r["best_val"] < r["val_start"]
    assert r["stop_reason"] in ("steps_done", "early_stopping")
    assert (tmp_path / "t-best.pt").exists() and (tmp_path / "t-last.pt").exists()
    rows = [json.loads(l) for l in (tmp_path / "p.jsonl").read_text().splitlines()]
    assert rows[-1]["step"] <= 120


def test_stop_file_and_resume(tmp_path):
    (tmp_path / "STOP").write_text("")
    m = Tiny()
    r = lt.long_train(m, _data(), _data(16), steps=100, out_dir=tmp_path, meta={"tag": "t"},
                      batch_size=4, eval_every=50, save_every=50, progress=tmp_path / "p.jsonl",
                      stop_file=tmp_path / "STOP", log=lambda *_: None)
    assert r["stop_reason"] == "stop_file" and r["steps_done"] == 1
    ck = torch.load(tmp_path / "t-last.pt", weights_only=False)
    r2 = lt.long_train(Tiny(), _data(), _data(16), steps=20, out_dir=tmp_path, meta={"tag": "t"},
                       batch_size=4, eval_every=10, save_every=10, resume=ck,
                       progress=tmp_path / "p.jsonl", stop_file=tmp_path / "STOP", log=lambda *_: None)
    assert r2["steps_done"] == 20


def test_load_tokens_caches(tmp_path):
    p = tmp_path / "train.txt"
    p.write_text("1 2 3 4\n5 6 7 8\n")
    a = lt.load_tokens(p)
    assert a.shape == (2, 4) and (tmp_path / "train.npy").exists()
    assert (lt.load_tokens(p) == a).all()
