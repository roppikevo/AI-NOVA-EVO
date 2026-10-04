"""Self-correction loop, end to end on a tiny CPU model (offline)."""

import json

import pytest
import torch
import torch.nn as nn

pytest.importorskip("tokenizers")

from evo.corpus.build_text_v1 import build
from evo.corpus.identity import identity_documents
from evo.corpus.sources import Document
from evo.learning import self_correction as sc
from nova.tokenizer import NovaTokenizer

TEXTS = {
    "sk": "Bratislava je hlavné mesto Slovenska a leží na Dunaji pri hraniciach s Rakúskom.",
    "cs": "Praha je hlavní město České republiky a leží na řece Vltavě ve středu Čech.",
    "pl": "Warszawa jest stolicą Polski i leży nad Wisłą w środkowej części kraju.",
    "en": "London is the capital city of England and stands on the River Thames today.",
}


class Tiny(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, 32)
        self.rnn = nn.GRU(32, 32, batch_first=True)
        self.head = nn.Linear(32, vocab)

    def forward(self, x, states=None):
        h, s = self.rnn(self.emb(x), states)
        return self.head(h), s


@pytest.fixture()
def world(tmp_path, monkeypatch):
    groups = {lang: [Document(lang, "wikipedia", f"{t} Odsek {i}. " * 3) for i in range(80)]
              for lang, t in TEXTS.items()}
    groups["identity"] = list(identity_documents(80))
    ds = tmp_path / "text_test"
    build(groups, ds, seq_len=128, vocab_size=600, log=lambda *_: None)
    monkeypatch.setattr(sc, "MISTAKES", tmp_path / "mistakes.jsonl")
    monkeypatch.setattr(sc, "WEIGHTS_DIR", tmp_path / "ckpt")
    tok = NovaTokenizer.load(ds / "tokenizer.json")
    torch.manual_seed(0)
    return ds, tok, Tiny(tok.vocab_size)


def test_question_pools_are_deterministic_and_split():
    qs = sc.identity_questions({"current_generation": 7}, "roppik", "NOVA")
    assert all(q.answer in q.options for q in qs)
    assert {q.pool for q in qs} <= {"practice", "exam"}
    assert qs[0].pool == sc.identity_questions(None, "roppik", "NOVA")[0].pool


def test_held_out_docs_and_questions(world):
    ds, tok, _ = world
    docs = sc._held_out_documents(tok, ds / "val.txt")
    assert docs and all(lang in sc.LANG_OPTIONS for lang, _ in docs)
    import random
    lq = sc.language_questions(docs, 5, random.Random(1))
    assert all(q.answer in q.options for q in lq)


def test_only_practice_mistakes_are_remembered(world, tmp_path):
    results = [
        {"key": "a", "pool": "practice", "correct": False, "confidence": 0.9},
        {"key": "b", "pool": "exam", "correct": False, "confidence": 0.9},
        {"key": "c", "pool": "practice", "correct": True, "confidence": 0.9},
    ]
    assert sc.remember_mistakes(results) == 1
    assert sc.remember_mistakes(results) == 0  # no duplicates
    rec = json.loads(sc.MISTAKES.read_text().splitlines()[0])
    assert rec["key"] == "a" and rec["confident_mistake"]


def test_full_round_runs_and_decides(world):
    ds, tok, model = world
    report = sc.self_correction_round(model, tok, ds, {"current_generation": 5},
                                      steps=5, log=lambda *_: None)
    assert report["decision"] in ("KEEP", "DISCARD", "NOTHING_TO_LEARN")
    assert "before" in report and report["questions"] > 5
    if report["decision"] != "NOTHING_TO_LEARN":
        assert "val_loss_after" in report and "after" in report


def test_malformed_mistake_records_are_skipped(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"key": "a", "pool": "practice"}\nnot json\n'
                 + json.dumps({"kind": "cloze", "question": "q", "answer": "a",
                               "nova_answer": "b", "lang": "sk"}) + "\n")
    assert len(sc.load_mistakes(p)) == 1
