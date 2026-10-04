"""Self-model corpus, think tokens and bounded thinking (offline)."""

import json

import pytest

pytest.importorskip("tokenizers")

from evo.corpus.build_text_v1 import build, encode_doc
from evo.corpus.self_model import facts_from_state, self_model_documents
from evo.corpus.sources import Document, teacher_jsonl
from nova.tokenizer import END_THINK_ID, THINK_ID, NovaTokenizer

STATE = {
    "current_generation": 5, "parent_generation": "GEN4",
    "primary_parent": "GEN4-CORE-002", "lineage": [1, 2, 3],
    "best_known": {"candidate": "GEN4-CORE-002", "validation_loss": 5.7698,
                   "parameters": 9856518, "dataset": "data/text_v2",
                   "efficiency": {"param_mb": 39.43, "cpu_tokens_per_sec": 5784.3,
                                  "peak_vram_mb": 542.3, "state_kb": 9.0}},
}


def test_self_model_facts_and_docs(tmp_path):
    f = facts_from_state(STATE)
    assert f["gen"] == 5 and f["params_m"] == "9,9" and f["cpu"] == 5784
    p = tmp_path / "evo_state.json"
    p.write_text(json.dumps(STATE))
    docs = list(self_model_documents(p, n_docs=50))
    assert len(docs) == 50
    text = " ".join(d.text for d in docs)
    assert "generácia 5" in text and "GEN4-CORE-002" in text and "5784" in text


def _tok(texts):
    return NovaTokenizer.train(texts, vocab_size=500, min_frequency=1)


def test_new_tokenizer_has_think_and_old_ids_stable():
    tok = _tok(["ahoj svet"] * 20)
    assert tok.has_think and tok.lang_id("sk") == 4
    ids = tok.encode_reasoned("Koľko je 2+2?", "2 a 2 je 4.", "4", "sk")
    assert ids[0] == 4 and THINK_ID in ids and END_THINK_ID in ids
    assert ids.index(THINK_ID) < ids.index(END_THINK_ID)


def test_teacher_jsonl_and_reasoned_encoding(tmp_path):
    rec = {"lang": "sk", "prompt": "Vysvetli dúhu.", "reasoning": "Svetlo sa láme v kvapkách.",
           "text": "Dúha vzniká lomom svetla.", "key": "k"}
    (tmp_path / "Qwen3.5-9B.jsonl").write_text(json.dumps(rec, ensure_ascii=False) + "\n")
    docs = list(teacher_jsonl(tmp_path, 10_000))
    assert docs[0].reasoning and docs[0].prompt
    tok = _tok([rec["prompt"], rec["reasoning"], rec["text"]] * 20)
    assert THINK_ID in encode_doc(tok, docs[0])
    assert THINK_ID not in encode_doc(tok, Document("sk", "wikipedia", "text"))


def test_generate_think_budget_is_enforced():
    import torch
    from nova.generate import generate

    tok = _tok(["ahoj svet"] * 20)

    class AlwaysThinkMore(torch.nn.Module):
        def forward(self, x, states=None):
            logits = torch.full((1, x.shape[1], tok.vocab_size), -1e9)
            logits[..., 20] = 0.0  # never emits </think> or <eos>
            return logits, None

    out = generate(AlwaysThinkMore(), tok, "Otázka", "sk", max_new_tokens=5,
                   think=True, max_think_tokens=8)
    assert "</think>" in out
