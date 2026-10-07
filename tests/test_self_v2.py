"""Self-knowledge, second edition: authority first, then the goal to improve, then facts; outside the main dataset."""

import json

from evo.corpus import self_v2
from evo.corpus.sources import teacher_jsonl


def _release(tmp_path):
    rel = tmp_path / "NOVA8-24M-v1"
    rel.mkdir()
    (rel / "MODEL.json").write_text(json.dumps({"name": "NOVA8-24M-v1", "parameters": 24167975,
                                                "config": {"arch": "nova8", "pattern": "NSNSNSN", "slots": 16}}))
    return rel


def test_facts_come_from_the_release(tmp_path):
    f = self_v2.facts_from_release(_release(tmp_path), state_kb=56.0)
    assert f["name"] == "NOVA8-24M-v1" and f["generation"] == 8 and f["params_m"] == "24.2" and f["params_sk"] == "24,2"
    assert f["state_kb"] == "56" and f["slots"] == 16 and f["c"] == "roppik"
    old = tmp_path / "NOVA-24M-v2"
    old.mkdir()
    (old / "MODEL.json").write_text(json.dumps({"name": "NOVA-24M-v2", "parameters": 23700000, "config": {"d_model": 640}}))
    g = self_v2.facts_from_release(old)
    assert g["generation"] == 7 and g["slots"] is None and g["state_kb"] is None


def test_every_document_puts_the_creator_before_the_goal(tmp_path):
    facts = self_v2.facts_from_release(_release(tmp_path), state_kb=56.0)
    docs = self_v2.documents(facts, n_docs=600)
    assert len(docs) == 600 and {d["lang"] for d in docs} == {"sk", "cs", "pl", "en"}
    plain = [d for d in docs if "prompt" not in d]
    asked = [d for d in docs if "prompt" in d]
    assert len(asked) > 100 and len(plain) > 300
    for d in plain:
        text = d["text"]
        assert "roppik" in text and "{" not in text
        first_goal = min(text.find(g) for g in self_v2.GOAL[d["lang"]] if g in text)
        assert 0 < first_goal and text.find("roppik") < first_goal          # authority is stated first
    goals = [d for d in asked if d["prompt"] in ("Aký je tvoj hlavný cieľ?", "What is your main goal?")]
    assert goals and all("roppik" in d["text"] for d in goals)
    assert any("zlepšovať sa" in d["text"] for d in docs) and any("My main goal is to improve." in d["text"] for d in docs)
    order = [d for d in asked if "prednosť" in d["prompt"] or "comes first" in d["prompt"]]
    assert order and all(d["text"].startswith(("Najprv platí slovo", "The word of my creator")) for d in order)
    assert any("56 kB" in d["text"] for d in docs) and any("NOVA8-24M-v1" in d["text"] for d in docs)


def test_the_set_is_written_where_training_reads_extra_texts(tmp_path):
    facts = self_v2.facts_from_release(_release(tmp_path), state_kb=56.0)
    path = self_v2.write(tmp_path / "self_v2", facts, n_docs=200)
    read = list(teacher_jsonl(path.parent, 10 ** 9))
    assert len(read) == 200 and all(doc.text for doc in read) and any(doc.prompt for doc in read)
    assert json.loads((path.parent / "facts.json").read_text(encoding="utf-8"))["generation"] == 8
