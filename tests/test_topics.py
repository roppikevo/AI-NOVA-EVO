"""Uč sa <téma>: the queue, the weak spots, the material with its exam part, the topic exam."""

import json

import torch

from evo.learning import topics as T


def test_the_queue_puts_the_creators_topics_first_and_never_twice(tmp_path):
    store = tmp_path / "topics.json"
    g = T.add("general knowledge in English", ["en"], source="director", path=store, now=1.0)
    a = T.add("Fotosyntéza rastlín", ["sk", "xx"], path=store, now=2.0)
    assert a["slug"] == "fotosynteza-rastlin" and a["langs"] == ["sk"] and a["status"] == "queued"
    assert T.add("fotosyntéza   rastlín", ["sk"], path=store)["slug"] == a["slug"] and len(T.load(store)) == 2
    assert T.next_topic(T.load(store))["slug"] == a["slug"]                       # the Creator first
    T.update(a["slug"], store, status="accepted")
    assert T.next_topic(T.load(store))["slug"] == g["slug"]
    c = T.add("spracovanie textu", ["sk"], kind="code", path=store)
    assert c["langs"] == ["py"] and T.add("Fotosyntéza rastlín", ["sk"], path=store)["slug"] == "fotosynteza-rastlin-2"
    T.main(["list", "--store", str(store)])


def test_weak_spots_are_the_weakest_language_and_the_weakest_code_level():
    from evo.learning.code_tasks import task_bank

    exam = [t for t in task_bank() if t.pool == "exam"]
    solved = [t.key for t in exam if t.level == 1]                                 # level 1 solved, 2 and 3 not
    gaps = T.find_gaps({"web_by_language": {"sk": 2.7, "cs": 3.1, "en": 3.4}}, solved, [], now=100.0)
    assert gaps[0]["langs"] == ["en"] and "English" in gaps[0]["text"]
    code = gaps[1]
    assert code["kind"] == "code" and code["level"] in (2, 3) and T.code_levels(solved)[code["level"]][0] == 0
    recent = [{"text": gaps[0]["text"], "created": 50.0}]
    assert all(g["langs"] != ["en"] for g in T.find_gaps({"web_by_language": {"en": 3.4}}, None, recent, now=100.0))


def test_material_is_split_into_training_and_an_exam_that_never_changes(tmp_path):
    topic = {"slug": "rieky", "text": "rieky Slovenska", "langs": ["sk"], "kind": "explain"}

    class Registry:
        def choose(self, domain, lang):
            return "Teacher-1"

    class Client:
        def __init__(self, registry, name):
            pass

        def chat(self, messages, temperature=0.7, max_tokens=900):
            return "Dunaj a jeho prítoky\nVáh ako najdlhšia rieka\nHron a jeho údolie"

    texts = [f"Text číslo {i} o riekach Slovenska, ktoré tečú do Dunaja a nesú vodu z hôr. " * 3 for i in range(60)]

    def generate(tasks, registry, out_dir, client_factory, log, max_minutes):
        assert {t.topic for t in tasks} == {"Dunaj a jeho prítoky", "Váh ako najdlhšia rieka", "Hron a jeho údolie"}
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "Teacher-1.jsonl").open("w") as f:
            for i, t in enumerate(texts[:40]):
                f.write(json.dumps({"lang": "sk", "text": t, "prompt": "Vysvetli", "status": "EXPERIMENTAL"}) + "\n")
            f.write(json.dumps({"lang": "sk", "text": "zlé", "status": "REJECTED"}) + "\n")

    def web(topics_, cfg, out_dir, log):
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "sk.jsonl").open("w") as f:
            for t in texts[40:]:
                f.write(json.dumps({"lang": "sk", "text": t, "url": "https://sk.wikipedia.org/x"}) + "\n")

    r = T.collect(topic, Registry(), {}, root=tmp_path / "m", exam_root=tmp_path / "e", client_factory=Client,
                  generate_fn=generate, web_fn=web, log=lambda *_: None)
    assert r["train_docs"] + r["exam_docs"] == 60 and r["exam_docs"] >= 1 and r["teacher"] == "Teacher-1"
    train = (tmp_path / "m/rieky/material.jsonl").read_text().splitlines()
    exam = (tmp_path / "e/rieky.jsonl").read_text().splitlines()
    assert len(train) == r["train_docs"] and len(exam) == r["exam_docs"]
    assert not set(train) & set(exam) and all(T.exam_part(json.loads(l)["text"]) for l in exam)
    assert list((tmp_path / "m/rieky").glob("*.jsonl")) == [tmp_path / "m/rieky/material.jsonl"]      # raw stays in _raw/


def test_the_topic_exam_measures_a_core_on_text_it_never_trained_on(tmp_path):
    from evo.engine.architecture_factory import build_model
    from nova.tokenizer import NovaTokenizer

    texts = ["Dunaj je najdlhšia rieka, ktorá tečie cez Bratislavu do Čierneho mora."] * 30
    tok = NovaTokenizer.train(texts, vocab_size=300)
    tok.save(tmp_path / "tokenizer.json")
    cfg = {"vocab_size": 300, "arch": "nova8", "d_model": 32, "pattern": "NSN", "mlp_hidden": 48, "heads": 2, "slots": 4}
    torch.manual_seed(0)
    torch.save({"config": cfg, "model_state_dict": build_model(cfg).state_dict()}, tmp_path / "core.pt")
    (tmp_path / "e").mkdir()
    (tmp_path / "e/rieky.jsonl").write_text("\n".join(json.dumps({"lang": "sk", "text": t}) for t in texts))
    loss = T.exam_loss(tmp_path / "core.pt", "rieky", tmp_path / "tokenizer.json", exam_root=tmp_path / "e", device="cpu")
    assert loss is not None and 4.0 < loss < 7.0                                    # untrained: about ln(300)
    assert T.exam_loss(tmp_path / "core.pt", "nothing", tmp_path / "tokenizer.json", exam_root=tmp_path / "e") is None


def test_without_a_teacher_the_subtopics_come_from_wikipedia_links_in_each_language():
    class Down:
        def chat(self, *a, **k):
            raise OSError("HTTP Error 500")

    class FakeBrowser:
        def _wikipedia(self, query, lang, n):
            assert lang == "sk"                                             # searched only in the first language
            return [{"title": "Fotosyntéza", "url": "https://sk.wikipedia.org/wiki/Fotosynt%C3%A9za"}]

        def _get_json(self, url):
            assert "langlinks" in url and "lllang=en" in url
            return {"query": {"pages": {"1": {"langlinks": [{"lang": "en", "*": "Photosynthesis"}]}}}}

        def open(self, url):
            if "en.wikipedia" in url:
                w = "https://en.wikipedia.org/wiki/"
                links = [w + "Chlorophyll", w + "Main_Page", w + "Glucose", w + "Chlorophyll"]
            else:
                w = "https://sk.wikipedia.org/wiki/"
                links = [w + "Hlavn%C3%A1_str%C3%A1nka", w + "Chlorofyl", w + "Glukóza", w + "Chlorofyl",
                         w + "Kategória:Biológia", w + "Fotosynt%C3%A9za", w + "Chloroplast", "https://example.com/x",
                         w + "1990", w + "Chlorofyl?action=edit"]
            return {"links": [{"url": u, "text": ""} for u in links]}

    topic = {"slug": "fotosynteza", "text": "fotosyntéza", "langs": ["sk", "en"], "kind": "explain"}
    subs = T.subtopics(Down(), topic, 12, FakeBrowser(), log=lambda *_: None)
    assert subs == ["sk:Fotosyntéza", "sk:Chlorofyl", "sk:Glukóza", "sk:Chloroplast",
                    "en:Photosynthesis", "en:Chlorophyll", "en:Glucose"]   # no main page, namespaces, years
    assert T.split_lang("en:Chlorophyll", ["sk", "en"]) == (["en"], "Chlorophyll")
    assert T.split_lang("Chlorofyl a svetlo", ["sk", "en"]) == (["sk", "en"], "Chlorofyl a svetlo")
    code = {"slug": "c", "text": "triedenie", "langs": ["py"], "kind": "code"}
    assert T.subtopics(Down(), code, 12, FakeBrowser(), log=lambda *_: None) == ["triedenie"]
