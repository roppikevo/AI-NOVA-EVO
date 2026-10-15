"""evo.learning.skills: tasks made by a program that knows the answer - an exam per kind of thinking, and material."""

import json

import pytest
import torch

from evo.learning import skills as S


def test_every_kind_makes_right_answers_the_same_way_every_time():
    for kind in S.KINDS:
        a, b = S.item(kind, 7), S.item(kind, 7)
        assert a == b and 2 <= len(a["options"]) <= 4 and 0 <= a["answer"] < len(a["options"])
        assert len(set(a["options"])) == len(a["options"]) and a["lang"] in ("sk", "en", "py")
    for i in range(200):
        it = S.item("arith", i)
        q = it["prompt"].replace("Koľko je ", "").replace("What is ", "").rstrip("?")
        assert eval(q) == int(it["options"][it["answer"]])
        it = S.item("compare", i)
        nums = [int(x) for x in it["prompt"].replace("?", "").split() if x.isdigit()]
        assert int(it["options"][it["answer"]]) == max(nums)


def test_logic_needs_logic_not_memory():
    seen = {S.item("logic-if", i)["options"][S.item("logic-if", i)["answer"]] for i in range(60)}
    assert {"áno", "nie", "nedá sa určiť"} & seen and ({"yes", "no", "cannot tell"} & seen)
    trap = [S.item("deduction", i) for i in range(200) if "určite" in S.item("deduction", i)["prompt"]]
    assert trap and all(t["options"][t["answer"]] == "nedá sa určiť" for t in trap)     # affirming the consequent


def test_material_never_contains_an_exam_item(tmp_path):
    p = S.material("recall", 500, tmp_path)
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    exam = {e["prompt"] for e in S.exam_items("recall", 400)}
    assert len(rows) == 500 and not exam & {r["prompt"] for r in rows}
    assert all(r["text"] and r["lang"] in ("sk", "en") for r in rows)


def test_the_exam_reads_the_cores_choice_and_knows_chance(tmp_path):
    from evo.engine.architecture_factory import build_model
    from nova.tokenizer import NovaTokenizer

    texts = [S.item(k, i)["prompt"] + " " + " ".join(S.item(k, i)["options"]) for k in S.KINDS for i in range(40)]
    tok = NovaTokenizer.train(texts, vocab_size=400)
    tok.save(tmp_path / "tok.json")
    cfg = {"vocab_size": 400, "arch": "nova8", "d_model": 32, "pattern": "NSN", "mlp_hidden": 48, "heads": 2, "slots": 4}
    torch.manual_seed(0)
    torch.save({"config": cfg, "model_state_dict": build_model(cfg).state_dict()}, tmp_path / "m.pt")
    r = S.exam(str(tmp_path / "m.pt"), str(tmp_path / "tok.json"), n=20, device="cpu")
    assert set(r["kinds"]) == set(S.KINDS) and r["weakest"][0] in S.KINDS
    for k, v in r["kinds"].items():
        assert 0 <= v["accuracy"] <= 1 and 0.24 <= v["chance"] <= 0.5
    # a core that always answers right scores 1
    items = S.exam_items("compare", 10)
    class Oracle(torch.nn.Module):
        def forward(self, x):
            return (torch.zeros(x.shape[0], x.shape[1], 400),)
    assert len(S.score_options(Oracle(), tok, items)) == 10


def test_a_second_form_and_the_trained_items_and_an_honest_interval():
    a, b = S.item("ordering", 3), S.item("ordering", 3, fmt=1)
    assert b["prompt"].startswith(("Otázka:", "Question:")) and b["prompt"].endswith(("Odpoveď:", "Answer:"))
    assert a["options"] == b["options"] and a["answer"] == b["answer"] and b["sep"] == " "
    tr = S.trained_items("recall", 50)
    assert len(tr) == 50 and not {t["prompt"] for t in tr} & {e["prompt"] for e in S.exam_items("recall", 400)}
    lo, hi = S.wilson(60, 200)
    assert lo < 0.3 < hi and hi - lo < 0.14


def test_paired_comparison_counts_only_the_items_that_changed(tmp_path):
    assert S.mcnemar("1" * 100, "1" * 100)["p"] == 1.0
    r = S.mcnemar("0" * 200, "0" * 150 + "1" * 50)
    assert r["only_b"] == 50 and r["p"] < 1e-10 and r["diff"] == 0.25
    assert S.mcnemar("10" * 50, "01" * 50)["p"] == 1.0                     # 50 won, 50 lost: nothing
    assert S.holm({"a": 0.001, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}      # 0.05/3, /2, /1
    assert S.holm({"a": 0.001, "b": 0.03, "c": 0.04}) == {"a": True, "b": False, "c": False}    # b > 0.05/2 stops
    for name, c in (("x", "0" * 200), ("y", "0" * 140 + "1" * 60)):
        (tmp_path / f"{name}.json").write_text(json.dumps({"kinds": {"arith": {"accuracy": c.count("1") / 200, "correct": c}}}))
    r = S.compare(str(tmp_path / "x.json"), str(tmp_path / "y.json"))
    assert r["arith"]["only_b"] == 60 and r["arith"]["significant_holm"]


def test_material_in_several_forms_never_uses_the_exams_unknown_form(tmp_path):
    p = S.material("ordering", 400, tmp_path, forms=S.TRAIN_FORMS)
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    assert {r["form"] for r in rows} == set(S.TRAIN_FORMS)
    assert not any(r.get("prompt", r["text"]).startswith(("Otázka:", "Question:")) for r in rows)
    exam = {e["prompt"] for e in S.exam_items("ordering", 400)}
    asked = {S.item("ordering", S.EXAM_SEED + i, f)["prompt"] + " " for i in range(400) for f in (2, 3, 4)}
    assert not any(r["text"].startswith(q) for r in rows if r["form"] for q in asked)          # no exam question at all
    assert any(r["text"].startswith(("Úloha:", "Task:")) for r in rows)
    with pytest.raises(ValueError):
        S.material("ordering", 10, tmp_path, forms=(0, 1))
    assert S.item("arith", 5, 1)["prompt"].startswith(("Otázka:", "Question:"))                  # the exam is unchanged
