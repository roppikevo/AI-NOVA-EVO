"""RuleLab 01: problems right by construction, a fair 'cannot tell', exam parts never in the material."""

import json
import random

from evo.learning import rulelab as R


def solve(p):
    """An independent solver: builds the order from the sentences themselves, answers or says cannot tell."""
    import re
    names = [n for n in p["options"] if n != R.CANT[p["lang"]]]
    above = set()
    words_more = {w for d in R.DIMS_SK.values() for w in d[:2]} | {d[0] for d in R.DIMS_EN.values()}
    text = " ".join(p["facts"])
    text = re.sub(r"(\w+), (?:ktorý|ktorá|who) (je|is)", r"\1. \1 \2", text)      # "B, who is" -> "B. B is"
    for m in re.finditer(r"(\w+) (?:je|is) (\w+) (?:ako|than) (\w+)", text):
        a, w, b = m.groups()
        above.add((a, b) if w in words_more else (b, a))
    closure = set(above)
    changed = True
    while changed:
        changed = False
        for (a, b) in list(closure):
            for (c, d) in list(closure):
                if b == c and (a, d) not in closure:
                    closure.add((a, d))
                    changed = True
    cant = R.CANT[p["lang"]]
    if p["q"] == "pair":
        x, y = [n for n in names if n in p["question"]]
        return x if (x, y) in closure else y if (y, x) in closure else cant
    tops = [n for n in names if not any((o, n) in closure for o in names)]
    bottoms = [n for n in names if not any((n, o) in closure for o in names)]
    pick = tops if p["q"] == "most" else bottoms
    return pick[0] if len(pick) == 1 else cant


def test_every_answer_is_what_an_independent_solver_finds():
    for part in R.PARTS:
        for i in range(300):
            p = R.problem(i, part)
            assert solve(p) == p["answer"], (part, p["prompt"], p["answer"])
            assert p["answer"] in p["options"] and len(set(p["options"])) == len(p["options"])
            assert all(n in " ".join(p["facts"]) for n in p["order"])            # nobody is never mentioned


def test_contrasts_are_right_too_and_cannot_tell_is_common_enough():
    cant = 0
    for s in range(300):
        rng = random.Random(f"rulelab-mat:{s}")
        p = R.problem(s, "same", rng)
        c = R.contrast(p, rng)
        names = c["order"]
        c2 = {**c, "options": [n for n in names] + [R.CANT[c["lang"]]]}
        assert solve(c2) == c["answer"], (c["prompt"], c["answer"])
        cant += c["cannot"] + p["cannot"]
    assert 60 < cant < 400


def test_the_exam_parts_differ_and_never_appear_in_material(tmp_path):
    words = " ".join(R.problem(R.EXAM_SEED + i, "words")["prompt"] for i in range(50))
    assert any(w in words for w in ("ťažš", "ľahš", "heavier", "richer", "bohatš", "poorer"))
    assert any(w in " ".join(R.problem(R.EXAM_SEED + i, "phrasing")["prompt"] for i in range(50)) for w in ("ktor", "who"))
    assert all(len(R.problem(R.EXAM_SEED + i, "deeper")["order"]) == 5 for i in range(20))
    for method in ("qa", "rule"):
        rows = [json.loads(l) for l in R.material(method, 800, tmp_path).read_text().splitlines()]
        text = " ".join(r["prompt"] + " " + r["text"] for r in rows)
        assert not any(w in text for w in ("ťažš", "ľahš", "heavier", "lighter", "richer", "poorer", "bohatš", "chudobnejš"))
        assert ", ktor" not in text and ", who" not in text
        exam = {R.problem(R.EXAM_SEED + i, "same")["prompt"] for i in range(400)}
        assert not exam & {r["prompt"] for r in rows}
    rule = [json.loads(l) for l in (tmp_path / "rule" / "material.jsonl").read_text().splitlines()]
    assert all(("Poradie:" in r["text"] or "Order:" in r["text"]) for r in rule)


def test_the_written_order_is_read_back_and_checked():
    assert R.parse_order("Poradie: Lea > Eva > Nina") == [["Lea", "Eva", "Nina"]]
    assert R.parse_order("Order: A > B | C > D (two separate pieces)") == [["A", "B"], ["C", "D"]]
    assert R.parse_order("Lea") is None
    for i in range(50):
        p = R.problem(i, "deeper")
        assert R.parse_order(R.order_line(p)) == ([p["order"]] if p["gap"] is None else [p["order"][:p["gap"] + 1], p["order"][p["gap"] + 1:]])
