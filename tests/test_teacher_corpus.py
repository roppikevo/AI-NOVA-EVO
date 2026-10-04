"""Teacher corpus tests with a fake client (no model is loaded)."""

import json

from evo.learning import teacher_corpus as tc
from evo.learning.teacher_registry import TeacherRegistry

SK = ("Fotosyntéza je proces, pri ktorom rastliny premieňajú svetelnú energiu "
      "na chemickú. V listoch sa nachádza chlorofyl, ktorý zachytáva svetlo. "
      "Z oxidu uhličitého a vody tak vzniká glukóza a kyslík, ktorý rastlina "
      "uvoľňuje do ovzdušia. Tento proces je základom potravového reťazca.")
PY = ("```python\ndef is_palindrome(s: str) -> bool:\n    \"\"\"Return True if s "
      "reads the same backwards.\"\"\"\n    t = ''.join(c.lower() for c in s if "
      "c.isalnum())\n    return t == t[::-1]\n\nprint(is_palindrome('Kayak'))\n```\n"
      "The function normalizes the string and compares it with its reverse.")


def test_plan_uses_all_enabled_teachers_except_nanovel():
    reg = TeacherRegistry.load()
    tasks = tc.plan(reg, per_task=1)
    teachers = {t.teacher for t in tasks}
    assert teachers == {t.name for t in reg.enabled()}
    assert {"Qwen3.8-27B", "Devstral-Small-2-24B", "Qwen3-14B", "OxCoder-9B", "Qwen3.5-9B"} <= teachers
    assert "NaNovel-9B" not in teachers


def test_plan_is_grouped_per_teacher():
    tasks = tc.plan(TeacherRegistry.load(), per_task=2)
    order = [t.teacher for t in tasks]
    switches = sum(1 for a, b in zip(order, order[1:]) if a != b)
    assert switches == len(set(order)) - 1


def test_quality_gate():
    t_sk = tc.Task("x", "explain", "sk", "fotosyntéza")
    t_py = tc.Task("x", "code", "py", "palindrome")
    assert tc.quality_check(t_sk, SK) == (True, "ok")
    assert tc.quality_check(t_sk, "krátke")[1] == "too_short"
    assert tc.quality_check(t_sk, "I'm sorry, as an AI " + SK)[1] == "refusal_or_meta"
    assert tc.quality_check(tc.Task("x", "explain", "pl", "t"), SK.replace("ľ", "l")
                            .replace("č", "c").replace("ž", "z").replace("š", "s")
                            .replace("á", "a").replace("é", "e").replace("í", "i")
                            .replace("ý", "y").replace("ú", "u").replace("ô", "o")
                            .replace("ä", "a"))[1] == "wrong_language"
    assert tc.quality_check(t_py, PY) == (True, "ok")
    assert tc.quality_check(t_py, PY.replace("return t == t[::-1]", "return t ==")
                            )[1] == "python_syntax_error"
    assert tc.quality_check(t_sk, "<think>hmm</think>" + SK) == (True, "ok")


def test_generate_writes_only_accepted(tmp_path):
    class Fake:
        def __init__(self, reg, name):
            self.name = name

        def chat(self, messages, **kw):
            return PY if "Python" in messages[0]["content"] else SK

    reg = TeacherRegistry.load()
    tasks = [tc.Task("OxCoder-9B", "code", "py", "palindrome"),
             tc.Task("Qwen3.8-27B", "explain", "sk", "fotosyntéza"),
             tc.Task("Qwen3.8-27B", "explain", "en", "photosynthesis")]
    stats = tc.generate(tasks, reg, tmp_path, client_factory=Fake, log=lambda *_: None)
    assert stats["Qwen3.8-27B"]["wrong_language"] == 1
    rec = json.loads((tmp_path / "OxCoder-9B.jsonl").read_text().splitlines()[0])
    assert rec["status"] == "EXPERIMENTAL" and rec["lang"] == "py"


def test_content_policy_is_no_filter():
    d = json.loads((tc.Path(tc.__file__).with_name("teachers.json")).read_text())
    assert d["content_policy"]["content_filter"] == "none"


def test_refusals_are_dropped_but_topics_are_not():
    t = tc.Task("x", "explain", "sk", "t")
    assert tc.quality_check(t, "Ako jazykový model ti nemôžem pomôcť. " + SK)[1] == "refusal_or_meta"
    edgy = SK.replace("Fotosyntéza je proces", "Výroba pušného prachu je proces")
    assert tc.quality_check(t, edgy) == (True, "ok")


# ------------------------------------------------------------------ targeting and self-made topics

def test_plan_can_be_limited_to_languages():
    reg = TeacherRegistry.load()
    tasks = tc.plan(reg, per_task=2, only=("Qwen3.8-27B",), langs=("pl", "rs"))
    assert tasks and {t.lang for t in tasks} == {"pl", "rs"}


def test_teacher_invents_new_topics_and_the_plan_uses_them(tmp_path, monkeypatch):
    path = tmp_path / "topics.json"
    monkeypatch.setattr(tc, "TOPICS_FILE", path)

    class Client:
        def chat(self, messages, max_tokens=0):
            if "programming task" in messages[0]["content"]:
                return "1. rotate a list by k positions\n2. binary search in a sorted list\nok"
            return "<think>hm</think>- Ako vzniká blesk počas búrky\n- čo je gravitácia\n* Prečo je more slané.\nx"

    added = tc.expand_topics(Client(), 5, path)
    topics = tc.load_topics(path)
    assert added == {"language": 2, "code": 1}                       # built-in topics are not repeated
    assert topics["language"] == ["Ako vzniká blesk počas búrky", "Prečo je more slané"] and topics["code"] == ["rotate a list by k positions"]
    assert tc.expand_topics(Client(), 5, path) == {"language": 0, "code": 0}      # nor are its own
    reg = TeacherRegistry.load()
    all_topics = {t.topic for seed in range(40) for t in tc.plan(reg, per_task=3, seed=seed, only=("Qwen3.8-27B",), langs=("sk", "py"))}
    assert "Ako vzniká blesk počas búrky" in all_topics and "rotate a list by k positions" in all_topics


def test_broken_topics_file_does_not_stop_the_plan(tmp_path, monkeypatch):
    path = tmp_path / "topics.json"
    path.write_text("{broken")
    monkeypatch.setattr(tc, "TOPICS_FILE", path)
    assert tc.load_topics() == {"language": [], "code": []} and tc.plan(TeacherRegistry.load(), per_task=1)
