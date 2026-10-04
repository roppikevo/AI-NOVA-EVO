"""Entrance exam for a new teacher (no network: the router is faked)."""

import io
import json

from evo.learning import teacher_check as chk
from evo.learning.teacher_registry import TeacherRegistry

SK = ("Fotosyntéza je proces, ktorým rastliny premieňajú svetlo na energiu, ktorá sa ukladá v cukroch. "
      "Rastliny pri tom prijímajú oxid uhličitý a vodu, pretože z nich tvoria glukózu, a uvoľňujú kyslík. "
      "Tento dej prebieha v chloroplastoch, ktoré sú plné zeleného farbiva, a je dôležitý aj pre zvieratá. ") * 2
CS = ("Obloha je modrá, protože se světlo v atmosféře rozptyluje, a krátké vlny se rozptylují nejvíce. "
      "Molekuly vzduchu, které jsou velmi malé, rozptylují modré světlo více než červené, a proto je vidíme. ") * 2
PL = ("Grawitacja jest siłą, która przyciąga do siebie ciała mające masę, i nie można się od niej uwolnić. "
      "Jest to oddziaływanie, które utrzymuje planety na orbitach oraz sprawia, że przedmioty spadają na ziemię. ") * 2
EN = ("The internet is a network of networks in which computers exchange data in small packets. "
      "Routers decide which way the packets are to travel, and the protocols make sure all of them arrive. ") * 2
PY_ANSWER = ("```python\ndef is_palindrome(s: str) -> bool:\n    \"\"\"True if s reads the same backwards.\"\"\"\n"
             "    t = ''.join(c.lower() for c in s if c.isalnum())\n    return t == t[::-1]\n\n"
             "print(is_palindrome('Kayak'))\n```\nThe function normalises the text and compares it with its reverse, "
             "which is the usual way to do it and it runs in linear time for every input that is given to it.")
RS_ANSWER = ("```rust\nfn gcd(a: u64, b: u64) -> u64 {\n    if b == 0 { a } else { gcd(b, a % b) }\n}\n\n"
             "fn main() {\n    println!(\"{}\", gcd(48, 18));\n}\n```\nEuclid's algorithm: the remainder replaces "
             "the larger number until nothing is left, and the last value that is not zero is the answer we want.")
GOOD = {"sk": SK, "cs": CS, "pl": PL, "en": EN, "py": PY_ANSWER, "rs": RS_ANSWER}


def fake_router(monkeypatch, answers, models=("New-Teacher",)):
    order = iter([lang for _, lang, _ in chk.EXAM])

    def fake_urlopen(req, timeout=0):
        if isinstance(req, str):
            return io.BytesIO(json.dumps({"data": [{"id": m} for m in models]}).encode())
        text = answers[next(order)]
        if isinstance(text, Exception):
            raise text
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": text}}], "usage": {"completion_tokens": 200},
                                      "timings": {"predicted_per_second": 14.0}}).encode())

    monkeypatch.setattr(chk.urllib.request, "urlopen", fake_urlopen)


def registry(tmp_path):
    p = tmp_path / "teachers.json"
    p.write_text(json.dumps({"router": "http://127.0.0.1:8081", "teachers": {"New-Teacher": {
        "enabled": False, "role": "x", "domains": ["language"], "languages": ["sk"], "priority": 2, "speed": "slow"}}}))
    return p


def test_good_teacher_passes_and_is_switched_on(monkeypatch, tmp_path):
    fake_router(monkeypatch, GOOD)
    result = chk.exam("http://127.0.0.1:8081", "New-Teacher")
    assert result["passed"] and result["ok"] == 6 and result["tok_s"] == 14.0
    path = registry(tmp_path)
    assert chk.record(result, enable=True, path=path) is True
    entry = json.loads(path.read_text())["teachers"]["New-Teacher"]
    assert entry["enabled"] is True and entry["speed"] == "fast" and entry["measured"]["ok"] == 6


def test_teacher_that_answers_in_the_wrong_language_stays_off(monkeypatch, tmp_path):
    fake_router(monkeypatch, {**GOOD, "sk": EN})          # English instead of Slovak
    result = chk.exam("http://127.0.0.1:8081", "New-Teacher")
    assert not result["passed"] and result["failed"] == {"sk": "wrong_language"}
    path = registry(tmp_path)
    assert chk.record(result, enable=True, path=path) is False
    assert json.loads(path.read_text())["teachers"]["New-Teacher"]["enabled"] is False


def test_model_that_cannot_be_loaded_fails_without_a_crash(monkeypatch):
    fake_router(monkeypatch, {k: OSError("HTTP 500") for k in GOOD})
    result = chk.exam("http://127.0.0.1:8081", "New-Teacher")
    assert not result["passed"] and result["ok"] == 0 and result["tok_s"] == 0.0


def test_measuring_without_enable_does_not_switch_on(monkeypatch, tmp_path):
    fake_router(monkeypatch, GOOD)
    path = registry(tmp_path)
    assert chk.record(chk.exam("http://127.0.0.1:8081", "New-Teacher"), enable=False, path=path) is False


def test_unknown_model_is_reported(monkeypatch, capsys):
    fake_router(monkeypatch, GOOD, models=("OxCoder-9B",))
    assert chk.main(["--name", "Qwen3.6-35B-A3B"]) == 2
    assert "models.ini" in capsys.readouterr().out


def test_new_teacher_is_registered_but_off_until_checked():
    reg = TeacherRegistry.load()
    t = reg.teachers["Qwen3.6-35B-A3B"]
    assert set(t.languages) == {"sk", "cs", "pl", "en", "py", "rs"} and "mentor" not in t.domains
    assert reg.choose("language", "sk") == "Qwen3.8-27B"
