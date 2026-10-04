"""Teacher registry tests (no network, nothing is loaded)."""

import io
import json

import pytest

from evo.learning import teacher_registry as tr


def test_registry_file_is_valid():
    reg = tr.TeacherRegistry.load()
    assert reg.router.startswith("http://127.0.0.1")
    assert "Qwen3.8-27B" in reg.teachers
    assert all(t.domains and t.languages for t in reg.teachers.values())


def test_choose_prefers_strongest_for_slovak():
    reg = tr.TeacherRegistry.load()
    assert reg.choose("language", "sk") == "Qwen3.8-27B"


def test_code_teachers_include_devstral_and_oxcoder():
    names = [t.name for t in tr.TeacherRegistry.load().candidates("code", "rs")]
    assert "Devstral-Small-2-24B" in names and "OxCoder-9B" in names


def test_disabled_teacher_not_chosen():
    reg = tr.TeacherRegistry.load()
    with pytest.raises(LookupError):
        reg.choose("creative")
    with pytest.raises(ValueError):
        tr.TeacherClient(reg, "NaNovel-9B")


def test_status_is_read_only_listing(monkeypatch):
    calls = []

    def fake_urlopen(url, timeout=0):
        calls.append(url)
        payload = {"data": [{"id": "OxCoder-9B", "status": {"value": "loaded"}},
                            {"id": "Qwen3.8-27B", "status": {"value": "unloaded"}}]}
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(tr.urllib.request, "urlopen", fake_urlopen)
    st = tr.TeacherRegistry.load().status()
    assert st["Qwen3.8-27B"] == "unloaded" and st["OxCoder-9B"] == "loaded"
    assert st["Devstral-Small-2-24B"] == "MISSING"
    assert calls == ["http://127.0.0.1:8081/v1/models"]


def test_mentor_order_devstral_first_oxcoder_fallback():
    reg = tr.TeacherRegistry.load()
    assert reg.mentors() == ["Devstral-Small-2-24B", "Qwen3.8-27B", "OxCoder-9B"]
    assert reg.timeout_factor("Devstral-Small-2-24B") == 4.0
    assert reg.timeout_factor("OxCoder-9B") == 1.0


def test_core_engine_falls_back_to_next_mentor():
    from evo.engine import core_evolution_engine as ce

    e = ce.CoreEvolutionEngine.__new__(ce.CoreEvolutionEngine)
    e._registry = tr.TeacherRegistry.load()
    e.mentors = e._registry.mentors()
    e.use_mentor(e.mentors[0])
    calls = []

    def ask(diag):
        calls.append(e.teacher_model)
        return {"response": "no code" if e.teacher_model != "Qwen3.8-27B" else "CODE"}

    def extract(resp):
        if resp != "CODE":
            raise ce.CoreEvolutionError("Teacher response did not contain a core source")
        return "source"

    e.ask_teacher, e.extract_source = ask, extract
    teacher, source = e.ask_teacher_with_fallback({})
    assert calls == ["Devstral-Small-2-24B", "Qwen3.8-27B"]
    assert teacher["mentor"] == "Qwen3.8-27B" and source == "source"
    assert e.teacher_timeout_factor == 4.0
