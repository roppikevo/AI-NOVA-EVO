"""GPU guard with fake nvidia-smi and router."""

import io
import json

from nova import gpu_guard as g


def test_no_action_when_enough_vram(monkeypatch):
    monkeypatch.setattr(g, "free_vram_mb", lambda: 6000)
    r = g.ensure_vram(need_mb=2500, log=lambda *_: None)
    assert r["unloaded"] == []


def test_unloads_router_models_until_free(monkeypatch):
    free = iter([700, 700, 5200])
    monkeypatch.setattr(g, "free_vram_mb", lambda: next(free))
    monkeypatch.setattr(g, "loaded_models", lambda router=g.ROUTER: ["Devstral-Small-2-24B"])
    calls = []
    monkeypatch.setattr(g, "unload", lambda m, router=g.ROUTER: calls.append(m) or True)
    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    r = g.ensure_vram(need_mb=2500, log=lambda *_: None)
    assert calls == ["Devstral-Small-2-24B"] and r["free_after_mb"] == 5200


def test_required_parameters_regex():
    import re
    src = "self.temperature = nn.Parameter(torch.ones(1))\nself.w: torch.Tensor = nn.Parameter(x)\nself.b = nn.Linear(2,2)"
    names = sorted(set(re.findall(r"self\.(\w+)\s*(?::[^=\n]+)?=\s*nn\.Parameter", src)))
    assert names == ["temperature", "w"]
