"""
Registry of teacher models served by the local llama.cpp router.

Registering a teacher never loads it. A model is only loaded by the
router when TeacherClient.chat() is actually called.

    from evo.learning.teacher_registry import TeacherRegistry
    reg = TeacherRegistry.load()
    reg.choose(domain="language", lang="sk")      -> "Qwen3.8-27B"
    reg.status()                                  -> read-only router listing
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REGISTRY_FILE = Path(__file__).with_name("teachers.json")


@dataclass
class Teacher:
    name: str
    enabled: bool
    role: str
    domains: list[str]
    languages: list[str]
    priority: int = 5
    extra: dict[str, Any] = field(default_factory=dict)


class TeacherRegistry:
    def __init__(self, router: str, teachers: dict[str, Teacher]) -> None:
        self.router = router.rstrip("/")
        self.teachers = teachers

    @classmethod
    def load(cls, path: str | Path = REGISTRY_FILE) -> "TeacherRegistry":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        teachers = {}
        for name, t in data["teachers"].items():
            known = {"enabled", "role", "domains", "languages", "priority"}
            teachers[name] = Teacher(
                name=name,
                enabled=bool(t.get("enabled", False)),
                role=t.get("role", ""),
                domains=list(t.get("domains", [])),
                languages=list(t.get("languages", [])),
                priority=int(t.get("priority", 5)),
                extra={k: v for k, v in t.items() if k not in known},
            )
        return cls(data["router"], teachers)

    def mentors(self, path: str | Path = REGISTRY_FILE) -> list[str]:
        """Ordered core-evolution mentors (enabled only)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        order = data.get("mentor_order") or [t.name for t in self.candidates("mentor")]
        return [n for n in order if n in self.teachers and self.teachers[n].enabled]

    def timeout_factor(self, name: str) -> float:
        speed = self.teachers[name].extra.get("speed", "medium")
        return {"fast": 1.0, "medium": 2.0, "slow": 4.0}.get(speed, 2.0)

    def enabled(self) -> list[Teacher]:
        return sorted(
            (t for t in self.teachers.values() if t.enabled),
            key=lambda t: (t.priority, t.name),
        )

    def candidates(self, domain: str, lang: str | None = None) -> list[Teacher]:
        return [
            t for t in self.enabled()
            if domain in t.domains and (lang is None or lang in t.languages)
        ]

    def choose(self, domain: str, lang: str | None = None) -> str:
        found = self.candidates(domain, lang)
        if not found:
            raise LookupError(f"No enabled teacher for domain={domain} lang={lang}")
        return found[0].name

    def status(self, timeout: float = 10.0) -> dict[str, str]:
        """Read-only: which registered teachers the router knows, and their state."""
        with urllib.request.urlopen(f"{self.router}/v1/models", timeout=timeout) as r:
            data = json.load(r)
        served = {
            m["id"]: (m.get("status") or {}).get("value", "unknown")
            for m in data.get("data", [])
        }
        return {name: served.get(name, "MISSING") for name in self.teachers}


class TeacherClient:
    """Chat with a registered teacher. Calling chat() makes the router load it."""

    def __init__(self, registry: TeacherRegistry, name: str, timeout: float = 600.0):
        if name not in registry.teachers or not registry.teachers[name].enabled:
            raise ValueError(f"Teacher {name} is not registered/enabled")
        self.url = f"{registry.router}/v1/chat/completions"
        self.name = name
        self.timeout = timeout

    def chat(self, messages, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        return self.chat_full(messages, temperature, max_tokens)[0]

    def chat_full(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> tuple[str, str]:
        """Returns (answer, reasoning). Reasoning is '' if the model gave none."""
        body = json.dumps({
            "model": self.name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.load(r)
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        if not reasoning and "<think>" in content:
            import re
            m = re.search(r"<think>(.*?)</think>", content, re.S)
            reasoning = m.group(1).strip() if m else ""
        return content, reasoning.strip()
