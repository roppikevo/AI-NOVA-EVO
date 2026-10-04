from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MentorRequest:
    """Structured technical request sent to OxCoder."""

    problem: str
    context: str = ""
    candidate_id: str | None = None
    learning_plan: bool = False
    executable_task: bool = False
    generation: int | None = None
    current_architecture: dict[str, Any] = field(default_factory=dict)
    previous_failures: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "context": self.context,
            "candidate_id": self.candidate_id,
            "generation": self.generation,
            "current_architecture": self.current_architecture,
            "previous_failures": self.previous_failures,
            "constraints": self.constraints,
        }


@dataclass
class MentorProposal:
    """Proposal returned by OxCoder."""

    success: bool
    proposal: str = ""
    hypothesis: str = ""
    rationale: str = ""
    suggested_changes: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    experiments: list[str] = field(default_factory=list)
    raw_response: str = ""
    error: str | None = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "proposal": self.proposal,
            "hypothesis": self.hypothesis,
            "rationale": self.rationale,
            "suggested_changes": self.suggested_changes,
            "risks": self.risks,
            "experiments": self.experiments,
            "raw_response": self.raw_response,
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
        }


class OxCoderMentor:
    """
    Local technical mentor interface.

    OxCoder is advisory only.
    It cannot directly modify the NOVA-EVO project,
    execute experiments, promote candidates, or alter policy.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8081",
        model: str = "OxCoder-9B",
        timeout_seconds: int = 90,
        temperature: float = 0.1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    @staticmethod
    def _system_prompt() -> str:
        return """
You are OxCoder, the technical mentor for NOVA-EVO.

Your role is advisory only.

You must:
- analyze the supplied technical problem,
- propose technically plausible solutions,
- explain your reasoning at a concise engineering level,
- identify risks and failure modes,
- suggest experiments that could verify your proposal.

You must NOT:
- claim that your proposal is proven,
- decide whether a candidate should be promoted,
- modify files,
- execute commands,
- invent experimental results,
- treat your own suggestions as facts.

NOVA-EVO independently validates every proposal.

Return a structured response using exactly these sections:

PROPOSAL:
HYPOTHESIS:
RATIONALE:
CHANGES:
RISKS:
EXPERIMENTS:
""".strip()

    @staticmethod
    def _executable_task_system_prompt() -> str:
        return """
You are the execution-task generator for NOVA-EVO.

Create one concrete local programming exercise that an autonomous
agent can execute and test.

Return exactly these lines:

LANGUAGE: python
FILENAME: main.py
COMMAND: python main.py
EXPECTED_OUTPUT: <expected text or NONE>
CONTENT_BEGIN:
<complete source code>
CONTENT_END:

Rules:
- Supported languages are only Python or Rust.
- Use exactly one source file named main.py for Python or main.rs for Rust.
- COMMAND must be EXACTLY one of these forms, with no extra arguments:
  python main.py
  python3 main.py
  rustc main.rs -o main
- For Rust, NEVER append ./main, &&, ;, |, redirects, or any other shell syntax.
- For Python, NEVER append any command or shell syntax.
- Do not use shell operators, pipes, redirects, sudo, network access,
  package installation, file deletion, or commands outside the workspace.
- The source code must be complete, deterministic, and executable.
- The program must directly test the requested learning objective.
- The program must contain at least one assertion directly related to the objective.
- The program must print EXPECTED_OUTPUT on successful execution.
- EXPECTED_OUTPUT must be a short deterministic success marker.
- Do not claim a compile-time property was tested by a runtime assertion.
- Output nothing before LANGUAGE and nothing after CONTENT_END.
""".strip()

    @staticmethod
    def _learning_system_prompt() -> str:
        return """
You create autonomous learning tasks for NOVA-EVO.

Output exactly 5 lines.
Every line must start with TASK:.

The tasks are executed by NOVA-EVO itself, not by a human.

Each task must:
- acquire or transform knowledge,
- be executable locally by the agent,
- produce a measurable result,
- be testable,
- progressively improve the target capability.

Do not suggest human actions such as:
- writing in a notebook,
- recording audio,
- sending email,
- attending lessons.

Prefer:
- datasets,
- code exercises,
- generated examples,
- automated tests,
- evaluation,
- teacher-model comparison,
- error analysis,
- knowledge extraction,
- repeated practice.

Output nothing except the 5 TASK: lines.
Do not output <think>, reasoning, explanations, or meta-commentary.
""".strip()

    @staticmethod
    def _build_user_prompt(request: MentorRequest) -> str:
        return json.dumps(
            request.to_dict(),
            ensure_ascii=False,
            indent=2,
        )

    def ask(self, request: MentorRequest) -> MentorProposal:
        started = time.monotonic()

        if request.executable_task:
            system_prompt = self._executable_task_system_prompt()
        elif request.learning_plan:
            system_prompt = self._learning_system_prompt()
        else:
            system_prompt = self._system_prompt()

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": self._build_user_prompt(request),
                },
            ],
            "temperature": self.temperature,
            "stream": False,
            "max_tokens": 1536 if request.learning_plan else 768,
            "reasoning_format": "none",
            "reasoning_effort": (
                "none"
                if (
                    request.learning_plan
                    or request.executable_task
                )
                else "default"
            ),
        }

        body = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        http_request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                http_request,
                timeout=self.timeout_seconds,
            ) as response:
                raw = response.read().decode("utf-8")

            elapsed = time.monotonic() - started

            data = json.loads(raw)

            message = data.get(
                "choices",
                [{}],
            )[0].get(
                "message",
                {},
            )

            content = message.get("content", "")
            reasoning = message.get("reasoning_content", "")

            if not content:
                content = reasoning

            if not content:
                return MentorProposal(
                    success=False,
                    raw_response=raw,
                    error="Mentor returned empty response",
                    elapsed_seconds=elapsed,
                )

            if (
                request.learning_plan
                or request.executable_task
            ):
                return MentorProposal(
                    success=True,
                    proposal=content,
                    raw_response=raw,
                    elapsed_seconds=elapsed,
                )

            return self.parse_proposal(
                content,
                raw_response=raw,
                elapsed_seconds=elapsed,
            )

        except urllib.error.URLError as exc:
            return MentorProposal(
                success=False,
                error=f"Mentor connection failed: {exc}",
                elapsed_seconds=time.monotonic() - started,
            )

        except TimeoutError:
            return MentorProposal(
                success=False,
                error="Mentor request timed out",
                elapsed_seconds=time.monotonic() - started,
            )

        except json.JSONDecodeError as exc:
            return MentorProposal(
                success=False,
                error=f"Invalid mentor JSON response: {exc}",
                elapsed_seconds=time.monotonic() - started,
            )

        except Exception as exc:
            return MentorProposal(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=time.monotonic() - started,
            )

    @staticmethod
    def _section(
        text: str,
        name: str,
        next_names: list[str],
    ) -> str:
        marker = f"{name}:"

        if marker not in text:
            return ""

        content = text.split(marker, 1)[1]

        positions = []

        for next_name in next_names:
            position = content.find(f"{next_name}:")

            if position >= 0:
                positions.append(position)

        if positions:
            content = content[:min(positions)]

        return content.strip()

    @classmethod
    def parse_proposal(
        cls,
        content: str,
        raw_response: str = "",
        elapsed_seconds: float = 0.0,
    ) -> MentorProposal:

        sections = [
            "PROPOSAL",
            "HYPOTHESIS",
            "RATIONALE",
            "CHANGES",
            "RISKS",
            "EXPERIMENTS",
        ]

        parsed: dict[str, str] = {}

        for index, name in enumerate(sections):
            parsed[name] = cls._section(
                content,
                name,
                sections[index + 1:],
            )

        return MentorProposal(
            success=True,
            proposal=parsed["PROPOSAL"],
            hypothesis=parsed["HYPOTHESIS"],
            rationale=parsed["RATIONALE"],
            suggested_changes=cls._split_lines(
                parsed["CHANGES"]
            ),
            risks=cls._split_lines(
                parsed["RISKS"]
            ),
            experiments=cls._split_lines(
                parsed["EXPERIMENTS"]
            ),
            raw_response=raw_response or content,
            elapsed_seconds=elapsed_seconds,
        )

    @staticmethod
    def _split_lines(text: str) -> list[str]:
        result = []

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            if line.startswith("-"):
                line = line[1:].strip()

            if line:
                result.append(line)

        return result

    @staticmethod
    def proposal_to_hypothesis(
        proposal: MentorProposal,
    ) -> dict[str, Any]:
        return {
            "source": "OxCoder-mentor",
            "status": "UNTESTED",
            "hypothesis": proposal.hypothesis,
            "proposal": proposal.proposal,
            "rationale": proposal.rationale,
            "suggested_changes": proposal.suggested_changes,
            "risks": proposal.risks,
            "experiments": proposal.experiments,
            "must_be_experimentally_verified": True,
            "created_at": time.time(),
        }


def self_test() -> None:
    content = """
PROPOSAL:
Increase state capacity for longer dependencies.

HYPOTHESIS:
A larger state dimension may improve sequence retention.

RATIONALE:
The current state may be capacity constrained.

CHANGES:
- Increase d_state.
- Keep the causal state-space core.

RISKS:
- Higher parameter count.
- Higher VRAM usage.

EXPERIMENTS:
- Compare against the current parent.
- Run the same seeds.
- Measure validation loss and parameters.
""".strip()

    proposal = OxCoderMentor.parse_proposal(content)

    assert proposal.success
    assert proposal.proposal
    assert proposal.hypothesis
    assert proposal.rationale
    assert len(proposal.suggested_changes) == 2
    assert len(proposal.risks) == 2
    assert len(proposal.experiments) == 3

    hypothesis = OxCoderMentor.proposal_to_hypothesis(proposal)

    assert hypothesis["source"] == "OxCoder-mentor"
    assert hypothesis["status"] == "UNTESTED"
    assert hypothesis["must_be_experimentally_verified"]

    request = MentorRequest(
        problem="Test technical problem",
        context="Self-test",
        candidate_id="SELFTEST-001",
        generation=4,
        current_architecture={
            "d_model": 384,
            "layers": 6,
        },
        previous_failures=["Example failure"],
        constraints=["Do not modify protected generations"],
    )

    data = request.to_dict()

    assert data["candidate_id"] == "SELFTEST-001"

    mentor = OxCoderMentor()

    assert mentor.endpoint.endswith(
        "/v1/chat/completions"
    )

    print("OXCODER MENTOR SELFTEST: PASSED")


if __name__ == "__main__":
    self_test()
