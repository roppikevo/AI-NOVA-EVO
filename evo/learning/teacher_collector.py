from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evo.engine.knowledge_store import KnowledgeStore
from evo.engine.oxcoder_mentor import (
    MentorProposal,
    MentorRequest,
    OxCoderMentor,
)


@dataclass
class TeacherLesson:
    success: bool
    goal: str
    topic: str
    domain: str
    teacher_response: str = ""
    knowledge_id: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "goal": self.goal,
            "topic": self.topic,
            "domain": self.domain,
            "teacher_response": self.teacher_response,
            "knowledge_id": self.knowledge_id,
            "error": self.error,
        }


class TeacherCollector:
    """
    Collect teaching material directly from OxCoder.

    OxCoder is a teacher, not an authority.
    Responses are stored as EXPERIMENTAL knowledge until
    independently verified or promoted by a later quality gate.
    """

    def __init__(
        self,
        mentor: OxCoderMentor | None = None,
        knowledge: KnowledgeStore | None = None,
    ) -> None:
        self.mentor = mentor or OxCoderMentor()
        self.knowledge = knowledge or KnowledgeStore()

    @staticmethod
    def _compose_response(
        proposal: MentorProposal,
    ) -> str:
        parts: list[str] = []

        if proposal.proposal.strip():
            parts.append(
                "PROPOSAL:\n"
                + proposal.proposal.strip()
            )

        if proposal.hypothesis.strip():
            parts.append(
                "HYPOTHESIS:\n"
                + proposal.hypothesis.strip()
            )

        if proposal.rationale.strip():
            parts.append(
                "RATIONALE:\n"
                + proposal.rationale.strip()
            )

        if proposal.suggested_changes:
            parts.append(
                "CHANGES:\n"
                + "\n".join(
                    f"- {item}"
                    for item in proposal.suggested_changes
                )
            )

        if proposal.risks:
            parts.append(
                "RISKS:\n"
                + "\n".join(
                    f"- {item}"
                    for item in proposal.risks
                )
            )

        if proposal.experiments:
            parts.append(
                "EXPERIMENTS:\n"
                + "\n".join(
                    f"- {item}"
                    for item in proposal.experiments
                )
            )

        return "\n\n".join(parts).strip()

    def teach(
        self,
        *,
        goal: str,
        topic: str,
        domain: str = "general",
        context: str = "",
    ) -> TeacherLesson:
        goal = goal.strip()
        topic = topic.strip()
        domain = domain.strip() or "general"

        if not goal:
            return TeacherLesson(
                success=False,
                goal=goal,
                topic=topic,
                domain=domain,
                error="Learning goal cannot be empty.",
            )

        if not topic:
            return TeacherLesson(
                success=False,
                goal=goal,
                topic=topic,
                domain=domain,
                error="Teacher topic cannot be empty.",
            )

        request = MentorRequest(
            problem=(
                "Teach NOVA-EVO the requested topic.\n\n"
                f"LEARNING GOAL:\n{goal}\n\n"
                f"TOPIC:\n{topic}\n\n"
                f"DOMAIN:\n{domain}"
            ),
            context=(
                "The answer will become training material for NOVA-EVO.\n"
                "Explain the concept precisely and technically.\n"
                "Include concrete examples where useful.\n"
                "Distinguish facts from assumptions.\n"
                "Do not invent experimental results.\n"
                "Do not claim that an untested statement is proven.\n"
                "The material must be useful as persistent teaching data.\n"
                + (
                    f"\nADDITIONAL CONTEXT:\n{context.strip()}"
                    if context.strip()
                    else ""
                )
            ),
            constraints=[
                "Teacher is advisory only.",
                "NOVA-EVO independently evaluates the material.",
                "Do not execute commands.",
                "Do not modify files.",
                "Do not invent benchmark results.",
            ],
        )

        proposal = self.mentor.ask(request)

        if not proposal.success:
            return TeacherLesson(
                success=False,
                goal=goal,
                topic=topic,
                domain=domain,
                error=proposal.error or "Teacher request failed.",
            )

        teacher_response = self._compose_response(proposal)

        if not teacher_response:
            return TeacherLesson(
                success=False,
                goal=goal,
                topic=topic,
                domain=domain,
                error="Teacher returned an empty learning response.",
            )

        record = self.knowledge.create(
            knowledge_type="mentor",
            title=f"Teacher lesson: {domain} / {topic}",
            content={
                "goal": goal,
                "topic": topic,
                "domain": domain,
                "prompt": request.problem,
                "teacher_response": teacher_response,
                "model": self.mentor.model,
                "verified": False,
                "verification_status": "UNVERIFIED",
                "teacher_elapsed_seconds": proposal.elapsed_seconds,
            },
            status="EXPERIMENTAL",
            source="oxcoder_teacher",
            confidence=0.8,
        )

        path = self.knowledge.save(record)

        return TeacherLesson(
            success=True,
            goal=goal,
            topic=topic,
            domain=domain,
            teacher_response=teacher_response,
            knowledge_id=record["id"],
        )


def self_test() -> None:
    import tempfile

    class FakeMentor:
        model = "FAKE-OXCODER"

        def ask(
            self,
            request: MentorRequest,
        ) -> MentorProposal:
            return MentorProposal(
                success=True,
                proposal=(
                    "Ownership určuje vlastníctvo hodnoty "
                    "a pravidlá jej presunu."
                ),
                hypothesis="Ownership podporuje pamäťovú bezpečnosť.",
                rationale=(
                    "Rust kontroluje pravidlá ownership "
                    "počas kompilácie."
                ),
                suggested_changes=[
                    "Rozlišuj move, borrow a mutable borrow."
                ],
                risks=[
                    "Nepomiešaj immutable a mutable borrow."
                ],
                experiments=[
                    "Over pravidlá ownership pomocou rustc."
                ],
                elapsed_seconds=0.01,
            )

    with tempfile.TemporaryDirectory() as tmp:
        knowledge = KnowledgeStore(tmp)

        collector = TeacherCollector(
            mentor=FakeMentor(),
            knowledge=knowledge,
        )

        result = collector.teach(
            goal="Nauč sa Rust",
            topic="Ownership",
            domain="rust",
        )

        assert result.success
        assert result.knowledge_id
        assert "Ownership" in result.teacher_response

        records = knowledge.search(
            knowledge_type="mentor"
        )

        assert len(records) == 1
        assert records[0]["source"] == "oxcoder_teacher"
        assert records[0]["status"] == "EXPERIMENTAL"
        assert records[0]["content"]["verified"] is False

    print("TEACHER COLLECTOR SELFTEST: PASSED")


if __name__ == "__main__":
    self_test()
