from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from evo.engine.knowledge_store import KnowledgeStore
from evo.learning.task_tokenizer import TaskTokenizer


class TaskLearningDataset:
    """
    Build an isolated text-learning dataset from validated learning
    experiences.

    Only successful LearningExecutor records containing source code
    are accepted. Splits are made per experience before chunking, so
    one task cannot leak into train and validation/test simultaneously.
    """

    def __init__(
        self,
        knowledge: KnowledgeStore | None = None,
        tokenizer: TaskTokenizer | None = None,
        seq_len: int = 128,
    ) -> None:
        self.knowledge = knowledge or KnowledgeStore()
        self.tokenizer = tokenizer or TaskTokenizer()
        self.seq_len = seq_len

        if seq_len < 16:
            raise ValueError("seq_len must be at least 16")

    @staticmethod
    def _identity(record: dict[str, Any]) -> str:
        content = record.get("content") or {}

        raw = json.dumps(
            {
                "goal": content.get("goal", ""),
                "task": content.get("task", ""),
                "language": content.get("language", ""),
                "source_code": content.get("source_code", ""),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

        return hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _render(record: dict[str, Any]) -> str:
        content = record.get("content") or {}

        goal = str(content.get("goal", "")).strip()
        task = str(content.get("task", "")).strip()
        language = str(content.get("language", "")).strip()
        source = str(content.get("source_code", "")).strip()
        expected = str(
            content.get("expected_output", "")
        ).strip()
        actual = str(
            content.get("actual_output", "")
        ).strip()

        return (
            "<GOAL>\n"
            f"{goal}\n"
            "</GOAL>\n"
            "<TASK>\n"
            f"{task}\n"
            "</TASK>\n"
            "<LANGUAGE>\n"
            f"{language}\n"
            "</LANGUAGE>\n"
            "<SOURCE>\n"
            f"{source}\n"
            "</SOURCE>\n"
            "<EXPECTED>\n"
            f"{expected}\n"
            "</EXPECTED>\n"
            "<ACTUAL>\n"
            f"{actual}\n"
            "</ACTUAL>"
        )

    def collect_records(
        self,
        *,
        min_examples: int = 32,
    ) -> list[dict[str, Any]]:
        records = []

        for record in self.knowledge.search(
            knowledge_type="success"
        ):
            content = record.get("content") or {}

            if record.get("status") != "VALIDATED":
                continue

            if record.get("source") != "learning_executor":
                continue

            if not content.get("source_code"):
                continue

            language = str(
                content.get("language", "")
            ).lower()

            if language not in {
                "python",
                "rust",
            }:
                continue

            records.append(record)

        unique: dict[str, dict[str, Any]] = {}

        for record in records:
            unique[self._identity(record)] = record

        records = list(unique.values())

        records.sort(
            key=lambda item: self._identity(item)
        )

        if len(records) < min_examples:
            raise ValueError(
                "Not enough validated learning examples: "
                f"{len(records)} available, "
                f"{min_examples} required."
            )

        return records

    def _split(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        splits = {
            "train": [],
            "val": [],
            "test": [],
        }

        for record in records:
            digest = hashlib.sha256(
                self._identity(record).encode("ascii")
            ).digest()

            bucket = int.from_bytes(
                digest[:8],
                "big",
            ) % 100

            if bucket < 70:
                split = "train"
            elif bucket < 85:
                split = "val"
            else:
                split = "test"

            splits[split].append(record)

        return splits

    def _encode_record(
        self,
        record: dict[str, Any],
    ) -> list[list[int]]:
        text = self._render(record)

        raw = self.tokenizer.encode(
            text,
            add_bos=False,
            add_eos=False,
        )

        payload_len = self.seq_len - 2

        chunks: list[list[int]] = []

        for start in range(
            0,
            len(raw),
            payload_len,
        ):
            chunk = raw[
                start:start + payload_len
            ]

            if not chunk:
                continue

            tokens = [
                self.tokenizer.config.bos_id,
                *chunk,
                self.tokenizer.config.eos_id,
            ]

            if len(tokens) < self.seq_len:
                tokens.extend(
                    [self.tokenizer.config.pad_id]
                    * (self.seq_len - len(tokens))
                )

            chunks.append(tokens)

        return chunks

    def build(
        self,
        output_dir: str | Path,
        *,
        min_examples: int = 32,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        records = self.collect_records(
            min_examples=min_examples
        )

        splits = self._split(records)

        counts: dict[str, int] = {}

        for split_name, split_records in splits.items():
            path = output_dir / f"{split_name}.txt"

            sample_count = 0

            with path.open(
                "w",
                encoding="utf-8",
            ) as f:
                for record in split_records:
                    for tokens in self._encode_record(
                        record
                    ):
                        f.write(
                            " ".join(
                                map(str, tokens)
                            )
                            + "\n"
                        )
                        sample_count += 1

            counts[split_name] = sample_count

        manifest = {
            "seq_len": self.seq_len,
            "vocab_size": self.tokenizer.vocab_size,
            "source": "KnowledgeStore / learning_executor",
            "validated_examples": len(records),
            "samples": counts,
            "split_policy": {
                "train": 0.70,
                "val": 0.15,
                "test": 0.15,
            },
            "final_holdout_used": False,
        }

        (output_dir / "manifest.json").write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return manifest


def self_test() -> None:
    import tempfile

    from evo.engine.knowledge_store import KnowledgeStore

    with tempfile.TemporaryDirectory() as tmp:
        knowledge = KnowledgeStore(tmp)

        for index in range(4):
            record = knowledge.create(
                knowledge_type="success",
                title=f"task-{index}",
                content={
                    "goal": "Rust learning",
                    "task": f"Ownership task {index}",
                    "language": "rust",
                    "task_type": "rust_generated",
                    "source_code": (
                        "fn main() { "
                        f"let value = {index}; "
                        f"assert_eq!(value, {index}); "
                        'println!("OK"); }'
                    ),
                    "expected_output": "OK",
                    "actual_output": "OK\n",
                },
                status="VALIDATED",
                source="learning_executor",
                confidence=1.0,
            )

            knowledge.save(record)

        dataset = TaskLearningDataset(
            knowledge=knowledge,
            seq_len=64,
        )

        try:
            dataset.build(
                Path(tmp) / "dataset",
                min_examples=32,
            )
        except ValueError as exc:
            assert "Not enough validated learning examples" in str(exc)
        else:
            raise AssertionError(
                "Dataset builder accepted too few examples."
            )

        records = dataset.collect_records(
            min_examples=4
        )

        assert len(records) == 4

        rendered = dataset._render(records[0])
        assert "<GOAL>" in rendered
        assert "<SOURCE>" in rendered

        chunks = dataset._encode_record(records[0])
        assert chunks
        assert all(
            len(chunk) == 64
            for chunk in chunks
        )

        print(
            "TASK LEARNING DATASET SELFTEST: PASSED"
        )


if __name__ == "__main__":
    self_test()
