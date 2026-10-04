from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizerConfig:
    vocab_size: int = 16384
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    byte_offset: int = 3


class TaskTokenizer:
    """
    Deterministic local byte-level tokenizer for NOVA-EVO task learning.

    Every UTF-8 byte is represented directly by one token.
    This requires no external tokenizer package and works for:
      - Slovak text and diacritics
      - Python source
      - Rust source
      - generated learning explanations
    """

    def __init__(
        self,
        config: TokenizerConfig | None = None,
    ) -> None:
        self.config = config or TokenizerConfig()

        if self.config.vocab_size < (
            self.config.byte_offset + 256
        ):
            raise ValueError(
                "Vocabulary is too small for byte tokenizer."
            )

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be str")

        tokens: list[int] = []

        if add_bos:
            tokens.append(self.config.bos_id)

        tokens.extend(
            self.config.byte_offset + byte
            for byte in text.encode("utf-8")
        )

        if add_eos:
            tokens.append(self.config.eos_id)

        return tokens

    def decode(
        self,
        tokens: list[int],
    ) -> str:
        raw = bytearray()

        for token in tokens:
            if token in {
                self.config.pad_id,
                self.config.bos_id,
                self.config.eos_id,
            }:
                continue

            byte = token - self.config.byte_offset

            if 0 <= byte <= 255:
                raw.append(byte)
            else:
                raise ValueError(
                    f"Token {token} is outside byte-token range."
                )

        return bytes(raw).decode(
            "utf-8",
            errors="replace",
        )

    def encode_fixed(
        self,
        text: str,
        *,
        seq_len: int = 128,
    ) -> list[int]:
        if seq_len < 2:
            raise ValueError(
                "seq_len must be at least 2"
            )

        tokens = self.encode(
            text,
            add_bos=True,
            add_eos=True,
        )

        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]

            if tokens[-1] != self.config.eos_id:
                tokens[-1] = self.config.eos_id

        elif len(tokens) < seq_len:
            tokens.extend(
                [self.config.pad_id]
                * (seq_len - len(tokens))
            )

        return tokens


def self_test() -> None:
    tokenizer = TaskTokenizer()

    text = (
        "NOVA-EVO Rust: vlastníctvo, borrowing, "
        "životnosť — test 42."
    )

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text
    assert all(
        0 <= token < tokenizer.vocab_size
        for token in encoded
    )

    fixed = tokenizer.encode_fixed(
        text,
        seq_len=128,
    )

    assert len(fixed) == 128
    assert fixed[0] == tokenizer.config.bos_id

    print("TASK TOKENIZER SELFTEST: PASSED")


if __name__ == "__main__":
    self_test()
