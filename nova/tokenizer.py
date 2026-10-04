"""
NOVA tokenizer — byte-level BPE shared by natural and programming languages.

Byte-level means every UTF-8 string (Slovak diacritics, Polish, code,
emoji) is representable; there is no unknown token in practice.

Special tokens have fixed ids so datasets and models stay compatible:

    0 <pad>   1 <unk>   2 <bos>   3 <eos>
    4 <sk>    5 <cs>    6 <pl>    7 <en>    8 <py>    9 <rs>

A document is encoded as:  <lang> text-tokens <eos>
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<sk>",
    "<cs>",
    "<pl>",
    "<en>",
    "<py>",
    "<rs>",
    "<think>",
    "</think>",
]

# ids 0-9 are required; <think> (10) and </think> (11) exist from text_v3 on
CORE_SPECIAL = 10
THINK_ID = 10
END_THINK_ID = 11

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

LANG_TAGS = {
    "sk": "<sk>",
    "cs": "<cs>",
    "pl": "<pl>",
    "en": "<en>",
    "py": "<py>",
    "rs": "<rs>",
}

DEFAULT_VOCAB_SIZE = 16384


class NovaTokenizer:
    def __init__(self, backend) -> None:
        self._tok = backend
        for i, name in enumerate(SPECIAL_TOKENS):
            got = self._tok.token_to_id(name)
            if got is None and i >= CORE_SPECIAL:
                continue  # older tokenizer without think tokens
            if got != i:
                raise ValueError(
                    f"Special token {name} has id {got}, expected {i}"
                )

    # -------------------------------------------------------------- build

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        min_frequency: int = 2,
    ) -> "NovaTokenizer":
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tok.train_from_iterator(texts, trainer=trainer)
        return cls(tok)

    @classmethod
    def load(cls, path: str | Path) -> "NovaTokenizer":
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    # ------------------------------------------------------------ encode

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self._tok.encode_batch(texts)]

    def encode_document(self, text: str, lang: str) -> list[int]:
        return [self.lang_id(lang)] + self.encode(text) + [EOS_ID]

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special)

    @property
    def has_think(self) -> bool:
        return self._tok.token_to_id("<think>") == THINK_ID

    def encode_reasoned(self, prompt: str, reasoning: str, answer: str, lang: str) -> list[int]:
        """<lang> prompt <think> reasoning </think> answer <eos>"""
        if not self.has_think:
            raise ValueError("tokenizer has no <think> tokens")
        return ([self.lang_id(lang)] + self.encode(prompt + "\n")
                + [THINK_ID] + self.encode(reasoning) + [END_THINK_ID]
                + self.encode(answer) + [EOS_ID])

    def lang_id(self, lang: str) -> int:
        return self._tok.token_to_id(LANG_TAGS[lang])


def pack_sequences(
    documents: Iterable[list[int]],
    seq_len: int,
) -> Iterator[list[int]]:
    """
    Concatenate encoded documents and cut them into fixed-length
    sequences (no padding). The trailing partial sequence is dropped.
    """
    buffer: list[int] = []
    for doc in documents:
        buffer.extend(doc)
        while len(buffer) >= seq_len:
            yield buffer[:seq_len]
            buffer = buffer[seq_len:]
