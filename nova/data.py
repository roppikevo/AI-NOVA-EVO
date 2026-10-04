from pathlib import Path

import torch
from torch.utils.data import Dataset


class TokenSequenceDataset(Dataset):
    """
    Simple fixed-length causal language-model dataset.

    Each line contains exactly seq_len integer tokens.
    """

    def __init__(
        self,
        path: str | Path,
        seq_len: int = 128,
    ):
        self.path = Path(path)
        self.seq_len = seq_len

        self.samples: list[list[int]] = []

        with self.path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                tokens = [int(x) for x in line.split()]

                if len(tokens) != seq_len:
                    raise ValueError(
                        f"{self.path}: line {line_number} "
                        f"contains {len(tokens)} tokens, "
                        f"expected {seq_len}"
                    )

                self.samples.append(tokens)

        if not self.samples:
            raise ValueError(f"Dataset is empty: {self.path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        tokens = torch.tensor(
            self.samples[index],
            dtype=torch.long,
        )

        input_ids = tokens[:-1]
        targets = tokens[1:]

        return input_ids, targets


def build_datasets(
    root: str | Path = "/opt/ai/work/nova-evo/data/gen0",
    seq_len: int = 128,
):
    root = Path(root)

    train = TokenSequenceDataset(
        root / "train.txt",
        seq_len=seq_len,
    )

    val = TokenSequenceDataset(
        root / "val.txt",
        seq_len=seq_len,
    )

    test = TokenSequenceDataset(
        root / "test.txt",
        seq_len=seq_len,
    )

    return train, val, test
