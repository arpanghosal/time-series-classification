"""Torch dataset over a directory of serialized signals."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..serialization import load_signal
from .preprocess import preprocess

SIGNAL_SUFFIX = ".tsig"


class SignalDataset(Dataset):
    """Loads every ``*.tsig`` file in ``root``; yields ``(tensor[channels, length], int label)``."""

    def __init__(self, root: str | Path, length: int | None = None, normalize: str = "zscore"):
        self.paths = sorted(Path(root).glob(f"*{SIGNAL_SUFFIX}"))
        if not self.paths:
            raise FileNotFoundError(f"no {SIGNAL_SUFFIX} files found in {root}")
        self.length = length
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sig = load_signal(self.paths[idx])
        x = preprocess(sig.data, length=self.length, normalize=self.normalize)
        return torch.from_numpy(x), int(sig.label)
