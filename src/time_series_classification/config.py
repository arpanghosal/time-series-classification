"""Default configuration values shared across the package."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Lightweight defaults suitable for local CPU runs."""

    data_dir: str = "data/synthetic"
    checkpoint: str = "checkpoints/model.pt"
    seq_len: int = 256
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    val_fraction: float = 0.2
    normalize: str = "zscore"  # "zscore", "minmax" or "none"
    seed: int = 0
