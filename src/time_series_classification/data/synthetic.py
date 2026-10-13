"""Synthetic signal generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..serialization import save_signal
from .dataset import SIGNAL_SUFFIX

NUM_CLASSES = 3


def generate_synthetic_dataset(
    out_dir: str | Path, n_samples: int = 300, length: int = 256, seed: int = 0
) -> list[Path]:
    """Write ``n_samples`` noisy signals (sine / square / sawtooth) and return their paths."""
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 1, length, endpoint=False)
    paths = []
    for i in range(n_samples):
        label = i % NUM_CLASSES
        phase = rng.uniform(0, 1)
        freq = rng.uniform(3, 6)
        arg = freq * t + phase
        if label == 0:
            wave = np.sin(2 * np.pi * arg)
        elif label == 1:
            wave = np.sign(np.sin(2 * np.pi * arg))
        else:
            wave = 2 * (arg % 1.0) - 1
        data = (wave + 0.2 * rng.standard_normal(length)).astype(np.float32)[None, :]
        path = out / f"sample_{i:05d}{SIGNAL_SUFFIX}"
        save_signal(path, data, label)
        paths.append(path)
    return paths
