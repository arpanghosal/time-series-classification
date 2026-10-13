"""Signal preprocessing: normalization, length and axis handling."""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def zscore(x: np.ndarray) -> np.ndarray:
    """Normalize each channel to zero mean and unit variance along the last axis."""
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return ((x - mean) / (std + _EPS)).astype(np.float32)


def minmax(x: np.ndarray) -> np.ndarray:
    """Scale each channel to [0, 1] along the last axis."""
    x = np.asarray(x, dtype=np.float32)
    lo = x.min(axis=-1, keepdims=True)
    hi = x.max(axis=-1, keepdims=True)
    return ((x - lo) / (hi - lo + _EPS)).astype(np.float32)


def pad_or_truncate(x: np.ndarray, length: int) -> np.ndarray:
    """Zero-pad or truncate the last axis to ``length``."""
    x = np.asarray(x)
    n = x.shape[-1]
    if n >= length:
        return x[..., :length]
    pad = [(0, 0)] * (x.ndim - 1) + [(0, length - n)]
    return np.pad(x, pad)


def to_channels_first(x: np.ndarray, time_axis: int = -1) -> np.ndarray:
    """Return a 2D ``(channels, time)`` array from a 1D or 2D signal."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :]
    if x.ndim != 2:
        raise ValueError(f"expected 1D or 2D signal, got {x.ndim}D")
    return np.moveaxis(x, time_axis, -1)


def preprocess(
    x: np.ndarray, length: int | None = None, normalize: str = "zscore", time_axis: int = -1
) -> np.ndarray:
    """Axis handling, pad/truncate, then normalization. Returns float32 ``(channels, length)``."""
    if normalize not in ("zscore", "minmax", "none"):
        raise ValueError(f"unknown normalize mode: {normalize!r}")
    x = to_channels_first(x, time_axis)
    if length is not None:
        x = pad_or_truncate(x, length)
    x = x.astype(np.float32)
    if normalize == "zscore":
        return zscore(x)
    if normalize == "minmax":
        return minmax(x)
    return x
