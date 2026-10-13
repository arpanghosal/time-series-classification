"""Baseline 1D CNN classifier."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class CNN1D(nn.Module):
    """Stack of Conv1d-BatchNorm-ReLU-MaxPool blocks, global average pooling, linear head.

    Global pooling makes the network independent of input length, as long as the
    length is at least ``2 ** len(channels)`` so pooling never empties the sequence.

    Input: ``(batch, in_channels, length)``. Output: logits ``(batch, num_classes)``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        channels: Sequence[int] = (16, 32, 64),
        kernel_size: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = tuple(channels)
        self.kernel_size = kernel_size
        self.dropout = dropout
        layers: list[nn.Module] = []
        prev = in_channels
        for c in self.channels:
            layers += [
                nn.Conv1d(prev, c, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(c),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            prev = c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(prev, num_classes))

    def config(self) -> dict:
        """Constructor arguments, stored in checkpoints to rebuild the model."""
        return {
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "channels": list(self.channels),
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected input (batch, {self.in_channels}, length), got {tuple(x.shape)}"
            )
        if x.shape[-1] < 2 ** len(self.channels):
            raise ValueError(f"input length {x.shape[-1]} too short for {len(self.channels)} pooling stages")
        return self.head(self.pool(self.features(x)).squeeze(-1))
