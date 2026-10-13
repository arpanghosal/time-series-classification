"""Evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data import SignalDataset
from ..models import CNN1D


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str = "cpu") -> tuple[float, float]:
    """Return (mean loss, accuracy) over ``loader``."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += loss_fn(out, y).item()
        correct += (out.argmax(1) == y).sum().item()
        n += y.numel()
    return total_loss / max(n, 1), correct / max(n, 1)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[CNN1D, dict]:
    """Rebuild a model from a checkpoint written by ``train``."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = CNN1D(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def evaluate_checkpoint(
    checkpoint: str | Path, data_dir: str | Path, batch_size: int = 64, device: str = "cpu"
) -> tuple[float, float]:
    """Evaluate a saved model on a held-out directory of serialized signals."""
    model, ckpt = load_checkpoint(checkpoint, device)
    ds = SignalDataset(data_dir, length=ckpt["seq_len"], normalize=ckpt["normalize"])
    return evaluate(model, DataLoader(ds, batch_size=batch_size), device)
