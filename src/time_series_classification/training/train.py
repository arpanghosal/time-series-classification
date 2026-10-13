"""Training loop."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from ..config import TrainConfig
from ..data import SignalDataset
from ..models import CNN1D
from .evaluate import evaluate


def train(cfg: TrainConfig, num_classes: int | None = None, device: str = "cpu") -> dict:
    """Train on ``cfg.data_dir`` with a train/val split and save the best checkpoint.

    Returns the final metrics dict.
    """
    torch.manual_seed(cfg.seed)
    ds = SignalDataset(cfg.data_dir, length=cfg.seq_len, normalize=cfg.normalize)
    first_x, _ = ds[0]
    if num_classes is None:
        num_classes = max(int(ds[i][1]) for i in range(len(ds))) + 1
    n_val = max(1, int(len(ds) * cfg.val_fraction))
    if n_val >= len(ds):
        raise ValueError("dataset too small for a train/val split")
    gen = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    model = CNN1D(in_channels=first_x.shape[0], num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc, metrics = -1.0, {}
    ckpt_path = Path(cfg.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        run_loss, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.numel()
            correct += (out.argmax(1) == y).sum().item()
            n += y.numel()
        val_loss, val_acc = evaluate(model, val_loader, device)
        metrics = {
            "epoch": epoch,
            "train_loss": run_loss / n,
            "train_acc": correct / n,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": model.config(),
                    "seq_len": cfg.seq_len,
                    "normalize": cfg.normalize,
                },
                ckpt_path,
            )
    metrics["best_val_acc"] = best_acc
    return metrics
