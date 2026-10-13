"""Command line interface: ``tsc generate | train | evaluate``."""

from __future__ import annotations

import argparse

from .config import TrainConfig
from .data.synthetic import generate_synthetic_dataset
from .training import evaluate_checkpoint, train


def build_parser() -> argparse.ArgumentParser:
    d = TrainConfig()
    p = argparse.ArgumentParser(prog="tsc", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate synthetic serialized signals")
    g.add_argument("--out", default=d.data_dir)
    g.add_argument("--n-samples", type=int, default=300)
    g.add_argument("--length", type=int, default=d.seq_len)
    g.add_argument("--seed", type=int, default=d.seed)

    t = sub.add_parser("train", help="train the CNN")
    t.add_argument("--data", default=d.data_dir)
    t.add_argument("--checkpoint", default=d.checkpoint)
    t.add_argument("--seq-len", type=int, default=d.seq_len)
    t.add_argument("--epochs", type=int, default=d.epochs)
    t.add_argument("--batch-size", type=int, default=d.batch_size)
    t.add_argument("--lr", type=float, default=d.lr)
    t.add_argument("--normalize", choices=["zscore", "minmax", "none"], default=d.normalize)
    t.add_argument("--seed", type=int, default=d.seed)

    e = sub.add_parser("evaluate", help="evaluate a checkpoint on a held-out directory")
    e.add_argument("--data", required=True)
    e.add_argument("--checkpoint", default=d.checkpoint)
    e.add_argument("--batch-size", type=int, default=64)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        paths = generate_synthetic_dataset(args.out, args.n_samples, args.length, seed=args.seed)
        print(f"wrote {len(paths)} signals to {args.out}")
    elif args.command == "train":
        cfg = TrainConfig(
            data_dir=args.data, checkpoint=args.checkpoint, seq_len=args.seq_len,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            normalize=args.normalize, seed=args.seed,
        )
        train(cfg)
        print(f"checkpoint saved to {cfg.checkpoint}")
    else:
        loss, acc = evaluate_checkpoint(args.checkpoint, args.data, args.batch_size)
        print(f"loss={loss:.4f} accuracy={acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
