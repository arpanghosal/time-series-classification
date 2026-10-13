# time-series-classification

Time-series signal processing with a compact binary serialization format and a
1D CNN classifier (PyTorch).

## Layout

- `serialization/` – binary signal format (`save_signal`, `load_signal`, `save_dataset`, `load_dataset`)
- `data/` – `SignalDataset`, preprocessing (z-score / min-max, pad/truncate, axis handling), synthetic generator
- `models/` – `CNN1D` baseline
- `training/` – training loop and evaluation
- `cli.py` – `tsc` command

## Data format

Little-endian. Header (24 bytes): magic `TSIG`, version (uint16), dtype code
(0=f32, 1=f64, 2=i32, 3=i64), reserved byte, label (int32), sample rate
(float32), channels (uint32), length (uint32); followed by the raw C-contiguous
`(channels, length)` signal bytes. Batch files (`TSDS`) hold a count and
length-prefixed records. Malformed input raises `SerializationError`.
Dataset directories use the `.tsig` extension.

## Quickstart

```bash
pip install -e ".[dev]"
tsc generate --out data/synthetic --n-samples 300
tsc train --data data/synthetic --checkpoint checkpoints/model.pt --epochs 5
tsc evaluate --data data/synthetic --checkpoint checkpoints/model.pt
```

(`python -m time_series_classification.cli ...` works too.) For a real
held-out evaluation, generate a second set with a different `--seed` and
evaluate on that directory.

## Development

```bash
ruff check .
pytest
```
