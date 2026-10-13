"""Binary format for time-series signals.

Layout (little endian)::

    offset  size  field
    0       4     magic bytes b"TSIG"
    4       2     version (uint16)
    6       1     dtype code (uint8, see ``DTYPE_CODES``)
    7       1     reserved (0)
    8       4     label / class id (int32)
    12      4     sample rate in Hz (float32)
    16      4     channel count (uint32)
    20      4     sequence length (uint32)
    24      ...   raw signal bytes, C-contiguous, shape (channels, length)

A dataset file is the magic b"TSDS", a uint16 version, a uint32 sample count
and then ``count`` records, each prefixed with a uint64 byte length and
containing one signal record as described above.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Sequence

import numpy as np

SIGNAL_MAGIC = b"TSIG"
DATASET_MAGIC = b"TSDS"
VERSION = 1

_HEADER = struct.Struct("<4sHBBifII")
_DS_HEADER = struct.Struct("<4sHI")
_LEN = struct.Struct("<Q")

DTYPE_CODES: dict[int, np.dtype] = {
    0: np.dtype("<f4"),
    1: np.dtype("<f8"),
    2: np.dtype("<i4"),
    3: np.dtype("<i8"),
}
_CODE_BY_DTYPE = {dt: code for code, dt in DTYPE_CODES.items()}


class SerializationError(ValueError):
    """Raised when binary data is malformed or cannot be serialized."""


@dataclass
class Signal:
    """A labelled multichannel signal with data of shape (channels, length)."""

    data: np.ndarray
    label: int
    sample_rate: float = 1.0


def _encode(signal: Signal) -> bytes:
    data = np.asarray(signal.data)
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2:
        raise SerializationError(f"signal must be 1D or 2D (channels, length), got {data.ndim}D")
    dtype = data.dtype.newbyteorder("<")
    if dtype not in _CODE_BY_DTYPE:
        raise SerializationError(f"unsupported dtype: {data.dtype}")
    channels, length = data.shape
    header = _HEADER.pack(
        SIGNAL_MAGIC, VERSION, _CODE_BY_DTYPE[dtype], 0,
        int(signal.label), float(signal.sample_rate), channels, length,
    )
    return header + np.ascontiguousarray(data, dtype=dtype).tobytes()


def _decode(buf: bytes) -> Signal:
    if len(buf) < _HEADER.size:
        raise SerializationError(f"truncated header: {len(buf)} < {_HEADER.size} bytes")
    magic, version, code, _, label, rate, channels, length = _HEADER.unpack_from(buf)
    if magic != SIGNAL_MAGIC:
        raise SerializationError(f"bad magic bytes: {magic!r}")
    if version != VERSION:
        raise SerializationError(f"unsupported version: {version}")
    if code not in DTYPE_CODES:
        raise SerializationError(f"unknown dtype code: {code}")
    dtype = DTYPE_CODES[code]
    expected = channels * length * dtype.itemsize
    payload = buf[_HEADER.size:]
    if len(payload) != expected:
        raise SerializationError(f"payload size mismatch: expected {expected} bytes, got {len(payload)}")
    data = np.frombuffer(payload, dtype=dtype).reshape(channels, length).copy()
    return Signal(data=data, label=label, sample_rate=rate)


def save_signal(path: str | Path, data: np.ndarray, label: int, sample_rate: float = 1.0) -> None:
    """Write a single signal to ``path``."""
    Path(path).write_bytes(_encode(Signal(np.asarray(data), label, sample_rate)))


def load_signal(path: str | Path) -> Signal:
    """Read a single signal from ``path``; raises :class:`SerializationError` if malformed."""
    return _decode(Path(path).read_bytes())


def save_dataset(path: str | Path, signals: Sequence[Signal]) -> None:
    """Write many signals into one batch file."""
    with open(path, "wb") as f:
        f.write(_DS_HEADER.pack(DATASET_MAGIC, VERSION, len(signals)))
        for s in signals:
            rec = _encode(s)
            f.write(_LEN.pack(len(rec)))
            f.write(rec)


def _read_exact(f: BinaryIO, n: int) -> bytes:
    buf = f.read(n)
    if len(buf) != n:
        raise SerializationError("unexpected end of file")
    return buf


def load_dataset(path: str | Path) -> list[Signal]:
    """Read a batch file written by :func:`save_dataset`."""
    with open(path, "rb") as f:
        magic, version, count = _DS_HEADER.unpack(_read_exact(f, _DS_HEADER.size))
        if magic != DATASET_MAGIC:
            raise SerializationError(f"bad magic bytes: {magic!r}")
        if version != VERSION:
            raise SerializationError(f"unsupported version: {version}")
        out = []
        for _ in range(count):
            (n,) = _LEN.unpack(_read_exact(f, _LEN.size))
            out.append(_decode(_read_exact(f, n)))
        return out
