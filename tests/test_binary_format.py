import numpy as np
import pytest

from time_series_classification.serialization import (
    SerializationError,
    Signal,
    load_dataset,
    load_signal,
    save_dataset,
    save_signal,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_round_trip(tmp_path, dtype):
    data = (np.random.default_rng(0).standard_normal((3, 50)) * 10).astype(dtype)
    save_signal(tmp_path / "a.tsig", data, label=2, sample_rate=250.0)
    s = load_signal(tmp_path / "a.tsig")
    assert s.label == 2 and s.sample_rate == 250.0
    assert s.data.dtype == data.dtype
    np.testing.assert_array_equal(s.data, data)


def test_1d_becomes_single_channel(tmp_path):
    save_signal(tmp_path / "a.tsig", np.arange(5, dtype=np.float32), 0)
    assert load_signal(tmp_path / "a.tsig").data.shape == (1, 5)


def test_dataset_round_trip(tmp_path):
    sigs = [Signal(np.full((2, 4), i, dtype=np.float32), i, 10.0) for i in range(3)]
    save_dataset(tmp_path / "d.tsds", sigs)
    out = load_dataset(tmp_path / "d.tsds")
    assert [s.label for s in out] == [0, 1, 2]
    np.testing.assert_array_equal(out[2].data, sigs[2].data)


def test_malformed(tmp_path):
    p = tmp_path / "a.tsig"
    save_signal(p, np.zeros((1, 8), dtype=np.float32), 0)
    raw = p.read_bytes()
    for bad in (b"", raw[:10], b"XXXX" + raw[4:], raw[:-1], raw[:4] + b"\x09\x00" + raw[6:]):
        p.write_bytes(bad)
        with pytest.raises(SerializationError):
            load_signal(p)


def test_unsupported_dtype(tmp_path):
    with pytest.raises(SerializationError):
        save_signal(tmp_path / "a.tsig", np.zeros((1, 4), dtype=np.int8), 0)
