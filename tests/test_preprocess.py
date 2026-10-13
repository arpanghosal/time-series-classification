import numpy as np
import pytest
from time_series_classification.data import minmax, pad_or_truncate, preprocess, to_channels_first, zscore


def test_zscore_stats():
    x = np.random.default_rng(0).standard_normal((2, 100)).astype(np.float32) * 5 + 3
    z = zscore(x)
    np.testing.assert_allclose(z.mean(-1), 0, atol=1e-5)
    np.testing.assert_allclose(z.std(-1), 1, atol=1e-3)


def test_constant_signal_is_stable():
    c = np.ones((1, 10), dtype=np.float32)
    assert np.isfinite(zscore(c)).all() and np.isfinite(minmax(c)).all()


def test_minmax_range():
    m = minmax(np.arange(10, dtype=np.float32)[None])
    assert m.min() == 0 and m.max() == pytest.approx(1, abs=1e-6)


def test_pad_truncate():
    x = np.ones((2, 5))
    assert pad_or_truncate(x, 8).shape == (2, 8)
    assert pad_or_truncate(x, 8)[:, 5:].sum() == 0
    assert pad_or_truncate(x, 3).shape == (2, 3)


def test_axes():
    assert to_channels_first(np.zeros(7)).shape == (1, 7)
    assert to_channels_first(np.zeros((7, 2)), time_axis=0).shape == (2, 7)


def test_preprocess_shape_dtype():
    out = preprocess(np.random.rand(4, 30), length=64)
    assert out.shape == (4, 64) and out.dtype == np.float32
    with pytest.raises(ValueError):
        preprocess(out, normalize="bogus")
