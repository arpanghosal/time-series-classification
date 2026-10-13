import pytest
import torch
from time_series_classification.data import SignalDataset

from time_series_classification.models import CNN1D


@pytest.mark.parametrize("length", [64, 256, 300])
def test_forward_shape(length):
    model = CNN1D(in_channels=2, num_classes=5).eval()
    assert model(torch.randn(4, 2, length)).shape == (4, 5)


def test_bad_input():
    model = CNN1D(in_channels=2, num_classes=3)
    with pytest.raises(ValueError):
        model(torch.randn(4, 1, 64))
    with pytest.raises(ValueError):
        model(torch.randn(4, 2, 4))


def test_dataset_and_cli_end_to_end(tmp_path):
    from time_series_classification.cli import main

    d, ck = str(tmp_path / "d"), str(tmp_path / "m.pt")
    assert main(["generate", "--out", d, "--n-samples", "30", "--length", "64"]) == 0
    x, y = SignalDataset(d, length=64)[0]
    assert x.shape == (1, 64) and isinstance(y, int)
    assert main(["train", "--data", d, "--checkpoint", ck, "--seq-len", "64", "--epochs", "1"]) == 0
    assert main(["evaluate", "--data", d, "--checkpoint", ck]) == 0
