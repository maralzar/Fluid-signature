import numpy as np
import pytest

from src.utils.normalization import minmax_scale, normalize_entropy, normalize_frequency


def test_normalize_frequency_sums_to_one():
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    normalized = normalize_frequency(values)
    assert np.isclose(normalized.sum(), 1.0)


def test_minmax_scale_bounds():
    values = np.array([0.0, 5.0, 10.0], dtype=np.float32)
    scaled = minmax_scale(values)
    assert scaled.min() == pytest.approx(0.0)
    assert scaled.max() == pytest.approx(1.0)


def test_normalize_entropy():
    entropy = np.array([0.0, np.log(2)], dtype=np.float32)
    normalized = normalize_entropy(entropy, num_bins=2)
    assert normalized[0] == pytest.approx(0.0)
    assert normalized[1] == pytest.approx(1.0)
