from __future__ import annotations

import numpy as np


def normalize_frequency(values: np.ndarray, axis: int | None = None, eps: float = 1e-8) -> np.ndarray:
    total = values.sum(axis=axis, keepdims=True)
    return values / np.maximum(total, eps)


def minmax_scale(values: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    if vmin is None:
        vmin = float(values.min()) if values.size else 0.0
    if vmax is None:
        vmax = float(values.max()) if values.size else 1.0
    if np.isclose(vmin, vmax):
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def normalize_entropy(entropy_values: np.ndarray, num_bins: int) -> np.ndarray:
    max_entropy = np.log(max(num_bins, 2))
    if max_entropy <= 0:
        return np.zeros_like(entropy_values, dtype=np.float32)
    return np.clip(entropy_values / max_entropy, 0.0, 1.0).astype(np.float32)
