"""Phase 2 cosine baseline."""

from __future__ import annotations

import numpy as np


def cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("cosine inputs must be same-sized one-dimensional vectors")
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        raise ValueError("cosine inputs must have nonzero magnitude")
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
