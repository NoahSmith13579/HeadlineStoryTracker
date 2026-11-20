"""Phase 2 cosine baseline."""

from __future__ import annotations


def cosine_score(a: object, b: object) -> float:
    raise TypeError("Use cosine_score_pairs for embeddings")


def cosine_score_pairs(a: list, b: list) -> float:
    import numpy as np

    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-12))
