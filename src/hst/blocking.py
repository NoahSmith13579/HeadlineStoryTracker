"""Top-k candidate blocking within the 72-hour window."""

from __future__ import annotations

from datetime import datetime

import numpy as np


def find_candidates(
    embeddings: np.ndarray,
    timestamps: list[datetime],
    ids: list[str],
    k: int = 15,
    window_hours: int = 72,
) -> list[dict]:
    similarities = embeddings @ embeddings.T
    np.fill_diagonal(similarities, -np.inf)
    pairs: dict[tuple[str, str], dict] = {}
    cutoff = window_hours * 3600

    for i in range(len(ids)):
        neighbors = np.argsort(-similarities[i], kind="stable")
        kept = 0
        for j in neighbors:
            if kept >= k:
                break
            if i == j:
                continue
            gap_s = abs((timestamps[i] - timestamps[j]).total_seconds())
            if gap_s <= cutoff:
                a, b = sorted((ids[i], ids[j]))
                pairs[(a, b)] = {
                    "a_id": a,
                    "b_id": b,
                    "cosine": float(np.clip(similarities[i, j], -1.0, 1.0)),
                    "time_gap_hours": gap_s / 3600.0,
                }
                kept += 1

    return [pairs[key] for key in sorted(pairs)]
