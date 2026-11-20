"""L2-normalized title embeddings with disk caching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def title_hash(title: str) -> str:
    return hashlib.sha256(title.encode("utf-8")).hexdigest()


def encode_titles(titles: list[str], cache_path: str | Path) -> np.ndarray:
    if not titles:
        return np.empty((0, 384), dtype=np.float32)
    cache = Path(cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    records: dict[str, list[float]] = {}
    if cache.exists():
        records = json.loads(cache.read_text(encoding="utf-8"))

    missing = [title for title in titles if title_hash(title) not in records]
    if missing:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        unique_missing = list(dict.fromkeys(missing))
        vectors = model.encode(
            unique_missing,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for title, vector in zip(unique_missing, vectors, strict=True):
            records[title_hash(title)] = np.asarray(vector, dtype=np.float32).tolist()
        cache.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    matrix = np.vstack(
        [np.asarray(records[title_hash(title)], dtype=np.float32) for title in titles]
    )
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix


def load_embeddings(ids: list[str], matrix_path: str | Path) -> np.ndarray:
    path = Path(matrix_path)
    matrix = np.load(path)
    ids_path = Path(f"{path}.ids.json")
    saved_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    positions = {headline_id: index for index, headline_id in enumerate(saved_ids)}
    missing = [headline_id for headline_id in ids if headline_id not in positions]
    if missing:
        raise ValueError(f"{len(missing)} headlines are missing cached embeddings")
    return matrix[[positions[headline_id] for headline_id in ids]]
