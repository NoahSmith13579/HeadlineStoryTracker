"""Health reporting implementation stub for Phase 1."""
import json
from pathlib import Path
from typing import Any

from hst.schemas import FeedHealth, HealthFile


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_health(
    path: str | Path,
    *,
    generated_at: str,
    duration_s: float,
    n_new_headlines: int,
    feeds: list[dict[str, Any]],
    quality: dict[str, float],
    clustering: dict[str, Any],
) -> dict[str, Any]:
    feed_models = [FeedHealth.model_validate(feed) for feed in feeds]
    payload = HealthFile(
        generated_at=generated_at,
        run={
            "duration_s": duration_s,
            "n_new_headlines": n_new_headlines,
            "n_feeds_ok": sum(feed.status == "ok" for feed in feed_models),
            "n_feeds_total": len(feed_models),
        },
        feeds=feed_models,
        quality=quality,
        clustering=clustering,
    ).model_dump(mode="json")
    write_json(path, payload)
    return payload
