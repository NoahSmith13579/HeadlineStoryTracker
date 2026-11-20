"""Validate, write, and update ingestion and clustering health reports."""
import json
import statistics
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
            "n_feeds_ok": sum(feed.status in ("ok", "not_modified") for feed in feed_models),
            "n_feeds_total": len(feed_models),
        },
        feeds=feed_models,
        quality=quality,
        clustering=clustering,
    ).model_dump(mode="json")
    write_json(path, payload)
    return payload


def update_clustering_health(
    data_dir: str | Path,
    clusters: list[list[str]],
    id_persistence: float,
) -> dict[str, Any]:
    state_dir = Path(data_dir) / "state"
    health_path = state_dir / "health.json"
    payload = json.loads(health_path.read_text(encoding="utf-8"))
    sizes = [len(cluster) for cluster in clusters]
    payload["clustering"] = {
        "n_clusters": len(clusters),
        "median_size": statistics.median(sizes) if sizes else 0.0,
        "singleton_rate": (
            sum(size == 1 for size in sizes) / len(sizes)
            if sizes else 0.0
        ),
        "id_persistence": id_persistence,
    }
    payload = HealthFile.model_validate(payload).model_dump(mode="json")
    write_json(health_path, payload)

    history_path = state_dir / "health_history.jsonl"
    if history_path.exists():
        lines = history_path.read_text(encoding="utf-8").splitlines()
        replaced = False
        for index, line in enumerate(lines):
            previous = json.loads(line)
            if previous.get("generated_at") == payload["generated_at"]:
                lines[index] = json.dumps(payload, ensure_ascii=False)
                replaced = True
        if replaced:
            history_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            with history_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    else:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload
