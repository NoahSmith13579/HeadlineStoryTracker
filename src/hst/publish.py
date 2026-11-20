"""Publish baseline stories and health into the static site."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from hst.schemas import StoriesFile, StoryHeadline, StoryObject


def read_headlines(data_dir: str | Path) -> list[dict]:
    path = Path(data_dir) / "headlines"
    records = []
    for file in sorted(path.glob("*.jsonl")):
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def medoid(cluster: list[dict], embeddings) -> dict:
    if not cluster:
        return {}
    positions = {record["id"]: i for i, record in enumerate(cluster)}
    if embeddings is None:
        return sorted(cluster, key=lambda record: (record["source_id"], record["title_clean"]))[0]
    selected = embeddings[list(positions.values())]
    scores = selected @ selected.T
    index = int(scores.sum(axis=1).argmax())
    return cluster[index]


def keywords_for_story(cluster: list[dict]) -> list[str]:
    counter: Counter[str] = Counter()
    for record in cluster:
        words = re.findall(r"[A-Za-z][A-Za-z0-9'-]{3,}", record["title_clean"].lower())
        counter.update(word for word in words if word not in {
            "about", "after", "says", "with", "from", "that", "have", "will",
        })
    return [word for word, _count in counter.most_common(5)]


def recency_weight(record: dict, now: datetime) -> float:
    timestamp = record.get("published_at") or record.get("first_seen_at")
    try:
        published = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if published.tzinfo is None:
            published = published.replace(tzinfo=UTC)
        age_hours = max(0.0, (now - published).total_seconds() / 3600)
        return 1.0 / (1.0 + age_hours / 72.0)
    except (AttributeError, ValueError):
        return 0.0


def publish(
    data_dir: str | Path,
    site_dir: str | Path,
    clusters: list[list[str]],
    tracked_stories: list[dict],
    embeddings,
    window_hours: int = 72,
) -> dict:
    data_path = Path(data_dir)
    site_path = Path(site_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    all_records = read_headlines(data_path)
    record_by_id = {record["id"]: record for record in all_records}

    now = datetime.now(UTC)
    cutoff = now - timedelta(hours=window_hours)
    in_window = []
    for record in all_records:
        timestamp = record.get("published_at") or record.get("first_seen_at")
        try:
            seen = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if seen.tzinfo is None:
                seen = seen.replace(tzinfo=UTC)
            if seen >= cutoff:
                in_window.append(record)
        except (AttributeError, ValueError):
            in_window.append(record)

    tracked_by_members = {
        frozenset(record["members"]): record
        for record in tracked_stories
    }
    story_objects: list[StoryObject] = []
    for cluster in sorted(clusters, key=lambda group: group[0]):
        members = [record_by_id[member] for member in cluster if member in record_by_id]
        if not members:
            continue
        title_record = medoid(members, embeddings)
        sources = sorted({member["source_id"] for member in members})
        timestamps = [
            member["published_at"] or member["first_seen_at"]
            for member in members
        ]
        ordered_timestamps = sorted(timestamps)
        hourly = [0] * 72
        for member in members:
            timestamp = member["published_at"] or member["first_seen_at"]
            try:
                seen = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                if seen.tzinfo is None:
                    seen = seen.replace(tzinfo=UTC)
                index = int((now - seen).total_seconds() // 3600)
                if 0 <= index < 72:
                    hourly[71 - index] += 1
            except (AttributeError, ValueError):
                pass
        tracked = tracked_by_members.get(frozenset(cluster), {})
        story_id = tracked.get("story_id") or "s_" + hashlib.sha1(
            "|".join(sorted(cluster)).encode("utf-8")
        ).hexdigest()[:8]
        headlines = []
        for member in sorted(members, key=lambda item: item["first_seen_at"], reverse=True)[:12]:
            headlines.append(StoryHeadline(
                id=member["id"],
                title=member["title_clean"],
                url=member["url"],
                source_id=member["source_id"],
                published_at=member["published_at"],
            ))
        story_objects.append(StoryObject(
            story_id=story_id,
            title=title_record["title_clean"],
            keywords=keywords_for_story(members),
            first_seen=min(ordered_timestamps),
            last_seen=max(ordered_timestamps),
            n_headlines=len(members),
            n_sources=len(sources),
            sources=sources,
            hourly_counts=hourly,
            headlines=headlines,
            merged_from=tracked.get("merged_from", []),
            split_from=tracked.get("split_from"),
            forecast=None,
        ))

    story_objects.sort(
        key=lambda story: (
            sum(recency_weight(record, now) for record in in_window
                if record["id"] in {
                    headline.id for headline in story.headlines
                }),
            story.n_headlines,
            story.last_seen,
        ),
        reverse=True,
    )
    top_stories = story_objects[:100]
    payload = StoriesFile(
        generated_at=now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        window_hours=window_hours,
        model_version="cosine-baseline",
        n_headlines_in_window=len(in_window),
        n_stories=len(top_stories),
        stories=top_stories,
    ).model_dump(mode="json")

    data_output = site_path / "data"
    data_output.mkdir(parents=True, exist_ok=True)
    stories_path = data_output / "stories.json"
    stories_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    if stories_path.stat().st_size > 500 * 1024:
        top_stories = top_stories[:50]
        payload["stories"] = top_stories
        payload["n_stories"] = len(top_stories)
        stories_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    detail_dir = data_output / "stories"
    detail_dir.mkdir(parents=True, exist_ok=True)
    for story in top_stories:
        (detail_dir / f"{story.story_id}.json").write_text(
            json.dumps(story.model_dump(mode="json"), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    health_path = data_path / "state" / "health.json"
    if health_path.exists():
        (data_output / "health.json").write_text(
            health_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    return payload
