"""Stable story identity tracking with Hungarian assignment."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scipy.optimize import linear_sum_assignment


def story_id(member_ids: list[str]) -> str:
    digest = hashlib.sha1("|".join(sorted(member_ids)).encode("utf-8")).hexdigest()[:8]
    return f"s_{digest}"


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def track_clusters(
    clusters: list[list[str]],
    state_path: str | Path,
    minimum_overlap: float = 0.5,
    now: datetime | None = None,
    death_hours: int = 72,
) -> tuple[list[dict], dict]:
    path = Path(state_path)
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"stories": {}}
    old_stories = state.get("stories", {})
    current_time = now or datetime.now(UTC)
    old_ids = list(old_stories)
    new_clusters = [set(cluster) for cluster in clusters]
    matches: list[tuple[int, str, float]] = []
    if old_ids and new_clusters:
        scores = [
            [
                jaccard(set(old_stories[old_id].get("members", [])), cluster)
                for cluster in new_clusters
            ]
            for old_id in old_ids
        ]
        rows, cols = linear_sum_assignment([[-score for score in row] for row in scores])
        for row, col in zip(rows, cols, strict=True):
            if scores[row][col] >= minimum_overlap:
                matches.append((int(col), old_ids[row], scores[row][col]))

    # Add every qualified old-story match for merge detection. A one-to-one
    # Hungarian match may represent only one branch of a merge or split.
    for col, cluster in enumerate(new_clusters):
        for _row, old_id in enumerate(old_ids):
            score = jaccard(set(old_stories[old_id].get("members", [])), cluster)
            if score >= minimum_overlap:
                already_matched = any(
                    match_col == col and match_old == old_id
                    for match_col, match_old, _ in matches
                )
                if not already_matched:
                    matches.append((col, old_id, score))

    old_to_new: dict[str, list[int]] = {}
    new_to_old: dict[int, list[str]] = {}
    for col, old_id, _score in matches:
        old_to_new.setdefault(old_id, []).append(col)
        new_to_old.setdefault(col, []).append(old_id)

    next_stories: dict[str, dict] = {}
    for col, cluster in enumerate(new_clusters):
        old_matches = new_to_old.get(col, [])
        if old_matches:
            largest_old = max(
                old_matches,
                key=lambda old_id: len(old_stories[old_id].get("members", [])),
            )
            record = dict(old_stories[largest_old])
            record.update({
                "members": sorted(cluster),
                "alive": True,
                "last_seen": current_time.isoformat().replace("+00:00", "Z"),
                "merged_from": (
                    sorted(old_id for old_id in old_matches if old_id != largest_old)
                ),
                "split_from": None,
            })
            next_stories[largest_old] = record
        else:
            new_id = story_id(sorted(cluster))
            next_stories[new_id] = {
                "members": sorted(cluster),
                "alive": True,
                "last_seen": current_time.isoformat().replace("+00:00", "Z"),
                "merged_from": [],
                "split_from": None,
            }

    # One old story matching multiple new clusters is a split.
    for old_id, cols in old_to_new.items():
        if len(cols) > 1:
            largest_col = max(cols, key=lambda col: len(new_clusters[col]))
            next_stories[old_id]["split_from"] = None
            for col in cols:
                if col == largest_col:
                    continue
                piece = new_clusters[col]
                new_id = story_id(sorted(piece))
                next_stories[new_id] = {
                    "members": sorted(piece),
                    "alive": True,
                    "last_seen": current_time.isoformat().replace("+00:00", "Z"),
                    "merged_from": [],
                    "split_from": old_id,
                }

    matched_old = set(old_to_new)
    cutoff = current_time - timedelta(hours=death_hours)
    for old_id, record in old_stories.items():
        if old_id in matched_old:
            continue
        last_seen = record.get("last_seen")
        if not last_seen:
            continue
        try:
            parsed = datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            if parsed >= cutoff:
                archived = dict(record)
                archived["alive"] = False
                next_stories[old_id] = archived
        except ValueError:
            continue

    state = {"stories": next_stories}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output = [
        {"story_id": story_id_value, **record}
        for story_id_value, record in sorted(next_stories.items())
    ]
    return output, state
