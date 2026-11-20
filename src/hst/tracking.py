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
    first_seen_by_id: dict[str, str] | None = None,
) -> tuple[list[dict], dict]:
    path = Path(state_path)
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old_stories = state.get("stories", {})
    archived = state.get("archived", {})
    current_time = now or datetime.now(UTC)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=UTC)
    current_time = current_time.astimezone(UTC)
    timestamp = current_time.isoformat(timespec="seconds").replace("+00:00", "Z")
    cutoff = current_time - timedelta(hours=death_hours)
    first_seen_by_id = first_seen_by_id or {}

    active_ids = []
    for old_id, record in old_stories.items():
        last_seen = record.get("last_seen")
        if last_seen:
            try:
                parsed = datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                if parsed < cutoff:
                    expired = dict(record)
                    expired.update({
                        "alive": False,
                        "archived_at": timestamp,
                        "archive_reason": "inactive",
                    })
                    archived[old_id] = expired
                    continue
            except ValueError:
                pass
        active_ids.append(old_id)

    old_members = {
        old_id: set(old_stories[old_id].get("members", []))
        for old_id in active_ids
    }
    new_members = [set(cluster) for cluster in clusters]
    scores = [
        [jaccard(old_members[old_id], cluster) for cluster in new_members]
        for old_id in active_ids
    ]

    primary_matches: set[tuple[int, int]] = set()
    if active_ids and new_members:
        rows, columns = linear_sum_assignment([[-score for score in row] for row in scores])
        primary_matches = {
            (int(row), int(column))
            for row, column in zip(rows, columns, strict=True)
            if scores[row][column] >= minimum_overlap
        }

    keeper_by_old: dict[str, int] = {}
    for row, old_id in enumerate(active_ids):
        overlapping_columns = [
            column for column, cluster in enumerate(new_members)
            if old_members[old_id] & cluster
        ]
        qualified_columns = [
            column for column in overlapping_columns
            if scores[row][column] >= minimum_overlap
        ]
        if qualified_columns:
            keeper_by_old[old_id] = max(
                qualified_columns,
                key=lambda column: (
                    len(old_members[old_id] & new_members[column]),
                    (row, column) in primary_matches,
                    -column,
                ),
            )
        elif len(overlapping_columns) > 1:
            keeper_by_old[old_id] = max(
                overlapping_columns,
                key=lambda column: (
                    len(old_members[old_id] & new_members[column]),
                    (row, column) in primary_matches,
                    -column,
                ),
            )

    next_stories: dict[str, dict] = {}
    current_stories: list[dict] = []
    consumed_old: set[str] = set()
    for column, cluster in enumerate(new_members):
        surviving_ids = [
            old_id for old_id, keeper_column in keeper_by_old.items()
            if keeper_column == column
        ]
        survivor = None
        if surviving_ids:
            survivor = max(
                surviving_ids,
                key=lambda old_id: (
                    len(old_members[old_id]),
                    old_id,
                ),
            )
            consumed_old.update(surviving_ids)

        if survivor is not None:
            record = dict(old_stories[survivor])
            record.update({
                "members": sorted(cluster),
                "alive": True,
                "first_seen": record.get("first_seen") or min(
                    (first_seen_by_id[item] for item in cluster if item in first_seen_by_id),
                    default=timestamp,
                ),
                "last_seen": timestamp,
                "merged_from": sorted(old_id for old_id in surviving_ids if old_id != survivor),
                "split_from": None,
            })
            for merged_id in record["merged_from"]:
                merged = dict(old_stories[merged_id])
                merged.update({
                    "alive": False,
                    "archived_at": timestamp,
                    "archive_reason": "merged",
                    "merged_into": survivor,
                })
                archived[merged_id] = merged
            story_key = survivor
        else:
            split_parents = [
                old_id for old_id, old_cluster in old_members.items()
                if (
                    old_cluster & cluster
                    and len([
                        candidate for candidate in new_members
                        if old_cluster & candidate
                    ]) > 1
                    and keeper_by_old.get(old_id) != column
                )
            ]
            parent = max(
                split_parents,
                key=lambda old_id: (len(old_members[old_id] & cluster), old_id),
                default=None,
            )
            story_key = story_id(sorted(cluster))
            record = {
                "members": sorted(cluster),
                "alive": True,
                "first_seen": min(
                    (first_seen_by_id[item] for item in cluster if item in first_seen_by_id),
                    default=timestamp,
                ),
                "last_seen": timestamp,
                "merged_from": [],
                "split_from": parent,
            }

        next_stories[story_key] = record
        current_stories.append({"story_id": story_key, **record})

    previous_present = {
        old_id
        for old_id, members in old_members.items()
        if any(members & cluster for cluster in new_members)
    }
    current_ids = {story["story_id"] for story in current_stories}
    kept_ids = previous_present & current_ids
    id_persistence = len(kept_ids) / len(previous_present) if previous_present else 1.0

    for old_id, record in old_stories.items():
        if old_id in consumed_old:
            continue
        if old_id in archived:
            continue
        last_seen = record.get("last_seen")
        if last_seen:
            try:
                parsed = datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                if parsed < cutoff:
                    continue
            except ValueError:
                pass
        inactive = dict(record)
        inactive["alive"] = False
        next_stories[old_id] = inactive

    new_state = {
        "stories": next_stories,
        "archived": archived,
        "id_persistence": id_persistence,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(new_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return current_stories, new_state
