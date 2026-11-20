"""Time-window filtering for stored headline records."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


def parse_headline_time(record: dict) -> datetime | None:
    value = record.get("published_at") or record.get("first_seen_at")
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def headlines_in_window(
    records: list[dict],
    window_hours: int,
    now: datetime | None = None,
) -> list[dict]:
    reference = now or datetime.now(UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    cutoff = reference.astimezone(UTC) - timedelta(hours=window_hours)
    return [
        record
        for record in records
        if (timestamp := parse_headline_time(record)) is not None
        and cutoff <= timestamp <= reference.astimezone(UTC) + timedelta(minutes=10)
    ]
