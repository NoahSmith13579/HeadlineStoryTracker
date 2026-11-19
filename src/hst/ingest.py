"""Ingestion implementation stub for Phase 1."""
import hashlib
import json
import time
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

import feedparser
import httpx

from hst.clean import canonicalize_url, clean_title, detect_language, normalize_title
from hst.config import load_sources
from hst.health import write_health
from hst.schemas import HeadlineRecord, SourceConfig

USER_AGENT = (
    "HeadlineStoryTracker/0.1 "
    "(RSS ingestion; https://github.com/NoahSmith13579/HeadlineStoryTracker)"
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_feed_date(entry: dict, now: datetime) -> str | None:
    parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if parsed is None:
        return None
    published = datetime(*parsed[:6], tzinfo=UTC)
    if published > now + timedelta(minutes=10) or published < now - timedelta(days=7):
        return None
    return published.isoformat(timespec="seconds").replace("+00:00", "Z")


def headline_id(url_canonical: str) -> str:
    return hashlib.sha1(url_canonical.encode("utf-8")).hexdigest()[:16]


def read_existing(data_dir: Path) -> tuple[set[str], dict[str, str]]:
    ids: set[str] = set()
    normalized_titles: dict[str, str] = {}
    for path in sorted((data_dir / "headlines").glob("**/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            ids.add(record["id"])
            title_key = normalize_title(record["title_clean"])
            normalized_titles.setdefault(title_key, record["id"])
    return ids, normalized_titles


def prune_headlines(data_dir: Path, retention_days: int) -> int:
    cutoff = utc_now() - timedelta(days=retention_days)
    removed = 0
    headline_dir = data_dir / "headlines"
    if not headline_dir.exists():
        return 0
    for path in headline_dir.glob("*.jsonl"):
        try:
            date = datetime.strptime(path.stem, "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            continue
        if date.date() < cutoff.date():
            path.unlink()
            removed += 1
    return removed


def fetch_feed(
    client: httpx.Client,
    source: SourceConfig,
    feed_state: dict,
) -> tuple[int | None, bytes, dict]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/atom+xml"}
    saved = feed_state.get(source.id, {})
    if saved.get("etag"):
        headers["If-None-Match"] = saved["etag"]
    if saved.get("last_modified"):
        headers["If-Modified-Since"] = saved["last_modified"]
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.get(source.feed_url, headers=headers)
            response_headers = {
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
            }
            return response.status_code, response.content, response_headers
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.25 * (2**attempt))
    raise RuntimeError(f"feed request failed after retries: {last_error}")


def append_records(data_dir: Path, records: Iterable[HeadlineRecord]) -> int:
    grouped: dict[str, list[HeadlineRecord]] = {}
    for record in records:
        grouped.setdefault(record.first_seen_at[:10], []).append(record)
    written = 0
    for date, date_records in grouped.items():
        path = data_dir / "headlines" / f"{date}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_ids = {
            json.loads(line)["id"]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        } if path.exists() else set()
        with path.open("a", encoding="utf-8") as stream:
            for record in date_records:
                if record.id in existing_ids:
                    continue
                stream.write(record.model_dump_json() + "\n")
                existing_ids.add(record.id)
                written += 1
    return written


def run_ingest(
    data_dir: str | Path = "data",
    sources_path: str | Path = "config/sources.yaml",
    prune: bool = False,
    retention_days: int = 60,
) -> dict:
    start = time.monotonic()
    now = utc_now()
    data_path = Path(data_dir)
    state_dir = data_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    sources = [source for source in load_sources(sources_path) if source.enabled]
    feed_state_path = state_dir / "feed_state.json"
    feed_state = (
        json.loads(feed_state_path.read_text(encoding="utf-8"))
        if feed_state_path.exists()
        else {}
    )

    existing_ids, title_map = read_existing(data_path)
    feed_health = []
    all_new: list[HeadlineRecord] = []

    with httpx.Client(timeout=15.0, follow_redirects=True) as client:
        for source in sources:
            feed_started = time.monotonic()
            try:
                status, content, response_headers = fetch_feed(client, source, feed_state)
                if status == 304:
                    saved = feed_state.get(source.id, {})
                    feed_health.append({
                        "id": source.id,
                        "status": "not_modified",
                        "http_status": 304,
                        "items": 0,
                        "new_items": 0,
                        "latency_ms": int((time.monotonic() - feed_started) * 1000),
                        "consecutive_failures": 0,
                        "last_success": saved.get("last_success"),
                    })
                    continue
                if status is None or status >= 400:
                    raise RuntimeError(f"HTTP {status}")

                parsed = feedparser.parse(content)
                if parsed.bozo and not parsed.entries:
                    raise RuntimeError("malformed feed")
                feed_state[source.id] = {
                    **response_headers,
                    "last_success": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
                }

                feed_new = 0
                for entry in parsed.entries:
                    title_raw = str(getattr(entry, "title", "")).strip()
                    url = str(getattr(entry, "link", "")).strip()
                    if not title_raw or not url:
                        continue
                    title_clean = clean_title(title_raw)
                    canonical = canonicalize_url(url)
                    if not canonical or not urlsplit(canonical).scheme:
                        continue
                    record_id = headline_id(canonical)
                    if record_id in existing_ids:
                        continue

                    published = parse_feed_date(entry, now)
                    published_at_source = "feed" if published else "first_seen"
                    first_seen = now.isoformat(timespec="seconds").replace("+00:00", "Z")
                    title_key = normalize_title(title_clean)
                    dup_of = title_map.get(title_key) if title_key else None
                    record = HeadlineRecord(
                        id=record_id,
                        source_id=source.id,
                        title_raw=title_raw,
                        title_clean=title_clean,
                        url=url,
                        url_canonical=canonical,
                        published_at=published,
                        published_at_source=published_at_source,
                        first_seen_at=first_seen,
                        lang=detect_language(title_clean),
                        dup_of=dup_of,
                    )
                    all_new.append(record)
                    existing_ids.add(record_id)
                    title_map.setdefault(title_key, record_id)
                    feed_new += 1

                feed_health.append({
                    "id": source.id,
                    "status": "ok",
                    "http_status": status,
                    "items": len(parsed.entries),
                    "new_items": feed_new,
                    "latency_ms": int((time.monotonic() - feed_started) * 1000),
                    "consecutive_failures": 0,
                    "last_success": feed_state[source.id]["last_success"],
                })
            except Exception:
                saved = feed_state.get(source.id, {})
                failures = saved.get("consecutive_failures", 0) + 1
                feed_state[source.id] = {
                    **saved,
                    "consecutive_failures": failures,
                }
                feed_health.append({
                    "id": source.id,
                    "status": "failed",
                    "http_status": None,
                    "items": 0,
                    "new_items": 0,
                    "latency_ms": int((time.monotonic() - feed_started) * 1000),
                    "consecutive_failures": failures,
                    "last_success": saved.get("last_success"),
                })

    n_written = append_records(data_path, all_new)
    feed_state_path.write_text(
        json.dumps(feed_state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if prune:
        prune_headlines(data_path, retention_days)

    records_for_quality = all_new
    quality = {
        "null_timestamp_rate": (
            sum(record.published_at is None for record in records_for_quality)
            / len(records_for_quality)
            if records_for_quality else 0.0
        ),
        "dup_rate": (
            sum(record.dup_of is not None for record in records_for_quality)
            / len(records_for_quality)
            if records_for_quality else 0.0
        ),
        "non_english_rate": (
            sum(record.lang not in ("en", "und") for record in records_for_quality)
            / len(records_for_quality)
            if records_for_quality else 0.0
        ),
    }
    health = write_health(
        state_dir / "health.json",
        generated_at=now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_s=round(time.monotonic() - start, 3),
        n_new_headlines=n_written,
        feeds=feed_health,
        quality=quality,
        clustering={
            "n_clusters": 0,
            "median_size": 0.0,
            "singleton_rate": 0.0,
            "id_persistence": 1.0,
        },
    )

    history_path = state_dir / "health_history.jsonl"
    with history_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(health, ensure_ascii=False) + "\n")
    return health
