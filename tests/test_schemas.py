import pytest

from hst.schemas import (
    FeedHealth,
    HeadlineRecord,
    HealthFile,
    MetricsFile,
    PairLabel,
    SourceConfig,
    StoriesFile,
    StoryObject,
)


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (
            SourceConfig,
            {
                "id": "wire",
                "name": "Example Wire",
                "feed_url": "https://example.com/rss",
                "region": "global",
                "category": "wire",
                "is_wire": True,
                "language": "en",
                "tos_note": "",
                "tos_checked": False,
                "enabled": True,
            },
        ),
        (
            HeadlineRecord,
            {
                "id": "0123456789abcdef",
                "source_id": "wire",
                "title_raw": "A headline",
                "title_clean": "A headline",
                "url": "https://example.com/story",
                "url_canonical": "https://example.com/story",
                "published_at": "2026-11-19T12:00:00Z",
                "published_at_source": "feed",
                "first_seen_at": "2026-11-19T12:01:00Z",
                "lang": "en",
                "dup_of": None,
            },
        ),
        (
            PairLabel,
            {
                "pair_id": "pair1",
                "a_id": "a",
                "b_id": "b",
                "label": 1,
                "ambiguous": False,
                "label_source": "human_pair",
                "round": 1,
                "labeled_at": "2026-11-19T12:00:00Z",
            },
        ),
        (
            StoryObject,
            {
                "story_id": "s_01234567",
                "title": "A headline",
                "keywords": ["headline"],
                "first_seen": "2026-11-19T12:00:00Z",
                "last_seen": "2026-11-19T13:00:00Z",
                "n_headlines": 1,
                "n_sources": 1,
                "sources": ["wire"],
                "hourly_counts": [0] * 72,
                "headlines": [],
                "merged_from": [],
                "split_from": None,
                "forecast": None,
            },
        ),
        (
            StoriesFile,
            {
                "generated_at": "2026-11-19T13:00:00Z",
                "window_hours": 72,
                "model_version": "baseline",
                "n_headlines_in_window": 1,
                "n_stories": 1,
                "stories": [],
            },
        ),
        (
            FeedHealth,
            {
                "id": "wire",
                "status": "ok",
                "http_status": 200,
                "items": 10,
                "new_items": 2,
                "latency_ms": 100,
                "consecutive_failures": 0,
                "last_success": "2026-11-19T12:00:00Z",
            },
        ),
        (
            HealthFile,
            {
                "generated_at": "2026-11-19T13:00:00Z",
                "run": {
                    "duration_s": 1,
                    "n_new_headlines": 2,
                    "n_feeds_ok": 1,
                    "n_feeds_total": 1,
                },
                "feeds": [],
                "quality": {"null_timestamp_rate": 0.0, "dup_rate": 0.0, "non_english_rate": 0.0},
                "clustering": {
                    "n_clusters": 0,
                    "median_size": 0.0,
                    "singleton_rate": 0.0,
                    "id_persistence": 1.0,
                },
            },
        ),
        (
            MetricsFile,
            {
                "agreement": None,
                "blocking_recall": None,
                "pair_level": None,
                "cluster_level": None,
                "calibration": None,
                "ablations": None,
                "learning_curve": None,
                "drift": None,
                "live_audit": None,
            },
        ),
    ],
)
def test_schema_round_trip(model, payload):
    instance = model.model_validate(payload)
    assert model.model_validate_json(instance.model_dump_json()) == instance
