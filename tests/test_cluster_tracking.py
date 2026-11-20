from datetime import UTC, datetime

from hst.cluster import cluster_headlines
from hst.tracking import jaccard, track_clusters


def test_jaccard_overlap():
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3


def test_cluster_baseline_keeps_near_pairs_separate():
    ids = ["a", "b", "c"]
    pairs = [
        {"a_id": "a", "b_id": "b", "cosine": 0.8, "time_gap_hours": 1},
        {"a_id": "b", "b_id": "c", "cosine": 0.1, "time_gap_hours": 1},
    ]
    clusters = cluster_headlines(ids, pairs, 0.65)
    assert ["a", "b"] in clusters
    assert ["c"] in clusters


def test_tracking_continuation(tmp_path):
    state = tmp_path / "stories.json"
    state.write_text(
        '{"stories":{"s_old":{"members":["a","b"],"alive":true}}}',
        encoding="utf-8",
    )
    tracked, _ = track_clusters([["a", "b", "c"]], state)
    assert "s_old" in {story["story_id"] for story in tracked}


def test_tracking_merge(tmp_path):
    state = tmp_path / "stories.json"
    state.write_text(
        '{"stories":{"s_large":{"members":["a","b","c"],"alive":true},'
        '"s_small":{"members":["c","d","e"],"alive":true}}}',
        encoding="utf-8",
    )
    tracked, _ = track_clusters([["a", "b", "c", "d", "e"]], state)
    large = next(story for story in tracked if story["story_id"] == "s_large")
    assert large["members"] == ["a", "b", "c", "d", "e"]
    assert "s_small" in large["merged_from"]


def test_tracking_split(tmp_path):
    state = tmp_path / "stories.json"
    state.write_text(
        '{"stories":{"s_old":{"members":["a","b","c","d"],"alive":true}}}',
        encoding="utf-8",
    )
    tracked, _ = track_clusters([["a", "b"], ["c", "d"]], state)
    ids = {story["story_id"] for story in tracked}
    split_ids = {story["split_from"] for story in tracked if story["split_from"]}
    assert "s_old" in ids
    assert split_ids == {"s_old"}


def test_tracking_death(tmp_path):
    state = tmp_path / "stories.json"
    state.write_text(
        '{"stories":{"s_old":{"members":["a","b"],"alive":true,'
        '"last_seen":"2026-11-01T00:00:00Z"}}}',
        encoding="utf-8",
    )
    tracked, _ = track_clusters(
        [],
        state,
        now=datetime(2026, 11, 5, tzinfo=UTC),
        death_hours=72,
    )
    assert tracked == []
