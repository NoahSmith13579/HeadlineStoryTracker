from datetime import UTC, datetime

import numpy as np

from hst.blocking import find_candidates
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


def test_cluster_clamps_cosine_rounding_above_one():
    pairs = [
        {"a_id": "a", "b_id": "b", "cosine": 1.0000001, "time_gap_hours": 1},
    ]
    assert cluster_headlines(["a", "b"], pairs, 0.65) == [["a", "b"]]


def test_blocking_respects_window_and_top_k():
    now = datetime(2026, 11, 20, tzinfo=UTC)
    embeddings = np.array([
        [1.0, 0.0],
        [0.99, 0.1],
        [0.8, 0.6],
        [1.0, 0.0],
    ])
    timestamps = [
        now,
        now,
        now,
        now.replace(day=16),
    ]
    pairs = find_candidates(
        embeddings,
        timestamps,
        ["a", "b", "c", "old"],
        k=1,
        window_hours=72,
    )
    assert all("old" not in (pair["a_id"], pair["b_id"]) for pair in pairs)
    assert pairs == sorted(pairs, key=lambda pair: (pair["a_id"], pair["b_id"]))
    assert ("a", "b") in {(pair["a_id"], pair["b_id"]) for pair in pairs}


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
        '{"stories":{"s_large":{"members":["a","b","c","x"],"alive":true},'
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
    tracked, _ = track_clusters([["a", "b", "c"], ["d"]], state)
    original = next(story for story in tracked if story["story_id"] == "s_old")
    split = next(story for story in tracked if story["split_from"] == "s_old")
    assert original["members"] == ["a", "b", "c"]
    assert split["members"] == ["d"]


def test_tracking_death(tmp_path):
    state = tmp_path / "stories.json"
    state.write_text(
        '{"stories":{"s_old":{"members":["a","b"],"alive":true,'
        '"last_seen":"2026-11-01T00:00:00Z"}}}',
        encoding="utf-8",
    )
    tracked, updated = track_clusters(
        [],
        state,
        now=datetime(2026, 11, 5, tzinfo=UTC),
        death_hours=72,
    )
    assert tracked == []
    assert "s_old" in updated["archived"]


def test_tracking_continuation_preserves_first_seen_and_id(tmp_path):
    state = tmp_path / "stories.json"
    state.write_text(
        '{"stories":{"s_old":{"members":["a","b"],"alive":true,'
        '"first_seen":"2026-11-18T12:00:00Z",'
        '"last_seen":"2026-11-19T12:00:00Z"}}}',
        encoding="utf-8",
    )
    tracked, _ = track_clusters(
        [["a", "b", "c"]],
        state,
        now=datetime(2026, 11, 20, tzinfo=UTC),
        first_seen_by_id={"c": "2026-11-20T00:00:00Z"},
    )
    assert tracked[0]["story_id"] == "s_old"
    assert tracked[0]["first_seen"] == "2026-11-18T12:00:00Z"


def test_tracking_persistence_excludes_inactive_old_ids(tmp_path):
    state = tmp_path / "stories.json"
    state.write_text(
        '{"stories":{"s_old":{"members":["a","b","c","d"],"alive":true}}}',
        encoding="utf-8",
    )
    _tracked, updated = track_clusters([["a", "b", "c", "d", "e", "f", "g", "h", "i"]], state)
    assert updated["id_persistence"] == 0.0
