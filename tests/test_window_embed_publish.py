import json
import sys
from datetime import UTC, datetime, timedelta
from types import ModuleType

import numpy as np
import pytest

from hst.embed import encode_titles, load_embeddings, title_hash
from hst.publish import keywords_by_story, medoid, publish
from hst.window import headlines_in_window


def test_headlines_in_window_uses_utc_and_excludes_old_items():
    now = datetime(2026, 11, 20, 12, tzinfo=UTC)
    records = [
        {"id": "inside", "published_at": (now - timedelta(hours=71)).isoformat()},
        {"id": "outside", "published_at": (now - timedelta(hours=73)).isoformat()},
        {"id": "undated", "published_at": None, "first_seen_at": now.isoformat()},
    ]
    assert [record["id"] for record in headlines_in_window(records, 72, now)] == [
        "inside",
        "undated",
    ]


def test_encode_titles_uses_hashed_cache(tmp_path):
    cache = tmp_path / "embeddings.json"
    cache.write_text(
        json.dumps({title_hash("headline"): [1.0, 0.0]}),
        encoding="utf-8",
    )
    vectors = encode_titles(["headline"], cache)
    np.testing.assert_array_equal(vectors, np.array([[1.0, 0.0]], dtype=np.float32))


def test_encode_titles_does_not_hide_model_failures(tmp_path, monkeypatch):
    class BrokenModel:
        def __init__(self, _name):
            pass

        def encode(self, *_args, **_kwargs):
            raise RuntimeError("model unavailable")

    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = BrokenModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    with pytest.raises(RuntimeError, match="model unavailable"):
        encode_titles(["headline"], tmp_path / "embeddings.json")


def test_load_embeddings_aligns_rows_by_headline_id(tmp_path):
    matrix_path = tmp_path / "embeddings.npy"
    np.save(matrix_path, np.array([[0.0], [1.0], [2.0]], dtype=np.float32))
    (tmp_path / "embeddings.npy.ids.json").write_text(
        json.dumps(["c", "a", "b"]),
        encoding="utf-8",
    )
    np.testing.assert_array_equal(
        load_embeddings(["b", "c"], matrix_path),
        np.array([[2.0], [0.0]], dtype=np.float32),
    )


def test_medoid_uses_global_embedding_positions():
    cluster = [
        {"id": "a", "source_id": "wire", "title_clean": "A"},
        {"id": "b", "source_id": "wire", "title_clean": "B"},
        {"id": "c", "source_id": "wire", "title_clean": "C"},
    ]
    embeddings = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.8, 0.6],
        [-1.0, 0.0],
    ])
    positions = {"a": 3, "b": 4, "c": 5}
    assert medoid(cluster, embeddings, positions)["id"] == "b"


def test_keywords_use_story_level_tfidf():
    stories = [
        [{
            "title_clean": "City council approves new harbor plan",
        }],
        [{
            "title_clean": "City council opens downtown library",
        }],
    ]
    keywords = keywords_by_story(stories)
    assert "harbor" in keywords[0]
    assert "library" in keywords[1]


def test_empty_embedding_input_has_expected_shape(tmp_path):
    assert encode_titles([], tmp_path / "empty.json").shape == (0, 384)


def test_publish_writes_bounded_story_and_detail_files(tmp_path):
    data_dir = tmp_path / "data"
    headline_dir = data_dir / "headlines"
    headline_dir.mkdir(parents=True)
    now = datetime.now(UTC)
    record = {
        "id": "0123456789abcdef",
        "source_id": "wire",
        "title_raw": "A verified event",
        "title_clean": "A verified event",
        "url": "https://example.com/event",
        "url_canonical": "https://example.com/event",
        "published_at": now.isoformat().replace("+00:00", "Z"),
        "published_at_source": "feed",
        "first_seen_at": now.isoformat().replace("+00:00", "Z"),
        "lang": "en",
        "dup_of": None,
    }
    (headline_dir / f"{now.date().isoformat()}.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )
    story = {
        "story_id": "s_01234567",
        "members": [record["id"]],
        "first_seen": record["first_seen_at"],
        "last_seen": record["first_seen_at"],
        "merged_from": [],
        "split_from": None,
    }
    result = publish(
        data_dir,
        tmp_path / "site",
        [[record["id"]]],
        [story],
        np.array([[1.0, 0.0]], dtype=np.float32),
        embedding_ids=[record["id"]],
    )
    stories_file = tmp_path / "site" / "data" / "stories.json"
    assert result["n_stories"] == 1
    assert stories_file.stat().st_size < 500 * 1024
    assert (tmp_path / "site" / "data" / "stories" / "s_01234567.json").exists()
