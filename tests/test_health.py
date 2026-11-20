import json

from hst.health import update_clustering_health, write_health


def test_clustering_health_replaces_current_history_entry(tmp_path):
    state_dir = tmp_path / "state"
    initial = write_health(
        state_dir / "health.json",
        generated_at="2026-11-20T12:00:00Z",
        duration_s=1.0,
        n_new_headlines=3,
        feeds=[],
        quality={
            "null_timestamp_rate": 0.0,
            "dup_rate": 0.0,
            "non_english_rate": 0.0,
        },
        clustering={
            "n_clusters": 0,
            "median_size": 0.0,
            "singleton_rate": 0.0,
            "id_persistence": 1.0,
        },
    )
    history = state_dir / "health_history.jsonl"
    history.write_text(json.dumps(initial) + "\n", encoding="utf-8")

    updated = update_clustering_health(
        tmp_path,
        [["a", "b"], ["c"]],
        id_persistence=0.95,
    )

    entries = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 1
    assert entries[0]["clustering"] == updated["clustering"]
    assert entries[0]["clustering"]["n_clusters"] == 2
    assert entries[0]["clustering"]["singleton_rate"] == 0.5
    assert entries[0]["clustering"]["id_persistence"] == 0.95
