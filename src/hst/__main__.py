import argparse
from pathlib import Path


def stub_command(name: str):
    def run(_args: argparse.Namespace) -> int:
        print(f"{name}: scaffold")
        return 0
    return run


def run_embed_command(args: argparse.Namespace) -> int:
    import json

    import numpy as np

    from hst.config import load_settings
    from hst.embed import encode_titles
    from hst.publish import read_headlines
    from hst.window import headlines_in_window

    settings = load_settings(args.settings)
    records = headlines_in_window(read_headlines(args.data_dir), settings["window_hours"])
    english = [record for record in records if record["lang"] == "en"]
    titles = [record["title_clean"] for record in english]
    embeddings = encode_titles(titles, args.cache)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings)
    Path(args.output + ".ids.json").write_text(
        json.dumps([record["id"] for record in english]),
        encoding="utf-8",
    )
    print(f"embed: encoded {len(titles)} English headlines")
    return 0


def run_cluster_command(args: argparse.Namespace) -> int:
    import json
    from datetime import datetime

    from hst.blocking import find_candidates
    from hst.cluster import cluster_headlines
    from hst.config import load_settings
    from hst.embed import load_embeddings
    from hst.health import update_clustering_health
    from hst.publish import read_headlines
    from hst.tracking import track_clusters

    settings = load_settings(args.settings)
    embedding_ids = json.loads(
        Path(f"{args.embeddings}.ids.json").read_text(encoding="utf-8")
    )
    records_by_id = {
        record["id"]: record
        for record in read_headlines(args.data_dir)
        if record["lang"] == "en"
    }
    missing_ids = [headline_id for headline_id in embedding_ids if headline_id not in records_by_id]
    if missing_ids:
        raise ValueError(f"{len(missing_ids)} embedded headlines are missing from storage")
    english = [records_by_id[headline_id] for headline_id in embedding_ids]
    ids = embedding_ids
    embeddings = load_embeddings(ids, args.embeddings)
    timestamps = [
        datetime.fromisoformat(
            (record["published_at"] or record["first_seen_at"]).replace("Z", "+00:00")
        )
        for record in english
    ]
    candidates = find_candidates(
        embeddings,
        timestamps,
        ids,
        k=settings["blocking_k"],
        window_hours=settings["window_hours"],
    )
    clusters = cluster_headlines(ids, candidates, settings["tau_baseline"])
    tracked, _state = track_clusters(
        clusters,
        args.state,
        settings["id_match_min_jaccard"],
        death_hours=settings["story_death_hours"],
        first_seen_by_id={record["id"]: record["first_seen_at"] for record in english},
    )
    health_path = Path(args.data_dir) / "state" / "health.json"
    if health_path.exists():
        update_clustering_health(args.data_dir, clusters, _state["id_persistence"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps({
            "clusters": clusters,
            "tracked": tracked,
            "candidates": candidates,
        }),
        encoding="utf-8",
    )
    print(f"cluster: {len(clusters)} stories")
    return 0


def run_publish_command(args: argparse.Namespace) -> int:
    import json

    from hst.embed import load_embeddings
    from hst.publish import publish

    result = json.loads(Path(args.cluster_output).read_text(encoding="utf-8"))
    embedding_ids = json.loads(
        Path(f"{args.embeddings}.ids.json").read_text(encoding="utf-8")
    )
    embeddings = load_embeddings(embedding_ids, args.embeddings)
    payload = publish(
        args.data_dir,
        args.site_dir,
        result["clusters"],
        result["tracked"],
        embeddings,
        embedding_ids=embedding_ids,
    )
    print(f"publish: {payload['n_stories']} stories")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m hst")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "embed", "cluster", "publish", "labeling", "eval"):
        subparser = subparsers.add_parser(name)
        subparser.set_defaults(handler=stub_command(name))
    ingest_parser = subparsers.choices["ingest"]
    ingest_parser.add_argument("--prune", action="store_true")
    ingest_parser.add_argument("--data-dir", default="data")
    ingest_parser.add_argument("--sources", default="config/sources.yaml")
    ingest_parser.add_argument("--settings", default="config/settings.yaml")
    ingest_parser.set_defaults(handler=run_ingest_command)
    embed_parser = subparsers.choices["embed"]
    embed_parser.add_argument("--data-dir", default="data")
    embed_parser.add_argument("--cache", default="data/state/embedding_cache.json")
    embed_parser.add_argument("--output", default="data/state/embeddings.npy")
    embed_parser.add_argument("--settings", default="config/settings.yaml")
    embed_parser.set_defaults(handler=run_embed_command)
    cluster_parser = subparsers.choices["cluster"]
    cluster_parser.add_argument("--data-dir", default="data")
    cluster_parser.add_argument("--embeddings", default="data/state/embeddings.npy")
    cluster_parser.add_argument("--state", default="data/state/stories_state.json")
    cluster_parser.add_argument("--settings", default="config/settings.yaml")
    cluster_parser.add_argument("--output", default="data/state/clusters.json")
    cluster_parser.set_defaults(handler=run_cluster_command)
    publish_parser = subparsers.choices["publish"]
    publish_parser.add_argument("--data-dir", default="data")
    publish_parser.add_argument("--embeddings", default="data/state/embeddings.npy")
    publish_parser.add_argument("--cluster-output", default="data/state/clusters.json")
    publish_parser.add_argument("--site-dir", default="site")
    publish_parser.set_defaults(handler=run_publish_command)
    return parser


def run_ingest_command(args: argparse.Namespace) -> int:
    from hst.config import load_settings
    from hst.ingest import run_ingest

    settings = load_settings(args.settings)
    health = run_ingest(
        args.data_dir,
        args.sources,
        args.prune,
        settings["retention_days"],
    )
    print(
        f"ingest: {health['run']['n_new_headlines']} new headlines "
        f"from {health['run']['n_feeds_ok']}/{health['run']['n_feeds_total']} feeds"
    )
    return 0


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
