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

    from hst.embed import encode_titles
    from hst.publish import read_headlines

    records = read_headlines(args.data_dir)
    english = [record for record in records if record["lang"] == "en"]
    titles = [record["title_clean"] for record in english]
    embeddings = encode_titles(titles, args.cache)
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

    import numpy as np

    from hst.blocking import find_candidates
    from hst.cluster import cluster_headlines
    from hst.config import load_settings
    from hst.publish import read_headlines
    from hst.tracking import track_clusters

    settings = load_settings(args.settings)
    records = read_headlines(args.data_dir)
    english = [record for record in records if record["lang"] == "en"]
    ids = [record["id"] for record in english]
    embeddings = np.load(args.embeddings)
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
    )
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

    import numpy as np

    from hst.publish import publish

    result = json.loads(Path(args.cluster_output).read_text(encoding="utf-8"))
    embeddings = np.load(args.embeddings)
    payload = publish(
        args.data_dir,
        args.site_dir,
        result["clusters"],
        result["tracked"],
        embeddings,
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
    ingest_parser.set_defaults(handler=run_ingest_command)
    embed_parser = subparsers.choices["embed"]
    embed_parser.add_argument("--data-dir", default="data")
    embed_parser.add_argument("--cache", default="data/state/embedding_cache.json")
    embed_parser.add_argument("--output", default="data/state/embeddings.npy")
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
    from hst.ingest import run_ingest

    health = run_ingest(args.data_dir, args.sources, args.prune)
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
