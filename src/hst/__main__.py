import argparse


def stub_command(name: str):
    def run(_args: argparse.Namespace) -> int:
        print(f"{name}: scaffold")
        return 0

    return run


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
    return parser


def run_ingest_command(args: argparse.Namespace) -> int:
    from hst.ingest import run_ingest

    health = run_ingest(args.data_dir, args.sources, args.prune)
    print(
        f"ingest: {health['run']['n_new_headlines']} new headlines "
        f"from {health['run']['n_feeds_ok']}/{health['run']['n_feeds_total']} feeds"
    )


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
