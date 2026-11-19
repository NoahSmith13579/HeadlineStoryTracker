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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
