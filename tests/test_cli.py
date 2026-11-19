import subprocess
import sys


def test_help_lists_subcommands():
    result = subprocess.run(
        [sys.executable, "-m", "hst", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for command in ("ingest", "embed", "cluster", "publish", "labeling", "eval"):
        assert command in result.stdout
