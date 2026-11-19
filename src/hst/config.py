from pathlib import Path
from typing import Any

import yaml

SEED = 1337


def load_settings(path: str | Path = "config/settings.yaml") -> dict[str, Any]:
    settings_path = Path(path)
    with settings_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if data.get("seed") != SEED:
        raise ValueError(f"settings.yaml seed must be {SEED}")
    return data
