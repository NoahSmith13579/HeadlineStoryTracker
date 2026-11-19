from pathlib import Path
from typing import Any

import yaml

from hst.schemas import SourceConfig

SEED = 1337


def load_settings(path: str | Path = "config/settings.yaml") -> dict[str, Any]:
    settings_path = Path(path)
    with settings_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if data.get("seed") != SEED:
        raise ValueError(f"settings.yaml seed must be {SEED}")
    return data


def load_sources(path: str | Path = "config/sources.yaml") -> list[SourceConfig]:
    sources_path = Path(path)
    with sources_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or []
    if isinstance(data, dict):
        data = data.get("sources", [])
    return [SourceConfig.model_validate(item) for item in data]
