from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceConfig(StrictModel):
    id: str
    name: str
    feed_url: str
    region: str
    category: Literal["wire", "national", "international", "regional", "specialty"]
    is_wire: bool
    language: str
    tos_note: str
    tos_checked: bool
    enabled: bool


class HeadlineRecord(StrictModel):
    id: str
    source_id: str
    title_raw: str
    title_clean: str
    url: str
    url_canonical: str
    published_at: str | None
    published_at_source: Literal["feed", "first_seen"]
    first_seen_at: str
    lang: str
    dup_of: str | None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if len(value) != 16:
            raise ValueError("id must contain 16 hexadecimal characters")
        int(value, 16)
        return value


class PairLabel(StrictModel):
    pair_id: str
    a_id: str
    b_id: str
    label: Literal[0, 1]
    ambiguous: bool
    label_source: Literal["human_story", "human_pair", "weak", "active"]
    round: int
    labeled_at: str

    @field_validator("b_id")
    @classmethod
    def validate_order(cls, value: str, info) -> str:
        a_id = info.data.get("a_id")
        if a_id is not None and a_id >= value:
            raise ValueError("a_id must be less than b_id")
        return value


class StoryHeadline(StrictModel):
    id: str
    title: str
    url: str
    source_id: str
    published_at: str | None


class StoryObject(StrictModel):
    story_id: str = Field(pattern=r"^s_[0-9a-f]{8}$")
    title: str
    keywords: list[str]
    first_seen: str
    last_seen: str
    n_headlines: int = Field(ge=0)
    n_sources: int = Field(ge=0)
    sources: list[str]
    hourly_counts: list[int] = Field(min_length=72, max_length=72)
    headlines: list[StoryHeadline] = Field(max_length=12)
    merged_from: list[str]
    split_from: str | None
    forecast: dict | None = None


class StoriesFile(StrictModel):
    generated_at: str
    window_hours: int
    model_version: str
    n_headlines_in_window: int = Field(ge=0)
    n_stories: int = Field(ge=0)
    stories: list[StoryObject]


class FeedHealth(StrictModel):
    id: str
    status: str
    http_status: int | None
    items: int = Field(ge=0)
    new_items: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    consecutive_failures: int = Field(ge=0)
    last_success: str | None


class HealthFile(StrictModel):
    generated_at: str
    run: dict
    feeds: list[FeedHealth]
    quality: dict
    clustering: dict


class MetricsFile(StrictModel):
    agreement: dict | None = None
    blocking_recall: dict | None = None
    pair_level: dict | None = None
    cluster_level: dict | None = None
    calibration: dict | None = None
    ablations: dict | None = None
    learning_curve: dict | None = None
    drift: dict | None = None
    live_audit: dict | None = None
