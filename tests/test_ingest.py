from pathlib import Path

import feedparser
import httpx
import pytest

from hst.ingest import append_records, fetch_feed, prune_headlines, read_existing, run_ingest
from hst.schemas import HeadlineRecord, SourceConfig

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_source() -> SourceConfig:
    return SourceConfig(
        id="fixture",
        name="Fixture",
        feed_url="https://example.com/rss",
        region="test",
        category="national",
        is_wire=False,
        language="en",
        tos_note="",
        tos_checked=False,
        enabled=True,
    )


def test_rss_fixture_parses():
    parsed = feedparser.parse((FIXTURES / "sample-rss.xml").read_text())
    assert len(parsed.entries) == 1
    assert "Mayors" in parsed.entries[0].title


def test_atom_fixture_parses():
    parsed = feedparser.parse((FIXTURES / "sample-atom.xml").read_text())
    assert len(parsed.entries) == 1
    assert "LIVE:" in parsed.entries[0].title


def test_malformed_fixture_has_no_usable_entry():
    parsed = feedparser.parse((FIXTURES / "malformed.xml").read_text())
    assert not parsed.entries or not getattr(parsed.entries[0], "title", "")


def test_fetch_feed_handles_304():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["If-None-Match"] == '"etag"'
        return httpx.Response(304, headers={"ETag": '"etag"'})

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        status, content, headers = fetch_feed(
            client,
            fixture_source(),
            {"fixture": {"etag": '"etag"', "last_modified": None}},
        )
    assert status == 304
    assert content == b""
    assert headers["etag"] == '"etag"'


def test_append_is_idempotent(tmp_path):
    record = HeadlineRecord(
        id="0123456789abcdef",
        source_id="fixture",
        title_raw="A story",
        title_clean="A story",
        url="https://example.com/a",
        url_canonical="https://example.com/a",
        published_at=None,
        published_at_source="first_seen",
        first_seen_at="2026-11-19T16:00:00Z",
        lang="en",
        dup_of=None,
    )
    assert append_records(tmp_path, [record]) == 1
    assert append_records(tmp_path, [record]) == 0
    ids, _ = read_existing(tmp_path)
    assert ids == {"0123456789abcdef"}


def test_prune_removes_old_daily_files(tmp_path):
    headline_dir = tmp_path / "headlines"
    headline_dir.mkdir()
    old = headline_dir / "2000-01-01.jsonl"
    old.write_text("{}", encoding="utf-8")
    recent = headline_dir / "2026-11-19.jsonl"
    recent.write_text("{}", encoding="utf-8")
    assert prune_headlines(tmp_path, 60) == 1
    assert not old.exists()
    assert recent.exists()


def test_ingest_fails_if_no_feed_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr("hst.ingest.load_sources", lambda _path: [])
    with pytest.raises(RuntimeError, match="no enabled feeds succeeded"):
        run_ingest(tmp_path)
