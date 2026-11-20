# Changelog

## 2026-11-19 - Phase 0

- Added the repository scaffold, schema models, CLI stubs, placeholder site, CI, and pipeline skeleton.

## 2026-11-19 - Phase 1

- Added RSS/Atom ingestion, cleaning, URL canonicalization, timestamp handling, duplicate tracking, retention, and health JSON.
- Added 25 candidate outlets with terms approval left unchecked.
- Added recorded feed fixtures and 56 passing tests.

## 2026-11-19 - Phase 2

- Added cosine embedding blocking, sparse average-linkage clustering, story tracking, and static publishing.
- Added a live story list and detail dialog.
- Changed the pipeline schedule to every 8 hours.
- Fixed pipeline package installation and verified a live Pages deployment.
- Moved the 8-hour schedule to minute 17 after no scheduled event appeared.
- Clamped cosine-derived distances to prevent floating-point scores above 1 from breaking clustering.
