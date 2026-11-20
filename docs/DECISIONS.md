# Decisions

| Date | Decision | Reason |
|---|---|---|
| 2026-11-19 | Implement one phase at a time, beginning with Phase 0. | The specification requires human confirmation before advancing phases. |
| 2026-11-19 | Pin dependencies to versions available at scaffold time and use Python 3.11 for Actions. | The specification requires pinned dependencies; local verification is intended to remain compatible with Python 3.11. |
| 2026-11-19 | Leave `config/sources.yaml` empty until Phase 1. | The spec prohibits guessing feed URLs and defers verified source discovery to Phase 1. |
| 2026-11-19 | Use 25 verified RSS feeds and keep all `tos_checked` values false. | The Phase 1 source gate requires official-feed verification while reserving legal/terms approval for the owner. |
| 2026-11-19 | Use `langdetect` with `SEED = 1337` for fixed-seed language detection. | It provides deterministic fixed-seed detection and marks malformed or short text as `und`. |
| 2026-11-19 | Schedule the pipeline every 8 hours rather than every 30 minutes. | This reflects the owner's lower-frequency operational preference while keeping scheduled workflows idempotent. |
