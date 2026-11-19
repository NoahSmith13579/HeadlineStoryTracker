# Decisions

| Date | Decision | Reason |
|---|---|---|
| 2026-11-19 | Implement one phase at a time, beginning with Phase 0. | The specification requires human confirmation before advancing phases. |
| 2026-11-19 | Pin dependencies to versions available at scaffold time and use Python 3.11 for Actions. | The specification requires pinned dependencies; local verification is intended to remain compatible with Python 3.11. |
| 2026-11-19 | Leave `config/sources.yaml` empty until Phase 1. | The spec prohibits guessing feed URLs and defers verified source discovery to Phase 1. |
