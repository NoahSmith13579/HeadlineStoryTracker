# Headline Story Tracker

An auditable static news story tracker. Phase 0 provides the repository scaffold and placeholder deployment.

## Local setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m hst --help
pytest
ruff check .
```

## Development

The implementation is staged according to `Headline Story Tracker Agent Implementation Spec.md`.
Later phases add ingestion, clustering, evaluation, and the live site.
