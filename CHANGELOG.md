# Changelog

All notable changes to Chart Assistant are recorded here. This project follows Semantic Versioning.

## [1.0.1] - 2026-08-11

### Changed

- Reworked the main README around the problem, live demo, supported files, local setup, and limits.
- Added a stable public Vercel demo and verified the deployed Azure-backed answering flow.
- Corrected the Vercel Python build configuration and pinned the required `uv` version.

## [1.0.0] - 2026-08-11

### Added

- Patient-chart question answering with patient-scoped hybrid retrieval and exact source quotes.
- PDF, DOCX, TXT, and Markdown ingestion with real processing progress.
- Natural patient-reference resolution with safe clarification for missing or ambiguous references.
- Five prompt revisions, grounding validation, bounded repair, health diagnostics, and detailed logs.
- Automated tests, retrieval evaluations, Vercel preview deployments, and tag-based releases.
