# Changelog

All notable changes to CivicGuide AI are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses semantic versioning for portfolio releases.

## [1.0.0] - 2026-09-21

### Added

- Public conversational interface for DMV, Social Security, Veterans Affairs
  and Federal Student Aid questions.
- Hybrid BM25 and dense retrieval fused with Reciprocal Rank Fusion.
- Atomic evidence planning, grounded generation and citation validation.
- Conversation routing between answer, clarification and safe abstention.
- Browser-local conversation history, source inspection and answer feedback.
- FastAPI liveness, readiness, retrieval, chat and feedback endpoints.
- Privacy-safe structured JSON logging and request correlation identifiers.
- Hardened non-root Docker image and Docker Compose runtime.
- GitHub Actions quality, test, dependency-audit and container gates.
- Native Vercel Git deployments, post-CI monitoring and scheduled uptime checks.
- Vercel rate limiting for paid AI routes and security response headers.
- Dependabot configuration for Python, Docker and GitHub Actions.
- Operational runbook covering release, incident response and rollback.

### Verified

- 59 automated tests pass.
- Four frozen runtime artifacts pass SHA-256 integrity checks.
- The dependency audit reports zero known vulnerabilities.
- The production container reaches Docker `healthy` status.
- A 40-request lexical load test completes without errors at p95 376 ms.
- Public readiness reports 1,273 chunks, vectors `[1273, 1536]` and generation
  availability.

### Known limitations

- Generated answers remain experimental and must be checked against the cited
  official sources.
- The strongest closed regression was reused during development and is not an
  independent production-quality estimate.
- The external human-validation protocol was not completed with the originally
  planned independent annotator and adjudicator roles.
- The service is a portfolio demonstrator, not an official government service.

[1.0.0]: https://github.com/dhiarabaaoui/CivicGuide-AI-Chatbot/releases/tag/v1.0.0
