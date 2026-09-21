# CivicGuide AI — Operations Runbook

This runbook describes how the public portfolio service is built, observed,
deployed and recovered. It complements the system and data cards; it does not
change the project's experimental quality status.

## Operational architecture

- **Production:** FastAPI serverless function on Vercel.
- **Portable runtime:** hardened, non-root Docker image with a read-only
  filesystem and writable `/tmp` only.
- **CI/CD:** GitHub Actions runs linting, tests, artifact verification,
  dependency auditing and a container smoke test. Preview and production
  deployments are available after the protected Vercel secrets are configured.
- **Availability probe:** GitHub Actions checks production every 30 minutes.
- **Observability:** privacy-safe JSON events are written to standard output and
  can be queried in Vercel Logs.
- **Abuse protection:** the published Vercel firewall rule
  `Protect paid AI endpoints` limits `/v1/chat` and `/v1/retrieve` to 10
  requests per IP and per 60-second fixed window.

## Health endpoints

| Endpoint | Meaning | External dependency |
|---|---|---|
| `GET /health/live` | The web process can answer HTTP requests | None |
| `GET /health/ready` | Runtime artifacts load and generation configuration is visible | No paid call |
| `GET /health` | Backward-compatible alias for readiness | No paid call |

Readiness must report `status: ready`, 1,273 chunks and vectors `[1273, 1536]`.
`generation_available` must be `true` in production.

## Logs and privacy

Every HTTP response includes `X-Request-ID`. The application emits one JSON
`http_request` event with method, path, status and latency. A RAG result emits a
`rag_response` event with trace ID, outcome, citation count, validation-error
count, cache hits and estimated cost. Feedback emits `user_feedback` with the
trace ID and rating.

Questions, answers, evidence text, API keys and personal information are never
written to operational logs. Local JSONL logs use the same reduced schema. Set
`LOG_LEVEL` to control verbosity and `APP_ENV` outside Vercel.

## Service objectives and alerts

These are portfolio operating objectives rather than contractual SLAs:

- production readiness succeeds on every scheduled probe;
- HTTP 5xx rate remains below 2% over 15 minutes;
- lexical retrieval p95 remains below 2.5 seconds in the local load test;
- every generated answer stays below the configured per-request cost ceiling;
- repeated 429 responses indicate that the abuse-control rule is working.

The firewall rule was verified on 21 September 2026 with intentionally invalid,
no-cost requests: the first 10 reached FastAPI and returned validation status
422, while requests 11 and 12 were stopped at the edge with status 429. Keep
this rule enabled whenever the public demo can reach paid OpenAI endpoints.

GitHub marks the scheduled uptime workflow as failed when readiness is invalid.
Vercel Logs should be filtered by `event`, `status_code`, `trace_id` and
`duration_ms`. OpenAI usage must also be reviewed by project in the platform
usage dashboard because an application-level per-request ceiling is not a
monthly account ceiling.

## CI/CD configuration

The deployment jobs remain safely disabled until repository variable
`ENABLE_VERCEL_DEPLOYMENTS=true` is created. Configure these GitHub environment
secrets for both `preview` and `production`:

- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`

Pull requests then receive Preview deployments after quality and container
jobs pass. A push to `main` produces a Production build and verifies its
readiness endpoint, moves `civicguide-ai-chatbot.vercel.app` to that immutable
deployment and verifies the public alias again. Protect `main` and require the
`Quality and tests` and `Container build and smoke test` checks before merging.

## Container operations

```powershell
docker compose build
docker compose up -d
docker compose ps
docker compose logs --follow
```

Open `http://127.0.0.1:8000/health/ready`. The compose service reads
`OPENAI_API_KEY` from the local environment or `.env`; the secret is never
baked into the image.

Run the no-cost lexical load test against the container:

```powershell
python scripts/load_test.py --requests 40 --concurrency 4
```

## Release checklist

1. Confirm CI lint, tests, dependency audit and container smoke test are green.
2. Review the Preview deployment and its `/health/ready` response.
3. Confirm no secret or `.env` file appears in the deployment manifest.
4. Merge into `main`; allow the production deployment and smoke test to finish.
5. Inspect Vercel 5xx logs and the uptime workflow.
6. Ask one controlled real question only when a paid end-to-end check is needed.

## Incident and rollback procedure

1. **Detect:** record the UTC time, request ID, failing endpoint and deployment.
2. **Triage:** inspect Vercel logs without copying user content or secrets.
3. **Contain:** enable Vercel Attack Mode for abuse, disable generation, or
   rotate `OPENAI_API_KEY` when credential exposure is suspected.
4. **Restore:** on Hobby, run `vercel rollback` to return to the immediately
   previous production deployment, then `vercel rollback status`.
5. **Verify:** check `/health/live`, `/health/ready`, the homepage and one
   no-cost lexical retrieval request.
6. **Resolve:** fix through a pull request, require green CI, deploy Preview,
   then promote the corrected release.
7. **Document:** record cause, impact, corrective action and a regression test.

Do not test rollback on the public alias during a portfolio demonstration
without announcing the short interruption. Vercel keeps immutable deployment
history, while the frozen artifact manifest identifies the exact RAG candidate.
