# CivicGuide AI — Portfolio case study

## One-minute overview

CivicGuide AI is a production-oriented retrieval-augmented generation system
for questions about US public services. It searches a frozen corpus of official
documents, selects traceable evidence, plans the claims an answer needs and
returns a cited response only after deterministic validation. The public demo
is backed by FastAPI, deployed on Vercel and independently reproducible through
a hardened Docker image.

## Problem

Administrative information is fragmented across long official pages, and a
generic language model can answer fluently without preserving the exact scope
or conditions in those pages. The engineering objective was therefore not only
to retrieve a relevant document, but to produce a useful conversational answer
whose factual claims remain connected to inspectable source spans.

## Selected architecture

1. A conversation router chooses `answer`, `ask_followup` or `abstain`.
2. BM25 finds exact terminology while OpenAI embeddings find semantic matches.
3. Reciprocal Rank Fusion combines the two rankings without mixing incompatible
   raw scores.
4. An evidence planner converts retrieved sections into short, traceable spans.
5. A claim planner defines which facts may appear and which evidence supports
   each fact.
6. A realizer writes the answer, after which deterministic checks validate the
   contract, evidence identifiers and inline citations.

The deployed candidate and its four runtime artifacts are frozen by SHA-256 so
that evaluation, Docker and Vercel use the same data.

## Engineering decisions

- **Hybrid over dense-only retrieval:** the measured hybrid configuration
  improved Recall@10 and MRR@10 over both standalone retrievers.
- **Evidence planning over prompt-only fixes:** error analysis showed that most
  failures came from missing evidence or the wrong action, not writing style.
- **Deterministic gates around generation:** invalid evidence identifiers and
  incomplete claim realizations are rejected before public exposure.
- **Three active prompts only:** rejected prompt variants and redundant
  experiments were removed from the active path.
- **Privacy-safe observability:** operational logs contain status, latency,
  counts and cost estimates, never the user's question or generated answer.

## Production evidence

| Area | Verified result |
|---|---|
| Automated quality | 59 tests passing in GitHub Actions |
| Supply chain | 0 known dependency vulnerabilities |
| Artifact integrity | 4 SHA-256-verified runtime artifacts |
| Container | Non-root, read-only and Docker `healthy` |
| Load check | 40/40 lexical requests, 0 errors, p95 376 ms |
| Deployment | Native Git-to-Vercel production delivery |
| Operations | Readiness checked post-CI and twice per hour |
| Abuse control | 10 requests per IP per 60 seconds on paid routes |

## What was difficult

The central difficulty was preserving multi-span scope: list introductions,
conditions and follow-up branches can be separated across chunks. Improvements
therefore moved beyond prompt tuning into source-order reconstruction, atomic
evidence selection and explicit conversation-state transitions. A second
challenge was reproducibility across Windows, Linux, Docker and serverless
environments; canonical line endings, pinned dependencies and artifact hashes
made the same candidate verifiable everywhere.

## Honest limitations

The system is deployed as an engineering demonstrator, not as an authoritative
government service. A reused 32-case regression reached 84.38%, while a separate
80-case automatically scored set remained materially below the desired quality
targets. The planned independent double annotation and adjudication protocol
was not completed. The interface therefore labels generated content as
experimental and directs users to the official cited sources.

## Two-minute interview narrative

> I built CivicGuide AI to make official US public-service documentation easier
> to navigate without hiding the risks of generated answers. I evaluated
> lexical, dense and hybrid retrieval, selected an RRF-based hybrid pipeline,
> and then added atomic evidence and claim planning because error analysis showed
> that missing evidence was a larger problem than wording. The runtime routes
> each turn between answering, clarifying and abstaining, and deterministic gates
> validate evidence IDs and citations before a response is shown. I packaged the
> frozen candidate in a non-root Docker image, added GitHub CI, dependency and
> artifact audits, Vercel CD, structured logs, rate limiting, health probes,
> monitoring and rollback documentation. I also kept the evaluation limitations
> visible: this is a production-engineered portfolio system, not an official or
> fully human-validated government assistant.

## CV-ready bullets

- Built and deployed a cited RAG assistant using FastAPI, BM25, OpenAI
  embeddings, RRF fusion and evidence-constrained generation across 1,273
  indexed chunks.
- Productionized the service with a hardened non-root Docker image, GitHub
  Actions CI, native Vercel CD, health probes, structured logging, rate limiting
  and automated post-deployment monitoring.
- Implemented reproducible ML evaluation and release controls with frozen
  SHA-256 artifacts, 59 automated tests, dependency auditing, cost ceilings and
  documented rollback procedures.
