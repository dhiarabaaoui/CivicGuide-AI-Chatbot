# CivicGuide AI — System Card

## Summary

CivicGuide AI is an experimental retrieval-augmented generation system for
questions about four US public-service domains: DMV, Social Security,
Veterans Affairs and federal student aid. It retrieves document passages,
selects atomic evidence, plans required claims and generates a cited answer.

This card describes the frozen runtime candidate
`rag_candidate_structural_v1_62383ac6c4ff` and the safeguards surrounding its
public portfolio demonstration.

## Intended use

- demonstrate an end-to-end, source-grounded RAG architecture;
- explore official public-service documents conversationally;
- inspect which passages support a generated answer;
- reproduce retrieval, planning, grounding and evaluation experiments.

## Out-of-scope use

- authoritative legal, financial, medical or benefits advice;
- emergency assistance;
- automated eligibility or entitlement decisions;
- replacing a government agency or qualified professional;
- processing confidential personal or case-specific records.

## System architecture

1. A conversation router selects `answer`, `ask_followup` or `abstain`.
2. BM25 and dense retrieval search 1,273 structure-aware document chunks.
3. Reciprocal Rank Fusion combines the two rankings.
4. Retrieved chunks are converted to short, traceable evidence spans.
5. A planner associates required claims with evidence identifiers.
6. A reviewer checks the plan when configured to do so.
7. A realizer writes only the supported claims.
8. Contract, grounding and citation validators accept or reject the output.

## Data

The research pipeline is based on MultiDoc2Dial. It includes dialogues grounded
in public documents across DMV, SSA, VA and StudentAid domains. Raw and
processed datasets are not committed. Frozen derived artifacts required by the
runtime are included with a SHA-256 manifest. See [DATA_CARD.md](DATA_CARD.md).

## Evaluation snapshot

| Evaluation | Result | Interpretation |
|---|---:|---|
| Closed 32-case regression | 84.38% overall success | Useful regression signal; reused during development |
| Closed regression faithfulness | 95.65% | Strong grounding on this reused set |
| Closed regression completeness | 86.96% | Not an independent estimate |
| Separate 80-case technical contract | 96.25% | Three invalid outputs used a safe fallback |
| Separate 80-case citation validity | 98.75% | Automated evaluator |
| Separate 80-case faithfulness | 98.75% | Automated evaluator |
| Separate 80-case acceptable correctness | 67.50% | Below the project target |
| Separate 80-case completeness | 33.75% | Main remaining limitation |
| Separate 80-case overall success | 33.75% | Fails the original production target |
| Public runtime smoke test | PASS | Four domain answers with citations + one clarification |

The 80-case scoring delivery was produced by an evaluator identified as AI, not
by two independent human annotators. It is therefore reported as automated
evaluation. A project-owner override allowed publication of the demonstrator,
but does not convert failed gates into successful validation.

## Safeguards

- inline citations linked to retrieved passages;
- deterministic schema and evidence-identifier checks;
- abstention when evidence is insufficient;
- explicit-domain mismatch detection;
- safe fallback for invalid generated output;
- maximum estimated cost per generated request;
- response cache for repeated requests;
- experimental-quality warning in the interface;
- reproducible frozen configuration and artifact hashes.

## Known limitations

- completeness is substantially weaker than citation validity;
- the corpus and source documents may become outdated;
- retrieval can miss the passage required for a complete answer;
- the router can choose an incorrect conversation action;
- multilingual questions may retrieve English evidence imperfectly;
- citations show provenance but do not guarantee that an answer is complete;
- serverless caches are ephemeral and are not shared across all instances;
- no independent dual-human validation and adjudication was completed.

## Release status

The application is a public experimental demonstrator. It is not described as
a production-validated public-service assistant. Changes to prompts, routing,
retrieval settings or runtime artifacts require a new candidate identifier and
a rerun of the automated checks.

