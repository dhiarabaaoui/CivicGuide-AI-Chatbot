# Notebook retention policy

## Purpose

The active repository keeps a notebook when it satisfies at least one of these
conditions:

1. it produces an artifact consumed by a later retained notebook;
2. it demonstrates a measured improvement that influenced the selected design;
3. it records a material go/no-go decision needed to understand the final
   system;
4. it introduces an engineering pattern that remains in the deployed pipeline.

Large caches, downloaded models and repeated variants may remain archived even
when the explanatory notebook is retained. Keeping a notebook documents the
research decision; it does not mean that its candidate was deployed.

## Restored after the retention audit

The 17 October 2026 audit restored nine notebooks that had been removed too
aggressively during the 13 September cleanup:

| Notebook | Why it remains useful |
|---|---|
| `04_bm25_retrieval_evaluation.ipynb` | Establishes the lexical baseline and produces inputs loaded by notebook 06. |
| `05_openai_dense_retrieval_evaluation.ipynb` | Measures the dense baseline and produces the second input loaded by notebook 06. |
| `07_conditional_query_rewriting_evaluation.ipynb` | Demonstrates a large retrieval gain on context-dependent turns and feeds the reranking/context experiments. |
| `08_cross_encoder_reranking_evaluation.ipynb` | Establishes the L2 reranking incumbent and quantifies its quality/latency trade-off. |
| `08b_cross_encoder_l6_challenger.ipynb` | Supplies the ranking consumed by notebook 09 and documents why the higher-quality reranker was too expensive for the final runtime. |
| `10_grounded_generation_pilot.ipynb` | Demonstrates the first complete retrieval-to-cited-answer chain and produces the answers evaluated by notebook 11. |
| `11_answer_quality_evaluation.ipynb` | Shows that syntactically valid citations were not sufficient and supplies the quality report analysed by notebook 12. |
| `13_grounded_generation_prompt_v2_canary.ipynb` | Introduces atomic claims, a design that later evolved into the planner/reviewer/realizer pipeline. |
| `14_prompt_v2_quality_canary.ipynb` | Measures the improvement from v1 to v2 while recording the failed production gates and the decision not to scale v2. |

These notebooks are historical research evidence. The deployed runtime remains
defined by `configs/runtime.json`, `rag_runtime/` and the three active prompts
in `prompts/`.

## Intentionally archive-only

Three notebooks remain archive-only because they are repeated, narrowly
targeted evaluation loops or candidates that failed their integration gates:

| Notebook | Why it remains archived |
|---|---|
| `15_prompt_v3_failure_canary.ipynb` | Tuned only on 15 known v2 failures; it recovered 10 cases but still failed faithfulness, citation and hallucination gates, so the apparent gain was not independent. |
| `16_prompt_v4_failure_canary.ipynb` | Tuned on the five remaining known failures and recovered only one, below the required four-of-five gate. |
| `17_conversation_state_resolver.ipynb` | Failed contract validity, polarity, surface-act and MRR non-regression gates; the recorded recommendation was `do_not_integrate_resolver`. |

Their conclusions remain summarized in `GUIDE_MEMOIRE_PROJET.md`, and their
original notebooks and heavy artifacts remain recoverable from the sibling
archive `RAG Assistant - archive experiments 2026-09-13`.

## Reproducibility boundary

The repository retains lightweight notebooks and configurations, but generated
datasets, paid-model caches, downloaded reranker weights and bulky reports are
not duplicated in Git. Re-running some historical notebooks can therefore
require rebuilding upstream artifacts, downloading the pinned model revision or
making explicitly authorized paid API calls. Existing executed outputs remain
in the notebooks so the original measured result is auditable without repeating
those calls.
