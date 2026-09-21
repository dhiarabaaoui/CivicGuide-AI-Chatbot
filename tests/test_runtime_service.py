from __future__ import annotations

import unittest
from types import SimpleNamespace

from rag_runtime.service import RAGRuntime


class RuntimeServiceTests(unittest.TestCase):
    def test_explicit_domain_detection_supports_french_questions(self) -> None:
        self.assertEqual(
            RAGRuntime._explicit_domain(
                "Comment demander une indemnisation d'invalidité auprès du VA ?"
            ),
            "va",
        )
        self.assertEqual(
            RAGRuntime._explicit_domain(
                "Comment remplir le formulaire FAFSA pour demander une aide fédérale ?"
            ),
            "studentaid",
        )

    def test_general_va_question_adds_retrieval_vocabulary(self) -> None:
        query = RAGRuntime._retrieval_query(
            "Comment demander une indemnisation d'invalidité auprès du VA ?",
            [],
            "va",
        )

        self.assertIn("how to prepare and file a claim", query)

    def test_domain_mismatch_stops_before_paid_retrieval(self) -> None:
        runtime = RAGRuntime.__new__(RAGRuntime)
        runtime.settings = SimpleNamespace(
            safeguards={"experimental_quality_warning": "warning"}
        )
        runtime._log = lambda row: None
        runtime.retrieve = lambda *args, **kwargs: self.fail("retrieval must not run")

        result = runtime.chat(
            "Comment remplir le formulaire FAFSA ?",
            domain="dmv",
        )

        self.assertEqual(result["status"], "clarification_required")
        self.assertEqual(result["suggested_domain"], "studentaid")
        self.assertEqual(result["cost_usd"], 0.0)
        self.assertEqual(result["citations"], [])

    def test_live_request_does_not_invent_a_gold_primary_evidence(self) -> None:
        runtime = RAGRuntime.__new__(RAGRuntime)
        prepared = runtime._prepare(
            "trace-test",
            "When can I renew my card?",
            [],
            [
                {
                    "evidence_id": "E1",
                    "span_id": "span-context",
                    "document_id": "doc-1",
                    "document_title": "Renewal guide",
                    "section_id": "section-context",
                    "section_title": "Special case",
                    "section_position": 0,
                    "parent_titles": [],
                    "priority_rank": 1,
                    "domain": "dmv",
                    "text": "Context about a special case.",
                },
                {
                    "evidence_id": "E2",
                    "span_id": "span-answer",
                    "document_id": "doc-1",
                    "document_title": "Renewal guide",
                    "section_id": "section-answer",
                    "section_title": "Renewal period",
                    "section_position": 0,
                    "parent_titles": [],
                    "priority_rank": 2,
                    "domain": "dmv",
                    "text": "You can renew one year before expiration.",
                },
            ],
            "dmv",
        )

        self.assertIn("PRIMARY_EVIDENCE_ID: NONE", prepared["request_body"]["input"])
        self.assertEqual(
            [item["evidence_id"] for item in prepared["evidence_catalog"]],
            ["E1", "E2"],
        )

    def test_guarded_failure_reports_embedding_cost_and_cache_hits(self) -> None:
        runtime = RAGRuntime.__new__(RAGRuntime)
        runtime.settings = SimpleNamespace(
            safeguards={"experimental_quality_warning": "warning"}
        )
        evidence = [
            {
                "evidence_id": "E1",
                "span_id": "span-1",
                "document_id": "doc-1",
                "document_title": "Document",
                "section_id": "section-1",
                "section_title": "Section",
                "section_position": 0,
                "parent_titles": [],
                "priority_rank": 1,
                "domain": "dmv",
                "text": "Relevant evidence.",
            }
        ]
        runtime.retrieve = lambda *args, **kwargs: {
            "evidence": evidence,
            "diagnostics": {
                "bm25_top_score": 1.0,
                "dense_top_similarity": None,
                "embedding": {"estimated_cost_usd": 0.002},
            },
        }
        runtime.generator = SimpleNamespace(
            available=True,
            generate=lambda prepared: {
                "payload": None,
                "errors": ["guard rejected output"],
                "cost_usd": 0.01,
                "cache_hits": ["planner"],
            },
        )
        runtime._log = lambda row: None

        result = runtime.chat("A sufficiently clear question")

        self.assertAlmostEqual(result["cost_usd"], 0.012)
        self.assertEqual(result["cache_hits"], ["planner"])

    def test_public_citations_are_deduplicated_by_evidence(self) -> None:
        runtime = RAGRuntime.__new__(RAGRuntime)
        evidence = [
            {
                "evidence_id": "E1",
                "text": "The complete source passage.",
                "span_id": "span-1",
                "document_id": "doc-1",
                "document_title": "Document",
                "section_title": "Section",
                "source_document_id": "source-1",
            }
        ]

        citations = runtime._citation_details(
            [
                {"evidence_id": "E1", "evidence_quote": "source"},
                {"evidence_id": "E1", "evidence_quote": "passage"},
            ],
            evidence,
        )

        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["evidence_quote"], "The complete source passage.")


if __name__ == "__main__":
    unittest.main()
