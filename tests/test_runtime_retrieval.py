from __future__ import annotations

import unittest

import numpy as np

from rag_runtime.retrieval import EvidenceIndex, HybridRetriever, tokenize
from rag_runtime.settings import load_settings


class RuntimeRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = load_settings()
        cls.retriever = HybridRetriever(cls.settings)
        cls.evidence = EvidenceIndex(cls.settings.path("documents"))

    def test_retained_artifacts_are_aligned(self) -> None:
        self.assertEqual(len(self.retriever.chunks), 1273)
        self.assertEqual(self.retriever.vectors.shape, (1273, 1536))

    def test_bm25_finds_vehicle_registration_evidence(self) -> None:
        hits, diagnostics = self.retriever.search(
            "What documents do I need to register a vehicle in New York?",
            domain="dmv",
        )
        self.assertEqual(diagnostics["mode"], "bm25")
        self.assertTrue(hits)
        titles = " ".join(str(hit.chunk["document_title"]) for hit in hits[:3]).casefold()
        self.assertIn("register", titles)

    def test_evidence_selection_returns_atomic_traceable_spans(self) -> None:
        query = "What documents do I need to register a vehicle?"
        hits, _ = self.retriever.search(query, domain="dmv")
        selected = self.evidence.select(query, hits, maximum=12)
        self.assertGreater(len(selected), 0)
        self.assertLessEqual(len(selected), 12)
        self.assertEqual([item["evidence_id"] for item in selected], [f"E{i}" for i in range(1, len(selected) + 1)])
        self.assertTrue(all(item["span_id"] in self.evidence.spans for item in selected))

    def test_zero_dimension_query_vector_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.retriever.search("vehicle", query_vector=np.zeros(1536, dtype=np.float32))

    def test_tokenizer_matches_ascii_word_contract(self) -> None:
        self.assertEqual(tokenize("CAFÉ / Vehicle-42"), ["caf", "vehicle", "42"])


if __name__ == "__main__":
    unittest.main()
