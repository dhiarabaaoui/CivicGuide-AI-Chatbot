from __future__ import annotations

import unittest

from rag_runtime.generation import OpenAIGenerator
from rag_runtime.settings import load_settings


class RuntimeGenerationTests(unittest.TestCase):
    def test_three_stage_contract_can_assemble_a_grounded_answer_without_network(self) -> None:
        generator = OpenAIGenerator(load_settings())
        responses = iter(
            [
                {
                    "output_text": '{"answer_objective":"Explain the requirement","mandatory_claims":[{"claim_id":"C1","claim_text":"You must bring identification.","claim_type":"direct_answer","evidence_ids":["E1"]}]}'
                },
                {
                    "output_text": '{"realizations":[{"claim_id":"C1","sentence":"You must bring identification.","citations":[{"evidence_id":"E1","evidence_quote":"must bring identification"}]}]}'
                },
            ]
        )

        def fake_call(kind, body, budget):
            return next(responses), False

        generator._call = fake_call  # type: ignore[method-assign]
        prepared = {
            "example_id": "offline_test",
            "domain": "dmv",
            "required_action": "answer",
            "expected_status": "answered",
            "evidence_catalog": [
                {
                    "evidence_id": "E1",
                    "span_id": "span_test",
                    "document_id": "doc_test",
                    "section_id": "section_test",
                    "section_position": 0,
                    "text": "Applicants must bring identification.",
                    "priority_rank": 1,
                }
            ],
            "structural_evidence_chains": [],
            "request_body": {
                "input": "\n".join(
                    [
                        "<EXECUTION_PLAN>",
                        "REQUIRED_ACTION: answer",
                        "BRANCH_MODE: direct_answer",
                        "PRIMARY_EVIDENCE_ID: E1",
                        "</EXECUTION_PLAN>",
                    ]
                )
            },
        }
        result = generator.generate(prepared)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["payload"]["status"], "answered")
        self.assertEqual(result["payload"]["citations"][0]["evidence_id"], "E1")


if __name__ == "__main__":
    unittest.main()
