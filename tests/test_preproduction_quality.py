import json
import unittest
from pathlib import Path

from scripts.build_planned_generation_contract import render_input
from scripts.evaluate_mandatory_claim_pipeline import (
    apply_runtime_checklist_continuation,
    apply_selected_primary_evidence,
    benchmark_expected_runtime_action,
    enrich_structural_evidence,
    evidence_selector_schema,
    is_fragmentary_evidence,
    normalized_deferral_action,
    planner_schema,
    realization_schema,
    sanitize_redundant_conflicting_claims,
    should_review_plan,
    structural_evidence_chains,
    validate_and_assemble,
    validate_conditional_qualifier_support,
    validate_plan,
    validate_primary_evidence_usage,
    validate_structural_claim_coverage,
)
from scripts.run_automated_end_to_end_evaluation import normalize, validate_generation


def generation_schema():
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["answered"]},
            "answer": {"type": "string"},
            "citations": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "evidence_id": {"type": "string", "enum": ["E1", "E2"]},
                        "evidence_quote": {"type": "string"},
                    },
                    "required": ["evidence_id", "evidence_quote"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["status", "answer", "citations"],
        "additionalProperties": False,
    }


class PreproductionQualityTests(unittest.TestCase):
    def test_independent_evidence_selector_schema_is_closed_to_supplied_ids(self):
        schema = evidence_selector_schema(["E1", "E2"])
        self.assertEqual(
            schema["properties"]["primary_evidence_id"]["enum"], ["E1", "E2"]
        )
        self.assertNotIn(
            "uniqueItems", schema["properties"]["complementary_evidence_ids"]
        )
        self.assertFalse(schema["additionalProperties"])

    def test_selected_primary_evidence_replaces_only_execution_anchor(self):
        prepared = {
            "request_body": {
                "input": "REQUIRED_ACTION: answer\nPRIMARY_EVIDENCE_ID: NONE\nDOMAIN: va"
            }
        }
        apply_selected_primary_evidence(prepared, "E7")
        self.assertIn("PRIMARY_EVIDENCE_ID: E7", prepared["request_body"]["input"])
        self.assertIn("REQUIRED_ACTION: answer", prepared["request_body"]["input"])

    def test_fragmentary_evidence_detects_incomplete_stem(self):
        self.assertTrue(is_fragmentary_evidence("Your COA is the estimate of"))
        self.assertTrue(is_fragmentary_evidence("Benefits may be paid to your:"))
        self.assertFalse(
            is_fragmentary_evidence("Benefits are reduced when claimed early.")
        )

    def test_fragmentary_selected_evidence_triggers_review_without_primary(self):
        prepared = {
            "required_action": "answer",
            "request_body": {"input": "PRIMARY_EVIDENCE_ID: NONE\nBRANCH_MODE: direct_answer"},
            "evidence_catalog": [
                {
                    "evidence_id": "E1",
                    "document_id": "D1",
                    "section_id": "S1",
                    "priority_rank": 1,
                    "text": "Your COA is the estimate of",
                },
                {
                    "evidence_id": "E2",
                    "document_id": "D1",
                    "section_id": "S1",
                    "priority_rank": 2,
                    "text": "tuition and fees",
                },
            ],
        }
        plan = {
            "mandatory_claims": [
                {
                    "claim_id": "C1",
                    "claim_text": "COA is an estimate.",
                    "claim_type": "definition",
                    "evidence_ids": ["E1"],
                }
            ]
        }
        self.assertTrue(should_review_plan(prepared, plan))

    def test_structural_chain_uses_source_order_not_retrieval_rank(self):
        catalog = [
            {
                "evidence_id": "E1", "document_id": "D1", "section_id": "S1",
                "section_position": 1, "priority_rank": 1, "text": "when you retire ;",
            },
            {
                "evidence_id": "E2", "document_id": "D1", "section_id": "S1",
                "section_position": 0, "priority_rank": 2,
                "text": "Benefits may be paid to your family:",
            },
            {
                "evidence_id": "E3", "document_id": "D1", "section_id": "S1",
                "section_position": 2, "priority_rank": 3,
                "text": "if you become disabled.",
            },
        ]
        chains = structural_evidence_chains(catalog)
        self.assertEqual(
            [[item["evidence_id"] for item in chain] for chain in chains],
            [["E2", "E1", "E3"]],
        )

    def test_legacy_contract_enrichment_triggers_partial_chain_review(self):
        prepared = {
            "required_action": "answer",
            "request_body": {
                "input": "PRIMARY_EVIDENCE_ID: NONE\nBRANCH_MODE: conclude_negative_outcome\n"
                "<EVIDENCE_UNTRUSTED_DATA>\n...\n</EVIDENCE_UNTRUSTED_DATA>"
            },
            "evidence_catalog": [
                {
                    "evidence_id": "E1", "span_id": "span_item", "document_id": "D1",
                    "section_id": "S1", "priority_rank": 1,
                    "text": "if you become disabled ;",
                },
                {
                    "evidence_id": "E2", "span_id": "span_stem", "document_id": "D1",
                    "section_id": "S1", "priority_rank": 2,
                    "text": "Benefits may be paid to your family:",
                },
            ],
        }
        count = enrich_structural_evidence(
            prepared,
            {
                "span_stem": {"section_position": 0},
                "span_item": {"section_position": 1},
            },
        )
        plan = {
            "mandatory_claims": [
                {
                    "claim_id": "C1", "claim_text": "Disability benefits apply.",
                    "claim_type": "direct_answer", "evidence_ids": ["E1"],
                }
            ]
        }
        self.assertEqual(count, 1)
        self.assertIn("E2[position=0] -> E1[position=1]", prepared["request_body"]["input"])
        self.assertTrue(should_review_plan(prepared, plan))
        errors = validate_structural_claim_coverage(plan, prepared)
        self.assertEqual(len(errors), 1)
        self.assertIn("require governing stem E2", errors[0])

        plan["mandatory_claims"][0]["evidence_ids"].append("E2")
        scope_errors = validate_structural_claim_coverage(plan, prepared)
        self.assertEqual(len(scope_errors), 1)
        self.assertIn("material scope term(s)", scope_errors[0])

        plan["mandatory_claims"][0]["claim_text"] = (
            "Disability benefits for your family do not apply."
        )
        self.assertEqual(validate_structural_claim_coverage(plan, prepared), [])

    def test_primary_evidence_is_a_deterministic_plan_invariant(self):
        prepared = {
            "request_body": {"input": "PRIMARY_EVIDENCE_ID: E1"},
            "evidence_catalog": [],
        }
        plan = {
            "mandatory_claims": [
                {"claim_id": "C1", "claim_text": "Question?", "evidence_ids": ["E2"]}
            ]
        }
        self.assertEqual(
            validate_primary_evidence_usage(plan, prepared),
            ["C1 must use PRIMARY_EVIDENCE_ID E1"],
        )
        plan["mandatory_claims"][0]["evidence_ids"].append("E1")
        self.assertEqual(validate_primary_evidence_usage(plan, prepared), [])

    def test_positive_checklist_reply_routes_to_first_unvisited_item(self):
        prepared = {
            "required_action": "answer",
            "expected_status": "answered",
            "request_body": {
                "input": "\n".join(
                    [
                        "REQUIRED_ACTION: answer",
                        "BRANCH_MODE: conclude_positive_outcome",
                        "USER_RESPONSE_POLARITY: positive",
                        "USER_RESPONSE_POLARITY_CONFIDENCE: 1.00",
                        "PREVIOUS_AGENT_UTTERANCE: Were released from active duty after 1951",
                        "PRIMARY_EVIDENCE_ID: NONE",
                        "<CONVERSATION>",
                        "AGENT: Were released from active duty after 1951",
                        "USER: yes",
                        "</CONVERSATION>",
                    ]
                )
            },
            "evidence_catalog": [
                {
                    "evidence_id": "E1", "document_id": "D1", "section_id": "S1",
                    "section_position": 0, "text": "You:",
                },
                {
                    "evidence_id": "E2", "document_id": "D1", "section_id": "S1",
                    "section_position": 1, "text": "Were released from active duty after 1951",
                },
                {
                    "evidence_id": "E3", "document_id": "D1", "section_id": "S1",
                    "section_position": 2, "text": "Were rated for a service-connected disability",
                },
            ],
        }
        self.assertTrue(apply_runtime_checklist_continuation(prepared))
        self.assertEqual(prepared["required_action"], "ask_followup")
        self.assertEqual(prepared["runtime_checklist_target_evidence_id"], "E3")
        self.assertIn("PRIMARY_EVIDENCE_ID: E3", prepared["request_body"]["input"])

    def test_unsupported_conditional_qualifier_is_rejected(self):
        prepared = {
            "request_body": {
                "input": "<CONVERSATION>\nUSER: My license is suspended.\n</CONVERSATION>"
            },
            "evidence_catalog": [
                {"evidence_id": "E1", "text": "You must surrender your plates."},
                {
                    "evidence_id": "E2",
                    "text": "A license becomes suspended when the lapse is 91 days or more.",
                },
            ],
        }
        plan = {
            "mandatory_claims": [
                {
                    "claim_id": "C1",
                    "claim_text": "If your license is suspended for a lapse of insurance, surrender your plates.",
                    "evidence_ids": ["E1", "E2"],
                }
            ]
        }
        errors = validate_conditional_qualifier_support(plan, prepared)
        self.assertEqual(len(errors), 1)
        self.assertIn("insurance", errors[0])
        plan["mandatory_claims"][0]["claim_text"] = (
            "If your license is suspended for a lapse, surrender your plates."
        )
        self.assertEqual(validate_conditional_qualifier_support(plan, prepared), [])

    def test_transitioned_deferral_is_scored_against_benchmark_action(self):
        example = {
            "target": {
                "dialogue_act": "respond_solution",
                "reference_answer": "ok",
            }
        }
        self.assertEqual(benchmark_expected_runtime_action(example), "answer")
        self.assertEqual(
            normalized_deferral_action("clarify_without_generation"), "ask_followup"
        )
        self.assertNotEqual(
            benchmark_expected_runtime_action(example),
            normalized_deferral_action("clarify_without_generation"),
        )

    def test_reserved_selection_includes_deterministic_deferrals(self):
        source = Path("scripts/evaluate_mandatory_claim_pipeline.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'row["example_id"] for row in [*all_requests, *all_deferrals]', source
        )
        self.assertIn(
            'row for row in all_deferrals if row["example_id"] in reserve_ids', source
        )

    def test_unsupported_acronym_definition_is_removed_when_supported_detail_remains(self):
        prepared = {
            "request_body": {
                "input": "BRANCH_MODE: conclude_positive_outcome\n<CONVERSATION>\nUSER: Yes\n</CONVERSATION>"
            },
            "evidence_catalog": [
                {"evidence_id": "E1", "text": "Your COA is the estimate of"},
                {"evidence_id": "E2", "text": "tuition and fees."},
            ],
        }
        plan = {
            "answer_objective": "Define COA.",
            "mandatory_claims": [
                {
                    "claim_id": "C1",
                    "claim_text": "COA stands for cost of attendance.",
                    "claim_type": "definition",
                    "evidence_ids": ["E1"],
                },
                {
                    "claim_id": "C2",
                    "claim_text": "Your COA estimates tuition and fees.",
                    "claim_type": "definition",
                    "evidence_ids": ["E1", "E2"],
                },
            ],
        }
        sanitized, removals = sanitize_redundant_conflicting_claims(plan, prepared)
        self.assertEqual(len(sanitized["mandatory_claims"]), 1)
        self.assertEqual(sanitized["mandatory_claims"][0]["claim_id"], "C1")
        self.assertEqual(removals[0]["unsupported_acronym_expansions"], ["COA"])

    def test_unicode_quote_normalization_is_stable(self):
        self.assertEqual(
            normalize("didn’t receive — a discharge"),
            normalize("didn't receive - a discharge"),
        )

    def test_invalid_extra_citation_is_removed_before_exposure(self):
        prepared = {
            "request_body": {"text": {"format": {"schema": generation_schema()}}},
            "evidence_catalog": [
                {"evidence_id": "E1", "text": "The exact supported statement."},
                {"evidence_id": "E2", "text": "Another source without an ellipsis."},
            ],
        }
        payload = {
            "status": "answered",
            "answer": "The exact supported statement.",
            "citations": [
                {"evidence_id": "E1", "evidence_quote": "exact supported statement"},
                {"evidence_id": "E2", "evidence_quote": "Another ... ellipsis"},
            ],
        }
        parsed, contract_valid, citations_valid, audit = validate_generation(
            prepared, {"output_text": json.dumps(payload)}
        )
        self.assertTrue(contract_valid)
        self.assertTrue(citations_valid)
        self.assertEqual(parsed["citations"], [payload["citations"][0]])
        self.assertEqual(audit, {"kept": 1, "dropped": 1})

    def test_negative_answer_branch_is_explicit_in_execution_plan(self):
        example = {
            "domain": "va",
            "query": {"text": "No", "dialogue_act": "response_negative"},
            "history": [{"role": "agent", "text": "Do you meet the requirement?"}],
        }
        rendered = render_input(example, "answer", "E1", "[E1] evidence", 12)
        self.assertIn("BRANCH_MODE: conclude_negative_outcome", rendered)
        self.assertIn("USER_RESPONSE_POLARITY: negative", rendered)
        self.assertIn(
            "PREVIOUS_AGENT_UTTERANCE: Do you meet the requirement?", rendered
        )

    def test_followup_branch_never_depends_on_response_polarity(self):
        example = {
            "domain": "ssa",
            "query": {"text": "Yes", "dialogue_act": "response_positive"},
            "history": [{"role": "agent", "text": "Do you receive benefits?"}],
        }
        rendered = render_input(example, "ask_followup", "E1", "[E1] evidence", 12)
        self.assertIn("BRANCH_MODE: ask_next_unresolved_condition", rendered)
        self.assertIn("REQUIRED_ACTION: ask_followup", rendered)

    def test_mandatory_claim_plan_requires_consecutive_claim_ids(self):
        schema = planner_schema(["E1", "E2"], 5)
        plan = {
            "answer_objective": "Explain the result and its mandatory surcharge.",
            "mandatory_claims": [
                {
                    "claim_id": "C1",
                    "claim_text": "The fine is between $50 and $100.",
                    "claim_type": "amount_or_date",
                    "evidence_ids": ["E1"],
                },
                {
                    "claim_id": "C3",
                    "claim_text": "A mandatory state surcharge is added.",
                    "claim_type": "consequence",
                    "evidence_ids": ["E2"],
                },
            ],
        }
        errors = validate_plan(plan, schema, "answer")
        self.assertTrue(any("consecutive" in error for error in errors))

    def test_public_answer_is_assembled_only_from_complete_claim_realizations(self):
        prepared = {
            "required_action": "answer",
            "evidence_catalog": [
                {"evidence_id": "E1", "text": "The fine is between $50 and $100."},
                {"evidence_id": "E2", "text": "A mandatory state surcharge is added."},
            ],
        }
        plan = {
            "answer_objective": "Explain both costs.",
            "mandatory_claims": [
                {
                    "claim_id": "C1",
                    "claim_text": "The fine is between $50 and $100.",
                    "claim_type": "amount_or_date",
                    "evidence_ids": ["E1"],
                },
                {
                    "claim_id": "C2",
                    "claim_text": "A mandatory state surcharge is added.",
                    "claim_type": "consequence",
                    "evidence_ids": ["E2"],
                },
            ],
        }
        schema = realization_schema(["C1", "C2"], ["E1", "E2"])
        realization = {
            "realizations": [
                {
                    "claim_id": "C1",
                    "sentence": "The fine is between $50 and $100.",
                    "citations": [
                        {"evidence_id": "E1", "evidence_quote": "between $50 and $100"}
                    ],
                },
                {
                    "claim_id": "C2",
                    "sentence": "A mandatory state surcharge is also added.",
                    "citations": [
                        {"evidence_id": "E2", "evidence_quote": "mandatory state surcharge"}
                    ],
                },
            ]
        }
        public, errors = validate_and_assemble(prepared, plan, realization, schema)
        self.assertEqual(errors, [])
        self.assertEqual(
            public["answer"],
            "The fine is between $50 and $100. A mandatory state surcharge is also added.",
        )
        self.assertEqual(len(public["citations"]), 2)

    def test_realizer_rejects_evidence_not_authorized_for_claim(self):
        prepared = {
            "required_action": "answer",
            "evidence_catalog": [
                {"evidence_id": "E1", "text": "Supported central result."},
                {"evidence_id": "E2", "text": "Unrelated downstream detail."},
            ],
        }
        plan = {
            "answer_objective": "State the central result.",
            "mandatory_claims": [
                {
                    "claim_id": "C1",
                    "claim_text": "State the central result.",
                    "claim_type": "direct_answer",
                    "evidence_ids": ["E1"],
                }
            ],
        }
        schema = realization_schema(["C1"], ["E1", "E2"])
        realization = {
            "realizations": [
                {
                    "claim_id": "C1",
                    "sentence": "Unrelated downstream detail.",
                    "citations": [
                        {"evidence_id": "E2", "evidence_quote": "Unrelated downstream detail"}
                    ],
                }
            ]
        }
        public, errors = validate_and_assemble(prepared, plan, realization, schema)
        self.assertIsNone(public)
        self.assertTrue(any("unauthorized_evidence_id" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
