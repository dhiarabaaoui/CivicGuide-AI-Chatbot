import unittest

from scripts.conversation_state import (
    build_train_transition_graph,
    classify_question_mode,
    resolve_conversation_state,
)


def reference(span_id: str) -> dict:
    return {"span_id": span_id}


def runtime_example(agent_text: str, response_act: str, response_text: str = "No") -> dict:
    return {
        "conversation_id": "conv_validation",
        "domain": "ssa",
        "history": [
            {
                "turn_id": "agent_1",
                "role": "agent",
                "dialogue_act": "query_condition",
                "text": agent_text,
            }
        ],
        "query": {
            "turn_id": "user_2",
            "role": "user",
            "dialogue_act": response_act,
            "text": response_text,
        },
    }


class ConversationStateTests(unittest.TestCase):
    def test_negative_service_offer_is_terminal_abstention(self):
        example = runtime_example(
            "Would you like to learn about other retirement planning websites then?",
            "response_negative",
        )
        source_turns = {"agent_1": {"references": [reference("span_offer")]}}
        state = resolve_conversation_state(example, source_turns, {}, "ask_followup", 1, 0.75, 0.5)
        self.assertEqual(
            classify_question_mode(example["history"][0]["text"]),
            "generic_continuation_offer",
        )
        self.assertEqual(state["resolved_action"], "abstain")
        self.assertEqual(state["resolution_source"], "negative_service_offer_terminal")
        self.assertTrue(state["override_applied"])

    def test_negative_specific_option_offer_is_not_forced_terminal(self):
        example = runtime_example("Would you like to see a Map of Regions?", "response_negative")
        source_turns = {"agent_1": {"references": [reference("span_map")]}}
        state = resolve_conversation_state(example, source_turns, {}, "ask_followup", 1, 0.75, 0.5)
        self.assertEqual(state["interaction_mode"], "specific_option_offer")
        self.assertEqual(state["resolved_action"], "ask_followup")
        self.assertFalse(state["override_applied"])

    def test_generic_offer_without_explicit_closure_is_not_forced_terminal(self):
        example = runtime_example("Do you want more information?", "response_negative", "No")
        source_turns = {"agent_1": {"references": [reference("span_more_info")]}}
        state = resolve_conversation_state(example, source_turns, {}, "ask_followup", 1, 0.75, 0.5)
        self.assertEqual(state["interaction_mode"], "generic_continuation_offer")
        self.assertEqual(state["resolved_action"], "ask_followup")
        self.assertFalse(state["override_applied"])

    def test_exact_checklist_transition_selects_unvisited_next_condition(self):
        example = runtime_example("Are you currently enrolled?", "response_positive", "Yes")
        source_turns = {"agent_1": {"references": [reference("span_condition_1")]}}
        graph = {
            ("span_condition_1", "response_positive"): {
                "support": 4,
                "predicted_action": "ask_followup",
                "action_purity": 1.0,
                "target_candidates": [
                    {"span_id": "span_condition_2", "count": 4, "purity": 1.0}
                ],
            }
        }
        state = resolve_conversation_state(example, source_turns, graph, "answer", 2, 0.9, 0.75)
        self.assertEqual(state["resolved_action"], "ask_followup")
        self.assertEqual(state["candidate_next_condition_span_id"], "span_condition_2")
        self.assertEqual(state["resolution_source"], "explicit_checklist_transition")

    def test_visited_only_followup_target_does_not_override(self):
        example = {
            "conversation_id": "conv_validation",
            "domain": "va",
            "history": [
                {
                    "turn_id": "agent_1",
                    "role": "agent",
                    "dialogue_act": "query_condition",
                    "text": "Condition one?",
                },
                {
                    "turn_id": "user_2",
                    "role": "user",
                    "dialogue_act": "response_positive",
                    "text": "Yes",
                },
                {
                    "turn_id": "agent_3",
                    "role": "agent",
                    "dialogue_act": "query_condition",
                    "text": "Condition two?",
                },
            ],
            "query": {
                "turn_id": "user_4",
                "role": "user",
                "dialogue_act": "response_positive",
                "text": "Yes",
            },
        }
        source_turns = {
            "agent_1": {"references": [reference("span_condition_1")]},
            "agent_3": {"references": [reference("span_condition_2")]},
        }
        graph = {
            ("span_condition_2", "response_positive"): {
                "support": 3,
                "predicted_action": "ask_followup",
                "action_purity": 1.0,
                "target_candidates": [
                    {"span_id": "span_condition_1", "count": 3, "purity": 1.0}
                ],
            }
        }
        state = resolve_conversation_state(example, source_turns, graph, "ask_followup", 1, 0.75, 0.5)
        self.assertEqual(state["resolved_action"], "answer")
        self.assertEqual(state["resolution_source"], "checklist_exhausted")
        self.assertEqual(
            state["visited_condition_span_ids"],
            ["span_condition_1", "span_condition_2"],
        )

    def test_train_graph_keeps_ordered_target_candidates(self):
        conversation = {
            "split": "train",
            "turns": [
                {
                    "index": 1,
                    "role": "agent",
                    "dialogue_act": "query_condition",
                    "utterance": "First condition?",
                    "references": [reference("span_1")],
                },
                {
                    "index": 2,
                    "role": "user",
                    "dialogue_act": "response_positive",
                    "utterance": "Yes",
                    "references": [],
                },
                {
                    "index": 3,
                    "role": "agent",
                    "dialogue_act": "query_condition",
                    "utterance": "Second condition?",
                    "references": [reference("span_2")],
                },
            ],
        }
        graph = build_train_transition_graph([conversation])
        node = graph[("span_1", "response_positive")]
        self.assertEqual(node["predicted_action"], "ask_followup")
        self.assertEqual(node["target_candidates"][0]["span_id"], "span_2")


if __name__ == "__main__":
    unittest.main()
