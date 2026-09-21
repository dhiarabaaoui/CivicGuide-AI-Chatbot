import unittest

from scripts.claim_grounding import (
    build_user_state,
    extract_personal_facts,
    fact_to_first_person,
    grounding_decision,
    infer_polarity,
    parse_conversation,
    question_to_user_fact,
    surface_fact_supported,
)


class ClaimGroundingTests(unittest.TestCase):
    def test_question_response_is_reconstructed_as_negative_fact(self):
        fact = question_to_user_fact(
            "Did your school close within 120 days after you withdrew?", "negative"
        )
        self.assertEqual(
            fact,
            "Your school close within 120 days after you withdrew did not happen.",
        )

    def test_user_state_contains_explicit_and_resolved_facts(self):
        turns = [
            {"role": "agent", "text": "Are you disabled?"},
            {"role": "user", "text": "No, I am not."},
        ]
        state = build_user_state(turns)
        self.assertIn("USER SAID: No, I am not.", state["premise"])
        self.assertIn("RESOLVED FACT: You are not disabled.", state["premise"])

    def test_prefaced_question_is_resolved(self):
        fact = question_to_user_fact(
            "Of course. Are you a New York state resident?", "positive"
        )
        self.assertEqual(fact, "You are a New York state resident.")

    def test_contracted_have_question_is_resolved(self):
        fact = question_to_user_fact(
            "You've already started receiving disability benefits?", "positive"
        )
        self.assertEqual(fact, "You have already started receiving disability benefits.")

    def test_generic_do_question_is_resolved(self):
        self.assertEqual(
            question_to_user_fact("Do you receive SSI?", "positive"),
            "You receive SSI.",
        )

    def test_named_condition_case_question_is_resolved(self):
        fact = question_to_user_fact(
            "Having an illness called a presumptive disease, is that your case?",
            "negative",
        )
        self.assertEqual(fact, "You do not have presumptive disease.")

    def test_causal_personal_fact_is_extracted(self):
        sentence = (
            "Because your school did not close within 120 days, "
            "that discharge condition is not met."
        )
        self.assertEqual(
            extract_personal_facts(sentence),
            ["your school did not close within 120 days"],
        )

    def test_general_instruction_is_not_personal_state(self):
        self.assertEqual(
            extract_personal_facts("You must contact your loan servicer immediately."),
            [],
        )

    def test_conditional_criterion_is_not_asserted_as_user_state(self):
        sentence = (
            "A discharge may apply only if you could not complete the program "
            "because your school closed."
        )
        self.assertEqual(extract_personal_facts(sentence), [])

    def test_policy_about_personal_account_is_not_biographical_state(self):
        sentence = "Your personal account is for your use only: only you can create it."
        self.assertEqual(extract_personal_facts(sentence), [])

    def test_assistant_fact_is_put_in_user_voice(self):
        self.assertEqual(
            fact_to_first_person("You are a New York State resident"),
            "I am a New York State resident",
        )

    def test_surface_support_ignores_temporal_filler(self):
        self.assertTrue(surface_fact_supported("you get benefits now", ["Yes, I get benefits."]))

    def test_surface_support_does_not_treat_request_as_personal_fact(self):
        self.assertFalse(
            surface_fact_supported(
                "you have to replace personalized plates",
                ["Tell me about how to replace personalized plates."],
            )
        )

    def test_explicit_negation_wins_over_leading_yes(self):
        self.assertEqual(infer_polarity("yes i don't have the inspection tag"), "negative")

    def test_low_entailment_blocks_public_exposure(self):
        decision = grounding_decision(
            [
                {
                    "fact": "you served at Camp Lejeune",
                    "entailment": 0.1,
                    "contradiction": 0.2,
                }
            ],
            minimum_entailment=0.6,
            maximum_contradiction=0.4,
        )
        self.assertEqual(decision["decision"], "block")
        self.assertEqual(decision["recommended_fallback"], "ask_followup_or_conditionalize")

    def test_conversation_parser_reads_only_tagged_turns(self):
        turns = parse_conversation(
            "prefix\n<CONVERSATION>\nUSER: Hello\nAGENT: Are you eligible?\n"
            "USER: No\n</CONVERSATION>\nsuffix"
        )
        self.assertEqual(len(turns), 3)
        self.assertEqual(turns[-1], {"role": "user", "text": "No"})


if __name__ == "__main__":
    unittest.main()
