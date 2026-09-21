from scripts.conversation_control import (
    branch_mode,
    infer_semantic_polarity,
    is_terminal_conversation_decline,
    is_specific_option_decline,
    needs_response_polarity_clarification,
    should_revert_unsupported_answer_override,
    should_abstain_on_uncertain_negative_answer,
    should_promote_classifier_abstention,
    should_revert_unsupported_followup_override,
)


def test_conditional_text_does_not_become_negative_personal_fact() -> None:
    result = infer_semantic_polarity(
        "response_negative", "If the school closed within 120 days, what happens?"
    )
    assert result["polarity"] == "unknown"
    assert branch_mode("answer", result) == "direct_answer"


def test_explicit_short_negative_is_preserved() -> None:
    result = infer_semantic_polarity("response_negative", "Not")
    assert result["polarity"] == "negative"
    assert branch_mode("answer", result) == "conclude_negative_outcome"


def test_negation_wins_over_leading_yes() -> None:
    result = infer_semantic_polarity("response_positive", "Yes, I don't have one")
    assert result["polarity"] == "negative"


def test_explicit_positive_is_preserved() -> None:
    result = infer_semantic_polarity("response_positive", "Yes")
    assert result["polarity"] == "positive"
    assert branch_mode("answer", result) == "conclude_positive_outcome"


def test_natural_terminal_decline_is_detected() -> None:
    example = {
        "history": [
            {
                "role": "agent",
                "text": "Would you like any related information or help with anything else?",
            }
        ],
        "query": {
            "dialogue_act": "response_negative",
            "text": "No, I don't think so, thanks.",
        },
    }
    assert is_terminal_conversation_decline(example)


def test_negative_answer_to_specific_condition_is_not_terminal() -> None:
    example = {
        "history": [{"role": "agent", "text": "Were you discharged dishonorably?"}],
        "query": {"dialogue_act": "response_negative", "text": "No"},
    }
    assert not is_terminal_conversation_decline(example)


def test_narrow_answer_override_guard() -> None:
    prediction = {
        "selected_action": "answer",
        "classifier_override_applied": True,
        "baseline_action": "defer_for_clarification",
    }
    shortlist = {
        "primary_evidence_accepted": False,
        "shortlist": [{"action_compatibility": 0.19}],
    }
    polarity = {"polarity": "positive", "confidence": 1.0}
    assert should_revert_unsupported_answer_override(
        prediction, shortlist, polarity, 0.25
    )
    assert not should_revert_unsupported_answer_override(
        prediction, shortlist, polarity, 0.15
    )


def test_negative_reply_to_option_is_not_an_eligibility_failure() -> None:
    result = infer_semantic_polarity("response_negative", "not")
    assert branch_mode("answer", result, "specific_option_offer") == "acknowledge_declined_option"


def test_uncertain_negative_answer_can_abstain() -> None:
    prediction = {
        "selected_action": "answer",
        "baseline_action": "answer",
        "graph_override_applied": False,
        "classifier_override_applied": False,
        "classifier_probabilities": {"answer": 0.405, "abstain": 0.361},
    }
    polarity = {"polarity": "negative", "confidence": 1.0}
    assert should_abstain_on_uncertain_negative_answer(
        prediction, polarity, 0.45, 0.05
    )


def test_conditional_response_requests_clarification() -> None:
    example = {
        "history": [{"role": "agent", "text": "Did it close within 120 days?"}],
        "query": {
            "dialogue_act": "response_negative",
            "text": "if it closed within 120 days",
        },
    }
    assert needs_response_polarity_clarification(example)


def test_declined_specific_option_is_acknowledged_without_rag() -> None:
    example = {
        "history": [{"role": "agent", "text": "Do you want a birth certificate?"}],
        "query": {"dialogue_act": "response_negative", "text": "not"},
    }
    assert is_specific_option_decline(example)


def test_classifier_abstention_promotion_requires_margin() -> None:
    prediction = {
        "selected_action": "answer",
        "baseline_action": "answer",
        "classifier_action": "abstain",
        "graph_override_applied": False,
        "classifier_probabilities": {
            "abstain": 0.49,
            "answer": 0.34,
            "ask_followup": 0.17,
        },
    }
    assert should_promote_classifier_abstention(prediction, 0.45, 0.10)


def test_low_confidence_followup_override_can_revert_to_answer() -> None:
    prediction = {
        "selected_action": "ask_followup",
        "classifier_override_applied": True,
        "baseline_action": "defer_for_clarification",
        "classifier_confidence": 0.53,
    }
    shortlist = {"shortlist": [{"action_compatibility": 0.95}]}
    polarity = {"polarity": "negative"}
    assert should_revert_unsupported_followup_override(
        prediction, shortlist, polarity, 0.90, 0.60
    )
