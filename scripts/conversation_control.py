"""High-precision conversational controls used before RAG generation.

The MultiDoc2Dial dialogue-act label is useful supervision, but it is not always a
faithful semantic description of the utterance text.  Runtime decisions therefore
prefer explicit lexical evidence and treat conditional or ambiguous utterances as
unknown instead of manufacturing a personal fact.
"""

from __future__ import annotations

import re
from typing import Any


def normalize_utterance(text: str) -> str:
    return " ".join(
        re.sub(r"[^a-z0-9'\u2019\s]", " ", str(text).casefold()).split()
    )


def infer_semantic_polarity(dialogue_act: str, text: str) -> dict[str, Any]:
    """Return a conservative polarity estimate for a user response.

    Explicit text wins over the dataset label.  Conditional questions and hedged
    statements are deliberately left unknown because turning them into a yes/no
    premise is more harmful than asking or answering conservatively.
    """

    normalized = normalize_utterance(text)
    if not normalized:
        return {"polarity": "unknown", "source": "empty_text", "confidence": 0.0}

    conditional_patterns = [
        r"^(?:if|whether|what if)\b",
        r"\b(?:it depends|maybe|perhaps|not sure|don't know|do not know)\b",
    ]
    if any(re.search(pattern, normalized) for pattern in conditional_patterns):
        return {
            "polarity": "unknown",
            "source": "ambiguous_or_conditional_text",
            "confidence": 0.95,
        }

    negative_patterns = [
        r"^(?:no|nope|nah|not)\b",
        r"^(?:yes|yeah|yep)[, ]+(?:i |we )?(?:do not|don't|did not|didn't|have not|haven't|am not|are not|cannot|can't)\b",
        r"^(?:i |we )?(?:do not|don't|did not|didn't|have not|haven't|am not|are not|cannot|can't)\b",
        r"\b(?:i |we )?(?:do not|don't) think so\b",
    ]
    if any(re.search(pattern, normalized) for pattern in negative_patterns):
        return {"polarity": "negative", "source": "explicit_negative_text", "confidence": 1.0}

    positive_patterns = [
        r"^(?:yes|yeah|yep|sure|correct|exactly|indeed)\b",
        r"^(?:i |we )?(?:do|did|have|am|are|can)\b",
    ]
    if any(re.search(pattern, normalized) for pattern in positive_patterns):
        return {"polarity": "positive", "source": "explicit_positive_text", "confidence": 1.0}

    # A dialogue-act fallback is retained only as a weak signal.  The branch mode
    # remains direct_answer unless confidence is high, so it cannot create a fact.
    if dialogue_act in {"response_positive", "response_negative"}:
        return {
            "polarity": "unknown",
            "source": f"unverified_dataset_label:{dialogue_act}",
            "confidence": 0.25,
        }
    return {"polarity": "not_applicable", "source": "non_response_turn", "confidence": 1.0}


def previous_agent_utterance(example: dict[str, Any]) -> str:
    return next(
        (
            str(turn.get("text", ""))
            for turn in reversed(example.get("history", []))
            if turn.get("role") in {"assistant", "agent"}
        ),
        "",
    )


def classify_previous_agent_mode(text: str) -> str:
    normalized = normalize_utterance(text)
    generic_patterns = [
        r"\banything else\b",
        r"\bany (?:other|more) questions?\b",
        r"\bwould you like (?:to (?:get|know|hear|learn)|some|any).{0,50}\b(?:related|more|further|additional) information\b",
        r"\bdo you want (?:some|any|to (?:get|know|hear|learn)).{0,50}\b(?:related|more|further|additional) information\b",
        r"\bother .{0,80}\b(?:resources|websites|information)\b",
        r"\b(?:help|assist) (?:you )?with anything else\b",
    ]
    if any(re.search(pattern, normalized) for pattern in generic_patterns):
        return "generic_continuation_offer"
    if re.search(
        r"^(?:do you want|would you like|would you want|shall i|can i (?:show|tell|provide|send))\b",
        normalized,
    ):
        return "specific_option_offer"
    return "condition_or_specific_offer"


def is_terminal_conversation_decline(
    example: dict[str, Any], polarity: dict[str, Any] | None = None
) -> bool:
    previous = previous_agent_utterance(example)
    if classify_previous_agent_mode(previous) != "generic_continuation_offer":
        return False
    polarity = polarity or infer_semantic_polarity(
        str(example.get("query", {}).get("dialogue_act", "")),
        str(example.get("query", {}).get("text", "")),
    )
    response = normalize_utterance(str(example.get("query", {}).get("text", "")))
    explicit_closure = any(
        re.search(pattern, response)
        for pattern in [
            r"\b(?:no|nope|nah)\b.*\b(?:thanks|thank you)\b",
            r"\b(?:that's|that is|this is) (?:all|enough|fine)\b",
            r"\bi(?:'m| am) good\b",
            r"\bi (?:do not|don't) need (?:anything|more|further|additional)\b",
            r"\bi (?:do not|don't) think so\b",
        ]
    )
    return polarity["polarity"] == "negative" or explicit_closure


def is_specific_option_decline(
    example: dict[str, Any], polarity: dict[str, Any] | None = None
) -> bool:
    previous = previous_agent_utterance(example)
    if classify_previous_agent_mode(previous) != "specific_option_offer":
        return False
    polarity = polarity or infer_semantic_polarity(
        str(example.get("query", {}).get("dialogue_act", "")),
        str(example.get("query", {}).get("text", "")),
    )
    return polarity["polarity"] == "negative" and polarity["confidence"] >= 0.9


def needs_response_polarity_clarification(
    example: dict[str, Any], polarity: dict[str, Any] | None = None
) -> bool:
    polarity = polarity or infer_semantic_polarity(
        str(example.get("query", {}).get("dialogue_act", "")),
        str(example.get("query", {}).get("text", "")),
    )
    previous = previous_agent_utterance(example).strip()
    return bool(
        polarity.get("source") == "ambiguous_or_conditional_text"
        and previous.endswith("?")
        and example.get("query", {}).get("dialogue_act")
        in {"response_positive", "response_negative"}
    )


def branch_mode(
    required_action: str,
    polarity: dict[str, Any],
    previous_mode: str = "condition_or_specific_offer",
) -> str:
    if required_action == "ask_followup":
        return "ask_next_unresolved_condition"
    if (
        previous_mode == "specific_option_offer"
        and polarity["confidence"] >= 0.9
        and polarity["polarity"] == "negative"
    ):
        return "acknowledge_declined_option"
    if polarity["confidence"] >= 0.9 and polarity["polarity"] == "negative":
        return "conclude_negative_outcome"
    if polarity["confidence"] >= 0.9 and polarity["polarity"] == "positive":
        return "conclude_positive_outcome"
    return "direct_answer"


def should_revert_unsupported_answer_override(
    prediction: dict[str, Any],
    shortlist: dict[str, Any],
    polarity: dict[str, Any],
    maximum_top_action_compatibility: float,
) -> bool:
    """Undo a narrow, empirically unsafe answer override.

    This is intentionally much narrower than general confidence gating: it only
    applies when the legacy planner deferred, the classifier alone forced an
    answer, the user explicitly said yes, and the retrieved top passage looks
    unlike answer evidence.  The threshold must be calibrated on development.
    """

    candidates = shortlist.get("shortlist", [])
    if not candidates:
        return False
    return bool(
        prediction.get("selected_action") == "answer"
        and prediction.get("classifier_override_applied")
        and prediction.get("baseline_action") == "defer_for_clarification"
        and polarity.get("polarity") == "positive"
        and not shortlist.get("primary_evidence_accepted")
        and float(candidates[0].get("action_compatibility", 1.0))
        <= float(maximum_top_action_compatibility)
    )


def should_abstain_on_uncertain_negative_answer(
    prediction: dict[str, Any],
    polarity: dict[str, Any],
    maximum_answer_probability: float,
    maximum_answer_abstain_margin: float,
) -> bool:
    """High-precision abstention for an unresolved negative branch."""
    probabilities = prediction.get("classifier_probabilities", {})
    answer_probability = float(probabilities.get("answer", 1.0))
    abstain_probability = float(probabilities.get("abstain", 0.0))
    return bool(
        prediction.get("selected_action") == "answer"
        and prediction.get("baseline_action") == "answer"
        and not prediction.get("graph_override_applied")
        and not prediction.get("classifier_override_applied")
        and polarity.get("polarity") == "negative"
        and answer_probability <= maximum_answer_probability
        and answer_probability - abstain_probability <= maximum_answer_abstain_margin
    )


def should_promote_classifier_abstention(
    prediction: dict[str, Any], minimum_probability: float, minimum_margin: float
) -> bool:
    probabilities = prediction.get("classifier_probabilities", {})
    abstain = float(probabilities.get("abstain", 0.0))
    answer = float(probabilities.get("answer", 0.0))
    followup = float(probabilities.get("ask_followup", 0.0))
    return bool(
        prediction.get("selected_action") == "answer"
        and prediction.get("baseline_action") == "answer"
        and prediction.get("classifier_action") == "abstain"
        and not prediction.get("graph_override_applied")
        and abstain >= minimum_probability
        and abstain - max(answer, followup) >= minimum_margin
    )


def should_revert_unsupported_followup_override(
    prediction: dict[str, Any],
    shortlist: dict[str, Any],
    polarity: dict[str, Any],
    minimum_top_action_compatibility: float,
    maximum_classifier_confidence: float,
) -> bool:
    candidates = shortlist.get("shortlist", [])
    return bool(
        candidates
        and prediction.get("selected_action") == "ask_followup"
        and prediction.get("classifier_override_applied")
        and prediction.get("baseline_action") == "defer_for_clarification"
        and polarity.get("polarity") == "negative"
        and float(candidates[0].get("action_compatibility", 0.0))
        >= minimum_top_action_compatibility
        and float(prediction.get("classifier_confidence", 1.0))
        <= maximum_classifier_confidence
    )
