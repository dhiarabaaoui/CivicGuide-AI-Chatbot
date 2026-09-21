"""Explicit checklist state tracking for MultiDoc2Dial-style conversations.

The resolver uses only information that exists at runtime: dialogue acts, the assistant's own
previous citations, the current user response, and a transition graph learned from train.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable


RESPONSE_ACTS = {"response_positive", "response_negative"}


def expected_action(dialogue_act: str, text: str) -> str:
    if "no relevant information is found" in text.casefold():
        return "abstain"
    return "ask_followup" if dialogue_act == "query_condition" else "answer"


def response_polarity(dialogue_act: str) -> str:
    if dialogue_act == "response_positive":
        return "positive"
    if dialogue_act == "response_negative":
        return "negative"
    return "unknown"


def classify_question_mode(text: str) -> str:
    normalized = " ".join(text.casefold().split())
    if re.match(
        r"^(?:do you have|have you|are you|were you|did you|is your|was your|can you|is everything|is (?:it|that|this) all|it['â€™]?s all clear)\b",
        normalized,
    ):
        return "condition_check"
    generic_continuation_patterns = [
        r"\brelated information\b",
        r"\bother .{0,80}\bthen\b",
        r"\b(?:learn|hear|know) about other .{0,80}\b(?:websites|resources|information)\b",
        r"\banything else\b",
        r"\b(?:more|further|additional) (?:help|information)\b",
    ]
    if any(re.search(pattern, normalized) for pattern in generic_continuation_patterns):
        return "generic_continuation_offer"
    offer_patterns = [
        r"\bwould you like\b",
        r"\bdo you want (?:me )?to\b",
        r"\bdo you want to (?:know|learn|hear|see)\b",
        r"\bwould you want\b",
        r"\bshall i\b",
        r"\bcan i (?:help|tell|show|provide)\b",
        r"\bneed (?:any )?(?:more|further|other) (?:help|information)\b",
    ]
    if any(re.search(pattern, normalized) for pattern in offer_patterns):
        return "specific_option_offer"
    return "condition_check"


def is_explicit_terminal_decline(question: str, response: str) -> bool:
    normalized_question = " ".join(question.casefold().split())
    normalized_response = " ".join(response.casefold().split())
    strong_question_patterns = [
        r"\brelated information\b",
        r"\babout other .{0,80}\bthen\b",
    ]
    closure_response_patterns = [
        r"\b(?:that|this) (?:is|will be)? ?(?:enough|sufficient)\b",
        r"\bthat will suffice\b",
        r"\bi(?:'| a)?m good\b",
        r"\bi (?:do not|don't) need (?:any )?(?:more|further|additional)? ?(?:help|information)?\b",
        r"\bi (?:do not|don't) want to\b",
        r"\bi (?:think|believe) i (?:understand|understood|get|got)\b",
        r"\b(?:understood|understand) all (?:the )?(?:info|information)\b",
    ]
    return bool(
        any(re.search(pattern, normalized_question) for pattern in strong_question_patterns)
        or any(re.search(pattern, normalized_response) for pattern in closure_response_patterns)
    )


def turn_span_ids(turn: dict[str, Any] | None) -> list[str]:
    if not turn:
        return []
    return sorted(
        {
            str(reference["span_id"])
            for reference in turn.get("references", [])
            if reference.get("span_id")
        }
    )


def build_train_transition_graph(
    conversations: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    raw: dict[tuple[str, str], dict[str, Counter[str]]] = {}
    for conversation in conversations:
        if conversation.get("split") != "train":
            continue
        turns = sorted(conversation["turns"], key=lambda item: item["index"])
        for previous, user, target in zip(turns, turns[1:], turns[2:]):
            if previous["role"] != "agent" or user["role"] != "user" or target["role"] != "agent":
                continue
            if user.get("dialogue_act") not in RESPONSE_ACTS:
                continue
            previous_spans = turn_span_ids(previous)
            target_spans = turn_span_ids(target)
            action = expected_action(target["dialogue_act"], target["utterance"])
            for previous_span in previous_spans:
                node = raw.setdefault(
                    (previous_span, user["dialogue_act"]),
                    {"actions": Counter(), "target_spans": Counter()},
                )
                node["actions"][action] += 1
                node["target_spans"].update(target_spans)
    graph: dict[tuple[str, str], dict[str, Any]] = {}
    for key, node in raw.items():
        action, action_count = node["actions"].most_common(1)[0]
        support = sum(node["actions"].values())
        target_candidates = [
            {"span_id": span_id, "count": count, "purity": count / support}
            for span_id, count in node["target_spans"].most_common()
        ]
        graph[key] = {
            "support": support,
            "predicted_action": action,
            "action_purity": action_count / support,
            "target_candidates": target_candidates,
        }
    return graph


def reconstruct_resolved_conditions(
    example: dict[str, Any], source_turns: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    timeline = [*example.get("history", []), example["query"]]
    resolved: list[dict[str, Any]] = []
    for previous, current in zip(timeline, timeline[1:]):
        if previous.get("role") != "agent" or current.get("role") != "user":
            continue
        if current.get("dialogue_act") not in RESPONSE_ACTS:
            continue
        source_turn = source_turns.get(previous["turn_id"])
        resolved.append(
            {
                "condition_turn_id": previous["turn_id"],
                "condition_text": previous["text"],
                "condition_span_ids": turn_span_ids(source_turn),
                "response_turn_id": current["turn_id"],
                "response_text": current["text"],
                "polarity": response_polarity(current["dialogue_act"]),
            }
        )
    return resolved


def resolve_conversation_state(
    example: dict[str, Any],
    source_turns: dict[str, dict[str, Any]],
    graph: dict[tuple[str, str], dict[str, Any]],
    baseline_action: str,
    minimum_support: int,
    minimum_action_purity: float,
    minimum_target_purity: float,
) -> dict[str, Any]:
    resolved = reconstruct_resolved_conditions(example, source_turns)
    latest = resolved[-1] if resolved else None
    visited_spans = {
        span_id
        for item in resolved
        for span_id in item["condition_span_ids"]
    }
    state = {
        "schema_version": "1.0.0",
        "conversation_id": example["conversation_id"],
        "domain": example["domain"],
        "active_user_turn_id": example["query"]["turn_id"],
        "surface_act": example["query"]["dialogue_act"],
        "resolved_conditions": resolved,
        "visited_condition_span_ids": sorted(visited_spans),
        "current_condition": latest,
        "interaction_mode": (
            classify_question_mode(latest["condition_text"]) if latest else "new_request"
        ),
        "candidate_next_condition_span_id": None,
        "baseline_action": baseline_action,
        "resolved_action": baseline_action,
        "resolution_source": "baseline_fallback",
        "override_applied": False,
        "confidence": "low",
        "transition_support": 0,
        "transition_action_purity": 0.0,
        "transition_target_purity": 0.0,
    }
    if example["query"]["dialogue_act"] not in RESPONSE_ACTS or latest is None:
        return state

    if (
        state["interaction_mode"] == "generic_continuation_offer"
        and latest["polarity"] == "negative"
        and is_explicit_terminal_decline(
            latest["condition_text"], latest["response_text"]
        )
    ):
        state.update(
            {
                "resolved_action": "abstain",
                "resolution_source": "negative_service_offer_terminal",
                "override_applied": baseline_action != "abstain",
                "confidence": "high",
            }
        )
        return state

    candidates = [
        (span_id, graph[(span_id, example["query"]["dialogue_act"])])
        for span_id in latest["condition_span_ids"]
        if (span_id, example["query"]["dialogue_act"]) in graph
    ]
    if not candidates:
        return state
    previous_span, transition = max(
        candidates,
        key=lambda item: (
            item[1]["support"],
            item[1]["action_purity"],
            item[0],
        ),
    )
    state["transition_support"] = int(transition["support"])
    state["transition_action_purity"] = float(transition["action_purity"])
    if (
        transition["support"] < minimum_support
        or transition["action_purity"] < minimum_action_purity
    ):
        return state

    predicted_action = transition["predicted_action"]
    next_candidate = next(
        (
            item
            for item in transition["target_candidates"]
            if item["span_id"] not in visited_spans and item["purity"] >= minimum_target_purity
        ),
        None,
    )
    if predicted_action == "ask_followup" and next_candidate is None:
        state.update(
            {
                "resolved_action": "answer",
                "resolution_source": "checklist_exhausted",
                "override_applied": baseline_action != "answer",
                "confidence": "medium",
                "transition_previous_span_id": previous_span,
            }
        )
        return state
    if next_candidate:
        state["candidate_next_condition_span_id"] = next_candidate["span_id"]
        state["transition_target_purity"] = float(next_candidate["purity"])
    state.update(
        {
            "resolved_action": predicted_action,
            "resolution_source": "explicit_checklist_transition",
            "override_applied": predicted_action != baseline_action,
            "confidence": "high" if transition["action_purity"] >= 0.9 else "medium",
            "transition_previous_span_id": previous_span,
        }
    )
    return state
