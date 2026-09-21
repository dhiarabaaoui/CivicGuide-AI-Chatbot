"""Build and validate Responses API-ready requests from locked RAG plans.

This stage performs no API call. It serializes the selected action and atomic evidence into
a strict structured-output contract, validates token budgets and hard regressions, and
creates a deterministic fallback artifact for planner deferrals.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tiktoken
from jsonschema import Draft202012Validator

try:
    from scripts.conversation_control import (
        branch_mode,
        classify_previous_agent_mode,
        infer_semantic_polarity,
        is_terminal_conversation_decline,
        is_specific_option_decline,
        needs_response_polarity_clarification,
        previous_agent_utterance,
        should_promote_classifier_abstention,
        should_revert_unsupported_answer_override,
        should_revert_unsupported_followup_override,
    )
except ModuleNotFoundError:
    from conversation_control import (  # type: ignore
        branch_mode,
        classify_previous_agent_mode,
        infer_semantic_polarity,
        is_terminal_conversation_decline,
        is_specific_option_decline,
        needs_response_polarity_clarification,
        previous_agent_utterance,
        should_promote_classifier_abstention,
        should_revert_unsupported_answer_override,
        should_revert_unsupported_followup_override,
    )
try:
    from scripts.evidence_anchor import numeric_anchors, promote_numeric_anchor_sources
except ModuleNotFoundError:
    from evidence_anchor import numeric_anchors, promote_numeric_anchor_sources  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "multidoc2dial_v1"
REPORT_ROOT = PROJECT_ROOT / "reports" / "generated" / "multidoc2dial_v1"
CONFIG_PATH = PROJECT_ROOT / "configs" / "planned_generation_contract.json"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def output_schema(
    config: dict[str, Any],
    evidence_ids: list[str],
    required_action: str,
    primary_evidence_accepted: bool,
) -> dict[str, Any]:
    if not evidence_ids:
        raise ValueError("Le schéma de génération exige au moins un evidence_id.")
    status_values = config["response_contract"]["statuses"]
    if primary_evidence_accepted:
        status_values = [
            "answered" if required_action == "answer" else "clarification_required"
        ]
    citation_schema: dict[str, Any] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "evidence_id": {"type": "string", "enum": evidence_ids},
                "evidence_quote": {"type": "string"},
            },
            "required": ["evidence_id", "evidence_quote"],
            "additionalProperties": False,
        },
    }
    if primary_evidence_accepted:
        citation_schema["minItems"] = 1
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": status_values},
            "answer": {"type": "string"},
            "citations": citation_schema,
        },
        "required": ["status", "answer", "citations"],
        "additionalProperties": False,
    }


def render_history(example: dict[str, Any], maximum_turns: int) -> str:
    turns = example.get("history", [])[-maximum_turns:]
    return "\n".join(
        f"{turn.get('role', 'unknown').upper()}: {turn.get('text', '')}" for turn in turns
    ) or "(no previous turns)"


def normalized_utterance(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def is_explicit_conversation_closure(
    example: dict[str, Any], policy: dict[str, Any]
) -> bool:
    if not policy.get("enabled", False):
        return False
    query = normalized_utterance(example["query"]["text"])
    if query not in {normalized_utterance(value) for value in policy["negative_responses"]}:
        return False
    history = example.get("history", [])
    previous_agent_text = next(
        (
            turn.get("text", "")
            for turn in reversed(history)
            if turn.get("role") in {"assistant", "agent"}
        ),
        "",
    )
    normalized_previous = normalized_utterance(previous_agent_text)
    return any(
        normalized_utterance(phrase) in normalized_previous
        for phrase in policy["continuation_phrases"]
    )


def render_evidence(sources: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    blocks: list[str] = []
    catalog: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        evidence_id = f"E{index}"
        text = source["text"].strip()
        blocks.append(
            "\n".join(
                [
                    f"[{evidence_id}]",
                    f"Document: {source['document_title']}",
                    f"Section: {source['title_path']}",
                    f"Section position: {source.get('section_position', 'unknown')}",
                    f"Span ID: {source['span_id']}",
                    f"Text: {text}",
                ]
            )
        )
        catalog.append(
            {
                "evidence_id": evidence_id,
                "span_id": source["span_id"],
                "document_id": source["document_id"],
                "section_id": source["section_id"],
                "text": text,
                "priority_rank": index,
                "section_position": source.get("section_position"),
                "evidence_origin": source.get("evidence_origin", "ranked"),
            }
        )
    return "\n\n---\n\n".join(blocks), catalog


def render_input(
    example: dict[str, Any], action: str, primary_evidence_id: str | None,
    evidence_text: str, history_maximum_turns: int,
    conversation_control: dict[str, Any] | None = None,
) -> str:
    control = conversation_control or infer_semantic_polarity(
        str(example.get("query", {}).get("dialogue_act", "")),
        str(example.get("query", {}).get("text", "")),
    )
    response_polarity = str(control["polarity"])
    previous_utterance = previous_agent_utterance(example) or "NONE"
    previous_mode = classify_previous_agent_mode(previous_utterance)
    resolved_branch_mode = branch_mode(action, control, previous_mode)
    return "\n".join(
        [
            "<EXECUTION_PLAN>",
            f"REQUIRED_ACTION: {action}",
            f"BRANCH_MODE: {resolved_branch_mode}",
            f"USER_RESPONSE_POLARITY: {response_polarity}",
            f"USER_RESPONSE_POLARITY_CONFIDENCE: {float(control['confidence']):.2f}",
            f"USER_RESPONSE_POLARITY_SOURCE: {control['source']}",
            f"PREVIOUS_AGENT_UTTERANCE: {previous_utterance}",
            f"PRIMARY_EVIDENCE_ID: {primary_evidence_id or 'NONE'}",
            f"DOMAIN: {example['domain']}",
            "</EXECUTION_PLAN>",
            "",
            "<CONVERSATION>",
            render_history(example, history_maximum_turns),
            f"USER: {example['query']['text']}",
            "</CONVERSATION>",
            "",
            "<EVIDENCE_UNTRUSTED_DATA>",
            evidence_text,
            "</EVIDENCE_UNTRUSTED_DATA>",
        ]
    )


def build_span_source_catalog() -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for document in iter_jsonl(DATASET_ROOT / "documents.jsonl"):
        for section in document["sections"]:
            parents = [
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in section.get("parent_titles", [])
            ]
            title_path = " > ".join(
                item for item in [*parents, section.get("title", "")] if item
            )
            spans = section["spans"]
            for position, span in enumerate(spans):
                neighbor_span_ids: list[str] = []
                for distance in range(1, 5):
                    for neighbor_position in (position - distance, position + distance):
                        if 0 <= neighbor_position < len(spans):
                            neighbor_span_ids.append(spans[neighbor_position]["id"])
                catalog[span["id"]] = {
                    "span_id": span["id"],
                    "document_id": document["id"],
                    "document_title": document["title"],
                    "domain": document["domain"],
                    "section_id": section["id"],
                    "section_title": section.get("title", ""),
                    "title_path": title_path,
                    "text": span.get("text", "").strip(),
                    "section_position": position,
                    "neighbor_span_ids": neighbor_span_ids,
                    "action_compatibility": 1.0,
                    "action_compatibility_source": "train_dialogue_transition_graph",
                }
    return catalog


def main() -> None:
    config = load_json(CONFIG_PATH)
    if config.get("locked_splits") != ["test"]:
        raise ValueError("Le split test doit rester verrouille.")
    prompt = (PROJECT_ROOT / config["prompt_path"]).read_text(encoding="utf-8").strip()
    benchmark_path = (
        DATASET_ROOT
        / "retrieval_benchmarks"
        / config["benchmark_experiment_id"]
        / f"{config['evaluation_split']}.jsonl"
    )
    legacy_predictions_path = (
        DATASET_ROOT
        / "answer_action_planner_experiments"
        / config["answer_action_planner_experiment_id"]
        / "evaluation"
        / "validation_predictions.jsonl"
    )
    tri_action_root = (
        DATASET_ROOT
        / "tri_action_planner_experiments"
        / config["tri_action_planner_experiment_id"]
    )
    tri_action_report_path = (
        REPORT_ROOT
        / "tri_action_planner_experiments"
        / config["tri_action_planner_experiment_id"]
        / "tri_action_planner_report.json"
    )
    shortlists_path = (
        DATASET_ROOT
        / "evidence_action_compatibility_experiments"
        / config["evidence_action_compatibility_experiment_id"]
        / "shortlists"
        / "validation_action_compatible_spans.jsonl"
    )
    transition_report_path = (
        REPORT_ROOT
        / "dialogue_transition_graph_experiments"
        / config["dialogue_transition_graph_experiment_id"]
        / "dialogue_transition_graph_report.json"
    )
    transition_report = load_json(transition_report_path)
    if not transition_report.get("gates_passed", False):
        raise RuntimeError("Le graphe de transitions selectionne ne passe pas ses gates.")
    tri_action_report = load_json(tri_action_report_path)
    if not tri_action_report.get("gates_passed", False):
        raise RuntimeError("Le planificateur tri-action selectionne ne passe pas ses gates.")
    action_controller_config = config.get("action_controller_v3", {})
    action_controller_report = None
    if action_controller_config.get("enabled", False):
        action_controller_report_path = (
            REPORT_ROOT
            / "action_controller_experiments"
            / action_controller_config["calibration_experiment_id"]
            / "report.json"
        )
        action_controller_report = load_json(action_controller_report_path)
        if not action_controller_report.get("gates_passed", False):
            raise RuntimeError("Le controleur d'action v3 selectionne ne passe pas ses gates.")
    benchmark = {row["id"]: row for row in iter_jsonl(benchmark_path)}
    legacy_predictions = {
        row["example_id"]: row for row in iter_jsonl(legacy_predictions_path)
    }
    predictions = {
        row["example_id"]: row
        for row in iter_jsonl(tri_action_root / "validation_tri_action_predictions.jsonl")
    }
    shortlists = {row["example_id"]: row for row in iter_jsonl(shortlists_path)}
    span_source_catalog = build_span_source_catalog()
    in_scope_prediction_ids = {
        example_id
        for example_id in predictions
        if example_id in benchmark and benchmark[example_id]["cohort"] == "direct_user_response"
    }
    excluded_out_of_scope_prediction_ids = sorted(set(predictions) - in_scope_prediction_ids)
    eligible_ids = sorted(
        set(shortlists) & set(benchmark) & set(predictions) & set(legacy_predictions)
    )
    if not eligible_ids:
        raise RuntimeError("Aucun plan accepte avec evidence atomique.")

    encoding = tiktoken.get_encoding(config["token_encoding"])
    generator = config["generator"]
    requests: list[dict[str, Any]] = []
    invalid_contracts = 0
    budget_violations = 0
    duplicate_evidence_ids = 0
    missing_evidence_texts = 0
    input_tokens: list[int] = []
    deferrals: list[dict[str, Any]] = []
    evidence_conflict_development_correct: list[bool] = []
    clarification_rescue_development_correct: list[bool] = []
    clarification_rescue_count = 0
    closure_ids: set[str] = set()
    transition_abstention_ids: set[str] = set()
    controller_abstention_ids: set[str] = set()
    ambiguous_response_clarification_ids: set[str] = set()
    specific_option_decline_ids: set[str] = set()
    transition_override_count = 0
    classifier_override_count = 0
    conversation_action_guard_count = 0
    classifier_abstention_promotion_count = 0
    unsupported_followup_reversion_count = 0
    transition_target_injection_count = 0
    numeric_anchor_promotion_count = 0
    for example_id in eligible_ids:
        example = benchmark[example_id]
        prediction = predictions[example_id]
        legacy_prediction = legacy_predictions[example_id]
        shortlist = shortlists[example_id]
        closure_policy = config["conversation_closure_policy"]
        conversation_control = infer_semantic_polarity(
            str(example.get("query", {}).get("dialogue_act", "")),
            str(example.get("query", {}).get("text", "")),
        )
        if closure_policy.get("enabled", False) and is_terminal_conversation_decline(
            example, conversation_control
        ):
            closure_ids.add(example_id)
            deferrals.append(
                {
                    "schema_version": "1.0.0",
                    "example_id": example_id,
                    "partition": prediction["partition"],
                    "domain": example["domain"],
                    "runtime_action": "close_conversation_without_generation",
                    "reason": "explicit_negative_response_to_continuation_offer",
                    "user_query": example["query"]["text"],
                    "deterministic_response": closure_policy["response"],
                }
            )
            continue
        if needs_response_polarity_clarification(example, conversation_control):
            ambiguous_response_clarification_ids.add(example_id)
            deferrals.append(
                {
                    "schema_version": "1.0.0",
                    "example_id": example_id,
                    "partition": prediction["partition"],
                    "domain": example["domain"],
                    "runtime_action": "clarify_without_generation",
                    "reason": "ambiguous_conditional_response_to_yes_no_question",
                    "user_query": example["query"]["text"],
                    "deterministic_response": (
                        "Could you clarify whether your answer to the previous question is yes or no?"
                    ),
                }
            )
            continue
        if is_specific_option_decline(example, conversation_control):
            specific_option_decline_ids.add(example_id)
            deferrals.append(
                {
                    "schema_version": "1.0.0",
                    "example_id": example_id,
                    "partition": prediction["partition"],
                    "domain": example["domain"],
                    "runtime_action": "acknowledge_without_generation",
                    "reason": "explicit_decline_of_specific_option",
                    "user_query": example["query"]["text"],
                    "deterministic_response": "Understood. I won't pursue that option.",
                }
            )
            continue
        transition_applied = bool(prediction.get("graph_override_applied"))
        transition_target_span_id = prediction.get("graph_target_span_id")
        classifier_applied = bool(prediction.get("classifier_override_applied"))
        required_action = prediction["selected_action"]
        action_guard_config = config.get("conversation_action_guard", {})
        conversation_action_guard_applied = bool(
            action_guard_config.get("enabled", False)
            and should_revert_unsupported_answer_override(
                prediction,
                shortlist,
                conversation_control,
                float(action_guard_config["maximum_top_action_compatibility"]),
            )
        )
        if conversation_action_guard_applied:
            required_action = "ask_followup"
            conversation_action_guard_count += 1
        classifier_abstention_promoted = bool(
            action_controller_config.get("enabled", False)
            and not conversation_action_guard_applied
            and should_promote_classifier_abstention(
                prediction,
                float(action_controller_config["minimum_abstain_probability"]),
                float(action_controller_config["minimum_abstain_margin"]),
            )
        )
        if classifier_abstention_promoted:
            controller_abstention_ids.add(example_id)
            classifier_abstention_promotion_count += 1
            deferrals.append(
                {
                    "schema_version": "1.0.0",
                    "example_id": example_id,
                    "partition": prediction["partition"],
                    "domain": example["domain"],
                    "runtime_action": "abstain_without_generation",
                    "reason": "calibrated_action_controller_v3_abstention",
                    "user_query": example["query"]["text"],
                    "deterministic_response": (
                        "I don't have enough relevant information to answer that reliably."
                    ),
                }
            )
            continue
        unsupported_followup_override_reverted = bool(
            action_controller_config.get("enabled", False)
            and not conversation_action_guard_applied
            and should_revert_unsupported_followup_override(
                prediction,
                shortlist,
                conversation_control,
                float(action_controller_config["minimum_followup_top_compatibility"]),
                float(action_controller_config["maximum_followup_classifier_confidence"]),
            )
        )
        if unsupported_followup_override_reverted:
            required_action = "answer"
            unsupported_followup_reversion_count += 1
        if required_action not in {"answer", "ask_followup"}:
            invalid_contracts += 1
            continue
        calibrated_action_accepted = bool(
            legacy_prediction["accepted"] or transition_applied or classifier_applied
        )
        if not calibrated_action_accepted:
            deferrals.append(
                {
                    "schema_version": "1.0.0",
                    "example_id": example_id,
                    "partition": prediction["partition"],
                    "domain": example["domain"],
                    "runtime_action": "clarify_without_generation",
                    "reason": "answer_action_probability_in_calibrated_deferral_band",
                    "user_query": example["query"]["text"],
                    "ask_followup_probability": legacy_prediction[
                        "ask_followup_probability"
                    ],
                }
            )
            continue
        conflict_config = config["evidence_conflict_gate"]
        working_sources = [dict(source) for source in shortlist["shortlist"]]
        transition_primary = False
        if transition_applied and transition_target_span_id:
            transition_source = next(
                (
                    source
                    for source in working_sources
                    if source["span_id"] == transition_target_span_id
                ),
                None,
            )
            if transition_source is None:
                transition_source = dict(span_source_catalog[transition_target_span_id])
            working_sources = [
                transition_source,
                *[
                    source
                    for source in working_sources
                    if source["span_id"] != transition_target_span_id
                ],
            ]
            for rank, source in enumerate(working_sources, start=1):
                source["rank"] = rank
            transition_primary = True
            transition_target_injection_count += 1
        top_source = working_sources[0]
        anchor_config = config.get("numeric_anchor_promotion", {})
        promoted_numeric_span_ids: list[str] = []
        if anchor_config.get("enabled", False):
            anchors = numeric_anchors(
                str(example.get("query", {}).get("text", "")),
                previous_agent_utterance(example),
            )
            working_sources, promoted_numeric_span_ids = promote_numeric_anchor_sources(
                working_sources,
                anchors,
                int(anchor_config.get("protected_primary_sources", 1)),
            )
            numeric_anchor_promotion_count += len(promoted_numeric_span_ids)
            top_source = working_sources[0]
        action_override_reason: str | None = None
        if transition_applied:
            action_override_reason = "train_dialogue_transition_graph"
            transition_override_count += 1
        elif classifier_applied:
            action_override_reason = "calibrated_tri_action_classifier"
            classifier_override_count += 1
        if conversation_action_guard_applied:
            action_override_reason = "conversation_action_guard_reverted_unsupported_answer"
        elif unsupported_followup_override_reverted:
            action_override_reason = "action_controller_v3_reverted_unsupported_followup"
        evidence_conflict = bool(
            conflict_config["enabled"]
            and required_action in set(conflict_config["actions"])
            and not shortlist.get("primary_evidence_accepted")
            and not transition_primary
            and float(top_source["action_compatibility"])
            < float(conflict_config["maximum_top_action_compatibility"])
        )
        positive_conflict_override = bool(
            evidence_conflict
            and conflict_config.get("positive_response_override_to_answer", False)
            and example["query"].get("dialogue_act")
            in set(conflict_config.get("positive_response_dialogue_acts", []))
        )
        if positive_conflict_override:
            required_action = "answer"
            action_override_reason = (
                "positive_response_with_strong_answer_compatible_top_evidence"
            )
            evidence_conflict = False
        rescue_config = config["clarification_evidence_rescue"]
        clarification_rescue = False
        if (
            evidence_conflict
            and rescue_config.get("enabled", False)
            and example["query"].get("dialogue_act")
            in set(rescue_config.get("dialogue_acts", []))
        ):
            rescue_source = max(
                working_sources,
                key=lambda source: (
                    float(source["action_compatibility"]),
                    -int(source["rank"]),
                ),
            )
            if float(rescue_source["action_compatibility"]) >= float(
                rescue_config["minimum_action_compatibility"]
            ):
                working_sources = [
                    rescue_source,
                    *[
                        source
                        for source in working_sources
                        if source["span_id"] != rescue_source["span_id"]
                    ],
                ]
                for rank, source in enumerate(working_sources, start=1):
                    source["rank"] = rank
                top_source = working_sources[0]
                clarification_rescue = True
                clarification_rescue_count += 1
                evidence_conflict = False
                if prediction["partition"] == "development":
                    clarification_rescue_development_correct.append(
                        rescue_source["span_id"] in set(example["gold"]["span_ids"])
                    )
        if evidence_conflict:
            top_is_gold = top_source["span_id"] in set(example["gold"]["span_ids"])
            if prediction["partition"] == "development":
                evidence_conflict_development_correct.append(top_is_gold)
            deferrals.append(
                {
                    "schema_version": "1.0.0",
                    "example_id": example_id,
                    "partition": prediction["partition"],
                    "domain": example["domain"],
                    "runtime_action": "clarify_without_generation",
                    "reason": "evidence_action_conflict_below_calibrated_threshold",
                    "user_query": example["query"]["text"],
                    "ask_followup_probability": prediction["classifier_probabilities"][
                        "ask_followup"
                    ],
                    "top_action_compatibility": top_source["action_compatibility"],
                }
            )
            continue
        primary_evidence_accepted = bool(
            shortlist.get("primary_evidence_accepted")
            or transition_primary
            or positive_conflict_override
            or clarification_rescue
            or conversation_action_guard_applied
        )
        evidence_depth = (
            int(config["evidence_depth_when_primary"])
            if primary_evidence_accepted
            else int(config["evidence_depth"])
        )
        sources = [dict(source) for source in working_sources[:evidence_depth]]
        for source in sources:
            source["evidence_origin"] = "ranked"
        expansion_config = config.get("adjacent_evidence_expansion", {})
        if expansion_config.get("enabled", False):
            maximum_distance = int(expansion_config.get("maximum_distance", 2))
            maximum_added = int(expansion_config.get("maximum_added_spans", 0))
            anchor_count = int(expansion_config.get("anchor_count", 0))
            existing_span_ids = {source["span_id"] for source in sources}
            added = 0
            for anchor in list(sources[:anchor_count]):
                anchor_catalog = span_source_catalog.get(anchor["span_id"], {})
                neighbor_ids = anchor_catalog.get("neighbor_span_ids", [])[
                    : maximum_distance * 2
                ]
                for neighbor_id in neighbor_ids:
                    if added >= maximum_added:
                        break
                    if neighbor_id in existing_span_ids or neighbor_id not in span_source_catalog:
                        continue
                    neighbor = dict(span_source_catalog[neighbor_id])
                    if not neighbor.get("text"):
                        continue
                    neighbor["rank"] = len(sources) + 1
                    neighbor["evidence_origin"] = "adjacent_structural_context"
                    sources.append(neighbor)
                    existing_span_ids.add(neighbor_id)
                    added += 1
                if added >= maximum_added:
                    break
        first_gold_rank = next(
            (
                rank
                for rank, source in enumerate(working_sources, start=1)
                if source["span_id"] in set(example["gold"]["span_ids"])
            ),
            None,
        )
        gold_in_request = bool(
            first_gold_rank is not None and first_gold_rank <= evidence_depth
        )
        ranked_source_count = sum(
            source.get("evidence_origin") == "ranked" for source in sources
        )
        while True:
            evidence_text, evidence_catalog = render_evidence(sources)
            rendered_input = render_input(
                example,
                required_action,
                "E1" if primary_evidence_accepted else None,
                evidence_text,
                int(config["history_maximum_turns"]),
                conversation_control,
            )
            token_count = len(encoding.encode(prompt)) + len(encoding.encode(rendered_input))
            if (
                token_count <= int(config["maximum_request_input_tokens"])
                or len(sources) <= ranked_source_count
            ):
                break
            sources.pop()
        evidence_ids = [item["evidence_id"] for item in evidence_catalog]
        schema = output_schema(
            config, evidence_ids, required_action, primary_evidence_accepted
        )
        Draft202012Validator.check_schema(schema)
        duplicate_evidence_ids += len(evidence_ids) - len(set(evidence_ids))
        missing_evidence_texts += sum(not item["text"] for item in evidence_catalog)
        input_tokens.append(token_count)
        if token_count > int(config["maximum_request_input_tokens"]):
            budget_violations += 1
        expected_status = (
            "answered"
            if required_action == "answer"
            else "clarification_required"
        )
        request_body = {
            "model": generator["model"],
            "instructions": prompt,
            "input": rendered_input,
            "reasoning": {"effort": generator["reasoning_effort"]},
            "text": {
                "verbosity": generator["text_verbosity"],
                "format": {
                    "type": "json_schema",
                    "name": config["response_contract"]["schema_name"],
                    "strict": True,
                    "schema": schema,
                },
            },
            "max_output_tokens": generator["max_output_tokens"],
            "store": generator["store"],
        }
        requests.append(
            {
                "schema_version": "1.0.0",
                "example_id": example_id,
                "partition": prediction["partition"],
                "domain": example["domain"],
                "required_action": required_action,
                "semantic_response_polarity": conversation_control["polarity"],
                "semantic_response_polarity_confidence": conversation_control["confidence"],
                "semantic_response_polarity_source": conversation_control["source"],
                "branch_mode": branch_mode(
                    required_action,
                    conversation_control,
                    classify_previous_agent_mode(previous_agent_utterance(example)),
                ),
                "promoted_numeric_anchor_span_ids": promoted_numeric_span_ids,
                "original_planner_action": prediction["baseline_action"],
                "action_override_reason": action_override_reason,
                "transition_graph_override_applied": transition_applied,
                "tri_action_classifier_override_applied": classifier_applied,
                "conversation_action_guard_applied": conversation_action_guard_applied,
                "unsupported_followup_override_reverted": unsupported_followup_override_reverted,
                "transition_target_span_id": transition_target_span_id,
                "clarification_evidence_rescue": clarification_rescue,
                "expected_status": expected_status,
                "planner_probability": prediction["classifier_probabilities"][
                    required_action
                ],
                "input_tokens": token_count,
                "evidence_catalog": evidence_catalog,
                "gold_available_in_context": bool(
                    shortlist["gold_available_in_context"]
                    or (
                        transition_target_span_id
                        and transition_target_span_id in set(example["gold"]["span_ids"])
                    )
                ),
                "gold_in_request": gold_in_request,
                "first_gold_rank": first_gold_rank,
                "evidence_depth": evidence_depth,
                "request_body": request_body,
            }
        )

    request_ids = {row["example_id"] for row in requests}
    deferral_ids = {row["example_id"] for row in deferrals}
    for example_id, prediction in sorted(predictions.items()):
        if example_id not in in_scope_prediction_ids:
            continue
        if example_id in request_ids or example_id in deferral_ids:
            continue
        example = benchmark.get(example_id)
        if example is None:
            continue
        legacy_prediction = legacy_predictions.get(example_id)
        if prediction["selected_action"] == "abstain":
            transition_abstention_ids.add(example_id)
            deferrals.append(
                {
                    "schema_version": "1.0.0",
                    "example_id": example_id,
                    "partition": prediction["partition"],
                    "domain": example["domain"],
                    "runtime_action": "abstain_without_generation",
                    "reason": (
                        "train_dialogue_transition_graph_abstention"
                        if prediction.get("graph_override_applied")
                        else "calibrated_tri_action_classifier_abstention"
                    ),
                    "user_query": example["query"]["text"],
                    "deterministic_response": (
                        "I don't have enough relevant information to continue this path."
                    ),
                    "transition_target_span_id": prediction.get("graph_target_span_id"),
                }
            )
            continue
        deferrals.append(
            {
                "schema_version": "1.0.0",
                "example_id": example_id,
                "partition": prediction["partition"],
                "domain": example["domain"],
                "runtime_action": "clarify_without_generation",
                "reason": (
                    "answer_action_probability_in_calibrated_deferral_band"
                    if legacy_prediction and not legacy_prediction["accepted"]
                    else "no_action_compatible_evidence_shortlist"
                ),
                "user_query": example["query"]["text"],
                "ask_followup_probability": prediction["classifier_probabilities"][
                    "ask_followup"
                ],
            }
        )

    accounted_in_scope_ids = request_ids | {row["example_id"] for row in deferrals}
    unaccounted_in_scope_ids = sorted(in_scope_prediction_ids - accounted_in_scope_ids)
    generation_coverage = len(request_ids) / len(in_scope_prediction_ids)
    non_generation_resolution_ids = (
        closure_ids | transition_abstention_ids | controller_abstention_ids
    )
    conflict_gate_population = in_scope_prediction_ids - non_generation_resolution_ids
    conflict_gate_generation_coverage = (
        len(request_ids) / len(conflict_gate_population) if conflict_gate_population else 0.0
    )
    development_conflict_top_gold_precision = (
        float(np.mean(evidence_conflict_development_correct))
        if evidence_conflict_development_correct
        else 0.0
    )
    clarification_rescue_development_precision = (
        float(np.mean(clarification_rescue_development_correct))
        if clarification_rescue_development_correct
        else 0.0
    )

    holdout_available = [
        row
        for row in requests
        if row["partition"] == "holdout" and row["gold_available_in_context"]
    ]
    holdout_gold_in_request_rate = (
        float(np.mean([row["gold_in_request"] for row in holdout_available]))
        if holdout_available
        else 0.0
    )
    hard_results: list[dict[str, Any]] = []
    hard_ok = True
    request_by_id = {row["example_id"]: row for row in requests}
    for example_id, expectation in config["hard_regression_expectations"].items():
        request = request_by_id.get(example_id)
        passed = bool(
            request
            and request["required_action"] == expectation["required_action"]
            and request["first_gold_rank"] is not None
            and request["first_gold_rank"] <= expectation["required_gold_rank_maximum"]
        )
        hard_ok = hard_ok and passed
        hard_results.append(
            {
                "example_id": example_id,
                "expected_action": expectation["required_action"],
                "actual_action": request["required_action"] if request else None,
                "first_gold_rank": request["first_gold_rank"] if request else None,
                "passed": passed,
            }
        )

    constraints = config["quality_constraints"]
    gates = {
        "request_contracts": invalid_contracts <= constraints["maximum_invalid_request_contracts"],
        "input_token_budget": budget_violations <= constraints["maximum_input_budget_violations"],
        "unique_evidence_ids": (
            duplicate_evidence_ids <= constraints["maximum_duplicate_evidence_ids"]
        ),
        "nonempty_evidence_texts": (
            missing_evidence_texts <= constraints["maximum_missing_evidence_texts"]
        ),
        "hard_regression_expectations": (
            hard_ok if constraints["require_all_hard_regression_expectations"] else True
        ),
        "holdout_gold_in_request_when_available": (
            holdout_gold_in_request_rate
            >= constraints["minimum_holdout_gold_in_request_rate_when_available"]
        ),
        "store_disabled": generator["store"] is False,
        "in_scope_accounting": not unaccounted_in_scope_ids,
        "evidence_conflict_generation_coverage": (
            conflict_gate_generation_coverage
            >= config["evidence_conflict_gate"]["minimum_overall_generation_coverage"]
        ),
        "evidence_conflict_development_precision": (
            not config["evidence_conflict_gate"].get("enabled", False)
            or development_conflict_top_gold_precision
            <= config["evidence_conflict_gate"][
                "maximum_development_deferred_top_gold_precision"
            ]
        ),
        "clarification_rescue_development_precision": (
            not config["clarification_evidence_rescue"].get("enabled", False)
            or clarification_rescue_development_precision
            >= config["clarification_evidence_rescue"][
                "minimum_development_promoted_gold_precision"
            ]
        ),
        "dialogue_transition_graph_source": transition_report["gates_passed"] is True,
        "tri_action_planner_source": tri_action_report["gates_passed"] is True,
        "action_controller_v3_source": (
            action_controller_report is None
            or action_controller_report.get("gates_passed", False) is True
        ),
        "test_split_locked": True,
    }
    gates_passed = all(gates.values())

    implementation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dependency_sha256 = {
        "conversation_control.py": hashlib.sha256(
            (PROJECT_ROOT / "scripts" / "conversation_control.py").read_bytes()
        ).hexdigest(),
        "evidence_anchor.py": hashlib.sha256(
            (PROJECT_ROOT / "scripts" / "evidence_anchor.py").read_bytes()
        ).hexdigest(),
    }
    fingerprint = stable_hash(
        {
            "configuration": config,
            "implementation_sha256": implementation_sha256,
            "dependency_sha256": dependency_sha256,
            "request_ids": sorted(request_ids),
            "prompt_hash": text_hash(prompt),
        }
    )
    experiment_id = f"{config['experiment_name']}_{fingerprint[:10]}"
    artifact_root = DATASET_ROOT / "planned_generation_contract_experiments" / experiment_id
    report_root = REPORT_ROOT / "planned_generation_contract_experiments" / experiment_id
    requests_path = artifact_root / "requests" / "validation_requests.jsonl"
    deferrals_path = artifact_root / "deferrals" / "validation_deferrals.jsonl"
    report_path = report_root / "planned_generation_contract_report.json"
    write_jsonl(requests_path, requests)
    write_jsonl(deferrals_path, deferrals)
    report = {
        "schema_version": "1.0.0",
        "pipeline_stage": "planned_generation_contract",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "implementation_sha256": implementation_sha256,
        "dependency_sha256": dependency_sha256,
        "official_api_contract": {
            "api": "Responses API",
            "structured_outputs": True,
            "strict_json_schema": True,
            "store": False,
        },
        "data_protocol": {
            "test_split_opened": False,
            "generation_requests_prepared": len(requests),
            "planner_deferrals": len(deferrals),
            "evidence_conflict_deferrals": sum(
                row["reason"] == "evidence_action_conflict_below_calibrated_threshold"
                for row in deferrals
            ),
            "planner_uncertainty_deferrals": sum(
                row["reason"] == "answer_action_probability_in_calibrated_deferral_band"
                for row in deferrals
            ),
            "generation_coverage": generation_coverage,
            "conflict_gate_generation_coverage_excluding_deterministic_closures": (
                conflict_gate_generation_coverage
            ),
            "in_scope_direct_user_response_predictions": len(in_scope_prediction_ids),
            "excluded_out_of_scope_predictions": len(excluded_out_of_scope_prediction_ids),
            "excluded_out_of_scope_prediction_ids": excluded_out_of_scope_prediction_ids,
            "unaccounted_in_scope_prediction_ids": unaccounted_in_scope_ids,
            "openai_calls_made": 0,
            "deterministic_conversation_closures": len(closure_ids),
            "ambiguous_response_clarifications": len(ambiguous_response_clarification_ids),
            "specific_option_decline_acknowledgments": len(specific_option_decline_ids),
            "dialogue_transition_graph_overrides": transition_override_count,
            "tri_action_classifier_overrides": classifier_override_count,
            "conversation_action_guard_overrides": conversation_action_guard_count,
            "classifier_abstention_promotions": classifier_abstention_promotion_count,
            "unsupported_followup_override_reversions": unsupported_followup_reversion_count,
            "numeric_anchor_promotions": numeric_anchor_promotion_count,
            "dialogue_transition_target_injections": transition_target_injection_count,
            "dialogue_transition_abstentions": len(transition_abstention_ids),
            "clarification_evidence_rescues": clarification_rescue_count,
        },
        "token_budget": {
            "configured_maximum": config["maximum_request_input_tokens"],
            "mean": float(np.mean(input_tokens)) if input_tokens else 0.0,
            "p95": float(np.percentile(input_tokens, 95)) if input_tokens else 0.0,
            "maximum": max(input_tokens) if input_tokens else 0,
            "violations": budget_violations,
        },
        "evidence_quality": {
            "holdout_gold_available_count": len(holdout_available),
            "holdout_gold_in_request_rate_when_available": holdout_gold_in_request_rate,
            "evidence_depth": config["evidence_depth"],
            "evidence_depth_when_primary": config["evidence_depth_when_primary"],
            "development_conflict_top_gold_precision": development_conflict_top_gold_precision,
            "clarification_rescue_development_promoted_gold_precision": (
                clarification_rescue_development_precision
            ),
        },
        "hard_regression_cases": hard_results,
        "validation_counts": {
            "invalid_contracts": invalid_contracts,
            "duplicate_evidence_ids": duplicate_evidence_ids,
            "missing_evidence_texts": missing_evidence_texts,
        },
        "gate_results": gates,
        "gates_passed": gates_passed,
        "recommendation": (
            "contract_ready_for_explicitly_authorized_minimal_generation_canary"
            if gates_passed
            else "keep_generation_blocked"
        ),
        "artifacts": {
            "requests": str(requests_path.relative_to(PROJECT_ROOT)),
            "deferrals": str(deferrals_path.relative_to(PROJECT_ROOT)),
        },
    }
    write_json(report_path, report)

    print(f"Experiment: {experiment_id}")
    print(f"Prepared requests / deterministic deferrals: {len(requests)} / {len(deferrals)}")
    print(
        "Input tokens mean / p95 / max: "
        f"{report['token_budget']['mean']:.1f} / {report['token_budget']['p95']:.1f} / "
        f"{report['token_budget']['maximum']}"
    )
    print(
        "Holdout gold in request when available: "
        f"{holdout_gold_in_request_rate:.4f}"
    )
    for row in hard_results:
        print(
            f"Hard case {row['example_id']}: action={row['actual_action']} "
            f"gold_rank={row['first_gold_rank']} passed={row['passed']}"
        )
    print(f"OpenAI calls made: 0")
    print(f"Gates passed: {gates_passed}")
    print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
