"""Evaluate a mandatory-claim planner and controlled realizer on a closed validation cohort.

The planner and realizer never receive reference answers or gold spans. Those fields are read only
after generation by the offline evaluator. The benchmark test split remains locked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tiktoken
from dotenv import load_dotenv
from jsonschema import Draft202012Validator

try:
    from scripts.plan_contract import (
        conflicting_years_with_conversation,
        unsupported_acronym_expansions,
        unsupported_organization_attributions,
    )
except ModuleNotFoundError:
    from plan_contract import (  # type: ignore
        conflicting_years_with_conversation,
        unsupported_acronym_expansions,
        unsupported_organization_attributions,
    )

try:
    from scripts.run_automated_end_to_end_evaluation import (
        JUDGE_INSTRUCTIONS,
        collect_previously_used_ids,
        extract_output_text,
        judge_schema,
        normalize,
        render_judge_input,
        usage_cost,
        wilson,
    )
except ModuleNotFoundError:
    from run_automated_end_to_end_evaluation import (  # type: ignore
        JUDGE_INSTRUCTIONS,
        collect_previously_used_ids,
        extract_output_text,
        judge_schema,
        normalize,
        render_judge_input,
        usage_cost,
        wilson,
    )


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "processed" / "multidoc2dial_v1"
REPORT_ROOT = ROOT / "reports" / "generated" / "multidoc2dial_v1"
CONFIG_PATH = ROOT / "configs" / "mandatory_claim_pipeline.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Evaluation configuration; defaults to the frozen v6 candidate configuration.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-maximum-cost-usd", type=float, default=0.0)
    parser.add_argument(
        "--maximum-cost-usd",
        type=float,
        help="Optional per-run ceiling; must be positive and no higher than the configured ceiling.",
    )
    parser.add_argument(
        "--allow-cache-resume",
        action="store_true",
        help="Allow a run whose uncached estimate exceeds the ceiling; the per-call ceiling still applies.",
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument(
        "--case-id-file",
        type=Path,
        help="JSON object containing an example_ids list for a targeted diagnostic.",
    )
    parser.add_argument("--fresh-confirmation", action="store_true")
    parser.add_argument(
        "--reserved-id-file",
        type=Path,
        help="Run an exact precommitted validation reserve from a JSON file.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def collect_mandatory_pipeline_used_ids() -> set[str]:
    used: set[str] = set()
    artifact_root = DATA_ROOT / "mandatory_claim_pipeline_experiments"
    for path in artifact_root.glob("*/case_results.jsonl"):
        for row in iter_jsonl(path):
            example_id = str(row.get("example_id", ""))
            if example_id.startswith("eval_"):
                used.add(example_id)
    report_root = REPORT_ROOT / "mandatory_claim_pipeline_experiments"
    for path in report_root.glob("*/mandatory_claim_pipeline_report.json"):
        report = load_json(path)
        for example_id in report.get("protocol", {}).get("selected_abstention_ids", []):
            if str(example_id).startswith("eval_"):
                used.add(str(example_id))
    return used


def planner_schema(evidence_ids: list[str], maximum_claims: int) -> dict[str, Any]:
    claim_ids = [f"C{index}" for index in range(1, maximum_claims + 1)]
    return {
        "type": "object",
        "properties": {
            "answer_objective": {"type": "string", "minLength": 1, "maxLength": 500},
            "mandatory_claims": {
                "type": "array",
                "minItems": 1,
                "maxItems": maximum_claims,
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string", "enum": claim_ids},
                        "claim_text": {"type": "string", "minLength": 1, "maxLength": 500},
                        "claim_type": {
                            "type": "string",
                            "enum": [
                                "direct_answer",
                                "consequence",
                                "condition",
                                "limitation",
                                "exception",
                                "amount_or_date",
                                "procedure",
                                "option",
                                "followup_question",
                            ],
                        },
                        "evidence_ids": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": min(5, len(evidence_ids)),
                            "items": {"type": "string", "enum": evidence_ids},
                        },
                    },
                    "required": ["claim_id", "claim_text", "claim_type", "evidence_ids"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["answer_objective", "mandatory_claims"],
        "additionalProperties": False,
    }


def evidence_selector_schema(evidence_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "central_objective": {"type": "string", "minLength": 1, "maxLength": 500},
            "primary_evidence_id": {"type": "string", "enum": evidence_ids},
            "complementary_evidence_ids": {
                "type": "array",
                "maxItems": min(4, max(0, len(evidence_ids) - 1)),
                "items": {"type": "string", "enum": evidence_ids},
            },
        },
        "required": ["central_objective", "primary_evidence_id", "complementary_evidence_ids"],
        "additionalProperties": False,
    }


def apply_selected_primary_evidence(prepared: dict[str, Any], evidence_id: str) -> None:
    prepared["request_body"]["input"] = re.sub(
        r"(?m)^PRIMARY_EVIDENCE_ID:.*$",
        f"PRIMARY_EVIDENCE_ID: {evidence_id}",
        prepared["request_body"]["input"],
        count=1,
    )


def realization_schema(claim_ids: list[str], evidence_ids: list[str]) -> dict[str, Any]:
    count = len(claim_ids)
    return {
        "type": "object",
        "properties": {
            "realizations": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string", "enum": claim_ids},
                        "sentence": {"type": "string", "minLength": 1, "maxLength": 1000},
                        "citations": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": min(5, len(evidence_ids)),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "evidence_id": {"type": "string", "enum": evidence_ids},
                                    "evidence_quote": {
                                        "type": "string",
                                        "minLength": 1,
                                        "maxLength": 1000,
                                    },
                                },
                                "required": ["evidence_id", "evidence_quote"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["claim_id", "sentence", "citations"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["realizations"],
        "additionalProperties": False,
    }


def parse_structured(response_data: dict[str, Any], schema: dict[str, Any]) -> tuple[Any, list[str]]:
    output_text = extract_output_text(response_data)
    try:
        parsed = json.loads(output_text) if output_text else None
    except json.JSONDecodeError as error:
        return None, [f"invalid_json: {error}"]
    if parsed is None:
        return None, ["empty_output"]
    errors = sorted(Draft202012Validator(schema).iter_errors(parsed), key=lambda item: list(item.path))
    return parsed, [error.message for error in errors]


def validate_plan(plan: Any, schema: dict[str, Any], required_action: str) -> list[str]:
    if plan is None:
        return ["missing_plan"]
    errors = [error.message for error in Draft202012Validator(schema).iter_errors(plan)]
    if errors:
        return errors
    claims = plan["mandatory_claims"]
    expected_ids = [f"C{index}" for index in range(1, len(claims) + 1)]
    actual_ids = [claim["claim_id"] for claim in claims]
    if actual_ids != expected_ids:
        errors.append(f"claim_ids_must_be_consecutive: expected={expected_ids}, actual={actual_ids}")
    if required_action == "ask_followup":
        if len(claims) != 1:
            errors.append("ask_followup_requires_exactly_one_claim")
        elif claims[0]["claim_type"] != "followup_question":
            errors.append("ask_followup_claim_type_must_be_followup_question")
    elif any(claim["claim_type"] == "followup_question" for claim in claims):
        errors.append("answer_must_not_contain_followup_question_claim")
    return errors


def execution_plan_value(prepared: dict[str, Any], key: str) -> str:
    prefix = f"{key}:"
    for line in prepared["request_body"]["input"].splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def prepared_conversation_text(prepared: dict[str, Any]) -> str:
    input_text = prepared["request_body"]["input"]
    match = re.search(
        r"<CONVERSATION>\s*(.*?)\s*</CONVERSATION>", input_text, flags=re.DOTALL
    )
    return match.group(1) if match else ""


def sanitize_redundant_conflicting_claims(
    plan: Any, prepared: dict[str, Any]
) -> tuple[Any, list[dict[str, Any]]]:
    """Drop only non-primary claims carrying a calibrated year conflict."""
    if not isinstance(plan, dict) or len(plan.get("mandatory_claims", [])) <= 1:
        return plan, []
    if execution_plan_value(prepared, "BRANCH_MODE") not in {
        "conclude_positive_outcome",
        "conclude_negative_outcome",
    }:
        return plan, []
    conversation = prepared_conversation_text(prepared)
    kept = []
    removed: list[dict[str, Any]] = []
    claims = plan["mandatory_claims"]
    evidence_catalog = prepared.get("evidence_catalog", [])
    for claim in claims:
        conflicts = conflicting_years_with_conversation(
            str(claim.get("claim_text", "")), conversation
        )
        unsupported_acronyms = unsupported_acronym_expansions(
            claim, evidence_catalog
        )
        removable_acronym_definition = bool(
            unsupported_acronyms and len(claims) > 1
        )
        removable_year_conflict = bool(conflicts and claim is not claims[0])
        if removable_year_conflict or removable_acronym_definition:
            removed.append({"claim_id": claim["claim_id"], "conflicting_years": conflicts})
            if unsupported_acronyms:
                removed[-1]["unsupported_acronym_expansions"] = unsupported_acronyms
        else:
            kept.append(claim)
    if not removed:
        return plan, []
    sanitized = {**plan, "mandatory_claims": []}
    for index, claim in enumerate(kept, start=1):
        sanitized["mandatory_claims"].append({**claim, "claim_id": f"C{index}"})
    return sanitized, removed


def validate_branch_outcome(plan: Any, prepared: dict[str, Any]) -> list[str]:
    """Require an explicit conclusion for a resolved yes/no branch."""
    if not isinstance(plan, dict) or not plan.get("mandatory_claims"):
        return []
    branch = execution_plan_value(prepared, "BRANCH_MODE")
    if branch in {"conclude_positive_outcome", "conclude_negative_outcome"}:
        if plan["mandatory_claims"][0].get("claim_type") != "direct_answer":
            return [f"{branch}_requires_direct_answer_as_C1"]
    if prepared.get("required_action") == "ask_followup":
        claim = plan["mandatory_claims"][0]
        evidence_by_id = {
            item["evidence_id"]: item["text"]
            for item in prepared.get("evidence_catalog", [])
        }
        source = " ".join(
            evidence_by_id.get(evidence_id, "")
            for evidence_id in claim.get("evidence_ids", [])
        ).casefold()
        claim_text = str(claim.get("claim_text", "")).casefold()
        source_has_required_negation = bool(
            re.search(r"\b(?:didn[ 'â€™]?t|did not|not receive|without)\b", source)
        )
        claim_preserves_negation = bool(
            re.search(r"\b(?:didn[ 'â€™]?t|did not|not receive|without)\b", claim_text)
        )
        if source_has_required_negation and not claim_preserves_negation:
            return ["followup_question_dropped_required_evidence_negation"]
    errors: list[str] = []
    for claim in plan["mandatory_claims"]:
        unsupported = unsupported_organization_attributions(
            claim, prepared.get("evidence_catalog", [])
        )
        if unsupported:
            errors.append(
                f"{claim['claim_id']}: unsupported_organization_attribution="
                + ",".join(unsupported)
            )
    return errors


def render_realizer_input(prepared: dict[str, Any], plan: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            prepared["request_body"]["input"],
            "<MANDATORY_CLAIM_PLAN>\n"
            + json.dumps(plan, ensure_ascii=False, indent=2)
            + "\n</MANDATORY_CLAIM_PLAN>",
        ]
    )


def primary_evidence_id(prepared: dict[str, Any]) -> str | None:
    for line in prepared["request_body"]["input"].splitlines():
        if line.startswith("PRIMARY_EVIDENCE_ID:"):
            value = line.split(":", 1)[1].strip()
            return None if value == "NONE" else value
    return None


def is_substantive_evidence(text: str) -> bool:
    normalized = normalize(text)
    if normalized in {"", "and", "or", "note", "if you", "you"}:
        return False
    return len(normalized.split()) >= 3


def is_fragmentary_evidence(text: str) -> bool:
    raw = text.strip().casefold()
    normalized = normalize(text).strip()
    if not normalized:
        return False
    return bool(
        re.search(r"(?:\bof|\bif|\bto|\band|\bor|[:,;])$", normalized)
        or (
            len(normalized.split()) <= 6
            and not re.search(r"[.!?]$", raw)
        )
    )


def build_span_structure_index(
    documents_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Return source ordering metadata without using gold answers or references."""
    path = documents_path or (DATA_ROOT / "documents.jsonl")
    index: dict[str, dict[str, Any]] = {}
    for document in iter_jsonl(path):
        for section in document["sections"]:
            for position, span in enumerate(section["spans"]):
                index[span["id"]] = {
                    "document_id": document["id"],
                    "section_id": section["id"],
                    "section_position": position,
                }
    return index


def structural_evidence_chains(
    evidence_catalog: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Find contiguous same-section spans that may reconstruct a sentence or list."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in evidence_catalog:
        if item.get("section_position") is None:
            continue
        key = (str(item["document_id"]), str(item["section_id"]))
        grouped.setdefault(key, []).append(item)

    chains: list[list[dict[str, Any]]] = []
    for items in grouped.values():
        ordered = sorted(items, key=lambda item: int(item["section_position"]))
        current: list[dict[str, Any]] = []
        for item in ordered:
            if current and int(item["section_position"]) != int(
                current[-1]["section_position"]
            ) + 1:
                if len(current) >= 2:
                    chains.append(current)
                current = []
            current.append(item)
        if len(current) >= 2:
            chains.append(current)

    return [
        chain
        for chain in chains
        if any(is_fragmentary_evidence(str(item["text"])) for item in chain)
    ]


def enrich_structural_evidence(
    prepared: dict[str, Any], span_structure: dict[str, dict[str, Any]]
) -> int:
    """Enrich legacy contracts and expose only safe source-order metadata."""
    for item in prepared.get("evidence_catalog", []):
        structure = span_structure.get(str(item.get("span_id", "")))
        if structure:
            item.setdefault("section_position", structure["section_position"])

    chains = structural_evidence_chains(prepared.get("evidence_catalog", []))
    if not chains:
        return 0
    lines = [
        "<STRUCTURAL_EVIDENCE_ORDER>",
        "Each chain is ordered as in its source section. It is only a reconstruction candidate; combine spans only when their grammar and meaning form one sentence, list, condition, or explanation.",
    ]
    for index, chain in enumerate(chains, start=1):
        ordered_ids = " -> ".join(
            f"{item['evidence_id']}[position={int(item['section_position'])}]"
            for item in chain
        )
        lines.append(f"CHAIN_{index}: {ordered_ids}")
    lines.append("</STRUCTURAL_EVIDENCE_ORDER>")
    metadata = "\n".join(lines)
    input_text = prepared["request_body"]["input"]
    closing = "</EVIDENCE_UNTRUSTED_DATA>"
    if "<STRUCTURAL_EVIDENCE_ORDER>" not in input_text:
        if closing in input_text:
            input_text = input_text.replace(closing, metadata + "\n" + closing, 1)
        else:
            input_text += "\n\n" + metadata
        prepared["request_body"]["input"] = input_text
    prepared["structural_evidence_chains"] = [
        [item["evidence_id"] for item in chain] for chain in chains
    ]
    return len(chains)


def validate_structural_claim_coverage(
    plan: Any, prepared: dict[str, Any]
) -> list[str]:
    """Require list children to carry their governing stem and its material scope."""
    if not isinstance(plan, dict):
        return []
    evidence_by_id = {
        item["evidence_id"]: item for item in prepared.get("evidence_catalog", [])
    }
    errors: list[str] = []
    for claim in plan.get("mandatory_claims", []):
        selected = set(claim.get("evidence_ids", []))
        for chain_ids in prepared.get("structural_evidence_chains", []):
            chain = [evidence_by_id[item] for item in chain_ids if item in evidence_by_id]
            for stem_index, stem in enumerate(chain[:-1]):
                if not str(stem["text"]).strip().endswith(":"):
                    continue
                child_ids = {
                    item["evidence_id"] for item in chain[stem_index + 1 :]
                }
                selected_children = sorted(selected.intersection(child_ids))
                if selected_children and stem["evidence_id"] not in selected:
                    errors.append(
                        f"{claim['claim_id']}: selected list item(s) "
                        f"{','.join(selected_children)} require governing stem "
                        f"{stem['evidence_id']}; cite the stem and preserve its material "
                        "subject/beneficiary scope in claim_text"
                    )
                    continue
                if selected_children:
                    scope_terms = set(
                        re.findall(
                            r"\byour\s+([^\W\d_]+)",
                            str(stem["text"]).casefold(),
                            flags=re.UNICODE,
                        )
                    )
                    claim_tokens = set(normalize(str(claim.get("claim_text", ""))).split())
                    if scope_terms and not scope_terms.intersection(claim_tokens):
                        errors.append(
                            f"{claim['claim_id']}: governing stem {stem['evidence_id']} has "
                            "material scope term(s) "
                            + ",".join(sorted(scope_terms))
                            + "; preserve that beneficiary/object scope explicitly in claim_text"
                        )
    return errors


def validate_primary_evidence_usage(
    plan: Any, prepared: dict[str, Any]
) -> list[str]:
    """Make the runtime-selected semantic anchor a real contract invariant."""
    if not isinstance(plan, dict) or not plan.get("mandatory_claims"):
        return []
    primary_id = primary_evidence_id(prepared)
    if primary_id and primary_id not in set(plan["mandatory_claims"][0]["evidence_ids"]):
        return [f"C1 must use PRIMARY_EVIDENCE_ID {primary_id}"]
    return []


def validate_conditional_qualifier_support(
    plan: Any, prepared: dict[str, Any]
) -> list[str]:
    """Reject unsupported causal qualifiers inside otherwise supported conditions."""
    if not isinstance(plan, dict):
        return []
    evidence_by_id = {
        item["evidence_id"]: str(item["text"])
        for item in prepared.get("evidence_catalog", [])
    }
    stopwords = {
        "a", "an", "and", "because", "due", "for", "of", "the", "to", "your",
    }

    def content_tokens(value: str) -> set[str]:
        return {
            token
            for token in normalize(value).split()
            if token not in stopwords and len(token) >= 4
        }

    errors: list[str] = []
    conversation = prepared_conversation_text(prepared)
    for claim in plan.get("mandatory_claims", []):
        claim_text = str(claim.get("claim_text", ""))
        match = re.search(
            r"(?i)\b(?:for\s+(?:a|an|the)?|due\s+to|because\s+of)\s+([^,;.]+)",
            claim_text,
        )
        if not match:
            continue
        qualifier_tokens = content_tokens(match.group(1))
        support_text = " ".join(
            [
                conversation,
                *[
                    evidence_by_id.get(evidence_id, "")
                    for evidence_id in claim.get("evidence_ids", [])
                ],
            ]
        )
        support_tokens = content_tokens(support_text)
        unsupported = sorted(
            token
            for token in qualifier_tokens
            if not any(
                token == supported
                or (
                    len(token) >= 6
                    and len(supported) >= 6
                    and token[:6] == supported[:6]
                )
                for supported in support_tokens
            )
        )
        if unsupported:
            errors.append(
                f"{claim['claim_id']}: unsupported conditional qualifier term(s) "
                + ",".join(unsupported)
                + "; remove the qualifier or attach evidence that states it"
            )
    return errors


def _evidence_tokens(value: str) -> set[str]:
    stopwords = {
        "a", "an", "and", "are", "did", "do", "for", "have", "i", "if", "in",
        "is", "it", "of", "on", "or", "the", "this", "to", "true", "was", "were",
        "you", "your",
    }
    return {
        token
        for token in normalize(value).split()
        if token not in stopwords and len(token) > 1
    }


def apply_runtime_checklist_continuation(prepared: dict[str, Any]) -> bool:
    """Continue an observable checklist when a short confirmation leaves an item unvisited."""
    if prepared.get("required_action") != "answer":
        return False
    if execution_plan_value(prepared, "USER_RESPONSE_POLARITY") != "positive":
        return False
    if float(execution_plan_value(prepared, "USER_RESPONSE_POLARITY_CONFIDENCE") or 0) < 0.9:
        return False

    catalog = prepared.get("evidence_catalog", [])
    checklist_sections = {
        (str(item["document_id"]), str(item["section_id"]))
        for item in catalog
        if normalize(str(item["text"])) in {"you", "all of these must be true"}
    }
    if not checklist_sections:
        return False

    previous = execution_plan_value(prepared, "PREVIOUS_AGENT_UTTERANCE")
    previous_normalized = normalize(previous)
    conversation = normalize(prepared_conversation_text(prepared))
    generic_confirmation = bool(re.search(r"\ball (?:of )?this true\b", previous_normalized))
    matched_previous = any(
        item.get("section_position") is not None
        and (str(item["document_id"]), str(item["section_id"])) in checklist_sections
        and len(_evidence_tokens(str(item["text"]))) >= 3
        and len(
            _evidence_tokens(str(item["text"])).intersection(_evidence_tokens(previous))
        )
        / len(_evidence_tokens(str(item["text"])))
        >= 0.75
        for item in catalog
    )
    if not (generic_confirmation or matched_previous):
        return False

    candidates: list[dict[str, Any]] = []
    for item in catalog:
        if (str(item["document_id"]), str(item["section_id"])) not in checklist_sections:
            continue
        if item.get("section_position") is None or not is_substantive_evidence(str(item["text"])):
            continue
        normalized = normalize(str(item["text"]))
        if normalized in {"you", "all of these must be true"}:
            continue
        tokens = _evidence_tokens(str(item["text"]))
        if not tokens:
            continue
        if len(tokens.intersection(_evidence_tokens(conversation))) / len(tokens) >= 0.75:
            continue
        candidates.append(item)
    if not candidates:
        return False

    target = min(candidates, key=lambda item: int(item["section_position"]))
    input_text = prepared["request_body"]["input"]
    replacements = {
        "REQUIRED_ACTION": "ask_followup",
        "BRANCH_MODE": "ask_next_unresolved_condition",
        "PRIMARY_EVIDENCE_ID": str(target["evidence_id"]),
    }
    for key, value in replacements.items():
        input_text = re.sub(
            rf"(?m)^{re.escape(key)}:.*$", f"{key}: {value}", input_text, count=1
        )
    prepared["request_body"]["input"] = input_text
    prepared["required_action"] = "ask_followup"
    prepared["expected_status"] = "clarification_required"
    prepared["runtime_checklist_continuation_applied"] = True
    prepared["runtime_checklist_target_evidence_id"] = target["evidence_id"]
    return True


def should_review_plan(prepared: dict[str, Any], plan: dict[str, Any]) -> bool:
    if prepared["required_action"] == "ask_followup":
        return True
    if (
        execution_plan_value(prepared, "BRANCH_MODE")
        in {"conclude_positive_outcome", "conclude_negative_outcome"}
        and len(plan.get("mandatory_claims", [])) > 1
    ):
        return True
    evidence_by_id = {item["evidence_id"]: item for item in prepared["evidence_catalog"]}
    selected = {
        evidence_id
        for claim in plan["mandatory_claims"]
        for evidence_id in claim["evidence_ids"]
    }
    primary_id = primary_evidence_id(prepared)
    primary = evidence_by_id.get(primary_id) if primary_id else None
    anchors = [primary] if primary else [evidence_by_id.get(item) for item in selected]
    anchors = [item for item in anchors if item is not None]
    if not anchors:
        return False
    if any(
        selected.intersection(chain) and not set(chain).issubset(selected)
        for chain in prepared.get("structural_evidence_chains", [])
    ):
        return True
    if any(
        is_fragmentary_evidence(anchor["text"])
        and any(
            item["evidence_id"] not in selected
            and item["document_id"] == anchor["document_id"]
            and item["section_id"] == anchor["section_id"]
            and int(item["priority_rank"]) <= 6
            and is_substantive_evidence(item["text"])
            for item in prepared["evidence_catalog"]
        )
        for anchor in anchors
    ):
        return True
    return any(
        item["evidence_id"] not in selected
        and any(
            item["document_id"] == anchor["document_id"]
            and item["section_id"] == anchor["section_id"]
            for anchor in anchors
        )
        and int(item["priority_rank"]) <= 3
        and is_substantive_evidence(item["text"])
        for item in prepared["evidence_catalog"]
    )


def validate_and_assemble(
    prepared: dict[str, Any], plan: dict[str, Any], parsed: Any, schema: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    if parsed is None:
        return None, ["missing_realization"]
    errors = [error.message for error in Draft202012Validator(schema).iter_errors(parsed)]
    if errors:
        return None, errors
    claims = plan["mandatory_claims"]
    expected_ids = [claim["claim_id"] for claim in claims]
    actual_ids = [item["claim_id"] for item in parsed["realizations"]]
    if actual_ids != expected_ids:
        errors.append(f"realization_order_or_coverage_invalid: expected={expected_ids}, actual={actual_ids}")
    claim_by_id = {claim["claim_id"]: claim for claim in claims}
    evidence_by_id = {item["evidence_id"]: item["text"] for item in prepared["evidence_catalog"]}
    all_citations: list[dict[str, str]] = []
    seen_citations: set[tuple[str, str]] = set()
    sentences: list[str] = []
    for item in parsed["realizations"]:
        claim = claim_by_id.get(item["claim_id"])
        if claim is None:
            continue
        sentence = item["sentence"].strip()
        sentences.append(sentence)
        allowed = set(claim["evidence_ids"])
        for citation in item["citations"]:
            evidence_id = citation["evidence_id"]
            quote = citation["evidence_quote"].strip()
            if evidence_id not in allowed:
                errors.append(f"{item['claim_id']}: unauthorized_evidence_id={evidence_id}")
                continue
            if not normalize(quote) or normalize(quote) not in normalize(evidence_by_id[evidence_id]):
                errors.append(f"{item['claim_id']}: non_contiguous_quote={evidence_id}")
                continue
            key = (evidence_id, quote)
            if key not in seen_citations:
                seen_citations.add(key)
                all_citations.append({"evidence_id": evidence_id, "evidence_quote": quote})
    if prepared["required_action"] == "ask_followup":
        answer = " ".join(sentences)
        if len(sentences) != 1 or not answer.endswith("?") or answer.count("?") != 1:
            errors.append("ask_followup_must_be_exactly_one_question")
    if errors:
        return None, errors
    return {
        "status": "clarification_required" if prepared["required_action"] == "ask_followup" else "answered",
        "answer": " ".join(sentences),
        "citations": all_citations,
    }, []


def make_response_body(
    stage: dict[str, Any], instructions: str, input_text: str, schema_name: str, schema: dict[str, Any]
) -> dict[str, Any]:
    return {
        "model": stage["model"],
        "instructions": instructions,
        "input": input_text,
        "reasoning": {"effort": stage["reasoning_effort"]},
        "text": {
            "verbosity": stage["text_verbosity"],
            "format": {"type": "json_schema", "name": schema_name, "strict": True, "schema": schema},
        },
        "max_output_tokens": stage["max_output_tokens"],
        "store": stage["store"],
    }


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows])) if rows else 0.0


def generated_metrics(rows: list[dict[str, Any]], minimum_correctness: int) -> dict[str, Any]:
    correctness = [int(row["correctness_0_to_4"]) for row in rows]
    return {
        "examples": len(rows),
        "contract_valid_rate": mean_bool(rows, "contract_valid"),
        "citation_valid_rate": mean_bool(rows, "citations_valid"),
        "action_accuracy": mean_bool(rows, "action_correct"),
        "routing_accuracy": mean_bool(rows, "deterministic_action_correct"),
        "faithfulness_pass_rate": mean_bool(rows, "faithfulness_pass"),
        "correctness_acceptable_rate": (
            float(np.mean([value >= minimum_correctness for value in correctness])) if rows else 0.0
        ),
        "correctness_mean_0_to_4": float(np.mean(correctness)) if correctness else 0.0,
        "completeness_pass_rate": mean_bool(rows, "completeness_pass"),
        "overall_pass_rate": mean_bool(rows, "overall_pass"),
        "hallucination_rate": 1.0 - mean_bool(rows, "faithfulness_pass"),
    }


def baseline_generated_metrics(rows: list[dict[str, Any]], minimum_correctness: int) -> dict[str, Any]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "contract_valid": bool(row["contract_valid"]),
                "citations_valid": bool(row["citations_valid"]),
                "action_correct": bool(row["action_correct"]),
                "deterministic_action_correct": bool(
                    row.get("deterministic_action_correct", row["action_correct"])
                ),
                "faithfulness_pass": bool(row["faithfulness_pass"]),
                "correctness_0_to_4": int(row["correctness_0_to_4"]),
                "completeness_pass": bool(row["completeness_pass"]),
                "overall_pass": bool(row["overall_pass"]),
            }
        )
    return generated_metrics(normalized, minimum_correctness)


def select_fresh_confirmation_cases(
    requests: list[dict[str, Any]],
    deferrals: list[dict[str, Any]],
    used_ids: set[str],
    fresh_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    generator = random.Random(int(fresh_config["sampling_seed"]))
    selected_requests: list[dict[str, Any]] = []
    selected_deferrals: list[dict[str, Any]] = []
    for domain, action_quotas in fresh_config["generated_per_domain"].items():
        for action, requested_count in action_quotas.items():
            count = int(requested_count)
            pool = [
                row
                for row in requests
                if row["partition"] == "holdout"
                and row["domain"] == domain
                and row["required_action"] == action
                and row["example_id"] not in used_ids
            ]
            generator.shuffle(pool)
            if len(pool) < count:
                raise RuntimeError(
                    f"Insufficient fresh generated cases: domain={domain}, action={action}, "
                    f"available={len(pool)}, requested={count}."
                )
            selected_requests.extend(pool[:count])
    for domain, requested_count in fresh_config["abstain_per_domain"].items():
        count = int(requested_count)
        pool = [
            row
            for row in deferrals
            if row["partition"] == "holdout"
            and row["domain"] == domain
            and row["runtime_action"] == "abstain_without_generation"
            and row["example_id"] not in used_ids
        ]
        generator.shuffle(pool)
        if len(pool) < count:
            raise RuntimeError(
                f"Insufficient fresh abstentions: domain={domain}, "
                f"available={len(pool)}, requested={count}."
            )
        selected_deferrals.extend(pool[:count])
    generator.shuffle(selected_requests)
    generator.shuffle(selected_deferrals)
    return selected_requests, selected_deferrals


def benchmark_expected_runtime_action(example: dict[str, Any]) -> str:
    if "no relevant information is found" in example["target"]["reference_answer"].casefold():
        return "abstain"
    return (
        "ask_followup"
        if example["target"]["dialogue_act"] == "query_condition"
        else "answer"
    )


def normalized_deferral_action(runtime_action: str) -> str:
    if runtime_action == "abstain_without_generation":
        return "abstain"
    if runtime_action == "clarify_without_generation":
        return "ask_followup"
    return "answer"


def main() -> None:
    args = parse_args()
    config = load_json(args.config.resolve())
    if config.get("evaluation_split") != "validation" or config.get("locked_splits") != ["test"]:
        raise ValueError("This experiment must use validation while keeping test locked.")

    planner_prompt = (ROOT / config["planner_prompt_path"]).read_text(encoding="utf-8")
    reviewer_prompt = (ROOT / config["reviewer_prompt_path"]).read_text(encoding="utf-8")
    realizer_prompt = (ROOT / config["realizer_prompt_path"]).read_text(encoding="utf-8")
    selector_config = config.get("evidence_selector") or {}
    selector_prompt = (
        (ROOT / selector_config["prompt_path"]).read_text(encoding="utf-8")
        if selector_config.get("enabled", False)
        else None
    )
    if config.get("planner_prompt_addendum_path"):
        planner_prompt += "\n\n" + (
            ROOT / config["planner_prompt_addendum_path"]
        ).read_text(encoding="utf-8")
    if config.get("reviewer_prompt_addendum_path"):
        reviewer_prompt += "\n\n" + (
            ROOT / config["reviewer_prompt_addendum_path"]
        ).read_text(encoding="utf-8")
    baseline_id = config["baseline_evaluation_experiment_id"]
    baseline_report_path = (
        REPORT_ROOT
        / "automated_end_to_end_evaluation_experiments"
        / baseline_id
        / "automated_end_to_end_evaluation_report.json"
    )
    baseline_report = load_json(baseline_report_path)
    requested_ids = set(args.case_id)
    if args.case_id_file is not None:
        case_id_payload = load_json(args.case_id_file.resolve())
        requested_ids.update(str(value) for value in case_id_payload.get("example_ids", []))
    fresh_confirmation = bool(args.fresh_confirmation)
    reserved_confirmation = args.reserved_id_file is not None
    if sum([bool(requested_ids), fresh_confirmation, reserved_confirmation]) > 1:
        raise ValueError(
            "--case-id, --fresh-confirmation, and --reserved-id-file are mutually exclusive."
        )
    contract_root = (
        DATA_ROOT
        / "planned_generation_contract_experiments"
        / config["generation_contract_experiment_id"]
    )
    all_requests = list(iter_jsonl(contract_root / "requests" / "validation_requests.jsonl"))
    all_deferrals = list(iter_jsonl(contract_root / "deferrals" / "validation_deferrals.jsonl"))
    previously_used_count = 0
    baseline_cohort_complete = True
    transitioned_deferrals: list[dict[str, Any]] = []
    if reserved_confirmation:
        reserve_payload = load_json(args.reserved_id_file)
        reserve_values = (
            reserve_payload.get("example_ids", [])
            if isinstance(reserve_payload, dict)
            else reserve_payload
        )
        reserve_ids = {str(value) for value in reserve_values}
        if not reserve_ids:
            raise ValueError("The reserved ID file is empty.")
        available = {
            row["example_id"] for row in [*all_requests, *all_deferrals]
        }
        unknown = reserve_ids - available
        if unknown:
            raise ValueError(f"Unknown reserved validation IDs: {sorted(unknown)}")
        previously_used = collect_previously_used_ids()
        if (
            isinstance(reserve_payload, dict)
            and reserve_payload.get("purpose") == "final_fresh_validation_v5_before_test"
            and reserve_payload.get("generation_contract_experiment_id")
            == config["generation_contract_experiment_id"]
        ):
            # collect_previously_used_ids scans every JSON config, including this
            # just-created precommit manifest.  Its own IDs are not observations.
            previously_used -= reserve_ids
        already_used = reserve_ids & (
            previously_used | collect_mandatory_pipeline_used_ids()
        )
        if already_used:
            raise ValueError(f"Reserved IDs were already used: {sorted(already_used)}")
        prepared_rows = [row for row in all_requests if row["example_id"] in reserve_ids]
        transitioned_deferrals = [
            row for row in all_deferrals if row["example_id"] in reserve_ids
        ]
        baseline_rows = []
        baseline_abstentions = []
        comparison_id = config["development_candidate_experiment_id"]
        comparison_report = load_json(
            REPORT_ROOT
            / "mandatory_claim_pipeline_experiments"
            / comparison_id
            / "mandatory_claim_pipeline_report.json"
        )
    elif fresh_confirmation:
        used_ids = collect_previously_used_ids() | collect_mandatory_pipeline_used_ids()
        previously_used_count = len(used_ids)
        prepared_rows, baseline_abstentions = select_fresh_confirmation_cases(
            all_requests, all_deferrals, used_ids, config["fresh_confirmation"]
        )
        baseline_rows: list[dict[str, Any]] = []
        comparison_id = config["development_candidate_experiment_id"]
        comparison_report = load_json(
            REPORT_ROOT
            / "mandatory_claim_pipeline_experiments"
            / comparison_id
            / "mandatory_claim_pipeline_report.json"
        )
    else:
        if requested_ids:
            comparison_id = config.get(
                "targeted_comparison_experiment_id",
                config["development_candidate_experiment_id"],
            )
            comparison_report = load_json(
                REPORT_ROOT
                / "mandatory_claim_pipeline_experiments"
                / comparison_id
                / "mandatory_claim_pipeline_report.json"
            )
            comparison_case_path = (
                DATA_ROOT
                / "mandatory_claim_pipeline_experiments"
                / comparison_id
                / "case_results.jsonl"
            )
            comparison_cases = list(iter_jsonl(comparison_case_path))
            targeted_by_id = {row["example_id"]: row for row in comparison_cases}
            baseline_by_id = {
                row["example_id"]: row
                for row in baseline_report["cases"]
                if row.get("case_type") == "generated"
            }
            available_ids = set(targeted_by_id) | set(baseline_by_id)
            unknown_ids = requested_ids - available_ids
            if unknown_ids:
                raise ValueError(f"Unknown targeted comparison IDs: {sorted(unknown_ids)}")
            baseline_rows = [
                targeted_by_id.get(example_id) or baseline_by_id[example_id]
                for example_id in sorted(requested_ids)
            ]
            if any(example_id not in targeted_by_id for example_id in requested_ids):
                comparison_id = f"mixed_targeted_with_{baseline_id}"
            baseline_abstentions = []
        else:
            baseline_rows = [
                row for row in baseline_report["cases"] if row["case_type"] == "generated"
            ]
            baseline_abstentions = [
                row
                for row in baseline_report["cases"]
                if row["case_type"] == "deterministic_abstention"
            ]
            comparison_id = baseline_id
            comparison_report = baseline_report
        cohort_ids_from_baseline = {row["example_id"] for row in baseline_rows}
        prepared_by_id = {
            row["example_id"]: row
            for row in all_requests
            if row["example_id"] in cohort_ids_from_baseline
        }
        excluded_non_generation_ids = sorted(
            cohort_ids_from_baseline - set(prepared_by_id)
        )
        if excluded_non_generation_ids:
            deferrals_by_id = {row["example_id"]: row for row in all_deferrals}
            transitioned_deferrals = [
                deferrals_by_id[example_id]
                for example_id in excluded_non_generation_ids
                if example_id in deferrals_by_id
            ]
            unresolved_ids = set(excluded_non_generation_ids) - {
                row["example_id"] for row in transitioned_deferrals
            }
            baseline_cohort_complete = not unresolved_ids
            label = "Targeted IDs" if requested_ids else "Baseline IDs"
            print(
                f"{label} now resolved without generation and scored as deterministic "
                "transitions: " + ", ".join(excluded_non_generation_ids)
            )
        baseline_rows = [
            row for row in baseline_rows if row["example_id"] in prepared_by_id
        ]
        prepared_rows = [prepared_by_id[row["example_id"]] for row in baseline_rows]
        if not prepared_rows:
            raise ValueError("No targeted cases remain in the generation path.")
    cohort_ids = {row["example_id"] for row in prepared_rows}
    transitioned_ids = {row["example_id"] for row in transitioned_deferrals}
    benchmark_path = (
        DATA_ROOT
        / "retrieval_benchmarks"
        / config["benchmark_experiment_id"]
        / "validation.jsonl"
    )
    benchmark_by_id = {
        row["id"]: row
        for row in iter_jsonl(benchmark_path)
        if row["id"] in cohort_ids | transitioned_ids
    }
    if set(benchmark_by_id) != cohort_ids | transitioned_ids:
        raise RuntimeError("Closed cohort cannot be reconstructed from validation artifacts.")

    if config.get("structural_evidence_guard", {}).get("enabled", False):
        span_structure = build_span_structure_index()
        for prepared in prepared_rows:
            enrich_structural_evidence(prepared, span_structure)
            if config.get("runtime_checklist_continuation_guard", {}).get(
                "enabled", False
            ):
                apply_runtime_checklist_continuation(prepared)

    encoding = tiktoken.get_encoding("cl100k_base")
    pricing = config["pricing"]
    expected_input_tokens = 0
    expected_output_tokens = 0
    for prepared in prepared_rows:
        evidence_ids = [item["evidence_id"] for item in prepared["evidence_catalog"]]
        if selector_prompt is not None:
            expected_input_tokens += len(
                encoding.encode(selector_prompt + prepared["request_body"]["input"])
            )
            expected_output_tokens += int(selector_config["expected_output_tokens"])
        plan_placeholder = {
            "answer_objective": "Directly resolve the active request.",
            "mandatory_claims": [
                {
                    "claim_id": "C1",
                    "claim_text": "State the central supported result.",
                    "claim_type": "direct_answer",
                    "evidence_ids": [evidence_ids[0]],
                }
            ],
        }
        expected_input_tokens += len(
            encoding.encode(planner_prompt + prepared["request_body"]["input"])
        )
        review_input = (
            prepared["request_body"]["input"]
            + "\n\n<MANDATORY_CLAIM_DRAFT>\n"
            + json.dumps(plan_placeholder, ensure_ascii=False)
            + "\n</MANDATORY_CLAIM_DRAFT>"
        )
        expected_input_tokens += len(encoding.encode(reviewer_prompt + review_input))
        realizer_input = render_realizer_input(prepared, plan_placeholder)
        expected_input_tokens += len(encoding.encode(realizer_prompt + realizer_input))
        placeholder_public = {
            "status": prepared["expected_status"],
            "answer": "A concise supported answer.",
            "citations": [{"evidence_id": evidence_ids[0], "evidence_quote": "supported"}],
        }
        expected_input_tokens += len(
            encoding.encode(
                JUDGE_INSTRUCTIONS
                + render_judge_input(benchmark_by_id[prepared["example_id"]], prepared, placeholder_public)
            )
        )
        expected_output_tokens += sum(
            int(config[name]["expected_output_tokens"])
            for name in ("planner", "reviewer", "realizer", "judge")
        )
    expected_cost = (
        expected_input_tokens * pricing["input_usd_per_million_tokens"]
        + expected_output_tokens * pricing["output_usd_per_million_tokens"]
    ) / 1_000_000
    configured_ceiling = float(pricing["maximum_authorized_cost_usd"])
    ceiling = configured_ceiling
    if args.maximum_cost_usd is not None:
        if args.maximum_cost_usd <= 0:
            raise ValueError("--maximum-cost-usd must be positive.")
        if args.maximum_cost_usd > configured_ceiling + 1e-12:
            raise ValueError(
                "--maximum-cost-usd cannot exceed the configured authorization ceiling."
            )
        ceiling = float(args.maximum_cost_usd)
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print(f"Closed validation cohort: {len(prepared_rows)} generated cases")
    print(f"Targeted diagnostic: {bool(requested_ids)}")
    print(f"Fresh confirmation: {fresh_confirmation}")
    print(
        "Deterministic outcomes: "
        f"{len(baseline_abstentions) + len(transitioned_deferrals)}"
    )
    print(f"Locked test split opened: false")
    print(f"Expected uncached cost: ${expected_cost:.6f}")
    print(f"Hard actual-cost ceiling: ${ceiling:.4f}")
    if expected_cost > ceiling:
        if not (requested_ids or args.allow_cache_resume):
            raise RuntimeError("Expected cost exceeds configured ceiling.")
        print(
            "Cache-resume run: uncached estimate exceeds this run ceiling; "
            "the per-call hard ceiling remains enforced."
        )
    if not args.execute:
        print("Dry-run complete: no OpenAI calls.")
        return
    if args.confirm_maximum_cost_usd + 1e-12 < ceiling:
        raise RuntimeError(f"Confirmation required: --confirm-maximum-cost-usd {ceiling:.2f}")

    load_dotenv(ROOT / config["api"]["dotenv_path"])
    api_key = os.getenv(config["api"]["api_key_environment_variable"])
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from .env.")
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        timeout=float(config["api"]["request_timeout_seconds"]),
        max_retries=int(config["api"]["max_retries"]),
    )
    cache_path = DATA_ROOT / "evaluation_cache" / config["api"]["cache_sqlite_filename"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(cache_path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS responses (cache_key TEXT PRIMARY KEY, kind TEXT NOT NULL, "
        "response_json TEXT NOT NULL, created_at_utc TEXT NOT NULL)"
    )
    connection.commit()

    actual_cost = 0.0
    call_counts: Counter[str] = Counter()
    cache_hits: Counter[str] = Counter()

    def cached_call(kind: str, body: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        nonlocal actual_cost
        key = stable_hash({"kind": kind, "body": body})
        cached = connection.execute(
            "SELECT response_json FROM responses WHERE cache_key=?", (key,)
        ).fetchone()
        if cached:
            cache_hits[kind] += 1
            return json.loads(cached[0]), True
        body_input_tokens = len(encoding.encode(json.dumps(body, ensure_ascii=False)))
        maximum_call_cost = (
            body_input_tokens * pricing["input_usd_per_million_tokens"]
            + int(body["max_output_tokens"]) * pricing["output_usd_per_million_tokens"]
        ) / 1_000_000
        if actual_cost + maximum_call_cost > ceiling + 1e-12:
            raise RuntimeError(
                f"Hard cost ceiling would be exceeded before {kind}: "
                f"actual=${actual_cost:.6f}, reserved_call=${maximum_call_cost:.6f}."
            )
        response = client.responses.create(**body)
        value = response.model_dump(mode="json")
        connection.execute(
            "INSERT OR REPLACE INTO responses VALUES (?,?,?,?)",
            (key, kind, json.dumps(value, ensure_ascii=False), datetime.now(timezone.utc).isoformat()),
        )
        connection.commit()
        call_counts[kind] += 1
        actual_cost += usage_cost(value.get("usage") or {}, pricing)
        return value, False

    results: list[dict[str, Any]] = []
    maximum_claims = int(config["maximum_mandatory_claims"])
    for index, prepared in enumerate(prepared_rows, start=1):
        example = benchmark_by_id[prepared["example_id"]]
        evidence_ids = [item["evidence_id"] for item in prepared["evidence_catalog"]]
        selector_output = None
        selector_errors: list[str] = []
        selector_cached = False
        selector_applied = False
        if selector_prompt is not None:
            selector_output_schema = evidence_selector_schema(evidence_ids)
            selector_body = make_response_body(
                selector_config,
                selector_prompt,
                prepared["request_body"]["input"],
                "mandatory_claim_evidence_selector_v1",
                selector_output_schema,
            )
            selector_data, selector_cached = cached_call(
                "mandatory_claim_evidence_selector", selector_body
            )
            selector_output, selector_errors = parse_structured(
                selector_data, selector_output_schema
            )
            if selector_output is not None:
                complementary = selector_output.get("complementary_evidence_ids", [])
                if len(complementary) != len(set(complementary)):
                    selector_errors.append("duplicate_complementary_evidence_id")
                if selector_output.get("primary_evidence_id") in complementary:
                    selector_errors.append("primary_repeated_as_complementary_evidence")
            if selector_output is not None and not selector_errors:
                apply_selected_primary_evidence(
                    prepared, str(selector_output["primary_evidence_id"])
                )
                selector_applied = True
        plan_schema = planner_schema(evidence_ids, maximum_claims)
        planner_body = make_response_body(
            config["planner"],
            planner_prompt,
            prepared["request_body"]["input"],
            "mandatory_claim_plan_v1",
            plan_schema,
        )
        planner_data, planner_cached = cached_call("mandatory_claim_planner", planner_body)
        plan, parse_errors = parse_structured(planner_data, plan_schema)
        plan, plan_sanitizations = sanitize_redundant_conflicting_claims(plan, prepared)
        plan_errors = [
            *parse_errors,
            *validate_plan(plan, plan_schema, prepared["required_action"]),
            *validate_branch_outcome(plan, prepared),
            *validate_primary_evidence_usage(plan, prepared),
            *validate_conditional_qualifier_support(plan, prepared),
            *validate_structural_claim_coverage(plan, prepared),
        ]
        planner_retry_used = False
        planner_retry_count = 0
        for repair_attempt in range(
            1, int(config.get("maximum_plan_repair_attempts", 1)) + 1
        ):
            if not plan_errors:
                break
            planner_retry_used = True
            planner_retry_count = repair_attempt
            retry_stage = {
                **config["planner"],
                "max_output_tokens": int(config["planner"]["max_output_tokens"]) * 2,
            }
            retry_input = "\n\n".join(
                [
                    prepared["request_body"]["input"],
                    "<PREVIOUS_VALIDATION_ERRORS>\n"
                    + "\n".join(plan_errors)
                    + "\n</PREVIOUS_VALIDATION_ERRORS>",
                    "The previous output was incomplete or invalid. Return one complete valid plan.",
                ]
            )
            retry_body = make_response_body(
                retry_stage,
                planner_prompt,
                retry_input,
                f"mandatory_claim_plan_retry_v{repair_attempt}",
                plan_schema,
            )
            retry_kind = (
                "mandatory_claim_planner_retry"
                if repair_attempt == 1
                else f"mandatory_claim_planner_retry_{repair_attempt}"
            )
            retry_data, retry_cached = cached_call(retry_kind, retry_body)
            planner_cached = planner_cached and retry_cached
            plan, parse_errors = parse_structured(retry_data, plan_schema)
            plan, retry_sanitizations = sanitize_redundant_conflicting_claims(plan, prepared)
            plan_sanitizations.extend(retry_sanitizations)
            plan_errors = [
                *parse_errors,
                *validate_plan(plan, plan_schema, prepared["required_action"]),
                *validate_branch_outcome(plan, prepared),
                *validate_primary_evidence_usage(plan, prepared),
                *validate_conditional_qualifier_support(plan, prepared),
                *validate_structural_claim_coverage(plan, prepared),
            ]
        draft_plan = plan
        review_applied = False
        review_cache_hit = False
        review_errors: list[str] = []
        if not plan_errors and should_review_plan(prepared, plan):
            review_applied = True
            review_input = "\n\n".join(
                [
                    prepared["request_body"]["input"],
                    "<MANDATORY_CLAIM_DRAFT>\n"
                    + json.dumps(plan, ensure_ascii=False, indent=2)
                    + "\n</MANDATORY_CLAIM_DRAFT>",
                ]
            )
            review_body = make_response_body(
                config["reviewer"],
                reviewer_prompt,
                review_input,
                "mandatory_claim_plan_review_v1",
                plan_schema,
            )
            review_data, review_cache_hit = cached_call(
                "mandatory_claim_plan_reviewer", review_body
            )
            reviewed_plan, review_parse_errors = parse_structured(review_data, plan_schema)
            reviewed_plan, review_sanitizations = sanitize_redundant_conflicting_claims(
                reviewed_plan, prepared
            )
            plan_sanitizations.extend(review_sanitizations)
            review_errors = [
                *review_parse_errors,
                *validate_plan(reviewed_plan, plan_schema, prepared["required_action"]),
                *validate_branch_outcome(reviewed_plan, prepared),
                *validate_primary_evidence_usage(reviewed_plan, prepared),
                *validate_conditional_qualifier_support(reviewed_plan, prepared),
                *validate_structural_claim_coverage(reviewed_plan, prepared),
            ]
            if not review_errors:
                plan = reviewed_plan

        public_payload = None
        realization = None
        realization_errors: list[str] = []
        realizer_cached = False
        repair_used = False
        review_fallback_used = False
        if not plan_errors:
            claim_ids = [claim["claim_id"] for claim in plan["mandatory_claims"]]
            real_schema = realization_schema(claim_ids, evidence_ids)
            realizer_input = render_realizer_input(prepared, plan)
            output_limits = config.get("adaptive_output_limits", {})
            realizer_stage = config["realizer"]
            if output_limits.get("enabled", False):
                adaptive_realizer_limit = int(output_limits["realizer_base_tokens"]) + int(
                    output_limits["realizer_tokens_per_claim"]
                ) * len(claim_ids)
                realizer_stage = {
                    **config["realizer"],
                    "max_output_tokens": min(
                        int(config["realizer"]["max_output_tokens"]),
                        adaptive_realizer_limit,
                    ),
                }
            realizer_body = make_response_body(
                realizer_stage,
                realizer_prompt,
                realizer_input,
                "mandatory_claim_realization_v1",
                real_schema,
            )
            realizer_data, realizer_cached = cached_call("mandatory_claim_realizer", realizer_body)
            realization, parse_errors = parse_structured(realizer_data, real_schema)
            public_payload, realization_errors = validate_and_assemble(
                prepared, plan, realization, real_schema
            )
            realization_errors = [*parse_errors, *realization_errors]
            if realization_errors and int(config["maximum_repair_attempts"]) > 0:
                repair_used = True
                repair_input = "\n\n".join(
                    [
                        realizer_input,
                        "<PREVIOUS_INVALID_OUTPUT>\n"
                        + json.dumps(realization, ensure_ascii=False)
                        + "\n</PREVIOUS_INVALID_OUTPUT>",
                        "<VALIDATION_ERRORS>\n"
                        + "\n".join(realization_errors)
                        + "\n</VALIDATION_ERRORS>",
                        "Return a corrected complete realization.",
                    ]
                )
                repair_body = make_response_body(
                    realizer_stage,
                    realizer_prompt,
                    repair_input,
                    "mandatory_claim_realization_repair_v1",
                    real_schema,
                )
                repair_data, repair_cached = cached_call("mandatory_claim_realizer_repair", repair_body)
                realizer_cached = realizer_cached and repair_cached
                realization, parse_errors = parse_structured(repair_data, real_schema)
                public_payload, realization_errors = validate_and_assemble(
                    prepared, plan, realization, real_schema
                )
                realization_errors = [*parse_errors, *realization_errors]

            # A reviewer may accidentally make a valid small draft less grounded.
            # If its reviewed realization still violates the strict citation contract
            # after repair, retry the already-validated draft instead of publishing
            # an empty/invalid response.
            if realization_errors and review_applied and draft_plan and draft_plan != plan:
                fallback_plan_errors = [
                    *validate_plan(draft_plan, plan_schema, prepared["required_action"]),
                    *validate_branch_outcome(draft_plan, prepared),
                    *validate_primary_evidence_usage(draft_plan, prepared),
                    *validate_conditional_qualifier_support(draft_plan, prepared),
                    *validate_structural_claim_coverage(draft_plan, prepared),
                ]
                if not fallback_plan_errors:
                    review_fallback_used = True
                    fallback_claim_ids = [
                        claim["claim_id"] for claim in draft_plan["mandatory_claims"]
                    ]
                    fallback_schema = realization_schema(fallback_claim_ids, evidence_ids)
                    fallback_input = render_realizer_input(prepared, draft_plan)
                    fallback_realizer_stage = config["realizer"]
                    if output_limits.get("enabled", False):
                        fallback_limit = int(output_limits["realizer_base_tokens"]) + int(
                            output_limits["realizer_tokens_per_claim"]
                        ) * len(fallback_claim_ids)
                        fallback_realizer_stage = {
                            **config["realizer"],
                            "max_output_tokens": min(
                                int(config["realizer"]["max_output_tokens"]),
                                fallback_limit,
                            ),
                        }
                    fallback_body = make_response_body(
                        fallback_realizer_stage,
                        realizer_prompt,
                        fallback_input,
                        "mandatory_claim_realization_draft_fallback_v1",
                        fallback_schema,
                    )
                    fallback_data, fallback_cached = cached_call(
                        "mandatory_claim_realizer_draft_fallback", fallback_body
                    )
                    realizer_cached = realizer_cached and fallback_cached
                    fallback_realization, fallback_parse_errors = parse_structured(
                        fallback_data, fallback_schema
                    )
                    fallback_payload, fallback_errors = validate_and_assemble(
                        prepared, draft_plan, fallback_realization, fallback_schema
                    )
                    fallback_errors = [*fallback_parse_errors, *fallback_errors]
                    if not fallback_errors:
                        plan = draft_plan
                        realization = fallback_realization
                        public_payload = fallback_payload
                        realization_errors = []

        contract_valid = not plan_errors and not realization_errors and public_payload is not None
        citations_valid = bool(contract_valid and public_payload["citations"])
        judge_output = None
        judge_contract_valid = False
        judge_cached = False
        judge_errors: list[str] = []
        if contract_valid:
            judge_stage = config["judge"]
            if config.get("adaptive_output_limits", {}).get("enabled", False):
                judge_stage = {
                    **config["judge"],
                    "max_output_tokens": min(
                        int(config["judge"]["max_output_tokens"]),
                        int(config["adaptive_output_limits"]["judge_max_output_tokens"]),
                    ),
                }
            judge_body = make_response_body(
                judge_stage,
                JUDGE_INSTRUCTIONS,
                render_judge_input(example, prepared, public_payload),
                "mandatory_claim_pipeline_judgment_v1",
                judge_schema(),
            )
            judge_data, judge_cached = cached_call("mandatory_claim_judge", judge_body)
            judge_output, judge_errors = parse_structured(judge_data, judge_schema())
            judge_contract_valid = not judge_errors and judge_output is not None
            if not judge_contract_valid:
                retry_stage = {
                    **judge_stage,
                    "max_output_tokens": int(judge_stage["max_output_tokens"]) * 2,
                }
                retry_body = make_response_body(
                    retry_stage,
                    JUDGE_INSTRUCTIONS,
                    render_judge_input(example, prepared, public_payload),
                    "mandatory_claim_pipeline_judgment_retry_v1",
                    judge_schema(),
                )
                retry_data, retry_cached = cached_call("mandatory_claim_judge_retry", retry_body)
                judge_cached = judge_cached and retry_cached
                judge_output, judge_errors = parse_structured(retry_data, judge_schema())
                judge_contract_valid = not judge_errors and judge_output is not None

        correctness = int(judge_output["correctness_0_to_4"]) if judge_contract_valid else 0
        faithfulness = bool(judge_output["faithfulness_pass"]) if judge_contract_valid else False
        completeness = bool(judge_output["completeness_pass"]) if judge_contract_valid else False
        action_correct = bool(judge_output["action_correct"]) if judge_contract_valid else False
        expected_runtime_action = benchmark_expected_runtime_action(example)
        deterministic_action_correct = prepared["required_action"] == expected_runtime_action
        overall_pass = bool(
            contract_valid
            and citations_valid
            and judge_contract_valid
            and deterministic_action_correct
            and action_correct
            and correctness >= int(config["minimum_acceptable_correctness"])
            and faithfulness
            and completeness
        )
        selected_span_ids = {
            item["span_id"]
            for item in prepared["evidence_catalog"]
            if plan and item["evidence_id"] in {
                evidence_id
                for claim in plan.get("mandatory_claims", [])
                for evidence_id in claim.get("evidence_ids", [])
            }
        }
        gold_ids = set(example["gold"]["span_ids"])
        gold_present = gold_ids.intersection(
            {item["span_id"] for item in prepared["evidence_catalog"]}
        )
        results.append(
            {
                "example_id": prepared["example_id"],
                "domain": prepared["domain"],
                "required_action": prepared["required_action"],
                "evidence_selector_applied": selector_applied,
                "evidence_selector_cache_hit": selector_cached,
                "evidence_selector_output": selector_output,
                "evidence_selector_errors": selector_errors,
                "planner_cache_hit": planner_cached,
                "planner_retry_used": planner_retry_used,
                "planner_retry_count": planner_retry_count,
                "review_applied": review_applied,
                "review_cache_hit": review_cache_hit,
                "realizer_cache_hit": realizer_cached,
                "judge_cache_hit": judge_cached,
                "repair_used": repair_used,
                "review_fallback_used": review_fallback_used,
                "plan_sanitizations": plan_sanitizations,
                "draft_plan": draft_plan,
                "plan": plan,
                "plan_errors": plan_errors,
                "review_errors": review_errors,
                "realization": realization,
                "realization_errors": realization_errors,
                "public_payload": public_payload,
                "contract_valid": contract_valid,
                "citations_valid": citations_valid,
                "judge_contract_valid": judge_contract_valid,
                "judge_errors": judge_errors,
                "judge": judge_output,
                "correctness_0_to_4": correctness,
                "faithfulness_pass": faithfulness,
                "completeness_pass": completeness,
                "action_correct": action_correct,
                "deterministic_action_correct": deterministic_action_correct,
                "overall_pass": overall_pass,
                "planned_claim_count": len(plan.get("mandatory_claims", [])) if plan else 0,
                "selected_evidence_count": len(selected_span_ids),
                "structural_evidence_chain_count": len(
                    prepared.get("structural_evidence_chains", [])
                ),
                "runtime_checklist_continuation_applied": bool(
                    prepared.get("runtime_checklist_continuation_applied", False)
                ),
                "runtime_checklist_target_evidence_id": prepared.get(
                    "runtime_checklist_target_evidence_id"
                ),
                "gold_evidence_recall_when_present": (
                    len(selected_span_ids.intersection(gold_present)) / len(gold_present)
                    if gold_present
                    else None
                ),
            }
        )
        print(
            f"[{index:02d}/{len(prepared_rows)}] {prepared['example_id']} "
            f"claims={results[-1]['planned_claim_count']} contract={contract_valid} "
            f"faith={faithfulness} complete={completeness} score={correctness} "
            f"cost=${actual_cost:.6f}",
            flush=True,
        )

    connection.close()
    transitioned_results = []
    for row in transitioned_deferrals:
        example = benchmark_by_id[row["example_id"]]
        runtime_action = normalized_deferral_action(str(row["runtime_action"]))
        expected_action = benchmark_expected_runtime_action(example)
        action_correct = runtime_action == expected_action
        transitioned_results.append(
            {
                "example_id": row["example_id"],
                "domain": row["domain"],
                "runtime_action": runtime_action,
                "source_runtime_action": row["runtime_action"],
                "expected_runtime_action": expected_action,
                "action_correct": action_correct,
                "overall_pass": action_correct,
                "reason": row.get("reason"),
                "deterministic_response": row.get("deterministic_response"),
            }
        )
    minimum_correctness = int(config["minimum_acceptable_correctness"])
    candidate_generated = generated_metrics(results, minimum_correctness)
    targeted_diagnostic = bool(requested_ids)
    if fresh_confirmation or reserved_confirmation:
        # There is no matched baseline generation for a never-before-used cohort.
        # Historical development metrics are useful context, but subtracting them
        # from this cohort would be a statistically invalid delta.
        baseline_generated = None
        baseline_total_overall = None
        baseline_total_action = None
        baseline_total_examples = None
        baseline_comparison_valid = False
    else:
        baseline_generated = baseline_generated_metrics(baseline_rows, minimum_correctness)
        baseline_comparison_valid = baseline_cohort_complete
        baseline_total_overall = (
            baseline_generated["overall_pass_rate"]
            if targeted_diagnostic
            else float(baseline_report["metrics"]["overall_pass_rate"])
        )
        baseline_total_action = (
            baseline_generated["action_accuracy"]
            if targeted_diagnostic
            else float(baseline_report["metrics"]["action_accuracy"])
        )
        baseline_total_examples = (
            len(baseline_rows) + len(transitioned_results)
            if targeted_diagnostic
            else int(baseline_report["metrics"]["examples"])
        )
    total_examples = len(results) + len(baseline_abstentions) + len(transitioned_results)
    deterministic_overall_passes = len(baseline_abstentions) + sum(
        row["overall_pass"] for row in transitioned_results
    )
    deterministic_action_passes = len(baseline_abstentions) + sum(
        row["action_correct"] for row in transitioned_results
    )
    candidate_total_overall = (
        sum(row["overall_pass"] for row in results) + deterministic_overall_passes
    ) / total_examples
    candidate_total_action = (
        sum(row["action_correct"] for row in results) + deterministic_action_passes
    ) / total_examples
    claim_counts = [row["planned_claim_count"] for row in results]
    recall_values = [
        row["gold_evidence_recall_when_present"]
        for row in results
        if row["gold_evidence_recall_when_present"] is not None
    ]
    planner_metrics = {
        "plan_valid_rate": mean_bool(
            [{"valid": not row["plan_errors"]} for row in results], "valid"
        ),
        "mean_planned_claims": float(np.mean(claim_counts)) if claim_counts else 0.0,
        "mean_selected_evidence": float(
            np.mean([row["selected_evidence_count"] for row in results])
        ),
        "mean_gold_evidence_recall_when_present": (
            float(np.mean(recall_values)) if recall_values else None
        ),
        "repair_rate": float(np.mean([row["repair_used"] for row in results])),
        "planner_retry_rate": float(
            np.mean([row["planner_retry_used"] for row in results])
        ),
        "review_rate": float(np.mean([row["review_applied"] for row in results])),
    }
    deltas = (
        {
            "generated_overall_pass_rate": candidate_generated["overall_pass_rate"] - baseline_generated["overall_pass_rate"],
            "total_overall_pass_rate": candidate_total_overall - baseline_total_overall,
            "total_action_accuracy": candidate_total_action - baseline_total_action,
            "faithfulness_pass_rate": candidate_generated["faithfulness_pass_rate"] - baseline_generated["faithfulness_pass_rate"],
            "correctness_acceptable_rate": candidate_generated["correctness_acceptable_rate"] - baseline_generated["correctness_acceptable_rate"],
            "completeness_pass_rate": candidate_generated["completeness_pass_rate"] - baseline_generated["completeness_pass_rate"],
        }
        if baseline_comparison_valid
        else None
    )
    gates = config["acceptance_gates"]
    gate_results = {
        "action_accuracy": candidate_total_action >= gates["minimum_action_accuracy"],
        "contract_valid_rate": candidate_generated["contract_valid_rate"]
        >= gates["minimum_contract_valid_rate"],
        "citation_valid_rate": candidate_generated["citation_valid_rate"]
        >= gates["minimum_citation_valid_rate"],
        "faithfulness_pass_rate": candidate_generated["faithfulness_pass_rate"]
        >= gates["minimum_faithfulness_pass_rate"],
        "correctness_acceptable_rate": candidate_generated["correctness_acceptable_rate"]
        >= gates["minimum_correctness_acceptable_rate"],
        "completeness_pass_rate": candidate_generated["completeness_pass_rate"]
        >= gates["minimum_completeness_pass_rate"],
        "total_overall_pass_rate": candidate_total_overall >= gates["minimum_overall_pass_rate"],
        "hallucination_rate": candidate_generated["hallucination_rate"]
        <= gates["maximum_hallucination_rate"],
        "mean_planned_claims": planner_metrics["mean_planned_claims"]
        <= gates["maximum_mean_planned_claims"],
        "cost_ceiling": actual_cost <= ceiling,
        "test_split_locked": True,
    }
    if not (fresh_confirmation or reserved_confirmation):
        gate_results["matched_baseline_comparison_available"] = baseline_comparison_valid
        if baseline_comparison_valid and deltas is not None:
            gate_results["overall_delta"] = (
                deltas["total_overall_pass_rate"]
                >= gates["minimum_delta_overall_pass_rate"]
            )
            gate_results["completeness_delta"] = (
                deltas["completeness_pass_rate"]
                >= gates["minimum_delta_completeness_pass_rate"]
            )
            gate_results["faithfulness_non_regression"] = (
                deltas["faithfulness_pass_rate"]
                >= -gates["maximum_faithfulness_regression"]
            )
    gates_passed = all(gate_results.values()) and not targeted_diagnostic
    experiment_payload = {
        "config": config,
        "targeted_case_ids": sorted(requested_ids),
        "fresh_confirmation": fresh_confirmation,
        "reserved_confirmation": reserved_confirmation,
        "cohort_ids": sorted(cohort_ids | transitioned_ids),
        "selector_prompt_sha256": stable_hash(selector_prompt) if selector_prompt else None,
        "planner_prompt_sha256": stable_hash(planner_prompt),
        "reviewer_prompt_sha256": stable_hash(reviewer_prompt),
        "realizer_prompt_sha256": stable_hash(realizer_prompt),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependency_sha256": {
            "plan_contract.py": hashlib.sha256(
                (ROOT / "scripts" / "plan_contract.py").read_bytes()
            ).hexdigest()
        },
    }
    suffix = (
        "reserve"
        if reserved_confirmation
        else ("fresh" if fresh_confirmation else ("targeted" if targeted_diagnostic else "full"))
    )
    experiment_id = f"{config['experiment_name']}_{suffix}_{stable_hash(experiment_payload)[:10]}"
    artifact_root = DATA_ROOT / "mandatory_claim_pipeline_experiments" / experiment_id
    report_root = REPORT_ROOT / "mandatory_claim_pipeline_experiments" / experiment_id
    write_jsonl(artifact_root / "case_results.jsonl", results)
    report = {
        "schema_version": "1.0.0",
        "pipeline_stage": "mandatory_claim_pipeline_evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "baseline_experiment_id": comparison_id,
        "protocol": {
            "split": "validation",
            "cohort": (
                "precommitted_final_validation_reserve"
                if reserved_confirmation
                else ("fresh_unused_validation_holdout" if fresh_confirmation else config["cohort"])
            ),
            "closed_regression_cohort": not (fresh_confirmation or reserved_confirmation),
            "baseline_comparison_valid": baseline_comparison_valid,
            "baseline_comparison_note": (
                "Matched example IDs and cohort."
                if baseline_comparison_valid
                else "No matched baseline exists for this fresh/reserved cohort; historical development metrics are not used as deltas."
            ),
            "fresh_confirmation": fresh_confirmation,
            "reserved_confirmation": reserved_confirmation,
            "sampling_seed": (
                config["fresh_confirmation"]["sampling_seed"] if fresh_confirmation else None
            ),
            "previously_used_ids_excluded": previously_used_count,
            "selected_generated_ids": [row["example_id"] for row in prepared_rows],
            "selected_abstention_ids": [row["example_id"] for row in baseline_abstentions],
            "selected_transition_deferral_ids": [
                row["example_id"] for row in transitioned_results
            ],
            "targeted_diagnostic": targeted_diagnostic,
            "targeted_case_ids": sorted(requested_ids),
            "planner_received_reference_answer": False,
            "planner_received_gold_evidence": False,
            "reviewer_received_reference_answer": False,
            "reviewer_received_gold_evidence": False,
            "realizer_received_reference_answer": False,
            "realizer_received_gold_evidence": False,
            "test_split_opened": False,
            "human_annotations_requested": 0,
        },
        "architecture": {
            "stages": [
                "fixed_action_and_retrieved_evidence",
                *(["independent_evidence_selector"] if selector_prompt is not None else []),
                "mandatory_claim_planner",
                "selective_mandatory_claim_plan_reviewer",
                "controlled_claim_realizer",
                "deterministic_claim_and_citation_validator",
                "unsupported_organization_attribution_validator",
                "one_shot_repair_on_contract_failure",
                "reviewed_plan_fallback_to_valid_draft",
                "offline_quality_judge",
            ],
            "public_answer_assembled_by_code": True,
        },
        "cost": {
            "expected_uncached_usd": expected_cost,
            "actual_usd": actual_cost,
            "authorized_ceiling_usd": ceiling,
            "configured_ceiling_usd": configured_ceiling,
            "new_calls_by_stage": dict(call_counts),
            "cache_hits_by_stage": dict(cache_hits),
        },
        "baseline_generated_metrics": baseline_generated,
        "candidate_generated_metrics": candidate_generated,
        "baseline_total_metrics": {
            "examples": baseline_total_examples,
            "action_accuracy": baseline_total_action,
            "overall_pass_rate": baseline_total_overall,
        },
        "candidate_total_metrics": {
            "examples": total_examples,
            "action_accuracy": candidate_total_action,
            "overall_pass_rate": candidate_total_overall,
            "overall_wilson_95": wilson(
                sum(row["overall_pass"] for row in results)
                + deterministic_overall_passes,
                total_examples,
            ),
        },
        "transitioned_deterministic_results": transitioned_results,
        "planner_metrics": planner_metrics,
        "deltas": deltas,
        "gate_results": gate_results,
        "gates_passed": gates_passed,
        "decision": (
            "targeted_development_diagnostic"
            if targeted_diagnostic
            else (
                (
                    "freeze_candidate_and_authorize_one_time_test"
                    if gates_passed
                    else "fresh_confirmation_failed_keep_test_locked"
                )
                if (fresh_confirmation or reserved_confirmation)
                else (
                    "accept_for_fresh_confirmation"
                    if gates_passed
                    else "reject_or_iterate_on_development"
                )
            )
        ),
        "cases": results,
        "limitations": [
            (
                "Fresh validation coverage has no DMV/VA follow-up cases and no VA abstentions "
                "because none remained unused."
                if fresh_confirmation
                else "This is a closed validation regression cohort, not a fresh confirmation sample."
            ),
            "The quality judge is automatic; test remains unopened.",
            "Gold-evidence recall is diagnostic only and is never exposed to planner or realizer.",
        ],
    }
    write_json(report_root / "mandatory_claim_pipeline_report.json", report)
    write_json(
        REPORT_ROOT / "mandatory_claim_pipeline" / "latest.json",
        {
            "experiment_id": experiment_id,
            "report_path": str(
                report_root.relative_to(ROOT) / "mandatory_claim_pipeline_report.json"
            ),
            "artifact_path": str(artifact_root.relative_to(ROOT) / "case_results.jsonl"),
        },
    )
    print("\n=== RESULT ===")
    print(f"Experiment: {experiment_id}")
    print(
        f"Baseline total overall: {baseline_total_overall:.4f}"
        if baseline_total_overall is not None
        else "Baseline total overall: unavailable (no matched cohort)"
    )
    print(f"Candidate total overall: {candidate_total_overall:.4f}")
    print(f"Generated faithfulness: {candidate_generated['faithfulness_pass_rate']:.4f}")
    print(f"Generated completeness: {candidate_generated['completeness_pass_rate']:.4f}")
    print(f"Actual cost: ${actual_cost:.6f}")
    print(f"Decision: {report['decision']}")


if __name__ == "__main__":
    main()
