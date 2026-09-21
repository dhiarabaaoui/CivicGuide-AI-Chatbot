"""Run the frozen structural candidate on externally locked human references.

This is a controlled-generation evaluation: the locked human action and supplied
official evidence are inputs to the frozen planner/reviewer/realizer. It does not
claim to evaluate autonomous retrieval or action routing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import tiktoken
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "processed" / "multidoc2dial_v1"
PACK_ROOT = DATA_ROOT / "clean_validation" / "external_annotation_pack_v1"
TASKS_PATH = PACK_ROOT / "annotation_tasks.jsonl"
REFERENCES_PATH = (
    PACK_ROOT
    / "human_reference_annotations_v2_traced"
    / "reference_annotations.jsonl"
)
CONFIG_PATH = ROOT / "configs" / "mandatory_claim_pipeline.json"
FREEZE_PATH = (
    DATA_ROOT
    / "candidate_freezes"
    / "rag_candidate_structural_v1_62383ac6c4ff"
    / "candidate_manifest.json"
)
FROZEN_PIPELINE_PATH = FREEZE_PATH.parent / "snapshot" / "scripts" / "evaluate_mandatory_claim_pipeline.py"

_pipeline_spec = importlib.util.spec_from_file_location(
    "frozen_evaluate_mandatory_claim_pipeline", FROZEN_PIPELINE_PATH
)
if _pipeline_spec is None or _pipeline_spec.loader is None:
    raise RuntimeError(f"Cannot load frozen pipeline module: {FROZEN_PIPELINE_PATH}")
_pipeline = importlib.util.module_from_spec(_pipeline_spec)
_pipeline_spec.loader.exec_module(_pipeline)
make_response_body = _pipeline.make_response_body
parse_structured = _pipeline.parse_structured
planner_schema = _pipeline.planner_schema
realization_schema = _pipeline.realization_schema
render_realizer_input = _pipeline.render_realizer_input
sanitize_redundant_conflicting_claims = _pipeline.sanitize_redundant_conflicting_claims
should_review_plan = _pipeline.should_review_plan
stable_hash = _pipeline.stable_hash
usage_cost = _pipeline.usage_cost
validate_and_assemble = _pipeline.validate_and_assemble
validate_branch_outcome = _pipeline.validate_branch_outcome
validate_conditional_qualifier_support = _pipeline.validate_conditional_qualifier_support
validate_plan = _pipeline.validate_plan
validate_primary_evidence_usage = _pipeline.validate_primary_evidence_usage
validate_structural_claim_coverage = _pipeline.validate_structural_claim_coverage

RESULT_ROOT = (
    DATA_ROOT
    / "clean_validation"
    / "external_candidate_runs"
    / "external_locked_reference_candidate_v1"
)
REPORT_ROOT = (
    ROOT
    / "reports"
    / "generated"
    / "multidoc2dial_v1"
    / "external_clean_validation"
    / "external_locked_reference_candidate_v1"
)
AUTHORIZED_MAXIMUM_COST_USD = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-maximum-cost-usd", type=float, default=0.0)
    parser.add_argument("--maximum-cost-usd", type=float, default=AUTHORIZED_MAXIMUM_COST_USD)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_text_sha256(path: Path) -> str:
    """Match the original LF bytes despite Windows snapshot newline expansion."""
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def verify_frozen_candidate() -> dict[str, Any]:
    manifest = read_json(FREEZE_PATH)
    errors = []
    for relative_path, metadata in manifest["artifacts"].items():
        path = FREEZE_PATH.parent / "snapshot" / relative_path
        if not path.exists():
            errors.append(f"missing:{relative_path}")
        elif frozen_text_sha256(path) != metadata["sha256"]:
            errors.append(f"hash_mismatch:{relative_path}")
    if errors:
        raise RuntimeError("Frozen candidate integrity failure: " + ", ".join(errors))
    return manifest


def conversation_for(task: dict[str, Any]) -> list[dict[str, str]]:
    conversation = task.get("conversation") or task.get("conversation_draft") or []
    if not conversation:
        raise ValueError(f"{task['case_id']}: missing conversation")
    return conversation


def build_prepared(task: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    action = reference["expected_action"]
    if action not in {"answer", "ask_followup"}:
        raise ValueError(f"{task['case_id']}: unsupported generated action {action}")
    evidence_catalog = []
    for index, evidence in enumerate(task["source_evidence"], start=1):
        evidence_catalog.append(
            {
                "evidence_id": f"E{index}",
                "span_id": evidence["id"],
                "document_id": task["source_document_id"],
                "section_id": evidence["id"],
                "text": evidence["text"],
                "title": evidence.get("title", ""),
                "priority_rank": index,
            }
        )
    if not evidence_catalog:
        raise ValueError(f"{task['case_id']}: no supplied evidence")

    conversation = conversation_for(task)
    conversation_text = "\n".join(
        f"{'AGENT' if turn['role'].lower() in {'assistant', 'agent'} else 'USER'}: {turn['text']}"
        for turn in conversation
    )
    previous_agent = next(
        (
            turn["text"]
            for turn in reversed(conversation[:-1])
            if turn["role"].lower() in {"assistant", "agent"}
        ),
        "NONE",
    )
    branch_mode = "ask_next_unresolved_condition" if action == "ask_followup" else "direct_answer"
    execution_plan = "\n".join(
        [
            "<EXECUTION_PLAN>",
            f"REQUIRED_ACTION: {action}",
            f"BRANCH_MODE: {branch_mode}",
            "USER_RESPONSE_POLARITY: unknown",
            "USER_RESPONSE_POLARITY_CONFIDENCE: 0.00",
            "USER_RESPONSE_POLARITY_SOURCE: none",
            f"PREVIOUS_AGENT_UTTERANCE: {previous_agent}",
            "PRIMARY_EVIDENCE_ID: E1",
            f"DOMAIN: {task['domain']}",
            "</EXECUTION_PLAN>",
        ]
    )
    evidence_blocks = []
    for item in evidence_catalog:
        evidence_blocks.append(
            "\n".join(
                [
                    f"[{item['evidence_id']}]",
                    f"Document: {task['source_document_id']}",
                    f"Section: {item['title']}",
                    f"Span ID: {item['span_id']}",
                    f"Text: {item['text']}",
                ]
            )
        )
    input_text = "\n\n".join(
        [
            execution_plan,
            f"<CONVERSATION>\n{conversation_text}\n</CONVERSATION>",
            "<EVIDENCE_UNTRUSTED_DATA>\n"
            + "\n\n---\n\n".join(evidence_blocks)
            + "\n</EVIDENCE_UNTRUSTED_DATA>",
        ]
    )
    return {
        "example_id": task["case_id"],
        "domain": task["domain"],
        "required_action": action,
        "expected_status": "clarification_required" if action == "ask_followup" else "answered",
        "evidence_catalog": evidence_catalog,
        "structural_evidence_chains": [],
        "request_body": {"input": input_text},
    }


def plan_validation_errors(
    plan: Any, schema: dict[str, Any], prepared: dict[str, Any], parse_errors: list[str]
) -> list[str]:
    return [
        *parse_errors,
        *validate_plan(plan, schema, prepared["required_action"]),
        *validate_branch_outcome(plan, prepared),
        *validate_primary_evidence_usage(plan, prepared),
        *validate_conditional_qualifier_support(plan, prepared),
        *validate_structural_claim_coverage(plan, prepared),
    ]


def scoring_row(
    task: dict[str, Any], reference: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    return {
        "task_id": task["case_id"],
        "domain": task["domain"],
        "scenario": task["scenario"],
        "conversation": conversation_for(task),
        "source_evidence": task["source_evidence"],
        "locked_human_reference": {
            "annotator_id": reference.get("annotator_identity"),
            "annotation_date": reference.get("annotation_date"),
            "expected_action": reference["expected_action"],
            "reference_answer": reference["reference_answer"],
            "supporting_evidence_ids": reference["supporting_evidence_ids"],
        },
        "candidate_response": result["candidate_response"],
        "candidate_execution_metadata": {
            "candidate_id": result["candidate_id"],
            "contract_valid": result["contract_valid"],
            "action_and_evidence_source": result["action_and_evidence_source"],
        },
        "human_candidate_evaluation": {
            "evaluator_id": None,
            "completed_at_utc": None,
            "action_correct": None,
            "contract_valid": None,
            "citation_valid": None,
            "faithfulness_pass": None,
            "correctness_0_to_4": None,
            "completeness_pass": None,
            "overall_pass": None,
            "notes": None,
        },
    }


def main() -> None:
    args = parse_args()
    if args.maximum_cost_usd <= 0 or args.maximum_cost_usd > AUTHORIZED_MAXIMUM_COST_USD + 1e-12:
        raise ValueError(f"Maximum cost must be in (0, {AUTHORIZED_MAXIMUM_COST_USD:.2f}].")
    manifest = verify_frozen_candidate()
    frozen_snapshot_root = FREEZE_PATH.parent / "snapshot"
    config = read_json(frozen_snapshot_root / "configs" / "mandatory_claim_pipeline.json")
    tasks = read_jsonl(TASKS_PATH)
    references = read_jsonl(REFERENCES_PATH)
    task_by_id = {row["case_id"]: row for row in tasks}
    reference_by_id = {row["task_id"]: row for row in references}
    if len(tasks) != 80 or set(task_by_id) != set(reference_by_id):
        raise RuntimeError("The locked 80-task pack and traced references do not match.")
    if not all(row.get("reference_locked") is True for row in references):
        raise RuntimeError("All human references must be locked before candidate execution.")

    planner_prompt = (frozen_snapshot_root / config["planner_prompt_path"]).read_text(encoding="utf-8")
    reviewer_prompt = (frozen_snapshot_root / config["reviewer_prompt_path"]).read_text(encoding="utf-8")
    realizer_prompt = (frozen_snapshot_root / config["realizer_prompt_path"]).read_text(encoding="utf-8")
    generated_prepared = [
        build_prepared(task, reference_by_id[task["case_id"]])
        for task in tasks
        if reference_by_id[task["case_id"]]["expected_action"] != "abstain"
    ]
    abstention_count = len(tasks) - len(generated_prepared)

    encoding = tiktoken.get_encoding("cl100k_base")
    pricing = config["pricing"]
    expected_input_tokens = 0
    expected_output_tokens = 0
    for prepared in generated_prepared:
        placeholder_plan = {
            "answer_objective": "Resolve the active request from supplied evidence.",
            "mandatory_claims": [
                {
                    "claim_id": "C1",
                    "claim_text": "State the supported response.",
                    "claim_type": (
                        "followup_question"
                        if prepared["required_action"] == "ask_followup"
                        else "direct_answer"
                    ),
                    "evidence_ids": ["E1"],
                }
            ],
        }
        source_input = prepared["request_body"]["input"]
        expected_input_tokens += len(encoding.encode(planner_prompt + source_input))
        expected_input_tokens += len(
            encoding.encode(
                reviewer_prompt
                + source_input
                + json.dumps(placeholder_plan, ensure_ascii=False)
            )
        )
        expected_input_tokens += len(
            encoding.encode(realizer_prompt + render_realizer_input(prepared, placeholder_plan))
        )
        expected_output_tokens += sum(
            int(config[name]["expected_output_tokens"])
            for name in ("planner", "reviewer", "realizer")
        )
    expected_cost = (
        expected_input_tokens * pricing["input_usd_per_million_tokens"]
        + expected_output_tokens * pricing["output_usd_per_million_tokens"]
    ) / 1_000_000
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print(f"Frozen candidate: {manifest['candidate_id']}")
    print(f"Generated cases: {len(generated_prepared)}")
    print(f"Deterministic abstentions: {abstention_count}")
    print("Evaluation scope: locked action + supplied official evidence -> frozen generation")
    print(f"Conservative uncached estimate: ${expected_cost:.6f}")
    print(f"Authorized hard cumulative ceiling: ${args.maximum_cost_usd:.2f}")
    if not args.execute:
        print("Dry-run complete: no API calls.")
        return
    if args.confirm_maximum_cost_usd + 1e-12 < args.maximum_cost_usd:
        raise RuntimeError(
            f"Confirmation required: --confirm-maximum-cost-usd {args.maximum_cost_usd:.2f}"
        )

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
    cache_path = DATA_ROOT / "evaluation_cache" / "external_locked_reference_candidate.sqlite3"
    connection = sqlite3.connect(cache_path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS responses (cache_key TEXT PRIMARY KEY, kind TEXT NOT NULL, "
        "response_json TEXT NOT NULL, created_at_utc TEXT NOT NULL)"
    )
    connection.commit()

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    ledger_path = RESULT_ROOT / "cost_ledger.json"
    ledger = (
        read_json(ledger_path)
        if ledger_path.exists()
        else {
            "schema_version": "1.0.0",
            "authorized_ceiling_usd": args.maximum_cost_usd,
            "cumulative_actual_cost_usd": 0.0,
            "new_api_calls": 0,
        }
    )
    if abs(float(ledger["authorized_ceiling_usd"]) - args.maximum_cost_usd) > 1e-12:
        raise RuntimeError("Existing ledger uses a different authorized cost ceiling.")
    cumulative_cost = float(ledger["cumulative_actual_cost_usd"])
    call_counts: Counter[str] = Counter()
    cache_hits: Counter[str] = Counter()

    def persist_ledger() -> None:
        ledger.update(
            {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "cumulative_actual_cost_usd": cumulative_cost,
                "new_api_calls": int(ledger.get("new_api_calls", 0)),
            }
        )
        write_json(ledger_path, ledger)

    def cached_call(kind: str, body: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        nonlocal cumulative_cost
        key = stable_hash({"kind": kind, "body": body})
        cached = connection.execute(
            "SELECT response_json FROM responses WHERE cache_key=?", (key,)
        ).fetchone()
        if cached:
            cache_hits[kind] += 1
            return json.loads(cached[0]), True
        body_tokens = len(encoding.encode(json.dumps(body, ensure_ascii=False)))
        reserved_cost = (
            body_tokens * pricing["input_usd_per_million_tokens"]
            + int(body["max_output_tokens"]) * pricing["output_usd_per_million_tokens"]
        ) / 1_000_000
        if cumulative_cost + reserved_cost > args.maximum_cost_usd + 1e-12:
            persist_ledger()
            raise RuntimeError(
                f"Hard cost ceiling would be exceeded before {kind}: "
                f"spent=${cumulative_cost:.6f}, reserved=${reserved_cost:.6f}."
            )
        response = client.responses.create(**body)
        value = response.model_dump(mode="json")
        connection.execute(
            "INSERT OR REPLACE INTO responses VALUES (?,?,?,?)",
            (
                key,
                kind,
                json.dumps(value, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        connection.commit()
        call_cost = usage_cost(value.get("usage") or {}, pricing)
        cumulative_cost += call_cost
        ledger["new_api_calls"] = int(ledger.get("new_api_calls", 0)) + 1
        call_counts[kind] += 1
        persist_ledger()
        return value, False

    results_path = RESULT_ROOT / "candidate_results.jsonl"
    existing_results = read_jsonl(results_path) if results_path.exists() else []
    results_by_id = {row["task_id"]: row for row in existing_results}
    maximum_claims = int(config["maximum_mandatory_claims"])

    for task_index, task in enumerate(tasks, start=1):
        task_id = task["case_id"]
        if task_id in results_by_id:
            print(f"[{task_index:02d}/80] {task_id} checkpoint=hit cost=${cumulative_cost:.6f}")
            continue
        reference = reference_by_id[task_id]
        action = reference["expected_action"]
        if action == "abstain":
            result = {
                "task_id": task_id,
                "domain": task["domain"],
                "candidate_id": manifest["candidate_id"],
                "required_action": "abstain",
                "action_and_evidence_source": "locked_human_reference_and_supplied_official_evidence",
                "contract_valid": True,
                "candidate_response": {
                    "status": "insufficient_evidence",
                    "answer": "The available official evidence does not contain enough information to answer this request.",
                    "citations": [],
                },
                "plan": None,
                "plan_errors": [],
                "realization_errors": [],
                "cache": {},
            }
        else:
            prepared = build_prepared(task, reference)
            evidence_ids = [item["evidence_id"] for item in prepared["evidence_catalog"]]
            plan_schema = planner_schema(evidence_ids, maximum_claims)
            planner_body = make_response_body(
                config["planner"],
                planner_prompt,
                prepared["request_body"]["input"],
                "external_locked_mandatory_claim_plan_v1",
                plan_schema,
            )
            planner_data, planner_cached = cached_call("external_locked_planner", planner_body)
            plan, parse_errors = parse_structured(planner_data, plan_schema)
            plan, sanitizations = sanitize_redundant_conflicting_claims(plan, prepared)
            plan_errors = plan_validation_errors(plan, plan_schema, prepared, parse_errors)
            planner_retries = 0
            for attempt in range(1, int(config.get("maximum_plan_repair_attempts", 1)) + 1):
                if not plan_errors:
                    break
                planner_retries = attempt
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
                        "Return one complete valid plan.",
                    ]
                )
                retry_body = make_response_body(
                    retry_stage,
                    planner_prompt,
                    retry_input,
                    f"external_locked_mandatory_claim_plan_retry_v{attempt}",
                    plan_schema,
                )
                retry_data, retry_cached = cached_call(
                    f"external_locked_planner_retry_{attempt}", retry_body
                )
                planner_cached = planner_cached and retry_cached
                plan, parse_errors = parse_structured(retry_data, plan_schema)
                plan, retry_sanitizations = sanitize_redundant_conflicting_claims(plan, prepared)
                sanitizations.extend(retry_sanitizations)
                plan_errors = plan_validation_errors(plan, plan_schema, prepared, parse_errors)

            draft_plan = plan
            review_applied = False
            review_cached = False
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
                    "external_locked_mandatory_claim_review_v1",
                    plan_schema,
                )
                review_data, review_cached = cached_call("external_locked_reviewer", review_body)
                reviewed_plan, review_parse_errors = parse_structured(review_data, plan_schema)
                reviewed_plan, review_sanitizations = sanitize_redundant_conflicting_claims(
                    reviewed_plan, prepared
                )
                sanitizations.extend(review_sanitizations)
                review_errors = plan_validation_errors(
                    reviewed_plan, plan_schema, prepared, review_parse_errors
                )
                if not review_errors:
                    plan = reviewed_plan

            public_payload = None
            realization = None
            realization_errors: list[str] = []
            realizer_cached = False
            repair_used = False
            if not plan_errors:
                claim_ids = [claim["claim_id"] for claim in plan["mandatory_claims"]]
                real_schema = realization_schema(claim_ids, evidence_ids)
                realizer_input = render_realizer_input(prepared, plan)
                limits = config.get("adaptive_output_limits", {})
                realizer_stage = config["realizer"]
                if limits.get("enabled", False):
                    adaptive_limit = int(limits["realizer_base_tokens"]) + int(
                        limits["realizer_tokens_per_claim"]
                    ) * len(claim_ids)
                    realizer_stage = {
                        **config["realizer"],
                        "max_output_tokens": min(
                            int(config["realizer"]["max_output_tokens"]), adaptive_limit
                        ),
                    }
                realizer_body = make_response_body(
                    realizer_stage,
                    realizer_prompt,
                    realizer_input,
                    "external_locked_mandatory_claim_realization_v1",
                    real_schema,
                )
                realizer_data, realizer_cached = cached_call(
                    "external_locked_realizer", realizer_body
                )
                realization, realization_parse_errors = parse_structured(
                    realizer_data, real_schema
                )
                public_payload, realization_errors = validate_and_assemble(
                    prepared, plan, realization, real_schema
                )
                realization_errors = [*realization_parse_errors, *realization_errors]
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
                        "external_locked_mandatory_claim_realization_repair_v1",
                        real_schema,
                    )
                    repair_data, repair_cached = cached_call(
                        "external_locked_realizer_repair", repair_body
                    )
                    realizer_cached = realizer_cached and repair_cached
                    realization, realization_parse_errors = parse_structured(
                        repair_data, real_schema
                    )
                    public_payload, realization_errors = validate_and_assemble(
                        prepared, plan, realization, real_schema
                    )
                    realization_errors = [*realization_parse_errors, *realization_errors]

            contract_valid = bool(not plan_errors and not realization_errors and public_payload)
            if not contract_valid:
                public_payload = {
                    "status": "insufficient_evidence",
                    "answer": "The system could not produce a response that passed its grounding contract.",
                    "citations": [],
                }
            citation_map = {
                item["evidence_id"]: item["span_id"] for item in prepared["evidence_catalog"]
            }
            public_payload["citations"] = [
                {
                    **citation,
                    "source_evidence_id": citation_map.get(citation["evidence_id"]),
                }
                for citation in public_payload.get("citations", [])
            ]
            result = {
                "task_id": task_id,
                "domain": task["domain"],
                "candidate_id": manifest["candidate_id"],
                "required_action": action,
                "action_and_evidence_source": "locked_human_reference_and_supplied_official_evidence",
                "contract_valid": contract_valid,
                "candidate_response": public_payload,
                "draft_plan": draft_plan,
                "plan": plan,
                "plan_errors": plan_errors,
                "review_applied": review_applied,
                "review_errors": review_errors,
                "plan_sanitizations": sanitizations,
                "planner_retries": planner_retries,
                "realization": realization,
                "realization_errors": realization_errors,
                "repair_used": repair_used,
                "cache": {
                    "planner": planner_cached,
                    "reviewer": review_cached,
                    "realizer": realizer_cached,
                },
            }

        results_by_id[task_id] = result
        ordered_results = [results_by_id[row["case_id"]] for row in tasks if row["case_id"] in results_by_id]
        write_jsonl(results_path, ordered_results)
        print(
            f"[{task_index:02d}/80] {task_id} action={action} "
            f"contract={result['contract_valid']} cost=${cumulative_cost:.6f}",
            flush=True,
        )

    connection.close()
    results = [results_by_id[task["case_id"]] for task in tasks]
    scoring_rows = [
        scoring_row(task, reference_by_id[task["case_id"]], results_by_id[task["case_id"]])
        for task in tasks
    ]
    write_jsonl(RESULT_ROOT / "candidate_scoring_tasks.jsonl", scoring_rows)
    summary = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "external_locked_reference_candidate_v1",
        "candidate_id": manifest["candidate_id"],
        "evaluation_scope": "controlled_generation_given_locked_human_action_and_supplied_official_evidence",
        "not_evaluated": ["autonomous_action_routing", "retrieval"],
        "tasks": len(results),
        "generated_tasks": sum(row["required_action"] != "abstain" for row in results),
        "deterministic_abstentions": sum(row["required_action"] == "abstain" for row in results),
        "contract_valid_count": sum(row["contract_valid"] for row in results),
        "contract_valid_rate": sum(row["contract_valid"] for row in results) / len(results),
        "human_candidate_scores_present": 0,
        "cumulative_actual_cost_usd": cumulative_cost,
        "authorized_cost_ceiling_usd": args.maximum_cost_usd,
        "new_api_calls_by_stage_this_process": dict(call_counts),
        "cache_hits_by_stage_this_process": dict(cache_hits),
        "test_split_opened": False,
        "next_required_step": "human_score_candidate_scoring_tasks_jsonl",
    }
    write_json(REPORT_ROOT / "run_summary.json", summary)
    write_json(RESULT_ROOT / "run_summary.json", summary)
    persist_ledger()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
