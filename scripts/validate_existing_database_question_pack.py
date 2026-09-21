"""Validate integrity and human-annotation progress for existing database questions."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "multidoc2dial_v1"
    / "clean_validation"
    / "existing_database_annotation_pack_v1"
)
REPORT_PATH = (
    ROOT
    / "reports"
    / "generated"
    / "multidoc2dial_v1"
    / "internal_human_validation"
    / "annotation_readiness.json"
)
ALLOWED_ACTIONS = {"answer", "ask_followup", "abstain"}


def reference_complete(annotation: dict[str, Any]) -> bool:
    return bool(
        annotation.get("annotator_id")
        and annotation.get("completed_at_utc")
        and annotation.get("authored_conversation_accepted") is True
        and annotation.get("expected_action") in ALLOWED_ACTIONS
        and annotation.get("reference_answer")
        and annotation.get("supporting_span_ids")
    )


def protected_hash(task: dict[str, Any]) -> str:
    protected = {
        key: task[key]
        for key in (
            "case_id",
            "domain",
            "scenario",
            "source_document_id",
            "source_document_version",
            "source_section_ids",
            "source_evidence",
            "conversation",
            "dataset_reference",
        )
    }
    return hashlib.sha256(
        json.dumps(protected, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def main() -> None:
    tasks = [
        json.loads(line)
        for line in (PACK_ROOT / "annotation_tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors: list[str] = []
    if len(tasks) != 80:
        errors.append(f"Expected 80 questions, found {len(tasks)}")
    if len({task["case_id"] for task in tasks}) != len(tasks):
        errors.append("Question IDs are not unique")
    domain_counts = Counter(task["domain"] for task in tasks)
    for domain in ("dmv", "ssa", "va", "studentaid"):
        if domain_counts[domain] != 20:
            errors.append(f"Expected 20 {domain} questions, found {domain_counts[domain]}")
    for task in tasks:
        if protected_hash(task) != task.get("precommitment_sha256"):
            errors.append(f"{task['case_id']}: protected question or evidence changed")
        if not task.get("conversation") or task["conversation"][-1].get("role") != "user":
            errors.append(f"{task['case_id']}: conversation must end with a user question")
        reference = task.get("dataset_reference") or {}
        if (
            reference.get("expected_action") not in ALLOWED_ACTIONS
            or not reference.get("reference_answer")
            or not isinstance(reference.get("supporting_span_ids"), list)
            or reference.get("hidden_from_candidate_runtime") is not True
        ):
            errors.append(f"{task['case_id']}: dataset reference is incomplete or unsafe")
        a, b = task.get("annotator_a") or {}, task.get("annotator_b") or {}
        if a.get("annotator_id") and a.get("annotator_id") == b.get("annotator_id"):
            errors.append(f"{task['case_id']}: A and B must be different people")
    complete_a = sum(reference_complete(task.get("annotator_a") or {}) for task in tasks)
    complete_b = sum(reference_complete(task.get("annotator_b") or {}) for task in tasks)
    dataset_references = sum(bool(task.get("dataset_reference", {}).get("reference_answer")) for task in tasks)
    candidate_responses = sum(task.get("candidate_response") is not None for task in tasks)
    automatic_evaluations = sum(task.get("automatic_evaluation") is not None for task in tasks)
    ready = dataset_references == 80 and not errors
    candidate_complete = candidate_responses == 80 and automatic_evaluations == 80
    report = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": len(tasks),
        "counts_by_domain": dict(domain_counts),
        "questions_already_authored": sum(bool(task.get("conversation")) for task in tasks),
        "annotator_a_complete": complete_a,
        "annotator_b_complete": complete_b,
        "dataset_references_available": dataset_references,
        "candidate_responses_available": candidate_responses,
        "automatic_evaluations_available": automatic_evaluations,
        "human_reference_annotation_required": False,
        "integrity_errors": errors,
        "ready_for_candidate_execution": ready and not candidate_complete,
        "candidate_execution_complete": candidate_complete,
        "verdict": (
            "INTERNAL_CANDIDATE_RUN_COMPLETE_READY_FOR_OPTIONAL_HUMAN_REVIEW"
            if ready and candidate_complete
            else ("READY_FOR_INTERNAL_CANDIDATE_RUN" if ready else "REFERENCE_OR_INTEGRITY_ERROR")
        ),
        "validation_claim": "INTERNAL_HUMAN_VALIDATION_ONLY_NOT_EXTERNAL_PRODUCTION_PROOF",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if ready else 2)


if __name__ == "__main__":
    main()
