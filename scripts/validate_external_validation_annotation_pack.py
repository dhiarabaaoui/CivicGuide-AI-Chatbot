"""Validate integrity and annotation readiness of the external production gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "processed" / "multidoc2dial_v1"
PACK_ROOT = DATA_ROOT / "clean_validation" / "external_annotation_pack_v1"
SOURCE_ROOT = DATA_ROOT / "clean_validation" / "external_sources_v1"
REPORT_PATH = (
    ROOT
    / "reports"
    / "generated"
    / "multidoc2dial_v1"
    / "external_clean_validation"
    / "annotation_readiness.json"
)
ALLOWED_ACTIONS = {"answer", "ask_followup", "abstain"}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def precommitment_hash(task: dict[str, Any]) -> str:
    value = {
        key: task[key]
        for key in (
            "case_id",
            "domain",
            "scenario",
            "source_document_id",
            "source_document_version",
            "source_section_ids",
            "source_evidence",
        )
    }
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def annotation_reference_complete(annotation: dict[str, Any]) -> bool:
    return (
        bool(annotation.get("annotator_id"))
        and bool(annotation.get("completed_at_utc"))
        and annotation.get("authored_conversation_accepted") is True
        and annotation.get("expected_action") in ALLOWED_ACTIONS
        and bool(annotation.get("reference_answer"))
        and isinstance(annotation.get("supporting_span_ids"), list)
    )


def annotation_scores_complete(annotation: dict[str, Any]) -> bool:
    boolean_fields = (
        "routing_correct",
        "contract_valid",
        "citation_valid",
        "faithfulness_pass",
        "completeness_pass",
        "overall_pass",
    )
    return (
        annotation_reference_complete(annotation)
        and all(isinstance(annotation.get(field), bool) for field in boolean_fields)
        and isinstance(annotation.get("correctness_0_to_4"), int)
        and 0 <= annotation["correctness_0_to_4"] <= 4
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pre-candidate", "post-candidate"), default="pre-candidate")
    args = parser.parse_args()
    tasks = list(iter_jsonl(PACK_ROOT / "annotation_tasks.jsonl"))
    manifest = json.loads((PACK_ROOT / "manifest.json").read_text(encoding="utf-8"))
    documents = {
        document["id"]: document
        for document in iter_jsonl(SOURCE_ROOT / "documents.jsonl")
    }
    errors: list[str] = []
    ids = [task["case_id"] for task in tasks]
    if len(tasks) != 80:
        errors.append(f"Expected 80 tasks, found {len(tasks)}")
    if len(set(ids)) != len(ids):
        errors.append("Case IDs are not unique")
    counts = Counter(task["domain"] for task in tasks)
    for domain in ("dmv", "ssa", "va", "studentaid"):
        if counts[domain] != 20:
            errors.append(f"Expected 20 {domain} tasks, found {counts[domain]}")

    authored = references_a = references_b = scored_a = scored_b = adjudicated = 0
    for task in tasks:
        case_id = task["case_id"]
        if precommitment_hash(task) != task.get("precommitment_sha256"):
            errors.append(f"{case_id}: precommitment hash changed")
        document = documents.get(task["source_document_id"])
        if document is None:
            errors.append(f"{case_id}: unknown source document")
            continue
        if task["source_document_version"] != document["version"]:
            errors.append(f"{case_id}: source version changed")
        available_sections = {section["id"] for section in document["sections"]}
        if not task["source_section_ids"] or not set(task["source_section_ids"]).issubset(available_sections):
            errors.append(f"{case_id}: invalid source section selection")
        if task.get("conversation") and task.get("case_author_id") and task.get("case_authored_at_utc"):
            authored += 1
        a = task.get("annotator_a") or {}
        b = task.get("annotator_b") or {}
        if annotation_reference_complete(a):
            references_a += 1
        if annotation_reference_complete(b):
            references_b += 1
        if a.get("annotator_id") and a.get("annotator_id") == b.get("annotator_id"):
            errors.append(f"{case_id}: annotators A and B must be different people")
        if task.get("case_author_id") in {a.get("annotator_id"), b.get("annotator_id")}:
            errors.append(f"{case_id}: case author must be independent from both annotators")
        if annotation_scores_complete(a):
            scored_a += 1
        if annotation_scores_complete(b):
            scored_b += 1
        adjudication = task.get("adjudication") or {}
        if adjudication.get("status") == "completed" and annotation_scores_complete(adjudication):
            if adjudication.get("adjudicator_id") in {a.get("annotator_id"), b.get("annotator_id")}:
                errors.append(f"{case_id}: adjudicator must be independent")
            else:
                adjudicated += 1

    references_locked = authored == references_a == references_b == 80
    candidate_responses = sum(task.get("candidate_response") is not None for task in tasks)
    pre_candidate_ready = references_locked and not errors and candidate_responses == 0
    post_candidate_ready = (
        candidate_responses == scored_a == scored_b == adjudicated == 80 and not errors
    )
    report = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "annotation_pack_id": manifest["annotation_pack_id"],
        "phase": args.phase,
        "tasks": len(tasks),
        "authored_conversations": authored,
        "annotator_a_reference_complete": references_a,
        "annotator_b_reference_complete": references_b,
        "candidate_responses": candidate_responses,
        "annotator_a_scores_complete": scored_a,
        "annotator_b_scores_complete": scored_b,
        "adjudications_complete": adjudicated,
        "integrity_errors": errors,
        "references_locked": references_locked,
        "pre_candidate_ready": pre_candidate_ready,
        "post_candidate_ready": post_candidate_ready,
        "verdict": (
            "READY_TO_EXECUTE_FROZEN_CANDIDATE"
            if pre_candidate_ready
            else (
                "EXTERNAL_VALIDATION_COMPLETE"
                if post_candidate_ready
                else "ANNOTATION_WORK_REQUIRED"
            )
        ),
    }
    write_json(REPORT_PATH, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    desired_ready = pre_candidate_ready if args.phase == "pre-candidate" else post_candidate_ready
    raise SystemExit(0 if desired_ready else 2)


if __name__ == "__main__":
    main()
