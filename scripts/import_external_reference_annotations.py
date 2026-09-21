"""Import and audit a user-supplied external reference-annotation JSONL file."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "multidoc2dial_v1"
    / "clean_validation"
    / "external_annotation_pack_v1"
    / "annotation_tasks.jsonl"
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir-name", default="human_reference_annotations_v1")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = TASKS_PATH.parent / args.output_dir_name
    input_bytes = input_path.read_bytes()
    supplied_rows = load_jsonl(input_path)
    task_rows = load_jsonl(TASKS_PATH)
    tasks_by_id = {row["case_id"]: row for row in task_rows}

    annotations = [row for row in supplied_rows if row.get("task_id")]
    summary_rows = [row for row in supplied_rows if not row.get("task_id")]
    ids = [row["task_id"] for row in annotations]
    duplicate_ids = sorted(key for key, count in Counter(ids).items() if count > 1)
    missing_ids = sorted(set(tasks_by_id) - set(ids))
    extra_ids = sorted(set(ids) - set(tasks_by_id))

    issues: list[dict] = []
    allowed_actions = {"answer", "ask_followup", "abstain"}
    for row in annotations:
        task_id = row["task_id"]
        if task_id not in tasks_by_id:
            continue
        task = tasks_by_id[task_id]
        allowed_evidence = {item["id"] for item in task.get("source_evidence", [])}
        evidence_ids = set(row.get("supporting_evidence_ids") or [])
        unknown_evidence = sorted(evidence_ids - allowed_evidence)
        if unknown_evidence:
            issues.append({"task_id": task_id, "type": "unknown_evidence", "values": unknown_evidence})
        if row.get("expected_action") not in allowed_actions:
            issues.append({"task_id": task_id, "type": "invalid_action"})
        if row.get("expected_action") == "answer" and not evidence_ids:
            issues.append({"task_id": task_id, "type": "answer_without_evidence"})
        if row.get("expected_action") == "ask_followup" and not row.get("missing_information"):
            issues.append({"task_id": task_id, "type": "followup_without_missing_information"})
        confidence = row.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            issues.append({"task_id": task_id, "type": "invalid_confidence"})

    complete_reference_set = (
        len(annotations) == len(task_rows) == 80
        and not duplicate_ids
        and not missing_ids
        and not extra_ids
        and not issues
    )
    candidate_evaluations_present = sum(row.get("candidate_evaluation") is not None for row in annotations)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_copy = output_dir / input_path.name
    shutil.copyfile(input_path, raw_copy)

    normalized_path = output_dir / "reference_annotations.jsonl"
    with normalized_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in annotations:
            human_metadata = row.get("human_annotation_metadata") or {}
            normalized = {
                "task_id": row["task_id"],
                "annotation_round": "single_human_reference",
                "provenance": "user_supplied_external_file",
                "annotator_identity": human_metadata.get("annotator_id"),
                "annotator_identity_recorded": bool(human_metadata.get("annotator_id")),
                "annotation_date": human_metadata.get("annotation_date"),
                "reference_locked": human_metadata.get("reference_locked") is True,
                "source_claimed_annotator_type": row.get("annotator_type"),
                "source_claimed_annotator_system": row.get("annotator_system"),
                "expected_action": row.get("expected_action"),
                "reference_answer": row.get("reference_answer"),
                "supporting_evidence_ids": row.get("supporting_evidence_ids") or [],
                "missing_information": row.get("missing_information"),
                "confidence": row.get("confidence"),
                "annotation_notes": row.get("annotation_notes"),
                "candidate_evaluation": row.get("candidate_evaluation"),
            }
            handle.write(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) + "\n")

    annotator_ids = sorted(
        {
            row.get("human_annotation_metadata", {}).get("annotator_id")
            for row in annotations
            if row.get("human_annotation_metadata", {}).get("annotator_id")
        }
    )
    annotation_dates = sorted(
        {
            row.get("human_annotation_metadata", {}).get("annotation_date")
            for row in annotations
            if row.get("human_annotation_metadata", {}).get("annotation_date")
        }
    )
    locked_references = sum(
        row.get("human_annotation_metadata", {}).get("reference_locked") is True
        for row in annotations
    )
    audit = {
        "schema_version": "1.0.0",
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_filename": input_path.name,
        "source_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "source_bytes": len(input_bytes),
        "task_count_expected": len(task_rows),
        "annotation_count": len(annotations),
        "summary_row_count": len(summary_rows),
        "duplicate_task_ids": duplicate_ids,
        "missing_task_ids": missing_ids,
        "extra_task_ids": extra_ids,
        "schema_evidence_issues": issues,
        "reference_annotation_complete": complete_reference_set,
        "expected_action_counts": dict(sorted(Counter(row.get("expected_action") for row in annotations).items())),
        "candidate_evaluations_present": candidate_evaluations_present,
        "candidate_evaluations_missing": len(annotations) - candidate_evaluations_present,
        "annotator_identity_recorded": len(annotator_ids) == 1 and len(annotations) == 80,
        "annotator_ids": annotator_ids,
        "annotation_dates": annotation_dates,
        "locked_reference_count": locked_references,
        "second_independent_annotation_present": False,
        "adjudication_present": False,
        "protocol_status": "REFERENCE_SET_COMPLETE_FULL_VALIDATION_INCOMPLETE",
        "next_required_step": "run_frozen_candidate_then_collect_candidate_scores",
    }
    (output_dir / "import_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
