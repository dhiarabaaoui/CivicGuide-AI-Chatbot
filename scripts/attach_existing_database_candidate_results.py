"""Attach one completed candidate evaluation to the existing-database annotation pack.

The immutable question, evidence, and hidden dataset-reference fields are preserved.  Only the
post-run ``candidate_response`` field and run-status metadata are populated.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "multidoc2dial_v1"
    / "clean_validation"
    / "existing_database_annotation_pack_v1"
)
DEFAULT_REPORT = (
    ROOT
    / "reports"
    / "generated"
    / "multidoc2dial_v1"
    / "mandatory_claim_pipeline_experiments"
    / "mandatory_claim_pipeline_v6_fragment_complete_reserve_c6530a994b"
    / "mandatory_claim_pipeline_report.json"
)
SUMMARY_PATH = (
    ROOT
    / "reports"
    / "generated"
    / "multidoc2dial_v1"
    / "internal_human_validation"
    / "candidate_evaluation_summary.json"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def generated_result(case: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "execution_mode": "generated",
        "runtime_action": (case.get("public_payload") or {}).get("status"),
        "payload": case.get("public_payload"),
    }


def generated_automatic_evaluation(case: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "contract_valid": case.get("contract_valid"),
        "citation_valid": case.get("citations_valid"),
        "faithfulness_pass": case.get("faithfulness_pass"),
        "correctness_0_to_4": case.get("correctness_0_to_4"),
        "completeness_pass": case.get("completeness_pass"),
        "action_correct": case.get("action_correct"),
        "overall_pass": case.get("overall_pass"),
        "automatic_judge": case.get("judge"),
    }


def deterministic_result(case: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "execution_mode": "deterministic",
        "runtime_action": case.get("runtime_action"),
        "payload": {
            "status": case.get("runtime_action"),
            "answer": case.get("deterministic_response"),
            "citations": [],
        },
    }


def deterministic_automatic_evaluation(case: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "contract_valid": True,
        "citation_valid": True,
        "faithfulness_pass": True,
        "correctness_0_to_4": 4 if case.get("overall_pass") else 0,
        "completeness_pass": True,
        "action_correct": case.get("action_correct"),
        "overall_pass": case.get("overall_pass"),
        "automatic_judge": None,
        "deterministic_reason": case.get("reason"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    report = json.loads(args.report.resolve().read_text(encoding="utf-8"))
    experiment_id = report["experiment_id"]
    results = {
        case["example_id"]: {
            "candidate_response": generated_result(case, experiment_id),
            "automatic_evaluation": generated_automatic_evaluation(case, experiment_id),
        }
        for case in report.get("cases", [])
    }
    for case in report.get("transitioned_deterministic_results", []):
        case_id = case["example_id"]
        if case_id in results:
            raise ValueError(f"Duplicate evaluation result for {case_id}")
        results[case_id] = {
            "candidate_response": deterministic_result(case, experiment_id),
            "automatic_evaluation": deterministic_automatic_evaluation(case, experiment_id),
        }

    tasks_path = PACK_ROOT / "annotation_tasks.jsonl"
    tasks = read_jsonl(tasks_path)
    task_ids = {task["case_id"] for task in tasks}
    if task_ids != set(results):
        missing = sorted(task_ids - set(results))
        unexpected = sorted(set(results) - task_ids)
        raise ValueError(f"Evaluation/pack mismatch: missing={missing}, unexpected={unexpected}")
    for task in tasks:
        result = results[task["case_id"]]
        task["candidate_response"] = result["candidate_response"]
        task["automatic_evaluation"] = result["automatic_evaluation"]
    write_jsonl(tasks_path, tasks)

    now = datetime.now(timezone.utc).isoformat()
    manifest_path = PACK_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "candidate_executed": True,
            "candidate_experiment_id": experiment_id,
            "candidate_run_status": "COMPLETED_INTERNAL_EVALUATION",
            "candidate_results_attached_at_utc": now,
            "candidate_decision": report["decision"],
        }
    )
    write_json(manifest_path, manifest)

    metrics = report["candidate_total_metrics"]
    generated_metrics = report["candidate_generated_metrics"]
    summary = {
        "schema_version": "1.0.0",
        "generated_at_utc": now,
        "experiment_id": experiment_id,
        "tasks": len(tasks),
        "generated_cases": len(report.get("cases", [])),
        "deterministic_cases": len(report.get("transitioned_deterministic_results", [])),
        "candidate_total_metrics": metrics,
        "candidate_generated_metrics": generated_metrics,
        "gates_passed": report["gates_passed"],
        "decision": report["decision"],
        "next_step": "HUMAN_ERROR_REVIEW_AND_PIPELINE_IMPROVEMENT",
        "validation_claim": "INTERNAL_HUMAN_VALIDATION_ONLY_NOT_EXTERNAL_PRODUCTION_PROOF",
        "source_report": str(args.report.resolve().relative_to(ROOT)).replace("\\", "/"),
    }
    write_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
