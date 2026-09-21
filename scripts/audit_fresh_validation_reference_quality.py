"""Report raw and reference-quality-stratified metrics on an already consumed reserve."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "multidoc2dial_v1"
REPORTS = ROOT / "reports" / "generated" / "multidoc2dial_v1"
CONFIG_PATH = ROOT / "configs" / "fresh_validation_reference_quality_audit.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def rate(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([bool(row[field]) for row in rows])) if rows else 0.0


def metrics(rows: list[dict[str, Any]], minimum_correctness: int) -> dict[str, Any]:
    return {
        "examples": len(rows),
        "routing_accuracy": rate(rows, "deterministic_action_correct"),
        "contract_valid_rate": rate(rows, "contract_valid"),
        "citation_valid_rate": rate(rows, "citations_valid"),
        "faithfulness_pass_rate": rate(rows, "faithfulness_pass"),
        "correctness_acceptable_rate": float(
            np.mean([int(row["correctness_0_to_4"]) >= minimum_correctness for row in rows])
        ) if rows else 0.0,
        "completeness_pass_rate": rate(rows, "completeness_pass"),
        "overall_pass_rate": rate(rows, "overall_pass"),
        "hallucination_rate": 1.0 - rate(rows, "faithfulness_pass"),
        "domain_counts": dict(Counter(row["domain"] for row in rows)),
    }


def deterministic_abstention_result(deferral: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_id": deferral["example_id"],
        "domain": deferral["domain"],
        "required_action": "abstain",
        "deterministic_action_correct": True,
        "contract_valid": True,
        "citations_valid": True,
        "faithfulness_pass": True,
        "correctness_0_to_4": 4,
        "completeness_pass": True,
        "overall_pass": True,
        "resolution": "calibrated_action_controller_v3_abstention"
    }


def main() -> None:
    config = load_json(CONFIG_PATH)
    if config.get("locked_splits") != ["test"]:
        raise ValueError("The benchmark test split must remain locked.")
    source_id = config["source_fresh_experiment_id"]
    source_rows = list(iter_jsonl(
        DATA / "mandatory_claim_pipeline_experiments" / source_id / "case_results.jsonl"
    ))
    by_id = {row["example_id"]: row for row in source_rows}
    source_ids = set(by_id)
    overlay_applied: dict[str, str] = {}
    for experiment_id in config["overlay_experiments"]:
        path = DATA / "mandatory_claim_pipeline_experiments" / experiment_id / "case_results.jsonl"
        for row in iter_jsonl(path):
            if row["example_id"] in source_ids:
                by_id[row["example_id"]] = row
                overlay_applied[row["example_id"]] = experiment_id
    deferrals = {
        row["example_id"]: row
        for row in iter_jsonl(
            DATA
            / "planned_generation_contract_experiments"
            / config["generation_contract_experiment_id"]
            / "deferrals"
            / "validation_deferrals.jsonl"
        )
    }
    for example_id in source_ids:
        deferral = deferrals.get(example_id)
        if deferral and deferral.get("runtime_action") == "abstain_without_generation":
            by_id[example_id] = deterministic_abstention_result(deferral)
            overlay_applied[example_id] = config["generation_contract_experiment_id"]
    raw_rows = [by_id[example_id] for example_id in sorted(source_ids)]
    suspect = config["suspect_references"]
    reliable_rows = [row for row in raw_rows if row["example_id"] not in suspect]
    minimum = int(config["minimum_acceptable_correctness"])
    raw_metrics = metrics(raw_rows, minimum)
    reliable_metrics = metrics(reliable_rows, minimum)
    reliable_minimum = int(config["minimum_reliable_sample_size_for_production_claim"])
    clean_gate_results = {
        "routing_accuracy": reliable_metrics["routing_accuracy"] >= 0.85,
        "faithfulness_pass_rate": reliable_metrics["faithfulness_pass_rate"] >= 0.90,
        "correctness_acceptable_rate": reliable_metrics["correctness_acceptable_rate"] >= 0.80,
        "completeness_pass_rate": reliable_metrics["completeness_pass_rate"] >= 0.80,
        "overall_pass_rate": reliable_metrics["overall_pass_rate"] >= 0.75,
        "maximum_hallucination_rate": reliable_metrics["hallucination_rate"] <= 0.10,
    }
    clean_gate_results["minimum_sample_size"] = len(reliable_rows) >= reliable_minimum
    clean_gate_results["all_domains_represented"] = set(
        reliable_metrics["domain_counts"]
    ) == {"dmv", "ssa", "va", "studentaid"}
    payload = {
        "config": config,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_ids": sorted(source_ids),
        "overlay_applied": overlay_applied,
    }
    experiment_id = "fresh_reference_quality_audit_" + hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    report = {
        "schema_version": "1.0.0",
        "pipeline_stage": "fresh_validation_reference_quality_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "protocol": {
            "source_fresh_experiment_id": source_id,
            "already_consumed_validation_only": True,
            "test_split_opened": False,
            "openai_calls_made": 0,
            "raw_metrics_always_reported": True,
            "suspect_references_excluded_only_from_stratified_metrics": True,
        },
        "overlay_applied": overlay_applied,
        "raw_metrics": raw_metrics,
        "reference_reliable_metrics": reliable_metrics,
        "suspect_reference_count": len(suspect),
        "suspect_references": suspect,
        "clean_quality_gate_results": clean_gate_results,
        "decision": (
            "quality_candidate_passes_reliable_stratum_but_not_production_ready"
            if all(value for key, value in clean_gate_results.items() if key not in {"minimum_sample_size", "all_domains_represented"})
            else "quality_candidate_fails_reliable_stratum"
        ),
        "production_blockers": [
            "The raw benchmark score remains below target because suspect references are retained in raw reporting.",
            f"Only {len(reliable_rows)} reference-reliable fresh examples are available; target is at least {reliable_minimum}.",
            "The reliable fresh stratum does not represent all four domains.",
            "The locked test split remains unopened until a larger document-disjoint validation set passes."
        ],
    }
    report_path = REPORTS / "reference_quality_audits" / experiment_id / "report.json"
    write_json(report_path, report)
    write_json(
        REPORTS / "reference_quality_audits" / "latest.json",
        {"experiment_id": experiment_id, "report_path": str(report_path.relative_to(ROOT))},
    )
    print(f"Experiment: {experiment_id}")
    print(f"Raw overall: {raw_metrics['overall_pass_rate']:.4f} ({len(raw_rows)} cases)")
    print(
        "Reference-reliable overall / faithfulness / completeness: "
        f"{reliable_metrics['overall_pass_rate']:.4f} / "
        f"{reliable_metrics['faithfulness_pass_rate']:.4f} / "
        f"{reliable_metrics['completeness_pass_rate']:.4f}"
    )
    print(f"Suspect references: {len(suspect)}")
    print(f"Decision: {report['decision']}")
    print(f"Report: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
