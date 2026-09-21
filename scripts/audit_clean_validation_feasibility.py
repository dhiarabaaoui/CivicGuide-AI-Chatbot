"""Audit whether an honest production-validation reserve can be built locally.

The strict reserve must be both evaluation-history document-disjoint and in the precommitted
holdout partition. Development rows are reported as a diagnostic only because reusing them for a
production claim would leak calibration decisions into the final metric.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.evaluate_mandatory_claim_pipeline import collect_mandatory_pipeline_used_ids
    from scripts.run_automated_end_to_end_evaluation import collect_previously_used_ids
except ModuleNotFoundError:
    from evaluate_mandatory_claim_pipeline import collect_mandatory_pipeline_used_ids  # type: ignore
    from run_automated_end_to_end_evaluation import collect_previously_used_ids  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "processed" / "multidoc2dial_v1"
REPORT_ROOT = ROOT / "reports" / "generated" / "multidoc2dial_v1"
BENCHMARK_ID = "multidoc2dial_retrieval_benchmark_v1_e8baf6e2ad"
CONTRACT_ID = "planned_generation_contract_v2_semantic_control_2f9f93633e"
DOMAINS = ("dmv", "ssa", "va", "studentaid")
DOMAIN_QUOTA = 20


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


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def span_document_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for document in iter_jsonl(DATA_ROOT / "documents.jsonl"):
        for section in document["sections"]:
            for span in section["spans"]:
                index[span["id"]] = document["id"]
    return index


def example_document_ids(
    example: dict[str, Any], span_to_document: dict[str, str]
) -> set[str]:
    return {
        span_to_document[span_id]
        for span_id in example["gold"]["span_ids"]
        if span_id in span_to_document
    }


def eligible_rows(
    rows: list[dict[str, Any]],
    benchmark_by_id: dict[str, dict[str, Any]],
    used_ids: set[str],
    used_document_ids: set[str],
    span_to_document: dict[str, str],
    partition: str | None,
) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for row in rows:
        example_id = row["example_id"]
        if partition is not None and row.get("partition") != partition:
            continue
        if example_id in used_ids or example_id not in benchmark_by_id:
            continue
        documents = example_document_ids(benchmark_by_id[example_id], span_to_document)
        if not documents or documents.intersection(used_document_ids):
            continue
        eligible.append(row)
    return eligible


def external_annotation_template() -> list[dict[str, Any]]:
    scenario_cycle = (
        "direct_answer",
        "clarification",
        "abstention",
        "multi_turn",
        "multi_span",
        "adversarial_or_insufficient_evidence",
    )
    rows: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for position in range(1, DOMAIN_QUOTA + 1):
            rows.append(
                {
                    "case_id": f"clean_{domain}_{position:02d}",
                    "domain": domain,
                    "scenario": scenario_cycle[(position - 1) % len(scenario_cycle)],
                    "source_document_id": "",
                    "source_document_version": "",
                    "source_section_ids": [],
                    "conversation": [],
                    "candidate_response": None,
                    "annotator_a": {
                        "expected_action": None,
                        "reference_answer": None,
                        "supporting_span_ids": [],
                        "routing_correct": None,
                        "contract_valid": None,
                        "citation_valid": None,
                        "faithfulness_pass": None,
                        "correctness_0_to_4": None,
                        "completeness_pass": None,
                        "overall_pass": None,
                        "notes": None,
                    },
                    "annotator_b": {
                        "expected_action": None,
                        "reference_answer": None,
                        "supporting_span_ids": [],
                        "routing_correct": None,
                        "contract_valid": None,
                        "citation_valid": None,
                        "faithfulness_pass": None,
                        "correctness_0_to_4": None,
                        "completeness_pass": None,
                        "overall_pass": None,
                        "notes": None,
                    },
                    "adjudication": {
                        "status": "pending",
                        "expected_action": None,
                        "reference_answer": None,
                        "supporting_span_ids": [],
                        "routing_correct": None,
                        "contract_valid": None,
                        "citation_valid": None,
                        "faithfulness_pass": None,
                        "correctness_0_to_4": None,
                        "completeness_pass": None,
                        "overall_pass": None,
                        "notes": None,
                    },
                }
            )
    return rows


def main() -> None:
    benchmark_path = DATA_ROOT / "retrieval_benchmarks" / BENCHMARK_ID / "validation.jsonl"
    benchmark_by_id = {row["id"]: row for row in iter_jsonl(benchmark_path)}
    contract_root = DATA_ROOT / "planned_generation_contract_experiments" / CONTRACT_ID
    contract_rows = [
        *iter_jsonl(contract_root / "requests" / "validation_requests.jsonl"),
        *iter_jsonl(contract_root / "deferrals" / "validation_deferrals.jsonl"),
    ]
    span_to_document = span_document_index()
    used_ids = (
        collect_previously_used_ids() | collect_mandatory_pipeline_used_ids()
    ).intersection(benchmark_by_id)
    used_document_ids = {
        document_id
        for example_id in used_ids
        for document_id in example_document_ids(
            benchmark_by_id[example_id], span_to_document
        )
    }

    strict = eligible_rows(
        contract_rows,
        benchmark_by_id,
        used_ids,
        used_document_ids,
        span_to_document,
        partition="holdout",
    )
    relaxed = eligible_rows(
        contract_rows,
        benchmark_by_id,
        used_ids,
        used_document_ids,
        span_to_document,
        partition=None,
    )
    strict_counts = Counter(row["domain"] for row in strict)
    relaxed_counts = Counter(row["domain"] for row in relaxed)
    quota_pass = all(strict_counts[domain] >= DOMAIN_QUOTA for domain in DOMAINS)

    report = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": BENCHMARK_ID,
        "generation_contract_experiment_id": CONTRACT_ID,
        "test_split_opened": False,
        "required_examples": DOMAIN_QUOTA * len(DOMAINS),
        "required_per_domain": DOMAIN_QUOTA,
        "previously_evaluated_validation_ids": len(used_ids),
        "documents_touched_by_previous_evaluations": len(used_document_ids),
        "strict_policy": {
            "partition": "holdout",
            "exclude_previously_evaluated_ids": True,
            "exclude_documents_touched_by_previous_evaluations": True,
            "counts_by_domain": {domain: strict_counts[domain] for domain in DOMAINS},
            "eligible_examples": len(strict),
            "quota_pass": quota_pass,
        },
        "diagnostic_only_relaxed_policy": {
            "warning": "Includes development rows used for calibration and cannot support a production claim.",
            "counts_by_domain": {domain: relaxed_counts[domain] for domain in DOMAINS},
            "eligible_examples": len(relaxed),
        },
        "verdict": (
            "LOCAL_STRICT_RESERVE_AVAILABLE"
            if quota_pass
            else "EXTERNAL_CLEAN_DOCUMENT_DISJOINT_COLLECTION_REQUIRED"
        ),
        "next_action": (
            "Freeze and precommit a stratified strict reserve."
            if quota_pass
            else "Collect 80 externally sourced cases using the generated dual-annotation template."
        ),
    }
    report_path = (
        REPORT_ROOT / "clean_validation_feasibility" / "clean_validation_feasibility.json"
    )
    template_path = (
        DATA_ROOT / "clean_validation" / "clean_document_disjoint_multidomain_v1_template.jsonl"
    )
    write_json(report_path, report)
    write_jsonl(template_path, external_annotation_template())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Annotation template: {template_path}")


if __name__ == "__main__":
    main()
