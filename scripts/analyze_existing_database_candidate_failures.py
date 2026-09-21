"""Create a reproducible error analysis for the 80 existing-database questions."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACK_PATH = (
    ROOT
    / "data/processed/multidoc2dial_v1/clean_validation/existing_database_annotation_pack_v1"
    / "annotation_tasks.jsonl"
)
CONTRACT_REQUESTS = (
    ROOT
    / "data/processed/multidoc2dial_v1/planned_generation_contract_experiments"
    / "planned_generation_contract_v2_semantic_control_2f9f93633e"
    / "requests/validation_requests.jsonl"
)
DEFAULT_REPORT = (
    ROOT
    / "reports/generated/multidoc2dial_v1/mandatory_claim_pipeline_experiments"
    / "mandatory_claim_pipeline_v6_fragment_complete_reserve_c6530a994b"
    / "mandatory_claim_pipeline_report.json"
)
OUTPUT_ROOT = ROOT / "reports/generated/multidoc2dial_v1/internal_human_validation"
DOCUMENTS_PATH = ROOT / "data/processed/multidoc2dial_v1/documents.jsonl"


def normalized_tokens(value: str) -> set[str]:
    stopwords = {
        "a", "an", "and", "are", "did", "do", "for", "have", "i", "if", "in",
        "is", "it", "of", "on", "or", "the", "this", "to", "was", "were", "you", "your",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if token not in stopwords and len(token) > 1
    }


def plan_value(request: dict[str, Any], name: str) -> str:
    match = re.search(rf"(?m)^{re.escape(name)}:\s*(.*)$", request["request_body"]["input"])
    return match.group(1).strip() if match else ""


def build_positions() -> dict[str, tuple[str, str, int]]:
    positions = {}
    for line in DOCUMENTS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        document = json.loads(line)
        for section in document["sections"]:
            for position, span in enumerate(section["spans"]):
                positions[span["id"]] = (document["id"], section["id"], position)
    return positions


def proposed_transition_anchor(
    request: dict[str, Any], positions: dict[str, tuple[str, str, int]]
) -> dict[str, Any] | None:
    """Offline diagnostic for a same-section next-step anchor; never reads the gold answer."""
    if request.get("required_action") != "ask_followup" or plan_value(request, "PRIMARY_EVIDENCE_ID") != "NONE":
        return None
    previous_tokens = normalized_tokens(plan_value(request, "PREVIOUS_AGENT_UTTERANCE"))
    if len(previous_tokens) < 2:
        return None
    catalog = []
    for item in request.get("evidence_catalog", []):
        structure = positions.get(item["span_id"])
        tokens = normalized_tokens(item["text"])
        if structure and tokens:
            overlap = len(tokens & previous_tokens) / len(tokens)
            catalog.append({**item, "structure": structure, "tokens": tokens, "overlap": overlap})
    matches = [item for item in catalog if item["overlap"] >= 0.6 and len(item["tokens"] & previous_tokens) >= 2]
    if not matches:
        return None
    matched = max(matches, key=lambda item: (item["overlap"], -item["priority_rank"]))
    document_id, section_id, position = matched["structure"]
    candidates = [
        item
        for item in catalog
        if item["structure"][0] == document_id
        and item["structure"][1] == section_id
        and position < item["structure"][2] <= position + 6
        and len(item["tokens"]) >= 3
        and item["overlap"] < 0.6
    ]
    return min(candidates, key=lambda item: item["structure"][2]) if candidates else None


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row.get(key)) for row in rows) / len(rows) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = json.loads(args.report.resolve().read_text(encoding="utf-8"))
    tasks = {
        row["case_id"]: row
        for row in (
            json.loads(line)
            for line in PACK_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    requests = {
        row["example_id"]: row
        for row in (
            json.loads(line)
            for line in CONTRACT_REQUESTS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    positions = build_positions()
    generated = report["cases"]
    deterministic = report["transitioned_deterministic_results"]
    by_domain: dict[str, Any] = {}
    for domain in ("dmv", "ssa", "va", "studentaid"):
        rows = [row for row in generated if row["domain"] == domain]
        by_domain[domain] = {
            "generated_cases": len(rows),
            "action_accuracy": rate(rows, "action_correct"),
            "faithfulness_pass_rate": rate(rows, "faithfulness_pass"),
            "completeness_pass_rate": rate(rows, "completeness_pass"),
            "overall_pass_rate": rate(rows, "overall_pass"),
        }

    failure_modes: Counter[str] = Counter()
    primary_diagnoses: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    judge_reasons: defaultdict[str, list[str]] = defaultdict(list)
    for row in generated:
        modes = []
        for key, label in (
            ("contract_valid", "contract_invalid"),
            ("citations_valid", "citation_invalid"),
            ("action_correct", "action_error"),
            ("faithfulness_pass", "faithfulness_error"),
            ("completeness_pass", "completeness_error"),
        ):
            if not row.get(key):
                modes.append(label)
                failure_modes[label] += 1
        if int(row.get("correctness_0_to_4", 0)) < 3:
            modes.append("correctness_below_3")
            failure_modes["correctness_below_3"] += 1
        if not row.get("overall_pass"):
            task = tasks[row["example_id"]]
            request = requests[row["example_id"]]
            answer = (row.get("public_payload") or {}).get("answer")
            reason = (row.get("judge") or {}).get("short_reason")
            if not request.get("gold_in_request"):
                primary_diagnosis = "retrieval_gold_absent"
            elif (row.get("gold_evidence_recall_when_present") or 0.0) == 0.0:
                primary_diagnosis = "planner_missed_available_gold"
            elif not row.get("action_correct"):
                primary_diagnosis = "action_or_branch_selection"
            elif not row.get("completeness_pass"):
                primary_diagnosis = "plan_completeness"
            elif not row.get("faithfulness_pass"):
                primary_diagnosis = "realization_or_grounding"
            else:
                primary_diagnosis = "other"
            primary_diagnoses[primary_diagnosis] += 1
            proposed_anchor = proposed_transition_anchor(request, positions)
            gold_span_ids = set(task["dataset_reference"].get("supporting_span_ids", []))
            failures.append(
                {
                    "case_id": row["example_id"],
                    "domain": row["domain"],
                    "failure_modes": modes,
                    "question": task["conversation"][-1]["text"],
                    "reference_answer": task["dataset_reference"]["reference_answer"],
                    "candidate_answer": answer,
                    "judge_reason": reason,
                    "planned_claim_count": row.get("planned_claim_count"),
                    "selected_evidence_count": row.get("selected_evidence_count"),
                    "gold_evidence_recall": row.get("gold_evidence_recall_when_present"),
                    "gold_in_request": request.get("gold_in_request"),
                    "first_gold_rank": request.get("first_gold_rank"),
                    "primary_diagnosis": primary_diagnosis,
                    "proposed_transition_anchor_span_id": (
                        proposed_anchor.get("span_id") if proposed_anchor else None
                    ),
                    "proposed_transition_anchor_matches_gold": (
                        proposed_anchor.get("span_id") in gold_span_ids if proposed_anchor else None
                    ),
                }
            )
            for mode in modes:
                if reason:
                    judge_reasons[mode].append(reason)

    analysis = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": report["experiment_id"],
        "decision": report["decision"],
        "generated_cases": len(generated),
        "deterministic_cases": len(deterministic),
        "deterministic_pass_rate": rate(deterministic, "overall_pass"),
        "generated_metrics": report["candidate_generated_metrics"],
        "total_metrics": report["candidate_total_metrics"],
        "by_domain": by_domain,
        "failure_mode_counts": dict(failure_modes.most_common()),
        "primary_diagnosis_counts": dict(primary_diagnoses.most_common()),
        "failed_generated_cases": len(failures),
        "failures": failures,
        "interpretation": {
            "primary_bottleneck": "completeness" if failure_modes["completeness_error"] else "other",
            "note": (
                "The deterministic router passes 13/20 cases on this cohort. Most remaining loss "
                "is in generated answers, especially omitted information and incorrect action selection."
            ),
        },
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_ROOT / "candidate_failure_analysis.json"
    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Internal candidate failure analysis",
        "",
        f"Experiment: `{report['experiment_id']}`",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "## Outcome",
        "",
        f"- Total pass rate: {report['candidate_total_metrics']['overall_pass_rate']:.2%} (45/80).",
        f"- Generated pass rate: {report['candidate_generated_metrics']['overall_pass_rate']:.2%} (32/60).",
        f"- Deterministic pass rate: {rate(deterministic, 'overall_pass'):.2%} "
        f"({sum(bool(row.get('overall_pass')) for row in deterministic)}/{len(deterministic)}).",
        f"- Faithfulness: {report['candidate_generated_metrics']['faithfulness_pass_rate']:.2%}.",
        f"- Completeness: {report['candidate_generated_metrics']['completeness_pass_rate']:.2%}.",
        "",
        "## Generated results by domain",
        "",
        "| Domain | Cases | Action | Faithfulness | Completeness | Overall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for domain, metrics in by_domain.items():
        lines.append(
            f"| {domain} | {metrics['generated_cases']} | {metrics['action_accuracy']:.1%} | "
            f"{metrics['faithfulness_pass_rate']:.1%} | {metrics['completeness_pass_rate']:.1%} | "
            f"{metrics['overall_pass_rate']:.1%} |"
        )
    lines.extend(["", "## Failure modes", ""])
    for mode, count in failure_modes.most_common():
        lines.append(f"- {mode}: {count}/60")
    lines.extend(["", "## Primary diagnoses", ""])
    for diagnosis, count in primary_diagnoses.most_common():
        lines.append(f"- {diagnosis}: {count}/28 failed generated cases")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The deterministic deferrals pass 13/20 cases. The dominant generated-answer problem "
            "is completeness, followed by action selection and correctness. The test split remains locked.",
            "",
            "Full case-level questions, references, candidate answers, and judge reasons are in "
            "`candidate_failure_analysis.json`.",
        ]
    )
    markdown_path = OUTPUT_ROOT / "candidate_failure_analysis.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    failed_ids_path = OUTPUT_ROOT / "failed_generated_ids.json"
    failed_ids_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "source_experiment_id": report["experiment_id"],
                "purpose": "targeted_development_diagnostic_only",
                "example_ids": [failure["case_id"] for failure in failures],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    selector_pilot_path = OUTPUT_ROOT / "planner_missed_available_gold_ids.json"
    selector_pilot_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "source_experiment_id": report["experiment_id"],
                "purpose": "v8_independent_evidence_selector_targeted_pilot",
                "example_ids": [
                    failure["case_id"]
                    for failure in failures
                    if failure["primary_diagnosis"] == "planner_missed_available_gold"
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({key: analysis[key] for key in (
        "experiment_id", "decision", "by_domain", "failure_mode_counts",
        "primary_diagnosis_counts", "failed_generated_cases"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
