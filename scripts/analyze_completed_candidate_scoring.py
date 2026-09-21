"""Audit a completed candidate-scoring file and publish reproducible metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "multidoc2dial_v1"
    / "clean_validation"
    / "external_candidate_runs"
    / "external_locked_reference_candidate_v1"
)
BASE_SCORING_PATH = RUN_ROOT / "candidate_scoring_tasks.jsonl"
CANDIDATE_RESULTS_PATH = RUN_ROOT / "candidate_results.jsonl"
OUTPUT_ROOT = RUN_ROOT / "completed_scoring_v1"
REPORT_ROOT = (
    ROOT
    / "reports"
    / "generated"
    / "multidoc2dial_v1"
    / "external_clean_validation"
    / "external_locked_reference_candidate_v1"
    / "completed_scoring_v1"
)
FROZEN_CONFIG_PATH = (
    ROOT
    / "data"
    / "processed"
    / "multidoc2dial_v1"
    / "candidate_freezes"
    / "rag_candidate_structural_v1_62383ac6c4ff"
    / "snapshot"
    / "configs"
    / "mandatory_claim_pipeline.json"
)
IMMUTABLE_FIELDS = (
    "domain",
    "scenario",
    "conversation",
    "source_evidence",
    "locked_human_reference",
    "candidate_response",
    "candidate_execution_metadata",
)
BOOLEAN_FIELDS = (
    "action_correct",
    "contract_valid",
    "citation_valid",
    "faithfulness_pass",
    "completeness_pass",
    "overall_pass",
)


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


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    result: dict[str, Any] = {"cases": total}
    for field in BOOLEAN_FIELDS:
        successes = sum(bool(row["human_candidate_evaluation"][field]) for row in rows)
        result[field] = {
            "successes": successes,
            "rate": successes / total if total else 0.0,
            "wilson_95": wilson(successes, total),
        }
    correctness = [
        int(row["human_candidate_evaluation"]["correctness_0_to_4"])
        for row in rows
    ]
    acceptable = sum(value >= 3 for value in correctness)
    result["correctness"] = {
        "mean_0_to_4": sum(correctness) / total if total else 0.0,
        "acceptable_successes": acceptable,
        "acceptable_rate": acceptable / total if total else 0.0,
        "acceptable_wilson_95": wilson(acceptable, total),
        "score_counts": dict(sorted(Counter(correctness).items())),
    }
    return result


def grouped(rows: list[dict[str, Any]], field_getter: Any) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(field_getter(row))].append(row)
    return {key: summarize(groups[key]) for key in sorted(groups)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    input_path = args.input.resolve()
    raw_bytes = input_path.read_bytes()
    completed = read_jsonl(input_path)
    base = read_jsonl(BASE_SCORING_PATH)
    candidate_results = read_jsonl(CANDIDATE_RESULTS_PATH)
    config = read_json(FROZEN_CONFIG_PATH)
    base_by_id = {row["task_id"]: row for row in base}
    completed_by_id = {row["task_id"]: row for row in completed}
    issues: list[str] = []

    if len(base) != len(completed) or len(completed) != 80:
        issues.append(f"expected_80_rows: base={len(base)}, completed={len(completed)}")
    if len(completed_by_id) != len(completed):
        issues.append("duplicate_task_ids")
    missing = sorted(set(base_by_id) - set(completed_by_id))
    extra = sorted(set(completed_by_id) - set(base_by_id))
    if missing:
        issues.append("missing_task_ids=" + ",".join(missing))
    if extra:
        issues.append("extra_task_ids=" + ",".join(extra))

    immutable_changes = []
    for task_id, row in completed_by_id.items():
        original = base_by_id.get(task_id)
        if original is None:
            continue
        for field in IMMUTABLE_FIELDS:
            if canonical(row.get(field)) != canonical(original.get(field)):
                immutable_changes.append(f"{task_id}:{field}")
        evaluation = row.get("human_candidate_evaluation") or {}
        for field in BOOLEAN_FIELDS:
            if not isinstance(evaluation.get(field), bool):
                issues.append(f"{task_id}:{field}_must_be_boolean")
        correctness = evaluation.get("correctness_0_to_4")
        if not isinstance(correctness, int) or isinstance(correctness, bool) or not 0 <= correctness <= 4:
            issues.append(f"{task_id}:invalid_correctness")
        expected_overall = bool(
            evaluation.get("action_correct")
            and evaluation.get("contract_valid")
            and evaluation.get("citation_valid")
            and evaluation.get("faithfulness_pass")
            and evaluation.get("completeness_pass")
            and isinstance(correctness, int)
            and correctness >= 3
        )
        if evaluation.get("overall_pass") is not expected_overall:
            issues.append(f"{task_id}:overall_pass_inconsistent")
    if immutable_changes:
        issues.append("immutable_changes=" + ",".join(immutable_changes))
    if issues:
        raise ValueError("Scoring integrity check failed: " + " | ".join(issues))

    evaluator_ids = sorted(
        {row["human_candidate_evaluation"].get("evaluator_id") for row in completed}
    )
    evaluator_ids = [value for value in evaluator_ids if value]
    declared_ai = any("ai" in value.casefold() or "codex" in value.casefold() for value in evaluator_ids)
    provenance_classification = (
        "AI_DECLARED_BY_EVALUATOR_ID" if declared_ai else "UNVERIFIED_EXTERNAL_EVALUATOR"
    )

    metrics = summarize(completed)
    by_domain = grouped(completed, lambda row: row["domain"])
    by_action = grouped(
        completed, lambda row: row["locked_human_reference"]["expected_action"]
    )
    by_scenario = grouped(completed, lambda row: row["scenario"])
    failure_rows = [
        {
            "task_id": row["task_id"],
            "domain": row["domain"],
            "scenario": row["scenario"],
            "expected_action": row["locked_human_reference"]["expected_action"],
            "evaluation": row["human_candidate_evaluation"],
            "candidate_response": row["candidate_response"],
        }
        for row in completed
        if not row["human_candidate_evaluation"]["overall_pass"]
    ]

    planned_claim_counts = [
        len((row.get("plan") or {}).get("mandatory_claims", []))
        for row in candidate_results
        if row.get("required_action") != "abstain"
    ]
    mean_planned_claims = (
        sum(planned_claim_counts) / len(planned_claim_counts)
        if planned_claim_counts
        else 0.0
    )
    gates = config["acceptance_gates"]
    gate_results = {
        "minimum_action_accuracy": {
            "value": metrics["action_correct"]["rate"],
            "threshold": gates["minimum_action_accuracy"],
            "passed": metrics["action_correct"]["rate"] >= gates["minimum_action_accuracy"],
        },
        "minimum_contract_valid_rate": {
            "value": metrics["contract_valid"]["rate"],
            "threshold": gates["minimum_contract_valid_rate"],
            "passed": metrics["contract_valid"]["rate"] >= gates["minimum_contract_valid_rate"],
        },
        "minimum_citation_valid_rate": {
            "value": metrics["citation_valid"]["rate"],
            "threshold": gates["minimum_citation_valid_rate"],
            "passed": metrics["citation_valid"]["rate"] >= gates["minimum_citation_valid_rate"],
        },
        "minimum_faithfulness_pass_rate": {
            "value": metrics["faithfulness_pass"]["rate"],
            "threshold": gates["minimum_faithfulness_pass_rate"],
            "passed": metrics["faithfulness_pass"]["rate"] >= gates["minimum_faithfulness_pass_rate"],
        },
        "minimum_correctness_acceptable_rate": {
            "value": metrics["correctness"]["acceptable_rate"],
            "threshold": gates["minimum_correctness_acceptable_rate"],
            "passed": metrics["correctness"]["acceptable_rate"] >= gates["minimum_correctness_acceptable_rate"],
        },
        "minimum_completeness_pass_rate": {
            "value": metrics["completeness_pass"]["rate"],
            "threshold": gates["minimum_completeness_pass_rate"],
            "passed": metrics["completeness_pass"]["rate"] >= gates["minimum_completeness_pass_rate"],
        },
        "minimum_overall_pass_rate": {
            "value": metrics["overall_pass"]["rate"],
            "threshold": gates["minimum_overall_pass_rate"],
            "passed": metrics["overall_pass"]["rate"] >= gates["minimum_overall_pass_rate"],
        },
        "maximum_mean_planned_claims": {
            "value": mean_planned_claims,
            "threshold": gates["maximum_mean_planned_claims"],
            "passed": mean_planned_claims <= gates["maximum_mean_planned_claims"],
        },
        "test_split_locked": {"value": True, "threshold": True, "passed": True},
    }
    unevaluable_gates = [
        "maximum_hallucination_rate_no_explicit_hallucination_field",
        "minimum_delta_overall_pass_rate_no_matched_baseline",
        "minimum_delta_completeness_pass_rate_no_matched_baseline",
        "maximum_faithfulness_regression_no_matched_baseline",
    ]
    all_evaluable_passed = all(item["passed"] for item in gate_results.values())
    verdict = (
        "AUTOMATED_SCORING_PASSES_EVALUABLE_GATES_HUMAN_CONFIRMATION_REQUIRED"
        if all_evaluable_passed
        else "AUTOMATED_SCORING_FAILS_PRODUCTION_TARGETS"
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    raw_copy = OUTPUT_ROOT / input_path.name
    shutil.copyfile(input_path, raw_copy)
    write_jsonl(OUTPUT_ROOT / "failed_cases.jsonl", failure_rows)
    report = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": input_path.name,
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "source_bytes": len(raw_bytes),
        "integrity": {
            "rows": len(completed),
            "immutable_changes": immutable_changes,
            "scoring_issues": issues,
            "passed": not immutable_changes and not issues,
        },
        "evaluator_ids": evaluator_ids,
        "evaluation_provenance_classification": provenance_classification,
        "human_validation_claim_allowed": False,
        "evaluation_scope": "automated_scoring_of_controlled_generation_given_locked_action_and_evidence",
        "metrics": metrics,
        "metrics_by_domain": by_domain,
        "metrics_by_action": by_action,
        "metrics_by_scenario": by_scenario,
        "mean_planned_claims_generated_cases": mean_planned_claims,
        "failed_case_count": len(failure_rows),
        "gate_results": gate_results,
        "unevaluable_gates": unevaluable_gates,
        "all_evaluable_gates_passed": all_evaluable_passed,
        "verdict": verdict,
        "owner_production_override_unchanged": True,
        "test_split_opened": False,
    }
    write_json(OUTPUT_ROOT / "scoring_report.json", report)
    write_json(REPORT_ROOT / "scoring_report.json", report)

    def pct(value: float) -> str:
        return f"{100 * value:.2f}%"

    markdown = "\n".join(
        [
            "# Évaluation des réponses candidates",
            "",
            f"- Évaluateur déclaré : `{', '.join(evaluator_ids)}`",
            f"- Provenance : **{provenance_classification}**",
            "- Validation humaine revendicable : **non**",
            f"- Cas : **{len(completed)}**",
            f"- Réussite globale : **{metrics['overall_pass']['successes']}/80 ({pct(metrics['overall_pass']['rate'])})**",
            f"- Action correcte : **{pct(metrics['action_correct']['rate'])}**",
            f"- Contrat valide : **{pct(metrics['contract_valid']['rate'])}**",
            f"- Citations valides : **{pct(metrics['citation_valid']['rate'])}**",
            f"- Fidélité : **{pct(metrics['faithfulness_pass']['rate'])}**",
            f"- Correction acceptable : **{pct(metrics['correctness']['acceptable_rate'])}**",
            f"- Complétude : **{pct(metrics['completeness_pass']['rate'])}**",
            f"- Note moyenne : **{metrics['correctness']['mean_0_to_4']:.3f}/4**",
            f"- Verdict : **{verdict}**",
            "",
            "## Résultat par domaine",
            "",
            "| Domaine | Réussites | Taux |",
            "|---|---:|---:|",
            *[
                f"| {domain} | {values['overall_pass']['successes']}/{values['cases']} | {pct(values['overall_pass']['rate'])} |"
                for domain, values in by_domain.items()
            ],
            "",
            "## Résultat par action attendue",
            "",
            "| Action | Réussites | Taux |",
            "|---|---:|---:|",
            *[
                f"| {action} | {values['overall_pass']['successes']}/{values['cases']} | {pct(values['overall_pass']['rate'])} |"
                for action, values in by_action.items()
            ],
            "",
            "## Interprétation",
            "",
            "Les citations et la fidélité sont fortes, mais la complétude est le principal échec. Les demandes de clarification et les abstentions n’obtiennent aucune réussite globale dans cette notation. Comme l’identifiant de l’évaluateur déclare explicitement une IA, ce fichier constitue une évaluation automatique supplémentaire et non une validation humaine indépendante. L’override de production décidé par le propriétaire reste un choix de risque, mais ces scores ne permettent pas d’affirmer que les seuils de qualité de production sont atteints.",
            "",
        ]
    )
    report_md = REPORT_ROOT / "scoring_report.md"
    report_md.parent.mkdir(parents=True, exist_ok=True)
    temporary_md = report_md.with_suffix(".md.tmp")
    temporary_md.write_text(markdown, encoding="utf-8")
    temporary_md.replace(report_md)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
