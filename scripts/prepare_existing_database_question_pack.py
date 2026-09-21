"""Prepare an 80-question internal human-annotation pack from existing database dialogues.

These rows have not appeared in previous evaluation artifacts and their source documents are
disjoint from prior evaluated documents. They remain an internal-development validation set because
they originate from MultiDoc2Dial, so they cannot replace the external production confirmation.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.audit_clean_validation_feasibility import (
        BENCHMARK_ID,
        CONTRACT_ID,
        DATA_ROOT,
        collect_mandatory_pipeline_used_ids,
        collect_previously_used_ids,
        eligible_rows,
        example_document_ids,
        span_document_index,
    )
except ModuleNotFoundError:
    from audit_clean_validation_feasibility import (  # type: ignore
        BENCHMARK_ID,
        CONTRACT_ID,
        DATA_ROOT,
        collect_mandatory_pipeline_used_ids,
        collect_previously_used_ids,
        eligible_rows,
        example_document_ids,
        span_document_index,
    )


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = DATA_ROOT / "clean_validation" / "existing_database_annotation_pack_v1"
REPORT_ROOT = ROOT / "reports" / "generated" / "multidoc2dial_v1" / "internal_human_validation"
DOMAINS = ("dmv", "ssa", "va", "studentaid")
REQUESTS_PER_DOMAIN = 15
DEFERRALS_PER_DOMAIN = 5
SAMPLING_SEED = 20260901


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


def empty_annotation() -> dict[str, Any]:
    return {
        "annotator_id": None,
        "completed_at_utc": None,
        "authored_conversation_accepted": None,
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
    }


def benchmark_conversation(example: dict[str, Any]) -> list[dict[str, str]]:
    turns = [*example.get("history", []), *example.get("intermediate_turns", []), example["query"]]
    turns = sorted(turns, key=lambda turn: int(turn.get("index", 0)))
    return [
        {
            "role": "assistant" if turn["role"] == "agent" else "user",
            "text": turn["text"],
        }
        for turn in turns
    ]


def dataset_reference(example: dict[str, Any]) -> dict[str, Any]:
    reference_answer = example["target"]["reference_answer"]
    expected_action = (
        "abstain"
        if "no relevant information is found" in reference_answer.casefold()
        else (
            "ask_followup"
            if example["target"]["dialogue_act"] == "query_condition"
            else "answer"
        )
    )
    return {
        "expected_action": expected_action,
        "reference_answer": reference_answer,
        "supporting_span_ids": list(example["gold"]["span_ids"]),
        "target_turn_id": example["target"]["turn_id"],
        "provenance": "multidoc2dial_v1_benchmark_target_and_gold",
        "hidden_from_candidate_runtime": True,
    }


def precommitment_hash(task: dict[str, Any]) -> str:
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


def request_task(row: dict[str, Any], example: dict[str, Any]) -> dict[str, Any]:
    evidence = [
        {
            "id": item["span_id"],
            "title": f"Document {item['document_id']} · section {item['section_id']}",
            "text": item["text"],
        }
        for item in row["evidence_catalog"]
    ]
    document_ids = sorted({item["document_id"] for item in row["evidence_catalog"]})
    task = {
        "case_id": row["example_id"],
        "domain": row["domain"],
        "scenario": "existing_rag_question",
        "case_type": "generated_request",
        "partition": row["partition"],
        "authoring_status": "existing_database_question_locked",
        "task_instruction": "Annoter cette conversation existante de la base sans consulter la réponse historique.",
        "source_document_id": "+".join(document_ids),
        "source_document_version": "multidoc2dial_v1_processed",
        "source_url": "Base locale MultiDoc2Dial",
        "source_section_ids": [item["id"] for item in evidence],
        "source_evidence": evidence,
        "conversation": benchmark_conversation(example),
        "dataset_reference": dataset_reference(example),
        "case_author_id": "dataset_multidoc2dial",
        "case_authored_at_utc": "dataset_provenance",
        "candidate_response": None,
        "annotator_a": empty_annotation(),
        "annotator_b": empty_annotation(),
        "adjudication": {**empty_annotation(), "status": "pending", "adjudicator_id": None},
    }
    task["precommitment_sha256"] = precommitment_hash(task)
    return task


def deferral_task(row: dict[str, Any], example: dict[str, Any]) -> dict[str, Any]:
    task = {
        "case_id": row["example_id"],
        "domain": row["domain"],
        "scenario": "existing_deterministic_case",
        "case_type": row["runtime_action"],
        "partition": row["partition"],
        "authoring_status": "existing_database_question_locked",
        "task_instruction": "Annoter l’action conversationnelle correcte. Aucun fait externe ne doit être inventé.",
        "source_document_id": "conversation_only",
        "source_document_version": "multidoc2dial_v1_processed",
        "source_url": "Base locale MultiDoc2Dial",
        "source_section_ids": ["conversation_only"],
        "source_evidence": [
            {
                "id": "conversation_only",
                "title": "Contexte conversationnel uniquement",
                "text": "Aucune preuve documentaire n’est nécessaire pour ce cas déterministe.",
            }
        ],
        "conversation": benchmark_conversation(example),
        "dataset_reference": dataset_reference(example),
        "case_author_id": "dataset_multidoc2dial",
        "case_authored_at_utc": "dataset_provenance",
        "candidate_response": None,
        "annotator_a": empty_annotation(),
        "annotator_b": empty_annotation(),
        "adjudication": {**empty_annotation(), "status": "pending", "adjudicator_id": None},
    }
    task["precommitment_sha256"] = precommitment_hash(task)
    return task


def sample(rows: list[dict[str, Any]], domain: str, count: int, salt: int) -> list[dict[str, Any]]:
    pool = sorted((row for row in rows if row["domain"] == domain), key=lambda row: row["example_id"])
    generator = random.Random(SAMPLING_SEED + salt)
    generator.shuffle(pool)
    if len(pool) < count:
        raise ValueError(f"Not enough {domain} rows: required {count}, found {len(pool)}")
    return pool[:count]


def main() -> None:
    benchmark_path = DATA_ROOT / "retrieval_benchmarks" / BENCHMARK_ID / "validation.jsonl"
    benchmark = {row["id"]: row for row in iter_jsonl(benchmark_path)}
    contract_root = DATA_ROOT / "planned_generation_contract_experiments" / CONTRACT_ID
    requests = list(iter_jsonl(contract_root / "requests" / "validation_requests.jsonl"))
    deferrals = list(iter_jsonl(contract_root / "deferrals" / "validation_deferrals.jsonl"))
    span_to_document = span_document_index()
    used_ids = (collect_previously_used_ids() | collect_mandatory_pipeline_used_ids()).intersection(benchmark)
    used_document_ids = {
        document_id
        for example_id in used_ids
        for document_id in example_document_ids(benchmark[example_id], span_to_document)
    }
    eligible_requests = eligible_rows(
        requests, benchmark, used_ids, used_document_ids, span_to_document, partition=None
    )
    eligible_deferrals = eligible_rows(
        deferrals, benchmark, used_ids, used_document_ids, span_to_document, partition=None
    )
    selected_requests: list[dict[str, Any]] = []
    selected_deferrals: list[dict[str, Any]] = []
    for index, domain in enumerate(DOMAINS):
        selected_requests.extend(sample(eligible_requests, domain, REQUESTS_PER_DOMAIN, index * 2))
        selected_deferrals.extend(sample(eligible_deferrals, domain, DEFERRALS_PER_DOMAIN, index * 2 + 1))
    tasks = [request_task(row, benchmark[row["example_id"]]) for row in selected_requests]
    tasks.extend(deferral_task(row, benchmark[row["example_id"]]) for row in selected_deferrals)
    tasks.sort(key=lambda task: (DOMAINS.index(task["domain"]), task["case_id"]))
    pack_hash = hashlib.sha256(
        "".join(task["precommitment_sha256"] for task in tasks).encode("ascii")
    ).hexdigest()
    manifest = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "annotation_pack_id": f"existing_database_annotation_pack_v1_{pack_hash[:12]}",
        "source_dataset": "multidoc2dial_v1",
        "tasks": len(tasks),
        "counts_by_domain": dict(Counter(task["domain"] for task in tasks)),
        "counts_by_case_type": dict(Counter(task["case_type"] for task in tasks)),
        "previously_evaluated_ids_excluded": len(used_ids),
        "previously_evaluated_documents_excluded": len(used_document_ids),
        "questions_already_authored": len(tasks),
        "case_author_required": False,
        "candidate_executed": False,
        "reference_answers_available": len(tasks),
        "references_hidden_from_candidate_runtime": True,
        "human_annotation_status": "OPTIONAL_POST_CANDIDATE_HUMAN_SCORING",
        "candidate_run_status": "READY_FOR_INTERNAL_CANDIDATE_EXECUTION",
        "validation_claim": "INTERNAL_HUMAN_VALIDATION_ONLY_NOT_EXTERNAL_PRODUCTION_PROOF",
    }
    write_jsonl(OUTPUT_ROOT / "annotation_tasks.jsonl", tasks)
    write_json(OUTPUT_ROOT / "manifest.json", manifest)
    write_json(
        OUTPUT_ROOT / "reserve_ids.json",
        {
            "schema_version": "1.0.0",
            "purpose": "internal_existing_database_validation_v1",
            "annotation_pack_id": manifest["annotation_pack_id"],
            "generation_contract_experiment_id": CONTRACT_ID,
            "example_ids": [task["case_id"] for task in tasks],
            "test_split_opened": False,
            "reference_answers_exposed_to_candidate_runtime": False,
        },
    )
    write_json(REPORT_ROOT / "existing_database_pack_report.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
