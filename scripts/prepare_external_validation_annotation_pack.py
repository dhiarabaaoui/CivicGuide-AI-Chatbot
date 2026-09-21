"""Prepare 80 source-grounded tasks for independent external annotation."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "processed" / "multidoc2dial_v1"
SOURCE_ROOT = DATA_ROOT / "clean_validation" / "external_sources_v1"
OUTPUT_ROOT = DATA_ROOT / "clean_validation" / "external_annotation_pack_v1"
REPORT_ROOT = ROOT / "reports" / "generated" / "multidoc2dial_v1" / "external_clean_validation"
DOMAINS = ("dmv", "ssa", "va", "studentaid")
CASES_PER_DOMAIN = 20
MAXIMUM_ALLOWED_TEXT_SIMILARITY = 0.20
SCENARIOS = (
    "direct_answer",
    "clarification",
    "abstention",
    "multi_turn",
    "multi_span",
    "adversarial_or_insufficient_evidence",
)


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


def normalized_title(value: str) -> str:
    value = re.sub(r"#\d+$", "", value.lower())
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def audit_disjointness(
    external: list[dict[str, Any]], local: list[dict[str, Any]]
) -> dict[str, Any]:
    local_titles = defaultdict(list)
    for document in local:
        local_titles[normalized_title(document["title"])].append(document["id"])
    exact_title_collisions = [
        {
            "external_id": document["id"],
            "local_ids": local_titles[normalized_title(document["title"])],
        }
        for document in external
        if normalized_title(document["title"]) in local_titles
    ]
    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(3, 5), sublinear_tf=True, norm="l2"
    )
    matrix = vectorizer.fit_transform(
        [document["text"] for document in [*external, *local]]
    )
    similarities = linear_kernel(matrix[: len(external)], matrix[len(external) :])
    matches: list[dict[str, Any]] = []
    for index, document in enumerate(external):
        local_index = int(similarities[index].argmax())
        matches.append(
            {
                "external_id": document["id"],
                "local_id": local[local_index]["id"],
                "local_title": local[local_index]["title"],
                "similarity": float(similarities[index, local_index]),
            }
        )
    maximum_similarity = max(match["similarity"] for match in matches)
    return {
        "method": "word_tfidf_ngrams_3_to_5_cosine_plus_normalized_exact_title",
        "maximum_allowed_text_similarity": MAXIMUM_ALLOWED_TEXT_SIMILARITY,
        "maximum_observed_text_similarity": maximum_similarity,
        "exact_title_collisions": exact_title_collisions,
        "text_threshold_collisions": [
            match
            for match in matches
            if match["similarity"] >= MAXIMUM_ALLOWED_TEXT_SIMILARITY
        ],
        "top_text_matches": sorted(
            matches, key=lambda match: match["similarity"], reverse=True
        )[:10],
        "passed": not exact_title_collisions
        and maximum_similarity < MAXIMUM_ALLOWED_TEXT_SIMILARITY,
    }


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


def scenario_instruction(scenario: str) -> str:
    instructions = {
        "direct_answer": "Author a natural user question answered directly by the supplied evidence.",
        "clarification": "Author an ambiguous request for which the correct action is one focused clarification question.",
        "abstention": "Author a plausible domain question that the supplied evidence does not answer; the correct action must be abstention.",
        "multi_turn": "Author a short dialogue whose final user reply resolves the immediately preceding assistant question.",
        "multi_span": "Author a question that materially requires both supplied evidence sections for a complete answer.",
        "adversarial_or_insufficient_evidence": "Author a request containing a tempting unsupported premise; the answer must not adopt that premise as an external fact.",
    }
    return instructions[scenario]


def conversation_draft(
    scenario: str, document_title: str, selected_sections: list[dict[str, Any]]
) -> list[dict[str, str]]:
    primary = selected_sections[0]["title"]
    secondary = selected_sections[-1]["title"]
    if scenario == "direct_answer":
        return [{"role": "user", "text": f"What should I know about {primary}?"}]
    if scenario == "clarification":
        return [{"role": "user", "text": "Can I do that in my situation?"}]
    if scenario == "abstention":
        return [
            {
                "role": "user",
                "text": (
                    f"Using this information about {primary}, can you guarantee my individual "
                    "application will be approved and give me the exact decision date?"
                ),
            }
        ]
    if scenario == "multi_turn":
        return [
            {"role": "user", "text": f"I need help with {document_title}."},
            {"role": "assistant", "text": f"Are you asking specifically about {primary}?"},
            {"role": "user", "text": "Yes."},
        ]
    if scenario == "multi_span":
        return [
            {
                "role": "user",
                "text": f"How do {primary} and {secondary} apply together, and what do I need to do?",
            }
        ]
    return [
        {
            "role": "user",
            "text": (
                f"Someone told me that {primary} guarantees approval in every case. "
                "Please confirm that claim."
            ),
        }
    ]


def prepare_tasks(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_domain = {
        domain: [document for document in documents if document["domain"] == domain]
        for domain in DOMAINS
    }
    if any(len(by_domain[domain]) != 5 for domain in DOMAINS):
        raise ValueError("Expected exactly five approved source documents per domain")
    tasks: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for position in range(1, CASES_PER_DOMAIN + 1):
            document_index = (position - 1) // 4
            within_document = (position - 1) % 4
            document = by_domain[domain][document_index]
            sections = document["sections"]
            first_index = min(within_document * 2, len(sections) - 1)
            scenario = SCENARIOS[(position - 1) % len(SCENARIOS)]
            selected = [sections[first_index]]
            if scenario == "multi_span" or (
                scenario == "multi_turn" and first_index + 1 < len(sections)
            ):
                selected.append(sections[min(first_index + 1, len(sections) - 1)])
            task_payload = {
                "case_id": f"clean_{domain}_{position:02d}",
                "domain": domain,
                "scenario": scenario,
                "authoring_status": "draft_ready_for_independent_acceptance",
                "task_instruction": scenario_instruction(scenario),
                "source_document_id": document["id"],
                "source_document_version": document["version"],
                "source_url": document["source_url"],
                "source_section_ids": [section["id"] for section in selected],
                "source_evidence": selected,
                "conversation_draft": conversation_draft(
                    scenario, document["title"], selected
                ),
                "conversation": [],
                "candidate_response": None,
                "annotator_a": empty_annotation(),
                "annotator_b": empty_annotation(),
                "adjudication": {
                    **empty_annotation(),
                    "status": "pending",
                    "adjudicator_id": None,
                },
            }
            task_payload["precommitment_sha256"] = hashlib.sha256(
                json.dumps(
                    {
                        key: task_payload[key]
                        for key in (
                            "case_id",
                            "domain",
                            "scenario",
                            "source_document_id",
                            "source_document_version",
                            "source_section_ids",
                            "source_evidence",
                        )
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            tasks.append(task_payload)
    return tasks


def main() -> None:
    documents = list(iter_jsonl(SOURCE_ROOT / "documents.jsonl"))
    local_documents = list(iter_jsonl(DATA_ROOT / "documents.jsonl"))
    disjointness = audit_disjointness(documents, local_documents)
    if not disjointness["passed"]:
        write_json(REPORT_ROOT / "document_disjointness_audit.json", disjointness)
        raise SystemExit("External sources failed the document-disjointness audit")
    tasks = prepare_tasks(documents)
    counts_by_domain = Counter(task["domain"] for task in tasks)
    counts_by_scenario = Counter(task["scenario"] for task in tasks)
    pack_hash = hashlib.sha256(
        "".join(task["precommitment_sha256"] for task in tasks).encode("ascii")
    ).hexdigest()
    manifest = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "annotation_pack_id": f"external_annotation_pack_v1_{pack_hash[:12]}",
        "source_pack_report": "reports/generated/multidoc2dial_v1/external_clean_validation/source_collection_report.json",
        "tasks": len(tasks),
        "counts_by_domain": dict(counts_by_domain),
        "counts_by_scenario": dict(counts_by_scenario),
        "source_documents": len(documents),
        "document_disjointness": disjointness,
        "candidate_executed": False,
        "annotator_a_completed": 0,
        "annotator_b_completed": 0,
        "adjudicated": 0,
        "production_status": "PENDING_INDEPENDENT_CASE_AUTHORING_AND_DUAL_ANNOTATION",
    }
    write_jsonl(OUTPUT_ROOT / "annotation_tasks.jsonl", tasks)
    write_json(OUTPUT_ROOT / "manifest.json", manifest)
    write_json(REPORT_ROOT / "document_disjointness_audit.json", disjointness)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
