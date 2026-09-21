"""Build transparent AI-annotation input files from the frozen external pack.

The generated files contain only task material and explicitly identify the
resulting work as automated, not human validation.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "data"
    / "processed"
    / "multidoc2dial_v1"
    / "clean_validation"
    / "external_annotation_pack_v1"
    / "annotation_tasks.jsonl"
)
OUTPUT_DIR = SOURCE.parent / "ai_automatic_annotation_inputs_v1"


HEADER = """AI AUTOMATIC ANNOTATION INPUT

Provenance: external_annotation_pack_v1
Validation type: automated_not_human
Important: outputs created from this material must never be represented as
independent human annotations or human-validated production metrics.

Use only the conversation and SOURCE_EVIDENCE supplied for each task. Do not
use external knowledge. Preserve every case_id and evidence ID exactly.
"""


def load_rows() -> list[dict]:
    rows = []
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def render_task(row: dict, index: int, total: int) -> str:
    conversation = row.get("conversation") or row.get("conversation_draft") or []
    candidate = row.get("candidate_response")
    parts = [
        f"=== TASK {index}/{total} ===",
        f"case_id: {row['case_id']}",
        f"domain: {row['domain']}",
        f"scenario: {row['scenario']}",
        f"task_instruction: {row['task_instruction']}",
        f"source_document_id: {row['source_document_id']}",
        f"source_document_version: {row['source_document_version']}",
        f"source_url: {row['source_url']}",
        "",
        "CONVERSATION_TO_ANNOTATE:",
    ]
    for turn in conversation:
        parts.append(f"- {turn['role']}: {turn['text']}")

    parts.extend(["", "SOURCE_EVIDENCE:"])
    for evidence in row.get("source_evidence", []):
        parts.extend(
            [
                f"[EVIDENCE_ID: {evidence['id']}]",
                f"title: {evidence.get('title', '')}",
                f"text: {evidence.get('text', '')}",
                "",
            ]
        )

    parts.append("CANDIDATE_RESPONSE:")
    if candidate is None:
        parts.append("null (create the reference annotation only; do not score a candidate)")
    elif isinstance(candidate, str):
        parts.append(candidate)
    else:
        parts.append(json.dumps(candidate, ensure_ascii=False, indent=2))
    parts.extend(["", "=== END TASK ===", ""])
    return "\n".join(parts)


def write_bundle(path: Path, rows: list[dict]) -> None:
    body = [HEADER, f"Task count: {len(rows)}", ""]
    for index, row in enumerate(rows, start=1):
        body.append(render_task(row, index, len(rows)))
    path.write_text("\n".join(body).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows()
    if len(rows) != 80:
        raise RuntimeError(f"Expected 80 tasks, found {len(rows)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_bundle(OUTPUT_DIR / "all_80_tasks.txt", rows)

    by_domain: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)

    for domain in sorted(by_domain):
        domain_rows = by_domain[domain]
        write_bundle(OUTPUT_DIR / f"{domain}_20_tasks.txt", domain_rows)

    manifest = {
        "schema_version": "1.0.0",
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "annotation_type": "automated_not_human",
        "total_tasks": len(rows),
        "tasks_by_domain": {key: len(value) for key, value in sorted(by_domain.items())},
        "full_file": "all_80_tasks.txt",
        "split_files": [f"{domain}_20_tasks.txt" for domain in sorted(by_domain)],
        "candidate_responses_present": sum(row.get("candidate_response") is not None for row in rows),
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Created {OUTPUT_DIR.relative_to(ROOT)}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
