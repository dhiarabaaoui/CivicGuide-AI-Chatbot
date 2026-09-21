"""Collect and freeze official sources for the external clean-validation set.

The source pack is intentionally separate from MultiDoc2Dial. It records the fetched bytes,
normalized text, stable section IDs, HTTP metadata, and exact-text overlap checks. It does not
create reference answers or pretend to provide the two independent human annotations required
for production authorization.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "processed" / "multidoc2dial_v1"
OUTPUT_ROOT = DATA_ROOT / "clean_validation" / "external_sources_v1"
REPORT_ROOT = ROOT / "reports" / "generated" / "multidoc2dial_v1"
ALLOWED_HOSTS = {
    "www.dmv.ca.gov",
    "www.ssa.gov",
    "www.va.gov",
    "studentaid.gov",
}
MINIMUM_TEXT_CHARACTERS = 800
MAXIMUM_DOWNLOAD_BYTES = 8_000_000


SOURCES = (
    # California DMV: deliberately different jurisdiction from the NY DMV material in the
    # original MultiDoc2Dial corpus.
    ("dmv", "ca_dmv_driver_cards", "https://www.dmv.ca.gov/portal/driver-licenses-identification-cards/"),
    ("dmv", "ca_dmv_real_id", "https://www.dmv.ca.gov/portal/driver-licenses-identification-cards/real-id/"),
    ("dmv", "ca_dmv_registration_renewal", "https://www.dmv.ca.gov/portal/vehicle-registration/vehicle-registration-renewal/"),
    ("dmv", "ca_dmv_license_renewal", "https://www.dmv.ca.gov/portal/driver-licenses-identification-cards/dl-renewal/"),
    ("dmv", "ca_dmv_insurance", "https://www.dmv.ca.gov/portal/vehicle-registration/insurance-requirements/"),
    ("ssa", "ssa_benefit_types", "https://www.ssa.gov/benefits"),
    ("ssa", "ssa_retirement", "https://www.ssa.gov/retirement"),
    ("ssa", "ssa_disability", "https://www.ssa.gov/disability"),
    ("ssa", "ssa_ssi", "https://www.ssa.gov/ssi"),
    ("ssa", "ssa_medicare_signup", "https://www.ssa.gov/medicare/sign-up"),
    ("va", "va_disability_claim", "https://www.va.gov/disability/how-to-file-claim/"),
    ("va", "va_travel_reimbursement", "https://www.va.gov/health-care/file-travel-pay-reimbursement/"),
    ("va", "va_change_address", "https://www.va.gov/change-address/"),
    ("va", "va_pact_act", "https://www.va.gov/resources/the-pact-act-and-your-va-benefits/"),
    ("va", "va_burial_allowance", "https://www.va.gov/burials-memorials/veterans-burial-allowance/"),
    ("studentaid", "fsa_fafsa_steps", "https://studentaid.gov/articles/fafsa-student-steps/"),
    ("studentaid", "fsa_aid_offers", "https://studentaid.gov/articles/evaluating-financial-aid-offers/"),
    ("studentaid", "fsa_repayment_calculator", "https://studentaid.gov/articles/repayment-calculator/"),
    ("studentaid", "fsa_loan_forgiveness", "https://studentaid.gov/articles/student-loan-forgiveness/"),
    ("studentaid", "fsa_account_facts", "https://studentaid.gov/articles/key-facts-accounts/"),
)


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


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


def fetch(url: str) -> tuple[bytes, dict[str, str], str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "RAG-quality-validation/1.0 (+offline research corpus)",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        final_url = response.geturl()
        final_host = (urlparse(final_url).hostname or "").lower()
        if final_host not in ALLOWED_HOSTS:
            raise ValueError(f"Redirected to non-approved host: {final_host}")
        body = response.read(MAXIMUM_DOWNLOAD_BYTES + 1)
        if len(body) > MAXIMUM_DOWNLOAD_BYTES:
            raise ValueError("Source exceeds maximum allowed download size")
        headers = {
            "content_type": response.headers.get("Content-Type", ""),
            "etag": response.headers.get("ETag", ""),
            "last_modified": response.headers.get("Last-Modified", ""),
        }
    return body, headers, final_url


def extract_document(source_id: str, html: bytes) -> tuple[str, str, list[dict[str, str]]]:
    soup = BeautifulSoup(html, "html.parser")
    for node in soup(["script", "style", "noscript", "svg", "nav", "footer", "form"]):
        node.decompose()
    title = normalize_text(soup.title.get_text(" ", strip=True) if soup.title else source_id)
    container = soup.find("main") or soup.find("article") or soup.body or soup
    sections: list[dict[str, str]] = []
    active_title = title
    active_parts: list[str] = []

    def flush() -> None:
        nonlocal active_parts
        text = normalize_text(" ".join(active_parts))
        if len(text) >= 40:
            position = len(sections) + 1
            sections.append(
                {
                    "id": f"extsec_{stable_hash(f'{source_id}:{position}:{active_title}:{text}')[:24]}",
                    "title": active_title,
                    "text": text,
                }
            )
        active_parts = []

    for node in container.find_all(["h1", "h2", "h3", "p", "li"], recursive=True):
        text = normalize_text(node.get_text(" ", strip=True))
        if not text:
            continue
        if node.name in {"h1", "h2", "h3"}:
            flush()
            active_title = text
        else:
            active_parts.append(text)
    flush()
    combined = normalize_text(" ".join(section["text"] for section in sections))
    return title, combined, sections


def local_text_hashes() -> set[str]:
    return {
        stable_hash(normalize_text(document.get("text", "")))
        for document in iter_jsonl(DATA_ROOT / "documents.jsonl")
    }


def assigned_case_shells(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenarios = (
        "direct_answer",
        "clarification",
        "abstention",
        "multi_turn",
        "multi_span",
        "adversarial_or_insufficient_evidence",
    )
    by_domain = {
        domain: [document for document in documents if document["domain"] == domain]
        for domain in ("dmv", "ssa", "va", "studentaid")
    }
    rows: list[dict[str, Any]] = []
    for domain, domain_documents in by_domain.items():
        for position in range(1, 21):
            document = domain_documents[(position - 1) % len(domain_documents)]
            section_ids = [section["id"] for section in document["sections"]]
            rows.append(
                {
                    "case_id": f"clean_{domain}_{position:02d}",
                    "domain": domain,
                    "scenario": scenarios[(position - 1) % len(scenarios)],
                    "source_document_id": document["id"],
                    "source_document_version": document["version"],
                    "source_url": document["source_url"],
                    "source_section_ids": section_ids[: min(3, len(section_ids))],
                    "conversation": [],
                    "candidate_response": None,
                    "annotator_a": None,
                    "annotator_b": None,
                    "adjudication": {"status": "pending"},
                }
            )
    return rows


def main() -> None:
    local_hashes = local_text_hashes()
    fetched_at = datetime.now(timezone.utc).isoformat()
    documents: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for domain, source_id, url in SOURCES:
        try:
            body, headers, final_url = fetch(url)
            title, text, sections = extract_document(source_id, body)
            text_hash = stable_hash(text)
            version = stable_hash(
                json.dumps(
                    {
                        "url": final_url,
                        "body_sha256": hashlib.sha256(body).hexdigest(),
                        "text_sha256": text_hash,
                    },
                    sort_keys=True,
                )
            )[:16]
            document = {
                "schema_version": "1.0.0",
                "id": f"external_{source_id}",
                "domain": domain,
                "title": title,
                "source_url": final_url,
                "requested_url": url,
                "fetched_at_utc": fetched_at,
                "version": version,
                "http_metadata": headers,
                "raw_sha256": hashlib.sha256(body).hexdigest(),
                "text_sha256": text_hash,
                "exact_text_present_in_local_corpus": text_hash in local_hashes,
                "text": text,
                "sections": sections,
            }
            if len(text) < MINIMUM_TEXT_CHARACTERS:
                raise ValueError(f"Only {len(text)} extracted text characters")
            if len(sections) < 3:
                raise ValueError(f"Only {len(sections)} substantive sections")
            snapshot_path = OUTPUT_ROOT / "raw" / f"{source_id}_{version}.html"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_bytes(body)
            documents.append(document)
            print(f"OK {domain:10s} {source_id:30s} chars={len(text):6d} sections={len(sections):3d}")
        except Exception as error:  # Preserve a complete audit trail across all sources.
            failures.append({"domain": domain, "source_id": source_id, "url": url, "error": str(error)})
            print(f"FAIL {domain:10s} {source_id:30s} {error}")

    write_jsonl(OUTPUT_ROOT / "documents.jsonl", documents)
    if len(documents) == len(SOURCES):
        write_jsonl(OUTPUT_ROOT / "case_assignment_shells.jsonl", assigned_case_shells(documents))
    report = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_pack_id": f"external_official_sources_v1_{stable_hash(json.dumps(documents, sort_keys=True))[:12]}",
        "required_sources": len(SOURCES),
        "collected_sources": len(documents),
        "counts_by_domain": {
            domain: sum(document["domain"] == domain for document in documents)
            for domain in ("dmv", "ssa", "va", "studentaid")
        },
        "all_hosts_official": all(
            (urlparse(document["source_url"]).hostname or "").lower() in ALLOWED_HOSTS
            for document in documents
        ),
        "exact_local_text_collisions": [
            document["id"] for document in documents if document["exact_text_present_in_local_corpus"]
        ],
        "failures": failures,
        "case_shells_written": len(documents) == len(SOURCES),
        "production_annotation_status": "PENDING_TWO_INDEPENDENT_ANNOTATORS",
    }
    write_json(REPORT_ROOT / "external_clean_validation" / "source_collection_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures or len(documents) != len(SOURCES) or report["exact_local_text_collisions"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
