"""Run a small, cost-capped, real end-to-end smoke test of the RAG runtime."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.rag_api import app
from rag_runtime.settings import load_settings


AUTHORIZED_SUITE_CEILING_USD = 0.12
REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "generated"
    / "multidoc2dial_v1"
    / "runtime"
    / "e2e_smoke"
)

CASES = [
    {
        "id": "dmv_renewal",
        "domain": "dmv",
        "message": (
            "When can I renew a non-driver ID card, and what happens if it has "
            "been expired for more than two years?"
        ),
    },
    {
        "id": "ssa_replacement",
        "domain": "ssa",
        "message": "Comment remplacer une carte Social Security perdue ?",
    },
    {
        "id": "va_disability",
        "domain": "va",
        "message": "Comment demander une indemnisation d'invalidité auprès du VA ?",
    },
    {
        "id": "studentaid_fafsa",
        "domain": "studentaid",
        "message": "Comment remplir le formulaire FAFSA pour demander une aide fédérale ?",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--maximum-cost-usd",
        type=float,
        default=AUTHORIZED_SUITE_CEILING_USD,
        help="Hard cumulative ceiling for this suite.",
    )
    parser.add_argument(
        "--confirm-maximum-cost-usd",
        type=float,
        default=0.0,
        help="Explicit confirmation; must be at least the selected ceiling.",
    )
    return parser.parse_args()


def public_case_result(case: dict[str, str], response: Any, elapsed: float) -> dict[str, Any]:
    payload = response.json()
    citations = payload.get("citations") if isinstance(payload.get("citations"), list) else []
    checks = {
        "http_200": response.status_code == 200,
        "answered": payload.get("status") == "answered",
        "answer_non_empty": bool(str(payload.get("answer") or "").strip()),
        "has_citations": bool(citations),
        "citations_traceable": bool(citations)
        and all(
            citation.get("evidence_id")
            and citation.get("evidence_quote")
            and citation.get("span_id")
            and citation.get("document_id")
            for citation in citations
        ),
        "warning_visible": bool(payload.get("warning")),
    }
    return {
        **case,
        "http_status": response.status_code,
        "runtime_status": payload.get("status"),
        "answer": payload.get("answer"),
        "citations": citations,
        "citation_count": len(citations),
        "cost_usd": float(payload.get("cost_usd") or 0.0),
        "cache_hits": payload.get("cache_hits") or [],
        "elapsed_seconds": round(elapsed, 3),
        "checks": checks,
        "passed": all(checks.values()),
        "validation_errors": payload.get("validation_errors") or [],
        "error_detail": payload.get("detail"),
    }


def main() -> int:
    args = parse_args()
    if args.maximum_cost_usd <= 0 or args.maximum_cost_usd > AUTHORIZED_SUITE_CEILING_USD:
        raise ValueError(
            f"--maximum-cost-usd must be in (0, {AUTHORIZED_SUITE_CEILING_USD:.2f}]."
        )
    if args.confirm_maximum_cost_usd + 1e-12 < args.maximum_cost_usd:
        raise RuntimeError(
            f"Confirmation required: --confirm-maximum-cost-usd {args.maximum_cost_usd:.2f}"
        )

    settings = load_settings()
    per_request_ceiling = float(settings.generation["per_request_maximum_cost_usd"])
    if len(CASES) * per_request_ceiling > args.maximum_cost_usd + 1e-12:
        raise RuntimeError("The suite cannot fit inside the confirmed worst-case cost ceiling.")

    client = TestClient(app)
    static_checks = {
        "root": client.get("/").status_code == 200,
        "styles": client.get("/static/styles.css").status_code == 200,
        "javascript": client.get("/static/app.js").status_code == 200,
    }
    health_response = client.get("/health")
    health = health_response.json()
    health_checks = {
        "http_200": health_response.status_code == 200,
        "ready": health.get("status") == "ready",
        "generation_available": health.get("generation_available") is True,
        "chunks_loaded": int(health.get("chunks") or 0) > 0,
    }

    spent = 0.0
    results: list[dict[str, Any]] = []
    for case in CASES:
        if spent + per_request_ceiling > args.maximum_cost_usd + 1e-12:
            raise RuntimeError(
                f"Cost guard stopped before {case['id']}: spent=${spent:.6f}, "
                f"reserved=${per_request_ceiling:.6f}."
            )
        started = time.perf_counter()
        response = client.post(
            "/v1/chat",
            json={
                "message": case["message"],
                "history": [],
                "domain": case["domain"],
                "use_dense": True,
            },
        )
        result = public_case_result(case, response, time.perf_counter() - started)
        spent += result["cost_usd"]
        results.append(result)
        print(
            f"[{case['id']}] status={result['runtime_status']} "
            f"citations={result['citation_count']} cost=${result['cost_usd']:.6f} "
            f"passed={result['passed']}"
        )

    clarification_response = client.post(
        "/v1/chat",
        json={"message": "oui", "history": [], "domain": None, "use_dense": True},
    )
    clarification_payload = clarification_response.json()
    clarification_checks = {
        "http_200": clarification_response.status_code == 200,
        "clarification_required": clarification_payload.get("status") == "clarification_required",
        "no_citations": clarification_payload.get("citations") == [],
        "zero_cost": float(clarification_payload.get("cost_usd") or 0.0) == 0.0,
    }
    mismatch_response = client.post(
        "/v1/chat",
        json={
            "message": "Comment remplir le formulaire FAFSA ?",
            "history": [],
            "domain": "dmv",
            "use_dense": True,
        },
    )
    mismatch_payload = mismatch_response.json()
    mismatch_checks = {
        "http_200": mismatch_response.status_code == 200,
        "clarification_required": mismatch_payload.get("status") == "clarification_required",
        "suggests_student_aid": mismatch_payload.get("suggested_domain") == "studentaid",
        "no_citations": mismatch_payload.get("citations") == [],
        "zero_cost": float(mismatch_payload.get("cost_usd") or 0.0) == 0.0,
    }

    report = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": settings.candidate_id,
        "suite": "runtime_real_e2e_smoke_v1",
        "confirmed_maximum_cost_usd": args.maximum_cost_usd,
        "actual_cost_usd": spent,
        "static_checks": static_checks,
        "health": health,
        "health_checks": health_checks,
        "cases": results,
        "clarification": {
            "payload": clarification_payload,
            "checks": clarification_checks,
            "passed": all(clarification_checks.values()),
        },
        "domain_mismatch": {
            "payload": mismatch_payload,
            "checks": mismatch_checks,
            "passed": all(mismatch_checks.values()),
        },
    }
    report["passed"] = (
        all(static_checks.values())
        and all(health_checks.values())
        and all(item["passed"] for item in results)
        and report["clarification"]["passed"]
        and report["domain_mismatch"]["passed"]
        and spent <= args.maximum_cost_usd + 1e-12
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "latest.json"
    markdown_path = REPORT_DIR / "latest.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rows = [
        "| Cas | Domaine | Statut | Citations | Coût USD | Résultat |",
        "|---|---|---|---:|---:|---|",
    ]
    for item in results:
        rows.append(
            f"| {item['id']} | {item['domain']} | {item['runtime_status']} | "
            f"{item['citation_count']} | {item['cost_usd']:.6f} | "
            f"{'PASS' if item['passed'] else 'FAIL'} |"
        )
    markdown = "\n".join(
        [
            "# Test end-to-end réel du runtime",
            "",
            f"- Candidat : `{settings.candidate_id}`",
            f"- Verdict : **{'PASS' if report['passed'] else 'FAIL'}**",
            f"- Coût réel : **${spent:.6f}**",
            f"- Plafond confirmé : **${args.maximum_cost_usd:.2f}**",
            f"- Frontend statique : **{'PASS' if all(static_checks.values()) else 'FAIL'}**",
            f"- Santé du runtime : **{'PASS' if all(health_checks.values()) else 'FAIL'}**",
            f"- Clarification sans coût : **{'PASS' if report['clarification']['passed'] else 'FAIL'}**",
            f"- Mauvais domaine bloqué sans coût : **{'PASS' if report['domain_mismatch']['passed'] else 'FAIL'}**",
            "",
            *rows,
            "",
            "Un cas passe seulement si l’API répond, produit une réponse non vide, conserve "
            "l’avertissement expérimental et fournit au moins une citation traçable.",
            "",
        ]
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    print(f"Report: {markdown_path}")
    print(f"Actual cumulative cost: ${spent:.6f}")
    print(f"Verdict: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
