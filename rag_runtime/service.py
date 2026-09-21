from __future__ import annotations

import json
import re
import threading
import unicodedata
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .generation import GenerationUnavailable, OpenAIGenerator
from .observability import emit_event
from .retrieval import EvidenceIndex, HybridRetriever, RetrievalHit, tokenize
from .settings import RuntimeSettings, load_settings


class RuntimeUnavailable(RuntimeError):
    pass


class RAGRuntime:
    def __init__(self, settings: RuntimeSettings | None = None):
        self.settings = settings or load_settings()
        self.retriever = HybridRetriever(self.settings)
        self.evidence_index = EvidenceIndex(self.settings.path("documents"))
        self.generator = OpenAIGenerator(self.settings)
        self._log_lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "candidate_id": self.settings.candidate_id,
            "chunks": len(self.retriever.chunks),
            "dense_vectors": list(self.retriever.vectors.shape),
            "generation_available": self.generator.available,
        }

    @staticmethod
    def _routing_text(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value.casefold())
        without_accents = "".join(
            character for character in normalized if not unicodedata.combining(character)
        )
        return " ".join(re.findall(r"[a-z0-9]+", without_accents))

    @classmethod
    def _explicit_domain(cls, message: str) -> str | None:
        text = f" {cls._routing_text(message)} "
        signals = {
            "dmv": (
                " dmv ",
                " real id ",
                " driver license ",
                " non driver id ",
                " permis de conduire ",
            ),
            "ssa": (
                " social security ",
                " ssa ",
                " securite sociale ",
            ),
            "va": (
                " veterans affairs ",
                " va disability ",
                " va compensation ",
                " indemnisation d invalidite ",
                " disability compensation ",
            ),
            "studentaid": (
                " fafsa ",
                " student aid ",
                " federal student aid ",
                " aide federale ",
            ),
        }
        matches = [
            domain
            for domain, phrases in signals.items()
            if any(phrase in text for phrase in phrases)
        ]
        return matches[0] if len(matches) == 1 else None

    @classmethod
    def _retrieval_query(
        cls,
        message: str,
        history: list[dict[str, str]],
        domain: str | None = None,
    ) -> str:
        contextual = [turn["content"] for turn in history[-4:] if turn.get("content")]
        query = "\n".join([*contextual, message]).strip()
        routing_text = cls._routing_text(message)
        asks_general_va_disability = (
            domain == "va"
            and any(
                phrase in routing_text
                for phrase in (
                    "indemnisation d invalidite",
                    "disability compensation",
                    "va disability",
                )
            )
            and not any(
                phrase in routing_text
                for phrase in ("inaptitude au travail", "unemployability", "can t work")
            )
        )
        if asks_general_va_disability:
            query += (
                "\nVA disability compensation how to prepare and file a claim "
                "online or by phone or mail"
            )
        return query

    @staticmethod
    def _needs_clarification(message: str, history: list[dict[str, str]]) -> bool:
        normalized = " ".join(tokenize(message))
        ambiguous = {"ça", "cela", "et moi", "pour moi", "comment", "pourquoi", "oui", "non"}
        return (normalized in ambiguous or len(normalized.split()) < 2) and not history

    def _query_vector(self, query: str, use_dense: bool) -> tuple[np.ndarray | None, dict[str, Any]]:
        if not use_dense:
            return None, {"used": False, "reason": "disabled_by_request"}
        if not self.generator.available:
            return None, {"used": False, "reason": "api_key_unavailable"}
        vector, usage = self.generator.embed(query)
        return np.asarray(vector, dtype=np.float32), {"used": True, **usage}

    @staticmethod
    def _serialize_hit(hit: RetrievalHit) -> dict[str, Any]:
        return {
            "chunk_id": hit.chunk_id,
            "rank": hit.rank,
            "score": round(hit.score, 8),
            "bm25_rank": hit.bm25_rank,
            "dense_rank": hit.dense_rank,
            "dense_similarity": (
                round(hit.dense_similarity, 6) if hit.dense_similarity is not None else None
            ),
            "domain": hit.chunk.get("domain"),
            "document_id": hit.chunk.get("document_id"),
            "document_title": hit.chunk.get("document_title"),
        }

    def retrieve(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        *,
        domain: str | None = None,
        use_dense: bool = True,
    ) -> dict[str, Any]:
        history = history or []
        query = self._retrieval_query(message, history, domain)
        query_vector, embedding = self._query_vector(query, use_dense)
        hits, diagnostics = self.retriever.search(query, domain=domain, query_vector=query_vector)
        evidence = self.evidence_index.select(
            query, hits, int(self.settings.retrieval["maximum_evidence_spans"])
        )
        diagnostics["embedding"] = embedding
        return {
            "query": query,
            "hits": [self._serialize_hit(hit) for hit in hits],
            "evidence": evidence,
            "diagnostics": diagnostics,
        }

    @staticmethod
    def _conversation_text(history: list[dict[str, str]], message: str) -> str:
        turns = [*history, {"role": "user", "content": message}]
        return "\n".join(
            f"{'AGENT' if turn['role'] == 'assistant' else 'USER'}: {turn['content']}"
            for turn in turns
        )

    def _prepare(
        self,
        trace_id: str,
        message: str,
        history: list[dict[str, str]],
        evidence: list[dict[str, Any]],
        domain: str | None,
    ) -> dict[str, Any]:
        previous_agent = next(
            (turn["content"] for turn in reversed(history) if turn["role"] == "assistant"),
            "NONE",
        )
        effective_domain = domain or (evidence[0]["domain"] if evidence else "unknown")
        execution_plan = "\n".join([
            "<EXECUTION_PLAN>",
            "REQUIRED_ACTION: answer",
            "BRANCH_MODE: direct_answer",
            "USER_RESPONSE_POLARITY: unknown",
            "USER_RESPONSE_POLARITY_CONFIDENCE: 0.00",
            "USER_RESPONSE_POLARITY_SOURCE: none",
            f"PREVIOUS_AGENT_UTTERANCE: {previous_agent}",
            # A live retrieval result has no human-labelled gold anchor. Forcing E1 here
            # can reject an otherwise grounded answer when the top atomic span is only
            # contextual. The planner must inspect and cite the relevant supplied spans.
            "PRIMARY_EVIDENCE_ID: NONE",
            f"DOMAIN: {effective_domain}",
            "</EXECUTION_PLAN>",
        ])
        blocks = []
        catalog = []
        for item in evidence:
            title_path = " > ".join([*item.get("parent_titles", []), item.get("section_title", "")]).strip(" >")
            blocks.append("\n".join([
                f"[{item['evidence_id']}]",
                f"Document: {item['document_title']}",
                f"Section: {title_path}",
                f"Span ID: {item['span_id']}",
                f"Text: {item['text']}",
            ]))
            catalog.append({
                "evidence_id": item["evidence_id"],
                "span_id": item["span_id"],
                "document_id": item["document_id"],
                "section_id": item["section_id"],
                "section_position": item["section_position"],
                "text": item["text"],
                "title": title_path,
                "priority_rank": item["priority_rank"],
            })
        input_text = "\n\n".join([
            execution_plan,
            "<CONVERSATION>\n" + self._conversation_text(history, message) + "\n</CONVERSATION>",
            "<EVIDENCE_UNTRUSTED_DATA>\n" + "\n\n---\n\n".join(blocks) + "\n</EVIDENCE_UNTRUSTED_DATA>",
        ])
        return {
            "example_id": trace_id,
            "domain": effective_domain,
            "required_action": "answer",
            "expected_status": "answered",
            "evidence_catalog": catalog,
            "structural_evidence_chains": [],
            "request_body": {"input": input_text},
        }

    def _has_sufficient_evidence(self, retrieval: dict[str, Any]) -> bool:
        if not retrieval["evidence"]:
            return False
        diagnostics = retrieval["diagnostics"]
        if diagnostics["bm25_top_score"] > 0:
            return True
        dense = diagnostics.get("dense_top_similarity")
        return dense is not None and dense >= float(self.settings.retrieval["minimum_dense_similarity"])

    def _citation_details(
        self, citations: list[dict[str, str]], evidence: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        by_id = {item["evidence_id"]: item for item in evidence}
        output = []
        seen: set[str] = set()
        for citation in citations:
            evidence_id = citation["evidence_id"]
            source = by_id.get(evidence_id)
            if not source or evidence_id in seen:
                continue
            seen.add(evidence_id)
            output.append({
                **citation,
                "evidence_quote": source["text"],
                "span_id": source["span_id"],
                "document_id": source["document_id"],
                "document_title": source["document_title"],
                "section_title": source["section_title"],
                "source_document_id": source.get("source_document_id"),
            })
        return output

    def _log(self, row: dict[str, Any]) -> None:
        if not self.settings.safeguards.get("log_requests_without_secrets", True):
            return
        path = self.settings.path("request_log")
        path.parent.mkdir(parents=True, exist_ok=True)
        retrieval = row.get("retrieval") or {}
        safe = {
            "created_at": row.get("created_at"),
            "trace_id": row.get("trace_id"),
            "status": row.get("status"),
            "suggested_domain": row.get("suggested_domain"),
            "cost_usd": round(float(row.get("cost_usd", 0.0) or 0.0), 8),
            "citation_count": len(row.get("citations") or []),
            "cache_hits": row.get("cache_hits") or [],
            "validation_error_count": len(row.get("validation_errors") or []),
            "retrieval": {
                key: retrieval.get(key)
                for key in (
                    "bm25_top_score",
                    "dense_top_similarity",
                    "candidate_count",
                    "returned_count",
                )
                if key in retrieval
            },
        }
        with self._log_lock, path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(safe, ensure_ascii=False) + "\n")
        emit_event("rag_response", **safe)

    def chat(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        *,
        domain: str | None = None,
        use_dense: bool = True,
    ) -> dict[str, Any]:
        history = history or []
        trace_id = uuid.uuid4().hex
        warning = self.settings.safeguards["experimental_quality_warning"]
        requested_domain = self._explicit_domain(message)
        if domain and requested_domain and requested_domain != domain:
            labels = {
                "dmv": "DMV",
                "ssa": "Social Security",
                "va": "Veterans Affairs",
                "studentaid": "Student Aid",
            }
            result = {
                "trace_id": trace_id,
                "status": "clarification_required",
                "answer": (
                    f"Votre question semble concerner {labels[requested_domain]}, "
                    f"mais le domaine sélectionné est {labels[domain]}. "
                    f"Sélectionnez {labels[requested_domain]} puis renvoyez votre question."
                ),
                "citations": [],
                "suggested_domain": requested_domain,
                "warning": warning,
                "cost_usd": 0.0,
            }
            self._log({**result, "created_at": datetime.now(timezone.utc).isoformat()})
            return result
        if self._needs_clarification(message, history):
            result = {
                "trace_id": trace_id,
                "status": "clarification_required",
                "answer": "Pouvez-vous préciser votre question et le service administratif concerné ?",
                "citations": [],
                "warning": warning,
                "cost_usd": 0.0,
            }
            self._log({**result, "created_at": datetime.now(timezone.utc).isoformat()})
            return result

        retrieval = self.retrieve(message, history, domain=domain, use_dense=use_dense)
        if not self._has_sufficient_evidence(retrieval):
            result = {
                "trace_id": trace_id,
                "status": "insufficient_evidence",
                "answer": "Je ne dispose pas de preuves suffisamment pertinentes pour répondre de façon fiable.",
                "citations": [],
                "warning": warning,
                "cost_usd": float(retrieval["diagnostics"]["embedding"].get("estimated_cost_usd", 0.0)),
                "retrieval": retrieval["diagnostics"],
            }
            self._log({**result, "created_at": datetime.now(timezone.utc).isoformat()})
            return result
        if not self.generator.available:
            raise RuntimeUnavailable("Generation requires OPENAI_API_KEY; lexical retrieval remains available.")

        prepared = self._prepare(trace_id, message, history, retrieval["evidence"], domain)
        try:
            generated = self.generator.generate(prepared)
        except GenerationUnavailable as error:
            raise RuntimeUnavailable(str(error)) from error
        payload = generated["payload"]
        if payload is None:
            result = {
                "trace_id": trace_id,
                "status": "insufficient_evidence",
                "answer": "Je ne dispose pas de sources suffisamment fiables pour répondre à cette question. Essayez de la reformuler ou choisissez un autre domaine.",
                "citations": [],
                "warning": warning,
                "cost_usd": generated["cost_usd"] + float(retrieval["diagnostics"]["embedding"].get("estimated_cost_usd", 0.0)),
                "cache_hits": generated["cache_hits"],
                "validation_errors": generated["errors"],
                "retrieval": retrieval["diagnostics"],
            }
        else:
            result = {
                "trace_id": trace_id,
                "status": payload["status"],
                "answer": payload["answer"],
                "citations": self._citation_details(payload["citations"], retrieval["evidence"]),
                "warning": warning,
                "cost_usd": generated["cost_usd"] + float(retrieval["diagnostics"]["embedding"].get("estimated_cost_usd", 0.0)),
                "cache_hits": generated["cache_hits"],
                "retrieval": retrieval["diagnostics"],
            }
        self._log({**result, "created_at": datetime.now(timezone.utc).isoformat()})
        return result
