from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

import tiktoken
from dotenv import load_dotenv

from scripts.evaluate_mandatory_claim_pipeline import (
    make_response_body,
    parse_structured,
    planner_schema,
    realization_schema,
    render_realizer_input,
    sanitize_redundant_conflicting_claims,
    should_review_plan,
    validate_and_assemble,
    validate_branch_outcome,
    validate_conditional_qualifier_support,
    validate_plan,
    validate_primary_evidence_usage,
    validate_structural_claim_coverage,
)
from scripts.run_automated_end_to_end_evaluation import usage_cost

from .settings import PROJECT_ROOT, RuntimeSettings


class GenerationUnavailable(RuntimeError):
    pass


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class OpenAIGenerator:
    def __init__(self, settings: RuntimeSettings):
        self.settings = settings
        self.pipeline = json.loads(settings.path("pipeline_config").read_text(encoding="utf-8"))
        self.planner_prompt = settings.path("planner_prompt").read_text(encoding="utf-8")
        self.reviewer_prompt = settings.path("reviewer_prompt").read_text(encoding="utf-8")
        self.realizer_prompt = settings.path("realizer_prompt").read_text(encoding="utf-8")
        load_dotenv(PROJECT_ROOT / self.pipeline["api"]["dotenv_path"])
        self.api_key = os.getenv(self.pipeline["api"]["api_key_environment_variable"])
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.cache_path = settings.path("response_cache")
        self._cache_lock = threading.Lock()
        self._client: Any = None

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.settings.generation.get("enabled", True))

    def _client_instance(self) -> Any:
        if not self.available:
            raise GenerationUnavailable("OPENAI_API_KEY is missing or generation is disabled.")
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                timeout=float(self.pipeline["api"]["request_timeout_seconds"]),
                max_retries=int(self.pipeline["api"]["max_retries"]),
            )
        return self._client

    def embed(self, text: str) -> tuple[Any, dict[str, Any]]:
        client = self._client_instance()
        response = client.embeddings.create(
            model=self.settings.retrieval["embedding_model"],
            input=text,
            encoding_format="float",
        )
        usage = getattr(response, "usage", None)
        tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        return response.data[0].embedding, {
            "input_tokens": tokens,
            "estimated_cost_usd": tokens * 0.02 / 1_000_000,
        }

    def _cached_response(self, kind: str, body: dict[str, Any]) -> dict[str, Any] | None:
        key = stable_hash({"candidate_id": self.settings.candidate_id, "kind": kind, "body": body})
        with self._cache_lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.cache_path) as connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS responses (cache_key TEXT PRIMARY KEY, kind TEXT NOT NULL, response_json TEXT NOT NULL, created_at_utc TEXT NOT NULL)"
                )
                row = connection.execute(
                    "SELECT response_json FROM responses WHERE cache_key=?", (key,)
                ).fetchone()
        return json.loads(row[0]) if row else None

    def _store_response(self, kind: str, body: dict[str, Any], value: dict[str, Any]) -> None:
        key = stable_hash({"candidate_id": self.settings.candidate_id, "kind": kind, "body": body})
        with self._cache_lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.cache_path) as connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS responses (cache_key TEXT PRIMARY KEY, kind TEXT NOT NULL, response_json TEXT NOT NULL, created_at_utc TEXT NOT NULL)"
                )
                connection.execute(
                    "INSERT OR REPLACE INTO responses VALUES (?,?,?,?)",
                    (key, kind, json.dumps(value, ensure_ascii=False), datetime.now(timezone.utc).isoformat()),
                )
                connection.commit()

    def _call(
        self, kind: str, body: dict[str, Any], budget: dict[str, float]
    ) -> tuple[dict[str, Any], bool]:
        cached = self._cached_response(kind, body)
        if cached is not None:
            return cached, True
        pricing = self.pipeline["pricing"]
        input_tokens = len(self.encoding.encode(json.dumps(body, ensure_ascii=False)))
        reserved = (
            input_tokens * float(pricing["input_usd_per_million_tokens"])
            + int(body["max_output_tokens"]) * float(pricing["output_usd_per_million_tokens"])
        ) / 1_000_000
        if budget["actual"] + reserved > budget["ceiling"] + 1e-12:
            raise RuntimeError("Per-request generation cost ceiling would be exceeded.")
        response = self._client_instance().responses.create(**body)
        value = response.model_dump(mode="json")
        self._store_response(kind, body, value)
        budget["actual"] += usage_cost(value.get("usage") or {}, pricing)
        return value, False

    @staticmethod
    def _plan_errors(plan: Any, schema: dict[str, Any], prepared: dict[str, Any], parse: list[str]) -> list[str]:
        return [
            *parse,
            *validate_plan(plan, schema, prepared["required_action"]),
            *validate_branch_outcome(plan, prepared),
            *validate_primary_evidence_usage(plan, prepared),
            *validate_conditional_qualifier_support(plan, prepared),
            *validate_structural_claim_coverage(plan, prepared),
        ]

    def generate(self, prepared: dict[str, Any]) -> dict[str, Any]:
        evidence_ids = [item["evidence_id"] for item in prepared["evidence_catalog"]]
        maximum_claims = int(self.pipeline["maximum_mandatory_claims"])
        schema = planner_schema(evidence_ids, maximum_claims)
        budget = {
            "actual": 0.0,
            "ceiling": float(self.settings.generation["per_request_maximum_cost_usd"]),
        }
        cache_hits: list[str] = []
        errors: list[str] = []

        body = make_response_body(
            self.pipeline["planner"], self.planner_prompt, prepared["request_body"]["input"],
            "runtime_mandatory_claim_plan_v1", schema,
        )
        value, cached = self._call("planner", body, budget)
        if cached:
            cache_hits.append("planner")
        plan, parse_errors = parse_structured(value, schema)
        plan, sanitizations = sanitize_redundant_conflicting_claims(plan, prepared)
        plan_errors = self._plan_errors(plan, schema, prepared, parse_errors)

        for attempt in range(int(self.settings.generation["maximum_plan_repair_attempts"])):
            if not plan_errors:
                break
            retry_input = "\n\n".join([
                prepared["request_body"]["input"],
                "<PREVIOUS_VALIDATION_ERRORS>\n" + "\n".join(plan_errors) + "\n</PREVIOUS_VALIDATION_ERRORS>",
                "Return one corrected complete valid plan.",
            ])
            retry_stage = {**self.pipeline["planner"], "max_output_tokens": int(self.pipeline["planner"]["max_output_tokens"]) * 2}
            body = make_response_body(retry_stage, self.planner_prompt, retry_input, f"runtime_plan_repair_v{attempt + 1}", schema)
            value, cached = self._call(f"planner_repair_{attempt + 1}", body, budget)
            if cached:
                cache_hits.append(f"planner_repair_{attempt + 1}")
            plan, parse_errors = parse_structured(value, schema)
            plan, more_sanitizations = sanitize_redundant_conflicting_claims(plan, prepared)
            sanitizations.extend(more_sanitizations)
            plan_errors = self._plan_errors(plan, schema, prepared, parse_errors)
        if plan_errors:
            return {"payload": None, "errors": plan_errors, "cost_usd": budget["actual"], "cache_hits": cache_hits}

        if should_review_plan(prepared, plan):
            review_input = "\n\n".join([
                prepared["request_body"]["input"],
                "<MANDATORY_CLAIM_DRAFT>\n" + json.dumps(plan, ensure_ascii=False, indent=2) + "\n</MANDATORY_CLAIM_DRAFT>",
            ])
            body = make_response_body(self.pipeline["reviewer"], self.reviewer_prompt, review_input, "runtime_plan_review_v1", schema)
            value, cached = self._call("reviewer", body, budget)
            if cached:
                cache_hits.append("reviewer")
            reviewed, parse_errors = parse_structured(value, schema)
            reviewed, more_sanitizations = sanitize_redundant_conflicting_claims(reviewed, prepared)
            sanitizations.extend(more_sanitizations)
            review_errors = self._plan_errors(reviewed, schema, prepared, parse_errors)
            if not review_errors:
                plan = reviewed
            else:
                errors.extend([f"reviewer:{item}" for item in review_errors])

        claim_ids = [item["claim_id"] for item in plan["mandatory_claims"]]
        realization_output_schema = realization_schema(claim_ids, evidence_ids)
        realizer_input = render_realizer_input(prepared, plan)
        output_limits = self.pipeline.get("adaptive_output_limits", {})
        realizer_stage = self.pipeline["realizer"]
        if output_limits.get("enabled"):
            adaptive = int(output_limits["realizer_base_tokens"]) + int(output_limits["realizer_tokens_per_claim"]) * len(claim_ids)
            realizer_stage = {**realizer_stage, "max_output_tokens": min(int(realizer_stage["max_output_tokens"]), adaptive)}
        body = make_response_body(realizer_stage, self.realizer_prompt, realizer_input, "runtime_claim_realization_v1", realization_output_schema)
        value, cached = self._call("realizer", body, budget)
        if cached:
            cache_hits.append("realizer")
        realization, parse_errors = parse_structured(value, realization_output_schema)
        payload, realization_errors = validate_and_assemble(prepared, plan, realization, realization_output_schema)
        realization_errors = [*parse_errors, *realization_errors]

        for attempt in range(int(self.settings.generation["maximum_realizer_repair_attempts"])):
            if not realization_errors:
                break
            retry_input = "\n\n".join([
                realizer_input,
                "<PREVIOUS_INVALID_OUTPUT>\n" + json.dumps(realization, ensure_ascii=False) + "\n</PREVIOUS_INVALID_OUTPUT>",
                "<VALIDATION_ERRORS>\n" + "\n".join(realization_errors) + "\n</VALIDATION_ERRORS>",
                "Return a corrected complete realization.",
            ])
            body = make_response_body(realizer_stage, self.realizer_prompt, retry_input, f"runtime_realization_repair_v{attempt + 1}", realization_output_schema)
            value, cached = self._call(f"realizer_repair_{attempt + 1}", body, budget)
            if cached:
                cache_hits.append(f"realizer_repair_{attempt + 1}")
            realization, parse_errors = parse_structured(value, realization_output_schema)
            payload, realization_errors = validate_and_assemble(prepared, plan, realization, realization_output_schema)
            realization_errors = [*parse_errors, *realization_errors]
        errors.extend(realization_errors)
        return {
            "payload": payload if not realization_errors else None,
            "errors": errors,
            "sanitizations": sanitizations,
            "cost_usd": budget["actual"],
            "cache_hits": cache_hits,
        }
