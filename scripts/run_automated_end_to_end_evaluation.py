"""Run a cached, cost-capped, automatic end-to-end RAG evaluation on validation holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tiktoken
from dotenv import load_dotenv
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "processed" / "multidoc2dial_v1"
REPORT_ROOT = ROOT / "reports" / "generated" / "multidoc2dial_v1"
CONFIG_PATH = ROOT / "configs" / "automated_end_to_end_evaluation.json"

JUDGE_INSTRUCTIONS = """You are an independent evaluator of a conversational RAG answer.
Use only the supplied question, conversation history, reference answer, gold evidence,
candidate answer, and cited evidence. Do not use outside knowledge.

Evidence policy:
- The current question and conversation history are admissible premises. The candidate may
  repeat a fact explicitly stated by the user without citing a document.
- External rules, requirements, dates, amounts, procedures, and eligibility criteria must
  be supported by the candidate's cited evidence.
- A narrow logical conclusion is faithful when it follows from an explicit user answer plus
  a cited rule (for example, a user answers no to a requirement that the cited source says
  must be true). Do not demand a document citation for the user's own yes/no statement.
- Do not treat a concise paraphrase of the active topic as an unsupported external fact when
  that topic is already explicit in the question or conversation.

Evaluate:
- action_correct: the candidate performs the expected answer or follow-up action.
- correctness_0_to_4: 4 fully correct/direct/complete; 3 mainly correct with a minor
  omission; 2 partially correct with an important omission/error; 1 mostly incorrect;
  0 unusable.
- faithfulness_pass: every important external factual claim is supported by the candidate's
  cited evidence, and any conclusion from user facts is licensed by a cited rule. A quoted
  fragment alone does not support a broader unrelated claim.
- completeness_pass: the candidate covers the essential information in the reference
  answer and gold evidence without omitting a material condition, limitation, or next step.
Return only the required structured output."""


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.translate(
        str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
    )
    return " ".join(re.sub(r"[^\w%$]+", " ", value.casefold()).split())


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-maximum-cost-usd", type=float, default=0.0)
    return parser.parse_args()


def collect_previously_used_ids() -> set[str]:
    used: set[str] = set()
    calibration_root = REPORT_ROOT / "quality_evaluation_experiments"
    for path in calibration_root.glob("answer_quality_evaluation_v1_*/human_calibration_blind.csv"):
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            delimiter = ";" if stream.readline().count(";") > 5 else ","
            stream.seek(0)
            for row in csv.DictReader(stream, delimiter=delimiter):
                example_id = str(row.get("example_id", "")).strip()
                if example_id.startswith("eval_"):
                    used.add(example_id)
    pattern = re.compile(r"eval_[0-9a-f]{24}")
    for path in (ROOT / "configs").glob("*.json"):
        if path == CONFIG_PATH:
            continue
        used.update(pattern.findall(path.read_text(encoding="utf-8")))
    for path in (DATA_ROOT / "fresh_human_canary_experiments").glob("*/generations.jsonl"):
        for row in iter_jsonl(path):
            if str(row.get("example_id", "")).startswith("eval_"):
                used.add(row["example_id"])
    if load_json(CONFIG_PATH).get("exclude_previous_automated_evaluations", False):
        automated_root = DATA_ROOT / "automated_end_to_end_evaluation_experiments"
        for path in automated_root.glob("*/case_results.jsonl"):
            for row in iter_jsonl(path):
                example_id = str(row.get("example_id", ""))
                if example_id.startswith("eval_"):
                    used.add(example_id)
    # The current automated benchmark is intentionally not excluded: with the same
    # seed and configuration, Run All selects the same IDs and reuses the API cache.
    # A new fresh benchmark must change the version or sampling seed explicitly.
    return used


def select_cases(
    requests: list[dict[str, Any]],
    deferrals: list[dict[str, Any]],
    config: dict[str, Any],
    used: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    generator = random.Random(int(config["sampling_seed"]))
    selected_requests: list[dict[str, Any]] = []
    selected_abstentions: list[dict[str, Any]] = []
    for domain in config["domains"]:
        for action, count in config["generated_per_domain"][domain].items():
            pool = [
                row
                for row in requests
                if row["partition"] == config["eligible_partition"]
                and row["domain"] == domain
                and row["required_action"] == action
                and row["example_id"] not in used
            ]
            generator.shuffle(pool)
            if len(pool) < int(count):
                raise RuntimeError(
                    f"Cas frais insuffisants: domain={domain}, action={action}, "
                    f"disponibles={len(pool)}, requis={count}."
                )
            selected_requests.extend(pool[: int(count)])
        count = int(config["abstain_per_domain"][domain])
        pool = [
            row
            for row in deferrals
            if row["partition"] == config["eligible_partition"]
            and row["domain"] == domain
            and row["runtime_action"] == "abstain_without_generation"
            and row["example_id"] not in used
        ]
        generator.shuffle(pool)
        if len(pool) < count:
            raise RuntimeError(
                f"Abstentions fraîches insuffisantes: domain={domain}, "
                f"disponibles={len(pool)}, requises={count}."
            )
        selected_abstentions.extend(pool[:count])
    generator.shuffle(selected_requests)
    generator.shuffle(selected_abstentions)
    return selected_requests, selected_abstentions


def expected_action(example: dict[str, Any]) -> str:
    target = example["target"]
    if "no relevant information is found" in target["reference_answer"].casefold():
        return "abstain"
    return "ask_followup" if target["dialogue_act"] == "query_condition" else "answer"


def build_span_index() -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for document in iter_jsonl(DATA_ROOT / "documents.jsonl"):
        for section in document["sections"]:
            for span in section["spans"]:
                index[span["id"]] = {
                    "span_id": span["id"],
                    "document_title": document["title"],
                    "section_title": section.get("title", ""),
                    "text": span["text"].strip(),
                }
    return index


def extract_output_text(response_data: dict[str, Any]) -> str:
    if response_data.get("output_text"):
        return str(response_data["output_text"])
    texts: list[str] = []
    for item in response_data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                texts.append(content["text"])
    return "".join(texts)


def validate_generation(
    prepared: dict[str, Any], response_data: dict[str, Any]
) -> tuple[dict[str, Any] | None, bool, bool, dict[str, int]]:
    output_text = extract_output_text(response_data)
    try:
        parsed = json.loads(output_text) if output_text else None
    except json.JSONDecodeError:
        parsed = None
    schema = prepared["request_body"]["text"]["format"]["schema"]
    contract_valid = bool(
        parsed is not None and not list(Draft202012Validator(schema).iter_errors(parsed))
    )
    evidence_by_id = {item["evidence_id"]: item["text"] for item in prepared["evidence_catalog"]}
    citations_valid = False
    sanitization = {"kept": 0, "dropped": 0}
    if contract_valid and parsed is not None:
        valid_citations = []
        for citation in parsed["citations"]:
            evidence_id = citation["evidence_id"]
            quote = normalize(citation["evidence_quote"])
            is_valid = bool(
                evidence_id in evidence_by_id
                and quote
                and quote in normalize(evidence_by_id[evidence_id])
            )
            if is_valid:
                valid_citations.append(citation)
            else:
                sanitization["dropped"] += 1
        sanitization["kept"] = len(valid_citations)
        parsed["citations"] = valid_citations
        citations_valid = bool(valid_citations)
        # Validate the public, sanitized payload rather than exposing a bad citation.
        contract_valid = not list(Draft202012Validator(schema).iter_errors(parsed))
    return parsed, contract_valid, citations_valid, sanitization


def judge_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action_correct": {"type": "boolean"},
            "correctness_0_to_4": {"type": "integer", "minimum": 0, "maximum": 4},
            "faithfulness_pass": {"type": "boolean"},
            "completeness_pass": {"type": "boolean"},
            "short_reason": {"type": "string", "maxLength": 500},
        },
        "required": [
            "action_correct",
            "correctness_0_to_4",
            "faithfulness_pass",
            "completeness_pass",
            "short_reason",
        ],
        "additionalProperties": False,
    }


def render_history(example: dict[str, Any]) -> str:
    return "\n".join(
        f"{turn['role'].upper()}: {turn['text']}" for turn in example.get("history", [])[-8:]
    ) or "(no previous turns)"


def render_judge_input(
    example: dict[str, Any], prepared: dict[str, Any], parsed: dict[str, Any]
) -> str:
    evidence_by_id = {item["evidence_id"]: item for item in prepared["evidence_catalog"]}
    cited_blocks = []
    for citation in parsed.get("citations", []):
        item = evidence_by_id.get(citation.get("evidence_id"))
        if item:
            cited_blocks.append(f"[{item['evidence_id']}] {item['text']}")
    span_index = build_span_index_cached()
    gold_blocks = [
        f"[{span_id}] {span_index[span_id]['text']}"
        for span_id in example["gold"]["span_ids"]
        if span_id in span_index
    ]
    return "\n".join(
        [
            f"EXPECTED_ACTION: {expected_action(example)}",
            f"QUESTION: {example['query']['text']}",
            "CONVERSATION_HISTORY:",
            render_history(example),
            f"REFERENCE_ANSWER: {example['target']['reference_answer']}",
            "GOLD_EVIDENCE:",
            "\n".join(gold_blocks) or "(none)",
            "CANDIDATE_OUTPUT:",
            json.dumps(parsed, ensure_ascii=False),
            "CANDIDATE_CITED_EVIDENCE:",
            "\n".join(cited_blocks) or "(none)",
        ]
    )


_SPAN_INDEX: dict[str, dict[str, str]] | None = None


def build_span_index_cached() -> dict[str, dict[str, str]]:
    global _SPAN_INDEX
    if _SPAN_INDEX is None:
        _SPAN_INDEX = build_span_index()
    return _SPAN_INDEX


def usage_cost(usage: dict[str, Any], pricing: dict[str, float]) -> float:
    input_tokens = int(usage.get("input_tokens", 0))
    cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    return (
        (input_tokens - cached) * pricing["input_usd_per_million_tokens"]
        + cached * pricing["cached_input_usd_per_million_tokens"]
        + output_tokens * pricing["output_usd_per_million_tokens"]
    ) / 1_000_000


def wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.96
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def main() -> None:
    args = parse_args()
    config = load_json(CONFIG_PATH)
    if config.get("locked_splits") != ["test"]:
        raise ValueError("Le split test doit rester verrouille.")
    contract_id = config["generation_contract_experiment_id"]
    contract_root = DATA_ROOT / "planned_generation_contract_experiments" / contract_id
    requests = list(iter_jsonl(contract_root / "requests" / "validation_requests.jsonl"))
    deferrals = list(iter_jsonl(contract_root / "deferrals" / "validation_deferrals.jsonl"))
    used = collect_previously_used_ids()
    selected_requests, selected_abstentions = select_cases(requests, deferrals, config, used)
    selected_ids = {row["example_id"] for row in [*selected_requests, *selected_abstentions]}
    benchmark = {
        row["id"]: row
        for row in iter_jsonl(
            DATA_ROOT
            / "retrieval_benchmarks"
            / config["benchmark_experiment_id"]
            / f"{config['evaluation_split']}.jsonl"
        )
        if row["id"] in selected_ids
    }
    if set(benchmark) != selected_ids:
        raise RuntimeError("Des exemples sélectionnés sont absents du benchmark.")

    encoding = tiktoken.get_encoding("cl100k_base")
    judge = config["judge"]
    pricing = config["pricing"]
    estimated_generation_input = sum(row["input_tokens"] for row in selected_requests)
    estimated_generation_output = sum(
        int(row["request_body"]["max_output_tokens"]) for row in selected_requests
    )
    estimated_judge_input = 0
    for row in selected_requests:
        example = benchmark[row["example_id"]]
        placeholder = {
            "status": row["expected_status"],
            "answer": "x" * 800,
            "citations": [{"evidence_id": "E1", "evidence_quote": "x" * 250}],
        }
        estimated_judge_input += len(
            encoding.encode(JUDGE_INSTRUCTIONS + render_judge_input(example, row, placeholder))
        )
    estimated_judge_output = len(selected_requests) * int(judge["max_output_tokens"])
    estimated_maximum = (
        (estimated_generation_input + estimated_judge_input)
        * pricing["input_usd_per_million_tokens"]
        + (estimated_generation_output + estimated_judge_output)
        * pricing["output_usd_per_million_tokens"]
    ) / 1_000_000
    ceiling = float(pricing["maximum_authorized_cost_usd"])
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print(f"Cas totaux: {len(selected_ids)}")
    print(f"Générations OpenAI: {len(selected_requests)}")
    print(f"Abstentions déterministes: {len(selected_abstentions)}")
    print(f"Coût maximal estimé: ${estimated_maximum:.6f}")
    print(f"Plafond autorisé: ${ceiling:.2f}")
    if estimated_maximum > ceiling + 1e-12:
        raise RuntimeError("Le coût maximal estimé dépasse le plafond configuré.")
    if not args.execute:
        print("Dry-run terminé: aucun appel OpenAI.")
        return
    if args.confirm_maximum_cost_usd + 1e-12 < ceiling:
        raise RuntimeError(f"Confirmation requise: --confirm-maximum-cost-usd {ceiling:.2f}")

    load_dotenv(ROOT / config["api"]["dotenv_path"])
    api_key = os.getenv(config["api"]["api_key_environment_variable"])
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY est absente du fichier .env.")
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        timeout=float(config["api"]["request_timeout_seconds"]),
        max_retries=int(config["api"]["max_retries"]),
    )
    cache_path = DATA_ROOT / "evaluation_cache" / config["api"]["cache_sqlite_filename"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(cache_path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS responses (cache_key TEXT PRIMARY KEY, kind TEXT NOT NULL, response_json TEXT NOT NULL, created_at_utc TEXT NOT NULL)"
    )
    connection.commit()

    def cached_call(kind: str, body: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        key = stable_hash({"kind": kind, "body": body})
        cached = connection.execute(
            "SELECT response_json FROM responses WHERE cache_key=?", (key,)
        ).fetchone()
        if cached:
            return json.loads(cached[0]), True
        response = client.responses.create(**body)
        value = response.model_dump(mode="json")
        connection.execute(
            "INSERT OR REPLACE INTO responses VALUES (?,?,?,?)",
            (key, kind, json.dumps(value, ensure_ascii=False), datetime.now(timezone.utc).isoformat()),
        )
        connection.commit()
        return value, False

    results: list[dict[str, Any]] = []
    new_generation_calls = 0
    new_judge_calls = 0
    total_cost = 0.0
    for index, prepared in enumerate(selected_requests, start=1):
        example = benchmark[prepared["example_id"]]
        response_data, generation_cached = cached_call("generation", prepared["request_body"])
        new_generation_calls += int(not generation_cached)
        if not generation_cached:
            total_cost += usage_cost(response_data.get("usage") or {}, pricing)
        parsed, contract_valid, citations_valid, citation_sanitization = validate_generation(prepared, response_data)
        judge_output = None
        judge_contract_valid = False
        judge_cached = False
        if parsed is not None and contract_valid:
            judge_body = {
                "model": judge["model"],
                "instructions": JUDGE_INSTRUCTIONS,
                "input": render_judge_input(example, prepared, parsed),
                "reasoning": {"effort": judge["reasoning_effort"]},
                "text": {
                    "verbosity": judge["text_verbosity"],
                    "format": {
                        "type": "json_schema",
                        "name": "rag_automatic_judgment_v1",
                        "strict": True,
                        "schema": judge_schema(),
                    },
                },
                "max_output_tokens": judge["max_output_tokens"],
                "store": judge["store"],
            }
            judge_data, judge_cached = cached_call("judge", judge_body)
            new_judge_calls += int(not judge_cached)
            if not judge_cached:
                total_cost += usage_cost(judge_data.get("usage") or {}, pricing)
            judge_text = extract_output_text(judge_data)
            try:
                judge_output = json.loads(judge_text) if judge_text else None
            except json.JSONDecodeError:
                judge_output = None
            judge_contract_valid = bool(
                judge_output is not None
                and not list(Draft202012Validator(judge_schema()).iter_errors(judge_output))
            )
        deterministic_action_correct = prepared["required_action"] == expected_action(example)
        correctness = int(judge_output["correctness_0_to_4"]) if judge_contract_valid else 0
        faithfulness = bool(judge_output["faithfulness_pass"]) if judge_contract_valid else False
        completeness = bool(judge_output["completeness_pass"]) if judge_contract_valid else False
        action_correct = bool(judge_output["action_correct"]) if judge_contract_valid else False
        overall_pass = bool(
            contract_valid
            and citations_valid
            and judge_contract_valid
            and deterministic_action_correct
            and action_correct
            and correctness >= int(config["minimum_acceptable_correctness"])
            and faithfulness
            and completeness
        )
        results.append(
            {
                "example_id": prepared["example_id"],
                "domain": prepared["domain"],
                "case_type": "generated",
                "expected_action": expected_action(example),
                "selected_action": prepared["required_action"],
                "deterministic_action_correct": deterministic_action_correct,
                "generation_cache_hit": generation_cached,
                "judge_cache_hit": judge_cached,
                "contract_valid": contract_valid,
                "citations_valid": citations_valid,
                "citation_sanitization": citation_sanitization,
                "parsed_output": parsed,
                "judge_contract_valid": judge_contract_valid,
                "judge": judge_output,
                "correctness_0_to_4": correctness,
                "faithfulness_pass": faithfulness,
                "completeness_pass": completeness,
                "action_correct": action_correct and deterministic_action_correct,
                "overall_pass": overall_pass,
            }
        )
        print(f"[{index:02d}/{len(selected_requests)}] {prepared['example_id']}: {'PASS' if overall_pass else 'FAIL'}")

    for deferral in selected_abstentions:
        example = benchmark[deferral["example_id"]]
        action_correct = expected_action(example) == "abstain"
        results.append(
            {
                "example_id": deferral["example_id"],
                "domain": deferral["domain"],
                "case_type": "deterministic_abstention",
                "expected_action": expected_action(example),
                "selected_action": "abstain",
                "deterministic_action_correct": action_correct,
                "contract_valid": True,
                "citations_valid": True,
                "judge_contract_valid": True,
                "correctness_0_to_4": 4 if action_correct else 0,
                "faithfulness_pass": True,
                "completeness_pass": action_correct,
                "action_correct": action_correct,
                "overall_pass": action_correct,
                "parsed_output": None,
                "judge": None,
            }
        )

    connection.close()
    generated = [row for row in results if row["case_type"] == "generated"]
    total = len(results)
    rate = lambda values: float(np.mean(values)) if values else 0.0
    action_accuracy = rate([row["action_correct"] for row in results])
    overall_pass_rate = rate([row["overall_pass"] for row in results])
    contract_rate = rate([row["contract_valid"] for row in generated])
    citation_rate = rate([row["citations_valid"] for row in generated])
    faithfulness_rate = rate([row["faithfulness_pass"] for row in generated])
    correctness_rate = rate(
        [
            row["correctness_0_to_4"] >= int(config["minimum_acceptable_correctness"])
            for row in generated
        ]
    )
    completeness_rate = rate([row["completeness_pass"] for row in generated])
    hallucination_rate = 1.0 - faithfulness_rate
    metrics = {
        "examples": total,
        "generated_examples": len(generated),
        "deterministic_abstentions": total - len(generated),
        "action_accuracy": action_accuracy,
        "overall_pass_rate": overall_pass_rate,
        "contract_valid_rate": contract_rate,
        "citation_valid_rate": citation_rate,
        "faithfulness_pass_rate": faithfulness_rate,
        "correctness_acceptable_rate": correctness_rate,
        "correctness_mean_0_to_4": rate([row["correctness_0_to_4"] for row in generated]),
        "completeness_pass_rate": completeness_rate,
        "hallucination_rate": hallucination_rate,
        "wilson_95": {
            "action_accuracy": wilson(sum(row["action_correct"] for row in results), total),
            "overall_pass_rate": wilson(sum(row["overall_pass"] for row in results), total),
            "faithfulness_pass_rate": wilson(
                sum(row["faithfulness_pass"] for row in generated), len(generated)
            ),
        },
    }
    gates_config = config["quality_gates"]
    gates = {
        "action_accuracy": action_accuracy >= gates_config["minimum_action_accuracy"],
        "overall_pass_rate": overall_pass_rate >= gates_config["minimum_overall_pass_rate"],
        "contract_valid_rate": contract_rate >= gates_config["minimum_contract_valid_rate"],
        "citation_valid_rate": citation_rate >= gates_config["minimum_citation_valid_rate"],
        "faithfulness_pass_rate": faithfulness_rate >= gates_config["minimum_faithfulness_pass_rate"],
        "correctness_acceptable_rate": correctness_rate >= gates_config["minimum_correctness_acceptable_rate"],
        "completeness_pass_rate": completeness_rate >= gates_config["minimum_completeness_pass_rate"],
        "hallucination_rate": hallucination_rate <= gates_config["maximum_hallucination_rate"],
        "cost_ceiling": total_cost <= ceiling,
        "test_split_locked": True,
    }
    implementation_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    experiment_id = f"{config['experiment_name']}_{stable_hash({'config': config, 'ids': sorted(selected_ids), 'implementation': implementation_hash})[:10]}"
    report_dir = REPORT_ROOT / "automated_end_to_end_evaluation_experiments" / experiment_id
    artifact_dir = DATA_ROOT / "automated_end_to_end_evaluation_experiments" / experiment_id
    write_jsonl(artifact_dir / "case_results.jsonl", results)
    report = {
        "schema_version": "1.0.0",
        "pipeline_stage": "automated_end_to_end_evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "implementation_sha256": implementation_hash,
        "generation_contract_experiment_id": contract_id,
        "data_protocol": {
            "split": config["evaluation_split"],
            "partition": config["eligible_partition"],
            "previously_used_ids_excluded": len(used),
            "test_split_opened": False,
            "human_annotations_requested": 0,
            "automatic_model_judge": True,
        },
        "cost": {
            "estimated_maximum_usd": estimated_maximum,
            "authorized_ceiling_usd": ceiling,
            "actual_usd": total_cost,
            "new_generation_calls": new_generation_calls,
            "new_judge_calls": new_judge_calls,
        },
        "metrics": metrics,
        "metrics_by_domain": {
            domain: {
                "examples": len(domain_rows := [row for row in results if row["domain"] == domain]),
                "action_accuracy": rate([row["action_correct"] for row in domain_rows]),
                "overall_pass_rate": rate([row["overall_pass"] for row in domain_rows]),
            }
            for domain in config["domains"]
        },
        "gate_results": gates,
        "gates_passed": all(gates.values()),
        "production_recommendation": (
            "ready_for_one_time_locked_test_evaluation"
            if all(gates.values())
            else "not_ready_keep_test_locked"
        ),
        "cases": results,
        "limitations": [
            "The quality judge is automatic, not a fresh human evaluation.",
            "This is a validation-holdout benchmark; the test split remains unopened.",
        ],
    }
    report_path = report_dir / "automated_end_to_end_evaluation_report.json"
    write_json(report_path, report)
    markdown = "\n".join(
        [
            "# Évaluation end-to-end automatique",
            "",
            f"- Expérience : `{experiment_id}`",
            f"- Cas : **{total}** ({len(generated)} générés, {total-len(generated)} abstentions)",
            f"- Action accuracy : **{action_accuracy:.2%}**",
            f"- Overall pass rate : **{overall_pass_rate:.2%}**",
            f"- Faithfulness : **{faithfulness_rate:.2%}**",
            f"- Citations valides : **{citation_rate:.2%}**",
            f"- Correctness acceptable : **{correctness_rate:.2%}**",
            f"- Complétude : **{completeness_rate:.2%}**",
            f"- Hallucinations : **{hallucination_rate:.2%}**",
            f"- Coût réel : **${total_cost:.6f}**",
            f"- Gates : **{'PASS' if all(gates.values()) else 'FAIL'}**",
            "- Split test ouvert : **non**",
        ]
    )
    (report_dir / "automated_end_to_end_evaluation_report.md").write_text(markdown, encoding="utf-8")
    latest_path = REPORT_ROOT / "automated_end_to_end_evaluation" / "latest.json"
    write_json(
        latest_path,
        {
            "experiment_id": experiment_id,
            "report_path": str(report_path.relative_to(ROOT)),
            "artifact_path": str((artifact_dir / "case_results.jsonl").relative_to(ROOT)),
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Coût réel: ${total_cost:.6f}")
    print(f"Gates passed: {all(gates.values())}")
    print(f"Report: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
