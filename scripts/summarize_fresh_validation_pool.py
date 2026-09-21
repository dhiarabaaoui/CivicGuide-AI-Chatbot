"""Summarize unused validation holdout cases without opening the locked test split."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

try:
    from scripts.evaluate_mandatory_claim_pipeline import collect_mandatory_pipeline_used_ids
    from scripts.run_automated_end_to_end_evaluation import (
        DATA_ROOT,
        collect_previously_used_ids,
        iter_jsonl,
    )
except ModuleNotFoundError:
    from evaluate_mandatory_claim_pipeline import collect_mandatory_pipeline_used_ids  # type: ignore
    from run_automated_end_to_end_evaluation import (  # type: ignore
        DATA_ROOT,
        collect_previously_used_ids,
        iter_jsonl,
    )


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ID = json.loads(
    (ROOT / "configs" / "mandatory_claim_pipeline.json").read_text(encoding="utf-8")
)["generation_contract_experiment_id"]
DOMAINS = ("dmv", "ssa", "va", "studentaid")


def main() -> None:
    used = collect_previously_used_ids() | collect_mandatory_pipeline_used_ids()
    contract_root = DATA_ROOT / "planned_generation_contract_experiments" / CONTRACT_ID
    requests = [
        row
        for row in iter_jsonl(contract_root / "requests" / "validation_requests.jsonl")
        if row["partition"] == "holdout" and row["example_id"] not in used
    ]
    deferrals = [
        row
        for row in iter_jsonl(contract_root / "deferrals" / "validation_deferrals.jsonl")
        if row["partition"] == "holdout"
        and row["runtime_action"] == "abstain_without_generation"
        and row["example_id"] not in used
    ]
    generated_counts = Counter((row["domain"], row["required_action"]) for row in requests)
    abstain_counts = Counter(row["domain"] for row in deferrals)
    print(f"Previously used validation IDs: {len(used)}")
    print(f"Fresh generated cases: {len(requests)}")
    print(f"Fresh deterministic abstentions: {len(deferrals)}")
    print("\nGenerated pool by domain/action:")
    for domain in DOMAINS:
        print(
            f"- {domain}: answer={generated_counts[(domain, 'answer')]}, "
            f"ask_followup={generated_counts[(domain, 'ask_followup')]}"
        )
    print("\nAbstention pool by domain:")
    for domain in DOMAINS:
        print(f"- {domain}: abstain={abstain_counts[domain]}")
    print("\nLocked test split opened: false")


if __name__ == "__main__":
    main()
