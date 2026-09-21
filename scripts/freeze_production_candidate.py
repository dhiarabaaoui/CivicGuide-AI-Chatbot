"""Freeze the exact runtime candidate before any confirmation or production evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "processed" / "multidoc2dial_v1"
CONFIG_PATH = ROOT / "configs" / "mandatory_claim_pipeline.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(payload.encode("utf-8"))


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("locked_splits") != ["test"]:
        raise ValueError("The benchmark test split must remain locked.")

    relative_paths = [
        "configs/mandatory_claim_pipeline.json",
        config["planner_prompt_path"],
        config["reviewer_prompt_path"],
        config["realizer_prompt_path"],
        "scripts/evaluate_mandatory_claim_pipeline.py",
        "scripts/build_planned_generation_contract.py",
        "scripts/conversation_state.py",
        "scripts/conversation_control.py",
        "scripts/claim_grounding.py",
        "scripts/evidence_anchor.py",
        "scripts/plan_contract.py",
    ]
    artifacts: dict[str, dict[str, Any]] = {}
    for relative_path in relative_paths:
        path = ROOT / relative_path
        content = path.read_bytes()
        artifacts[relative_path] = {
            "sha256": sha256_bytes(content),
            "bytes": len(content),
        }

    identity = {
        "schema_version": "1.0.0",
        "generation_contract_experiment_id": config["generation_contract_experiment_id"],
        "benchmark_experiment_id": config["benchmark_experiment_id"],
        "structural_evidence_guard": config.get("structural_evidence_guard"),
        "artifacts": artifacts,
    }
    candidate_id = f"rag_candidate_structural_v1_{stable_hash(identity)[:12]}"
    output_root = DATA_ROOT / "candidate_freezes" / candidate_id

    for relative_path in relative_paths:
        source = ROOT / relative_path
        snapshot = output_root / "snapshot" / relative_path
        write_text_atomic(snapshot, source.read_text(encoding="utf-8"))

    manifest = {
        **identity,
        "candidate_id": candidate_id,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "targeted_confirmation_then_external_clean_validation",
        "test_split_opened": False,
        "reference_answers_in_runtime": False,
        "gold_evidence_in_runtime": False,
        "runtime_feature_flags_enabled": False,
        "mutation_policy": "Any artifact hash change requires a new candidate_id.",
    }
    manifest_path = output_root / "candidate_manifest.json"
    write_text_atomic(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    print(f"Candidate: {candidate_id}")
    print(f"Manifest: {manifest_path}")
    print(f"Frozen artifacts: {len(artifacts)}")
    print("Test split opened: false")


if __name__ == "__main__":
    main()
