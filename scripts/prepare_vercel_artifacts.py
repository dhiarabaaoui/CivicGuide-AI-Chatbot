"""Copy only the frozen runtime artifacts required by the Vercel deployment."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = ROOT / "configs" / "runtime.json"
TARGET_DIRECTORY = ROOT / "runtime_artifacts"
MAXIMUM_TOTAL_BYTES = 100 * 1024 * 1024

ARTIFACT_NAMES = {
    "chunks": "chunks.jsonl",
    "documents": "documents.jsonl",
    "dense_vectors": "chunk_vectors.npy",
    "dense_index": "chunk_index.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    TARGET_DIRECTORY.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "candidate_id": config["candidate_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {},
    }
    total_bytes = 0
    for key, target_name in ARTIFACT_NAMES.items():
        source_relative = Path(config["artifacts"][key])
        source = ROOT / source_relative
        if not source.is_file():
            raise FileNotFoundError(f"Missing frozen runtime artifact: {source}")
        target = TARGET_DIRECTORY / target_name
        shutil.copy2(source, target)
        size = target.stat().st_size
        total_bytes += size
        manifest["artifacts"][key] = {
            "source": source_relative.as_posix(),
            "target": target.relative_to(ROOT).as_posix(),
            "size_bytes": size,
            "sha256": sha256(target),
        }
    if total_bytes > MAXIMUM_TOTAL_BYTES:
        raise RuntimeError(
            f"Runtime artifacts use {total_bytes / 1024 / 1024:.2f} MB; "
            "the deployment guard is 100 MB."
        )
    manifest["total_bytes"] = total_bytes
    manifest["total_megabytes"] = round(total_bytes / 1024 / 1024, 3)
    (TARGET_DIRECTORY / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(ARTIFACT_NAMES)} artifacts ({manifest['total_megabytes']} MB).")


if __name__ == "__main__":
    main()
