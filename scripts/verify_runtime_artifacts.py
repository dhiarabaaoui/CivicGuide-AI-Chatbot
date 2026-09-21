"""Verify the size and SHA-256 of every packaged runtime artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "runtime_artifacts" / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures: list[str] = []
    for name, metadata in manifest["artifacts"].items():
        path = ROOT / metadata["target"]
        if not path.is_file():
            failures.append(f"{name}: missing {path}")
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        if actual_size != int(metadata["size_bytes"]):
            failures.append(f"{name}: size {actual_size} != {metadata['size_bytes']}")
        if actual_hash != metadata["sha256"]:
            failures.append(f"{name}: sha256 {actual_hash} != {metadata['sha256']}")

    if failures:
        print("Runtime artifact verification failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"Verified {len(manifest['artifacts'])} runtime artifacts for {manifest['candidate_id']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
