from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RuntimeSettings:
    raw: dict[str, Any]

    @property
    def candidate_id(self) -> str:
        return str(self.raw["candidate_id"])

    def path(self, key: str) -> Path:
        configured = Path(str(self.raw["artifacts"][key]))
        if os.getenv("VERCEL") and key in {"response_cache", "request_log"}:
            return Path(tempfile.gettempdir()) / "rag-assistant" / configured.name
        return PROJECT_ROOT / configured

    @property
    def retrieval(self) -> dict[str, Any]:
        return self.raw["retrieval"]

    @property
    def generation(self) -> dict[str, Any]:
        return self.raw["generation"]

    @property
    def safeguards(self) -> dict[str, Any]:
        return self.raw["safeguards"]


def load_settings(path: Path | None = None) -> RuntimeSettings:
    if path is None:
        config_name = "runtime.vercel.json" if os.getenv("VERCEL") else "runtime.json"
        config_path = PROJECT_ROOT / "configs" / config_name
    else:
        config_path = path
    value = json.loads(config_path.read_text(encoding="utf-8"))
    return RuntimeSettings(value)
