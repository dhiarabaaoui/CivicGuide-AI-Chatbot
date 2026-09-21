from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_runtime.settings import load_settings


class VercelDeploymentTests(unittest.TestCase):
    def test_vercel_configuration_uses_packaged_runtime_artifacts(self) -> None:
        with patch.dict(os.environ, {"VERCEL": "1"}):
            settings = load_settings()

        self.assertIn("runtime_artifacts", settings.path("chunks").parts)
        self.assertTrue(str(settings.path("dense_vectors")).endswith("chunk_vectors.npy"))

    def test_vercel_writes_only_to_temporary_storage(self) -> None:
        with patch.dict(os.environ, {"VERCEL": "1"}):
            settings = load_settings()
            cache = settings.path("response_cache")
            request_log = settings.path("request_log")

        temporary_root = Path(tempfile.gettempdir())
        self.assertTrue(cache.is_relative_to(temporary_root))
        self.assertTrue(request_log.is_relative_to(temporary_root))
        self.assertNotIn("reports", request_log.parts)


if __name__ == "__main__":
    unittest.main()
