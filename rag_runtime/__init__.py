"""Production-oriented runtime for the frozen RAG candidate."""

from .service import RAGRuntime, RuntimeUnavailable

__all__ = ["RAGRuntime", "RuntimeUnavailable"]
