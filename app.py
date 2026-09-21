"""Vercel entrypoint for the FastAPI application."""

from apps.rag_api import app

__all__ = ["app"]
