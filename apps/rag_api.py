from __future__ import annotations

import re
import uuid
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from rag_runtime import RAGRuntime, RuntimeUnavailable
from rag_runtime.observability import emit_event
from rag_runtime.settings import load_settings


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class RetrievalRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=12)
    domain: Literal["dmv", "ssa", "va", "studentaid"] | None = None
    use_dense: bool = False

    @field_validator("message")
    @classmethod
    def non_blank_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value.strip()


class ChatRequest(RetrievalRequest):
    use_dense: bool = True


class FeedbackRequest(BaseModel):
    trace_id: str = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    rating: Literal["helpful", "not_helpful"]
    reason: Literal["inaccurate", "incomplete", "unclear", "citation_issue", "other"] | None = None


app = FastAPI(
    title="CivicGuide AI API",
    version="0.1.0",
    description="Runtime API for the frozen structural RAG candidate.",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_ROOT), name="static")


def _request_id(request: Request) -> str:
    supplied = request.headers.get("x-request-id", "")
    if re.fullmatch(r"[A-Za-z0-9_-]{8,64}", supplied):
        return supplied
    return uuid.uuid4().hex


@app.middleware("http")
async def observe_request(request: Request, call_next):
    request_id = _request_id(request)
    started = perf_counter()
    try:
        response = await call_next(request)
    except Exception as error:
        emit_event(
            "http_request",
            level="error",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=500,
            duration_ms=round((perf_counter() - started) * 1000, 2),
            exception_type=type(error).__name__,
        )
        raise
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; base-uri 'self'; frame-ancestors 'none'; "
        "form-action 'self'; img-src 'self' data:; object-src 'none'"
    )
    if request.url.path.startswith("/v1/"):
        response.headers["Cache-Control"] = "no-store"
    if request.headers.get("x-forwarded-proto") == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    emit_event(
        "http_request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round((perf_counter() - started) * 1000, 2),
    )
    return response


@lru_cache(maxsize=1)
def runtime() -> RAGRuntime:
    return RAGRuntime(load_settings())


def turns(value: list[ChatTurn]) -> list[dict[str, str]]:
    return [item.model_dump() for item in value]


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(FRONTEND_ROOT / "index.html")


@app.get("/health")
def health() -> dict:
    try:
        return runtime().health()
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/health/live")
def liveness() -> dict:
    return {"status": "alive"}


@app.get("/health/ready")
def readiness() -> dict:
    return health()


@app.post("/v1/retrieve")
def retrieve(request: RetrievalRequest) -> dict:
    try:
        return runtime().retrieve(
            request.message,
            turns(request.history),
            domain=request.domain,
            use_dense=request.use_dense,
        )
    except RuntimeUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Retrieval failed.") from error


@app.post("/v1/chat")
def chat(request: ChatRequest) -> dict:
    try:
        return runtime().chat(
            request.message,
            turns(request.history),
            domain=request.domain,
            use_dense=request.use_dense,
        )
    except RuntimeUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="The guarded generation pipeline failed.") from error


@app.post("/v1/feedback", status_code=status.HTTP_202_ACCEPTED)
def feedback(request: FeedbackRequest) -> dict:
    emit_event(
        "user_feedback",
        trace_id=request.trace_id,
        rating=request.rating,
        reason=request.reason,
    )
    return {"status": "accepted", "trace_id": request.trace_id}
