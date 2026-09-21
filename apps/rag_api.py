from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from rag_runtime import RAGRuntime, RuntimeUnavailable
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


app = FastAPI(
    title="CivicGuide AI API",
    version="0.1.0",
    description="Runtime API for the frozen structural RAG candidate.",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_ROOT), name="static")


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
