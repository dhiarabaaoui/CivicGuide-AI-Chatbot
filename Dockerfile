FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VERCEL=1 \
    APP_ENV=container \
    LOG_LEVEL=INFO

WORKDIR /app

RUN addgroup --system civicguide \
    && adduser --system --ingroup civicguide --home /nonexistent civicguide

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip==26.2.1 \
    && python -m pip install --no-cache-dir --requirement requirements.txt

COPY app.py ./app.py
COPY apps/rag_api.py ./apps/rag_api.py
COPY rag_runtime ./rag_runtime
COPY frontend ./frontend
COPY runtime_artifacts ./runtime_artifacts
COPY configs/runtime.vercel.json configs/mandatory_claim_pipeline.json ./configs/
COPY prompts/mandatory_claim_planner_v5.txt prompts/mandatory_claim_plan_reviewer_v3.txt prompts/mandatory_claim_realizer_v1.txt ./prompts/
COPY scripts/plan_contract.py scripts/evaluate_mandatory_claim_pipeline.py scripts/run_automated_end_to_end_evaluation.py ./scripts/

RUN mkdir -p /tmp/rag-assistant \
    && chown -R civicguide:civicguide /tmp/rag-assistant

USER civicguide

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"]

CMD ["python", "-m", "uvicorn", "apps.rag_api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
