from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from rag_system.config import load_settings
from rag_system.evaluation import run_ragas_evaluation
from rag_system.factory import build_services
from rag_system.schemas import EvaluationResult, IngestStats, QueryResponse

app = FastAPI(title="Production RAG System", version="0.1.0")
_services = None


class IngestRequest(BaseModel):
    path: str
    recursive: bool = True


class IngestUrlRequest(BaseModel):
    url: str


class QueryRequest(BaseModel):
    question: str


class EvalRequest(BaseModel):
    dataset: str = "data/eval/golden_qa.jsonl"
    threshold: float = 0.85


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestStats)
def ingest(request: IngestRequest) -> IngestStats:
    return _get_services()["ingestion"].ingest_path(request.path, recursive=request.recursive)


@app.post("/ingest-url", response_model=IngestStats)
def ingest_url(request: IngestUrlRequest) -> IngestStats:
    return _get_services()["ingestion"].ingest_url(request.url)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    return _get_services()["query_engine"].query(request.question)


@app.post("/eval", response_model=EvaluationResult)
def evaluate(request: EvalRequest) -> EvaluationResult:
    return run_ragas_evaluation(
        dataset_path=Path(request.dataset),
        query_engine=_get_services()["query_engine"],
        threshold=request.threshold,
    )


def _get_services():
    global _services
    if _services is None:
        _services = build_services(load_settings())
    return _services
