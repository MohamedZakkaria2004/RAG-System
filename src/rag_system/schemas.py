from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    text: str
    source: str
    title: str | None = None
    page: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    title: str | None = None
    page: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(DocumentChunk):
    score: float = 0.0
    vector_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None


class Citation(BaseModel):
    chunk_id: str
    source: str
    page: int | None = None
    title: str | None = None
    excerpt: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    supported: bool
    refused: bool


class IngestStats(BaseModel):
    documents_loaded: int
    chunks_indexed: int
    sources: list[str]


class EvaluationResult(BaseModel):
    samples: int
    metrics: dict[str, float]
    passed: bool
    threshold: float
