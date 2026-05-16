from __future__ import annotations

import math
import re

from rag_system.schemas import DocumentChunk, RetrievedChunk


def hybrid_retrieve(
    query: str,
    vector_store,
    top_k: int = 8,
    candidate_count: int = 25,
) -> list[RetrievedChunk]:
    vector_results = vector_store.similarity_search(query, k=candidate_count)
    all_chunks = vector_store.all_chunks()
    bm25_results = bm25_search(query, all_chunks, k=candidate_count)

    fused = reciprocal_rank_fusion([vector_results, bm25_results])
    by_id: dict[str, RetrievedChunk] = {}
    for chunk in vector_results + bm25_results:
        existing = by_id.get(chunk.chunk_id)
        if existing is None:
            by_id[chunk.chunk_id] = chunk
        else:
            existing.vector_score = existing.vector_score or chunk.vector_score
            existing.bm25_score = existing.bm25_score or chunk.bm25_score

    ranked: list[RetrievedChunk] = []
    for chunk_id, score in fused:
        chunk = by_id[chunk_id]
        chunk.score = score
        ranked.append(chunk)
    return ranked[: max(top_k, candidate_count)]


def bm25_search(query: str, chunks: list[DocumentChunk], k: int) -> list[RetrievedChunk]:
    if not chunks or k <= 0:
        return []
    tokenized_docs = [_tokenize(chunk.text) for chunk in chunks]
    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    scores = _rank_bm25_scores(tokenized_docs, tokenized_query)
    ranked = sorted(zip(chunks, scores, strict=True), key=lambda item: item[1], reverse=True)
    return [
        RetrievedChunk(**chunk.model_dump(), score=float(score), bm25_score=float(score))
        for chunk, score in ranked[:k]
        if score > 0
    ]


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _rank_bm25_scores(tokenized_docs: list[list[str]], tokenized_query: list[str]) -> list[float]:
    try:
        from rank_bm25 import BM25Okapi

        bm25_scores = [
            float(score) for score in BM25Okapi(tokenized_docs).get_scores(tokenized_query)
        ]
        lexical_scores = [_lexical_overlap_score(doc, tokenized_query) for doc in tokenized_docs]
        if all(score <= 0 for score in bm25_scores):
            return lexical_scores
        return [
            bm25 + (0.001 * lexical)
            for bm25, lexical in zip(bm25_scores, lexical_scores, strict=True)
        ]
    except Exception:
        return [_lexical_overlap_score(doc, tokenized_query) for doc in tokenized_docs]


def _lexical_overlap_score(document_tokens: list[str], query_tokens: list[str]) -> float:
    if not document_tokens:
        return 0.0
    doc_counts: dict[str, int] = {}
    for token in document_tokens:
        doc_counts[token] = doc_counts.get(token, 0) + 1
    score = 0.0
    for token in query_tokens:
        if token in doc_counts:
            score += 1.0 + math.log(doc_counts[token])
    return score


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())
