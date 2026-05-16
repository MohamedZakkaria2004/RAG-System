from __future__ import annotations

from rag_system.generation import REFUSAL_MESSAGE, AnswerGenerator, extract_citation_ids
from rag_system.retrieval import hybrid_retrieve
from rag_system.schemas import Citation, QueryResponse, RetrievedChunk


class QueryEngine:
    def __init__(
        self,
        vector_store,
        reranker,
        generator: AnswerGenerator,
        top_k: int = 8,
        candidate_count: int = 25,
    ) -> None:
        self.vector_store = vector_store
        self.reranker = reranker
        self.generator = generator
        self.top_k = top_k
        self.candidate_count = candidate_count

    def query(self, question: str) -> QueryResponse:
        candidates = hybrid_retrieve(
            question,
            self.vector_store,
            top_k=self.top_k,
            candidate_count=self.candidate_count,
        )
        chunks = self.reranker.rerank(question, candidates, self.top_k)
        if not chunks:
            return self._refusal(question, chunks)

        answer = self.generator.generate(question, chunks).strip()
        citation_ids = extract_citation_ids(answer)
        available_ids = {chunk.chunk_id for chunk in chunks}
        supported = bool(citation_ids) and citation_ids.issubset(available_ids)
        if not supported or answer == REFUSAL_MESSAGE:
            return self._refusal(question, chunks)

        cited_chunks = [chunk for chunk in chunks if chunk.chunk_id in citation_ids]
        return QueryResponse(
            question=question,
            answer=answer,
            citations=[_to_citation(chunk) for chunk in cited_chunks],
            retrieved_chunks=chunks,
            supported=True,
            refused=False,
        )

    def _refusal(self, question: str, chunks: list[RetrievedChunk]) -> QueryResponse:
        return QueryResponse(
            question=question,
            answer=REFUSAL_MESSAGE,
            citations=[],
            retrieved_chunks=chunks,
            supported=False,
            refused=True,
        )


def _to_citation(chunk: RetrievedChunk) -> Citation:
    excerpt = chunk.text[:500].strip()
    return Citation(
        chunk_id=chunk.chunk_id,
        source=chunk.source,
        page=chunk.page,
        title=chunk.title,
        excerpt=excerpt,
    )
