from rag_system.generation import REFUSAL_MESSAGE
from rag_system.pipeline import QueryEngine
from rag_system.reranker import PassthroughReranker
from rag_system.schemas import DocumentChunk, RetrievedChunk


class FakeStore:
    def __init__(self, chunks):
        self.chunks = chunks

    def similarity_search(self, query, k):
        return [RetrievedChunk(**chunk.model_dump(), score=1.0) for chunk in self.chunks[:k]]

    def all_chunks(self):
        return self.chunks


class CitedGenerator:
    def generate(self, question, chunks):
        return f"Refunds are available for 30 days [{chunks[0].chunk_id}]."


class UnsupportedGenerator:
    def generate(self, question, chunks):
        return "Refunds are available for 30 days."


def test_query_engine_returns_citations_for_supported_answer():
    chunk = DocumentChunk(
        chunk_id="chunk1",
        text="Refunds are available for 30 days.",
        source="policy.md",
        chunk_index=0,
    )
    engine = QueryEngine(FakeStore([chunk]), PassthroughReranker(), CitedGenerator(), top_k=1)

    response = engine.query("What is the refund window?")

    assert response.supported is True
    assert response.refused is False
    assert response.citations[0].chunk_id == "chunk1"


def test_query_engine_refuses_uncited_answer():
    chunk = DocumentChunk(
        chunk_id="chunk1",
        text="Refunds are available for 30 days.",
        source="policy.md",
        chunk_index=0,
    )
    engine = QueryEngine(FakeStore([chunk]), PassthroughReranker(), UnsupportedGenerator(), top_k=1)

    response = engine.query("What is the refund window?")

    assert response.refused is True
    assert response.answer == REFUSAL_MESSAGE
