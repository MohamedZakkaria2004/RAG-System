from rag_system.retrieval import bm25_search, hybrid_retrieve, reciprocal_rank_fusion
from rag_system.schemas import DocumentChunk, RetrievedChunk


class FakeVectorStore:
    def __init__(self, chunks):
        self.chunks = chunks

    def similarity_search(self, query, k):
        return [
            RetrievedChunk(**chunk.model_dump(), score=1.0 / (index + 1), vector_score=1.0)
            for index, chunk in enumerate(self.chunks[:k])
        ]

    def all_chunks(self):
        return self.chunks


def test_bm25_search_finds_specific_term():
    chunks = [
        DocumentChunk(chunk_id="a", text="alpha beta", source="a.md", chunk_index=0),
        DocumentChunk(
            chunk_id="b",
            text="refund window is thirty days",
            source="b.md",
            chunk_index=0,
        ),
    ]

    results = bm25_search("refund", chunks, k=2)

    assert results[0].chunk_id == "b"
    assert results[0].bm25_score is not None


def test_hybrid_retrieve_combines_vector_and_keyword_results():
    chunks = [
        DocumentChunk(chunk_id="a", text="semantic policy", source="a.md", chunk_index=0),
        DocumentChunk(chunk_id="b", text="exact serial ABC-123", source="b.md", chunk_index=0),
    ]

    results = hybrid_retrieve("ABC-123", FakeVectorStore(chunks), top_k=2, candidate_count=2)

    assert {result.chunk_id for result in results} == {"a", "b"}


def test_reciprocal_rank_fusion_prefers_consensus():
    a = RetrievedChunk(chunk_id="a", text="a", source="a", chunk_index=0)
    b = RetrievedChunk(chunk_id="b", text="b", source="b", chunk_index=0)

    fused = reciprocal_rank_fusion([[a, b], [a]])

    assert fused[0][0] == "a"
