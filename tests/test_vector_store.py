from pathlib import Path

from rag_system.schemas import DocumentChunk
from rag_system.vector_store import ChromaVectorStore


class FakeEmbedder:
    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        lowered = text.lower()
        return [
            float("refund" in lowered),
            float("shipping" in lowered),
            float(len(text) % 10),
        ]


def test_chroma_vector_store_round_trips_chunk_metadata(tmp_path: Path):
    store = ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        collection_name="test_chunks",
        embedder=FakeEmbedder(),
    )
    chunk = DocumentChunk(
        chunk_id="chunk1",
        text="Refunds are available for 30 days.",
        source="policy.md",
        page=2,
        title="Policy",
        chunk_index=0,
        metadata={"kind": "md"},
    )

    indexed = store.upsert_chunks([chunk])
    results = store.similarity_search("refund", k=1)
    all_chunks = store.all_chunks()

    assert indexed == 1
    assert results[0].chunk_id == "chunk1"
    assert results[0].source == "policy.md"
    assert results[0].page == 2
    assert all_chunks[0].metadata["kind"] == "md"
