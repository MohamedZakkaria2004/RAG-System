from rag_system.chunking import TokenChunker
from rag_system.schemas import RawDocument


def test_chunker_creates_overlapping_chunks():
    text = " ".join(f"token{i}" for i in range(200))
    chunker = TokenChunker(chunk_size=40, overlap=10)
    chunks = chunker.chunk_document(RawDocument(text=text, source="fixture.txt"))

    assert len(chunks) > 1
    assert all(chunk.source == "fixture.txt" for chunk in chunks)
    assert all(chunk.chunk_id for chunk in chunks)
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunker.token_count(chunks[0].text) <= 40
