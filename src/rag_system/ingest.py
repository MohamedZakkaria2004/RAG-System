from __future__ import annotations

from rag_system.chunking import TokenChunker
from rag_system.loaders import load_path, load_url
from rag_system.schemas import IngestStats, RawDocument


class IngestionService:
    def __init__(self, vector_store, chunker: TokenChunker) -> None:
        self.vector_store = vector_store
        self.chunker = chunker

    def ingest_path(self, path: str, recursive: bool = True) -> IngestStats:
        return self._ingest_documents(load_path(path, recursive=recursive))

    def ingest_url(self, url: str) -> IngestStats:
        return self._ingest_documents(load_url(url))

    def _ingest_documents(self, documents: list[RawDocument]) -> IngestStats:
        chunks = self.chunker.chunk_documents(documents)
        indexed = self.vector_store.upsert_chunks(chunks)
        return IngestStats(
            documents_loaded=len(documents),
            chunks_indexed=indexed,
            sources=sorted({document.source for document in documents}),
        )
