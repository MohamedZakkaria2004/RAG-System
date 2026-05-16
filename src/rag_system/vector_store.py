from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from rag_system.schemas import DocumentChunk, RetrievedChunk


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise ImportError("Install langchain-openai to use OpenAI embeddings.") from exc
        self._embeddings = OpenAIEmbeddings(model=model, api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)


class GeminiEmbedder:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as exc:
            raise ImportError("Install langchain-google-genai to use Gemini embeddings.") from exc
        self._embeddings = GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str,
        embedder: Embedder,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError("Install chromadb to use ChromaVectorStore.") from exc

        self.embedder = embedder
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert_chunks(self, chunks: list[DocumentChunk], batch_size: int = 64) -> int:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embeddings = self.embedder.embed_documents([chunk.text for chunk in batch])
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                embeddings=embeddings,
                documents=[chunk.text for chunk in batch],
                metadatas=[_chunk_to_metadata(chunk) for chunk in batch],
            )
        return len(chunks)

    def similarity_search(self, query: str, k: int) -> list[RetrievedChunk]:
        if k <= 0:
            return []
        query_embedding = self.embedder.embed_query(query)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        chunks: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=True,
        ):
            score = 1.0 / (1.0 + float(distance))
            chunks.append(
                _metadata_to_chunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata or {},
                    score=score,
                    vector_score=score,
                )
            )
        return chunks

    def all_chunks(self) -> list[DocumentChunk]:
        result = self.collection.get(include=["documents", "metadatas"])
        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        return [
            _metadata_to_document_chunk(chunk_id, text, metadata or {})
            for chunk_id, text, metadata in zip(ids, documents, metadatas, strict=True)
        ]


def _chunk_to_metadata(chunk: DocumentChunk) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "source": chunk.source,
        "chunk_index": chunk.chunk_index,
        "metadata_json": json.dumps(chunk.metadata, ensure_ascii=True),
    }
    if chunk.title is not None:
        metadata["title"] = chunk.title
    if chunk.page is not None:
        metadata["page"] = chunk.page
    return metadata


def _metadata_to_document_chunk(
    chunk_id: str,
    text: str,
    metadata: dict[str, Any],
) -> DocumentChunk:
    try:
        extra = json.loads(metadata.get("metadata_json", "{}"))
    except json.JSONDecodeError:
        extra = {}
    return DocumentChunk(
        chunk_id=str(metadata.get("chunk_id", chunk_id)),
        text=text,
        source=str(metadata.get("source", "")),
        title=metadata.get("title"),
        page=metadata.get("page"),
        chunk_index=int(metadata.get("chunk_index", 0)),
        metadata=extra,
    )


def _metadata_to_chunk(
    chunk_id: str,
    text: str,
    metadata: dict[str, Any],
    score: float,
    vector_score: float | None = None,
) -> RetrievedChunk:
    base = _metadata_to_document_chunk(chunk_id, text, metadata)
    return RetrievedChunk(**base.model_dump(), score=score, vector_score=vector_score)
