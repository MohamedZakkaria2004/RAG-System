from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from rag_system.schemas import DocumentChunk, RawDocument


class TokenChunker:
    def __init__(self, chunk_size: int = 700, overlap: int = 100) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._encoding = self._load_encoding()

    def chunk_documents(self, documents: Iterable[RawDocument]) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks

    def chunk_document(self, document: RawDocument) -> list[DocumentChunk]:
        tokens = self._encode(document.text)
        if not tokens:
            return []

        chunks: list[DocumentChunk] = []
        step = self.chunk_size - self.overlap
        for chunk_index, start in enumerate(range(0, len(tokens), step)):
            token_slice = tokens[start : start + self.chunk_size]
            if not token_slice:
                continue
            text = self._decode(token_slice).strip()
            if not text:
                continue
            chunk_id = self._stable_chunk_id(document, chunk_index, text)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=text,
                    source=document.source,
                    title=document.title,
                    page=document.page,
                    chunk_index=chunk_index,
                    metadata=document.metadata,
                )
            )
            if start + self.chunk_size >= len(tokens):
                break
        return chunks

    def token_count(self, text: str) -> int:
        return len(self._encode(text))

    def _stable_chunk_id(self, document: RawDocument, chunk_index: int, text: str) -> str:
        payload = f"{document.source}|{document.page}|{chunk_index}|{text}".encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def _load_encoding(self):
        try:
            import tiktoken

            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None

    def _encode(self, text: str) -> list[int] | list[str]:
        if self._encoding is not None:
            return self._encoding.encode(text)
        return re.findall(r"\S+\s*", text)

    def _decode(self, tokens: list[int] | list[str]) -> str:
        if self._encoding is not None:
            return self._encoding.decode(tokens)
        return "".join(tokens)
