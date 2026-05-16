from __future__ import annotations

from rag_system.schemas import RetrievedChunk


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        model = self._get_model()
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = model.predict(pairs)
        rescored: list[RetrievedChunk] = []
        for chunk, score in zip(chunks, scores, strict=True):
            updated = chunk.model_copy()
            updated.rerank_score = float(score)
            updated.score = float(score)
            rescored.append(updated)
        return sorted(rescored, key=lambda chunk: chunk.rerank_score or 0.0, reverse=True)[:top_k]

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "Install sentence-transformers to use cross-encoder reranking."
                ) from exc
            self._model = CrossEncoder(self.model_name)
        return self._model


class PassthroughReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return chunks[:top_k]
