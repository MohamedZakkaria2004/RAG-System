from __future__ import annotations

from rag_system.chunking import TokenChunker
from rag_system.config import RagSettings, load_settings
from rag_system.generation import GeminiAnswerGenerator, OpenAIAnswerGenerator
from rag_system.ingest import IngestionService
from rag_system.pipeline import QueryEngine
from rag_system.prompts import PromptRegistry
from rag_system.reranker import CrossEncoderReranker
from rag_system.vector_store import ChromaVectorStore, GeminiEmbedder, OpenAIEmbedder


def build_services(settings: RagSettings | None = None):
    settings = settings or load_settings()
    provider = settings.provider.lower()
    if provider == "gemini":
        embedder = GeminiEmbedder(
            model=settings.gemini_embedding_model,
            api_key=settings.google_api_key,
        )
    elif provider == "openai":
        embedder = OpenAIEmbedder(model=settings.embedding_model, api_key=settings.openai_api_key)
    else:
        raise ValueError(f"Unsupported provider: {settings.provider}")

    vector_store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.collection_name,
        embedder=embedder,
    )
    chunker = TokenChunker(
        chunk_size=settings.chunk_size_tokens,
        overlap=settings.chunk_overlap_tokens,
    )
    prompt_registry = PromptRegistry(settings.prompt_dir)
    if provider == "gemini":
        generator = GeminiAnswerGenerator(
            model=settings.gemini_chat_model,
            prompt_registry=prompt_registry,
            api_key=settings.google_api_key,
        )
    else:
        generator = OpenAIAnswerGenerator(
            model=settings.chat_model,
            prompt_registry=prompt_registry,
            api_key=settings.openai_api_key,
        )
    reranker = CrossEncoderReranker(settings.reranker_model)
    return {
        "settings": settings,
        "ingestion": IngestionService(vector_store=vector_store, chunker=chunker),
        "query_engine": QueryEngine(
            vector_store=vector_store,
            reranker=reranker,
            generator=generator,
            top_k=settings.retrieval_top_k,
            candidate_count=settings.rerank_candidate_count,
        ),
    }
