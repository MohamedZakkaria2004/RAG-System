from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RagSettings(BaseSettings):
    provider: str = Field(default="openai")
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "RAG_OPENAI_API_KEY"),
    )
    google_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    )
    config_path: str | None = Field(
        default="config/default.yaml",
        validation_alias=AliasChoices("RAG_CONFIG_PATH", "CONFIG_PATH"),
    )
    embedding_model: str = Field(default="text-embedding-3-small")
    chat_model: str = Field(default="gpt-4.1-mini")
    gemini_embedding_model: str = Field(default="models/gemini-embedding-001")
    gemini_chat_model: str = Field(default="gemini-2.5-flash")
    collection_name: str = Field(default="rag_chunks")
    chroma_persist_dir: str = Field(default="data/chroma")
    chunk_size_tokens: int = Field(default=700, ge=500, le=800)
    chunk_overlap_tokens: int = Field(default=100, ge=0)
    retrieval_top_k: int = Field(default=8, ge=1)
    rerank_candidate_count: int = Field(default=25, ge=1)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    faithfulness_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    prompt_dir: str = Field(default="prompts")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


def load_settings(config_path: str | Path | None = None) -> RagSettings:
    settings = RagSettings()
    path = Path(config_path or settings.config_path or "")
    if path and path.exists():
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        settings = settings.model_copy(update=data)
    if settings.chunk_overlap_tokens >= settings.chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
    return settings
