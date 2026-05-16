from __future__ import annotations

import re
from typing import Protocol

from rag_system.prompts import PromptRegistry
from rag_system.schemas import RetrievedChunk

REFUSAL_MESSAGE = (
    "I don't have enough supported information in the indexed documents to answer that."
)


class AnswerGenerator(Protocol):
    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str: ...


class OpenAIAnswerGenerator:
    def __init__(
        self,
        model: str,
        prompt_registry: PromptRegistry,
        api_key: str | None = None,
    ) -> None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError("Install langchain-openai to use OpenAI answer generation.") from exc
        self.prompt_template = prompt_registry.load("answer_generation", "v1")
        self.llm = ChatOpenAI(model=model, api_key=api_key, temperature=0)

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        prompt = self.prompt_template.format(question=question, context=format_context(chunks))
        response = self.llm.invoke(prompt)
        return str(response.content).strip()


class GeminiAnswerGenerator:
    def __init__(
        self,
        model: str,
        prompt_registry: PromptRegistry,
        api_key: str | None = None,
    ) -> None:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError("Install langchain-google-genai to use Gemini generation.") from exc
        self.prompt_template = prompt_registry.load("answer_generation", "v1")
        self.llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        prompt = self.prompt_template.format(question=question, context=format_context(chunks))
        response = self.llm.invoke(prompt)
        return str(response.content).strip()


def format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for chunk in chunks:
        location = f"source={chunk.source}"
        if chunk.page is not None:
            location += f", page={chunk.page}"
        parts.append(f"[{chunk.chunk_id}] {location}\n{chunk.text}")
    return "\n\n".join(parts)


def extract_citation_ids(answer: str) -> set[str]:
    return set(re.findall(r"\[([A-Za-z0-9_.:/\\-]+)\]", answer))
