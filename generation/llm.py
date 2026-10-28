"""LLM construction for the configured provider."""

from __future__ import annotations

from config import Settings


def create_llm(settings: Settings):
    """Create the configured chat model."""

    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
        )

    if settings.llm_provider == "vllm":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value()
            if settings.llm_api_key
            else "not-needed",
            temperature=settings.llm_temperature,
        )

    if settings.llm_provider == "groq":
        if settings.llm_api_key is None:
            raise ValueError("LLM_API_KEY is required when LLM_PROVIDER=groq")
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.llm_model,
            api_key=settings.llm_api_key.get_secret_value(),
            temperature=settings.llm_temperature,
        )

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
