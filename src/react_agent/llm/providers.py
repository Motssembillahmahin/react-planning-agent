"""Pick an LLM implementation by name. Keys are read from environment variables only."""

from __future__ import annotations

import os

from react_agent.llm.anthropic_llm import AnthropicLLM
from react_agent.llm.base import LLM, LLMError
from react_agent.llm.openai_llm import OpenAICompatibleLLM


def available_provider() -> str | None:
    """First provider whose key is set, or None."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
        return "openai"
    return None


def make_llm(provider: str, model: str | None = None, base_url: str | None = None) -> LLM:
    if provider == "auto":
        found = available_provider()
        if found is None:
            raise LLMError("no provider key set (ANTHROPIC_API_KEY / OPENAI_API_KEY)")
        provider = found
    if provider == "anthropic":
        return AnthropicLLM(model) if model else AnthropicLLM()
    if provider in ("openai", "ollama"):
        if provider == "ollama":
            base_url = base_url or "http://localhost:11434/v1"
            model = model or "llama3.1"
        return OpenAICompatibleLLM(model or "gpt-4o-mini", base_url=base_url)
    raise LLMError(f"unknown provider '{provider}'")
