"""
LLM infrastructure for Famou 2.0.

Provides:
- LLMClient protocol (base interface)
- OpenAIClient implementation (with retry logic)
- AsyncOpenAIClient for parallel generation
- GeminiClient for Google Gemini models
- MockLLMClient for testing
- FallbackLLMClient for automatic failover across providers
"""

from famou.infrastructure.llm.base import LLMClient, LLMResponse
from famou.infrastructure.llm.fallback_client import FallbackLLMClient
from famou.infrastructure.llm.gemini_client import GeminiClient
from famou.infrastructure.llm.mock_client import MockLLMClient
from famou.infrastructure.llm.openai_client import AsyncOpenAIClient, OpenAIClient

__all__ = [
    "LLMClient",
    "LLMResponse",
    "OpenAIClient",
    "AsyncOpenAIClient",
    "GeminiClient",
    "MockLLMClient",
    "FallbackLLMClient",
]
