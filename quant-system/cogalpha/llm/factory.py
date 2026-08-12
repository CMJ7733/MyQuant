"""Client factory: :class:`~cogalpha.config.LLMConfig` -> :class:`LLMClient`."""

from __future__ import annotations

from typing import Optional

from cogalpha.config import LLMConfig
from cogalpha.llm.base import LLMClient, LLMError
from cogalpha.llm.mock_client import MockLLMClient
from cogalpha.llm.recorder import CallRecorder


def build_client(
    cfg: LLMConfig,
    recorder: Optional[CallRecorder] = None,
    max_calls: Optional[int] = None,
) -> LLMClient:
    """Instantiate the configured backend, resolving secrets first."""
    cfg = cfg.resolve_secrets()

    if cfg.provider == "mock":
        return MockLLMClient(
            model=f"mock:{cfg.model}",
            default_temperature=cfg.checker_temperature,
            max_tokens=cfg.max_tokens,
            seed=cfg.mock_seed,
            recorder=recorder,
            max_calls=max_calls,
            max_concurrency=cfg.max_concurrency,
        )

    if cfg.provider == "openai":
        from cogalpha.llm.openai_client import OpenAIClient

        if not cfg.api_key:
            raise LLMError(
                "no API key for provider 'openai'. Provide one of:\n"
                "  cp configs/llm.yaml.example configs/llm.yaml   # then edit, it is git-ignored\n"
                "  export COGALPHA_API_KEY=...   (and COGALPHA_API_BASE)\n"
                "  llm.key_set: /path/to/key_set.yaml\n"
                "Or run offline with --llm-provider mock."
            )
        if not cfg.api_base:
            raise LLMError(
                "no api_base for provider 'openai'; set llm.api_base "
                "(e.g. https://qianfan.baidubce.com/v2) or COGALPHA_API_BASE"
            )

        return OpenAIClient(
            model=cfg.model,
            api_key=cfg.api_key or "",
            api_base=cfg.api_base,
            default_temperature=cfg.checker_temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
            recorder=recorder,
            max_calls=max_calls,
            max_concurrency=cfg.max_concurrency,
        )

    raise LLMError(f"unknown llm.provider '{cfg.provider}'; expected 'openai' or 'mock'")
