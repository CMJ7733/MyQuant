"""
OpenAI LLM client implementation.

Provides production-ready OpenAI API integration with:
- Automatic retry logic for transient failures
- Rate limit handling
- Timeout management
- Token usage tracking
"""

import time
from typing import Any, Dict, Optional

from openai import AuthenticationError, OpenAI, RateLimitError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from famou.infrastructure.llm.base import BaseLLMClient, LLMResponse


class OpenAIClient(BaseLLMClient):
    """
    OpenAI API client with robust error handling.

    Features:
    - Automatic retry with exponential backoff
    - Rate limit handling
    - Configurable timeouts
    - Support for all OpenAI models (GPT-4, GPT-4o, o1, etc.)
    - Token usage tracking

    Example:
        >>> client = OpenAIClient(
        ...     api_key="sk-...",
        ...     model="gpt-4o",
        ...     temperature=0.7,
        ...     timeout=60
        ... )
        >>> response = client.generate(
        ...     prompt="Write a Python function to sort a list",
        ...     system="You are an expert Python programmer"
        ... )
        >>> print(response.text)
        >>> print(f"Tokens used: {response.usage['total_tokens']}")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 60,
        max_retries: int = 3,
        langfuse: Optional["Langfuse"] = None,  # type: ignore
        logger: Optional["Logger"] = None,  # type: ignore
        **default_kwargs,
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model identifier (gpt-4, gpt-4o, gpt-4o-mini, o1, etc.)
            api_base: Optional API base URL (for proxies or compatible APIs)
            temperature: Default sampling temperature
            max_tokens: Default maximum tokens to generate
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            langfuse: Optional Langfuse client for observability (passed to BaseLLMClient)
            logger: Optional logger for warnings (passed to BaseLLMClient)
            **default_kwargs: Additional default parameters for API calls
        """
        # Initialize parent (BaseLLMClient) with Langfuse/logger
        super().__init__(langfuse=langfuse, logger=logger)

        # Store config for pickle reconstruction
        self._api_key = api_key
        self._api_base = api_base

        # Initialize OpenAI client
        client_kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if api_base:
            client_kwargs["base_url"] = api_base

        self.client = OpenAI(**client_kwargs)
        self.provider = "openai"
        self.model = model
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_kwargs = default_kwargs

    def __getstate__(self):
        """Pickle: exclude the httpx-backed OpenAI client (contains _thread.RLock)."""
        state = self.__dict__.copy()
        del state["client"]
        return state

    def __setstate__(self, state):
        """Unpickle: restore all fields then rebuild the OpenAI client."""
        self.__dict__.update(state)
        client_kwargs: Dict[str, Any] = {"api_key": self._api_key, "timeout": self.timeout}
        if self._api_base:
            client_kwargs["base_url"] = self._api_base
        self.client = OpenAI(**client_kwargs)

    def _generate_impl(
        self,
        prompt: str,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate text using OpenAI API (provider-specific implementation).

        This method is called by the parent's generate() template method,
        which handles Langfuse instrumentation automatically.

        Automatically retries on:
        - Rate limit errors (429) with exponential backoff
        - Server errors (500, 502, 503, 504)
        - Timeout errors

        Args:
            prompt: User prompt/instruction
            system: System prompt to set context
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens (overrides default)
            **kwargs: Additional OpenAI API parameters
                - top_p: Nucleus sampling
                - frequency_penalty: Penalize frequent tokens
                - presence_penalty: Penalize present tokens
                - stop: Stop sequences
                - seed: Random seed (for reproducibility)
                - etc.

        Returns:
            LLMResponse with generated text and metadata

        Raises:
            OpenAI API errors after retries exhausted
        """
        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Merge parameters (kwargs > method args > instance defaults)
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            **self.default_kwargs,
            **kwargs,
        }

        # Call API with retry logic
        response = self._call_with_retry(**params)

        # Extract response data
        choice = response.choices[0]
        message = choice.message

        # Handle thinking/reasoning content from different models
        # - OpenAI o1: message.reasoning
        # - DeepSeek: message.reasoning_content or similar
        # - Other providers may use different field names
        thinking = None
        for field in ["reasoning", "reasoning_content", "thinking", "deepseek_reasoning"]:
            if hasattr(message, field):
                thinking = getattr(message, field)
                if thinking:
                    break

        # Also check in raw response dict for nested thinking fields
        if not thinking and isinstance(message, dict):
            for key in ["reasoning", "reasoning_content", "thinking", "deepseek_reasoning"]:
                if key in message:
                    thinking = message[key]
                    if thinking:
                        break

        return LLMResponse(
            text=message.content or "",
            thinking=thinking,
            raw_response=response.model_dump(),
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            model=response.model,
            provider=self.provider,
            finish_reason=choice.finish_reason,
        )

    def _call_with_retry(self, **params):
        """
        Call OpenAI API with automatic retry.

        Uses exponential backoff: 2s → 4s → 8s between retries.

        Args:
            **params: Parameters to pass to chat.completions.create()

        Returns:
            OpenAI ChatCompletion response

        Raises:
            OpenAI API errors after max retries
        """
        request_id = self._new_request_id()
        max_attempts = max(1, int(self.max_retries))
        for attempt in range(1, max_attempts + 1):
            started_at = time.time()
            try:
                response = self.client.chat.completions.create(**params)
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
                completion_tokens = (
                    getattr(usage, "completion_tokens", None) if usage is not None else None
                )
                self._emit_request_hook(
                    self._build_request_log_entry(
                        request_id=request_id,
                        attempt=attempt,
                        max_retries=max_attempts,
                        status="success",
                        duration_seconds=time.time() - started_at,
                        prompt_tokens=prompt_tokens,
                        response_tokens=completion_tokens,
                    )
                )
                return response
            except Exception as exc:
                error_text = str(exc)
                status = "timeout" if "timeout" in exc.__class__.__name__.lower() else "error"
                self._emit_request_hook(
                    self._build_request_log_entry(
                        request_id=request_id,
                        attempt=attempt,
                        max_retries=max_attempts,
                        status=status,
                        duration_seconds=time.time() - started_at,
                        error_message=error_text,
                    )
                )
                if isinstance(exc, AuthenticationError):
                    if attempt >= max_attempts:
                        from famou.core.types import FatalRolloutError
                        raise FatalRolloutError(
                            f"LLM authentication failed (401) after {attempt} attempts"
                        ) from exc
                    time.sleep(min(max(2, 2 ** attempt), 10))
                    continue
                if attempt >= max_attempts:
                    raise
                time.sleep(min(max(2, 2 ** attempt), 10))

    def __repr__(self) -> str:
        """Concise representation."""
        return (
            f"OpenAIClient(model={self.model}, "
            f"temperature={self.temperature}, "
            f"max_tokens={self.max_tokens})"
        )


class AsyncOpenAIClient:
    """
    Async OpenAI client for parallel generation.

    Similar to OpenAIClient but uses async/await for concurrent requests.
    Useful when generating multiple programs in parallel.

    Example:
        >>> import asyncio
        >>> client = AsyncOpenAIClient(api_key="sk-...")
        >>>
        >>> async def generate_many():
        ...     prompts = ["Write sort", "Write search", "Write reverse"]
        ...     tasks = [client.generate(p, system="Code expert") for p in prompts]
        ...     responses = await asyncio.gather(*tasks)
        ...     return responses
        >>>
        >>> responses = asyncio.run(generate_many())
        >>> len(responses)
        3
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 60,
        max_retries: int = 3,
        **default_kwargs,
    ):
        """Initialize async OpenAI client."""
        from openai import AsyncOpenAI

        client_kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if api_base:
            client_kwargs["base_url"] = api_base

        self.client = AsyncOpenAI(**client_kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_kwargs = default_kwargs

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate text asynchronously.

        Args:
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated text
        """
        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Merge parameters
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            **self.default_kwargs,
            **kwargs,
        }

        # Call API (async)
        response = await self._call_with_retry(**params)

        # Extract response
        choice = response.choices[0]
        message = choice.message

        # Handle thinking/reasoning content from different models
        # - OpenAI o1: message.reasoning
        # - DeepSeek: message.reasoning_content or similar
        # - Other providers may use different field names
        thinking = None
        for field in ["reasoning", "reasoning_content", "thinking", "deepseek_reasoning"]:
            if hasattr(message, field):
                thinking = getattr(message, field)
                if thinking:
                    break

        # Also check in raw response dict for nested thinking fields
        if not thinking and isinstance(message, dict):
            for key in ["reasoning", "reasoning_content", "thinking", "deepseek_reasoning"]:
                if key in message:
                    thinking = message[key]
                    if thinking:
                        break

        return LLMResponse(
            text=message.content or "",
            thinking=thinking,
            raw_response=response.model_dump(),
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            model=response.model,
            finish_reason=choice.finish_reason,
        )

    async def _call_with_retry(self, **params):
        """Call OpenAI API with retry (async)."""
        retryer = AsyncRetrying(
            retry=retry_if_exception_type((Exception,)) & retry_if_not_exception_type((AuthenticationError,)),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            reraise=True,
        )
        async for attempt in retryer:
            with attempt:
                return await self.client.chat.completions.create(**params)
        raise RuntimeError("Async retry loop exited unexpectedly")

    def __repr__(self) -> str:
        """Concise representation."""
        return (
            f"AsyncOpenAIClient(model={self.model}, "
            f"temperature={self.temperature}, "
            f"max_tokens={self.max_tokens})"
        )