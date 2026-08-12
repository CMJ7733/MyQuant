"""
Google Gemini LLM client implementation.

Provides production-ready Gemini API integration with:
- Automatic retry logic for transient failures
- Rate limit handling
- Timeout management
- Token usage tracking
- Thinking/reasoning content extraction (for thinking models)
"""

import base64
import time
from typing import Any, Dict, Optional

from famou.infrastructure.llm.base import BaseLLMClient, LLMResponse


class GeminiClient(BaseLLMClient):
    """
    Google Gemini API client with robust error handling.

    Features:
    - Automatic retry with exponential backoff
    - Rate limit handling
    - Configurable timeouts
    - Support for all Gemini models (gemini-2.0-flash, gemini-2.5-pro, etc.)
    - Token usage tracking
    - Thinking content extraction for thinking-capable models

    Requires the `google-genai` package:
        pip install google-genai

    Example:
        >>> client = GeminiClient(
        ...     api_key="AIza...",
        ...     model="gemini-2.0-flash",
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
        model: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 60,
        max_retries: int = 3,
        proxy: Optional[str] = None,
        langfuse: Optional["Langfuse"] = None,  # type: ignore
        logger: Optional["Logger"] = None,  # type: ignore
        **default_kwargs,
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Google AI API key
            model: Model identifier (gemini-2.0-flash, gemini-2.5-pro, etc.)
            temperature: Default sampling temperature
            max_tokens: Default maximum output tokens
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            langfuse: Optional Langfuse client for observability (passed to BaseLLMClient)
            logger: Optional logger for warnings (passed to BaseLLMClient)
            **default_kwargs: Additional default parameters for API calls
        """
        super().__init__(langfuse=langfuse, logger=logger)

        self._api_key = api_key
        self.provider = "google"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.proxy = proxy
        self.default_kwargs = default_kwargs

        self._build_client()

    def _build_client(self) -> None:
        """Construct the underlying google-genai client."""
        from google import genai
        from google.genai import types

        client_kwargs = {"api_key": self._api_key, "vertexai": True}
        if self.proxy:
            client_kwargs["http_options"] = types.HttpOptions(
                client_args={"proxy": self.proxy},
                async_client_args={"proxy": self.proxy},
            )
        self.client = genai.Client(**client_kwargs)

    def __getstate__(self) -> Dict[str, Any]:
        """Pickle: exclude the google-genai client (may contain non-serializable objects)."""
        state = self.__dict__.copy()
        del state["client"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Unpickle: restore all fields then rebuild the Gemini client."""
        self.__dict__.update(state)
        self._build_client()

    def _generate_impl(
        self,
        prompt: str,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate text using Gemini API (provider-specific implementation).

        This method is called by the parent's generate() template method,
        which handles Langfuse instrumentation automatically.

        Automatically retries on:
        - Rate limit errors (429)
        - Server errors (500, 502, 503, 504)
        - Timeout errors

        Args:
            prompt: User prompt/instruction
            system: System prompt to set context
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens (overrides default)
            **kwargs: Additional Gemini API GenerateContentConfig parameters
                - top_p: Nucleus sampling
                - top_k: Top-k sampling
                - stop_sequences: Stop sequences
                - seed: Random seed (for reproducibility)
                - etc.

        Returns:
            LLMResponse with generated text and metadata

        Raises:
            google.genai errors after retries exhausted
        """
        from google.genai import types

        # Merge default_kwargs with call-level kwargs (call-level wins)
        merged_kwargs = {**self.default_kwargs, **kwargs}

        config = types.GenerateContentConfig(
            system_instruction=system if system else None,
            temperature=temperature if temperature is not None else self.temperature,
            max_output_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            **merged_kwargs,
        )

        response = self._call_with_retry(prompt=prompt, config=config)

        candidate = response.candidates[0]

        # Extract text and thinking content separately
        # Thinking models (e.g. gemini-2.5-pro) return parts with thought=True
        text_parts = []
        thinking_parts = []
        for part in candidate.content.parts:
            if getattr(part, "thought", False):
                thinking_parts.append(part.text or "")
            else:
                text_parts.append(part.text or "")

        text = "".join(text_parts)
        thinking = "".join(thinking_parts) if thinking_parts else None

        # Token usage
        usage_meta = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage_meta, "total_token_count", 0) or 0

        # Raw response serialization
        # Gemini SDK model_dump() may contain bytes (protobuf internals);
        # sanitize recursively so json.dumps() downstream never fails.
        try:
            raw_response = _sanitize_for_json(response.model_dump())
        except Exception:
            raw_response = {"model": self.model}

        return LLMResponse(
            text=text,
            thinking=thinking,
            raw_response=raw_response,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            model=self.model,
            provider=self.provider,
            finish_reason=str(candidate.finish_reason) if candidate.finish_reason else None,
        )

    def _call_with_retry(self, *, prompt: str, config: Any) -> Any:
        """
        Call Gemini API with automatic retry.

        Uses exponential backoff: 2s -> 4s -> 8s between retries.

        Args:
            prompt: User prompt string
            config: GenerateContentConfig instance

        Returns:
            Gemini GenerateContentResponse

        Raises:
            google.genai errors after max retries
        """
        request_id = self._new_request_id()
        max_attempts = max(1, int(self.max_retries))
        for attempt in range(1, max_attempts + 1):
            started_at = time.time()
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                usage_meta = getattr(response, "usage_metadata", None)
                prompt_tokens = getattr(usage_meta, "prompt_token_count", None) if usage_meta else None
                completion_tokens = getattr(usage_meta, "candidates_token_count", None) if usage_meta else None
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
                if _is_auth_error(exc):
                    if attempt >= max_attempts:
                        from famou.core.types import FatalRolloutError
                        raise FatalRolloutError(
                            f"LLM authentication failed (401) after {attempt} attempts"
                        ) from exc
                    time.sleep(min(max(2, 2**attempt), 10))
                    continue
                if attempt >= max_attempts:
                    raise
                time.sleep(min(max(2, 2**attempt), 10))

    def __repr__(self) -> str:
        """Concise representation."""
        return (
            f"GeminiClient(model={self.model}, "
            f"temperature={self.temperature}, "
            f"max_tokens={self.max_tokens})"
        )


def _is_auth_error(exc: Exception) -> bool:
    """Check if the exception is an authentication/authorization error (401/403).

    The google-genai SDK raises ``google.genai.errors.ClientError`` for 4xx
    responses.  We inspect the status code attribute first, then fall back to
    substring matching in the string representation.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status in (401, 403):
        return True
    msg = str(exc).lower()
    return "401" in msg or "unauthorized" in msg or "authentication" in msg


def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert bytes values to base64 strings so the result
    is always JSON-serializable.  Gemini SDK model_dump() may embed
    raw protobuf bytes inside the response dict.
    """
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("ascii")
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj
