"""Base LLM client protocol, request hooks, and response models."""

import datetime
import json
import time
import uuid
from abc import ABC, abstractmethod
from contextvars import ContextVar
from pathlib import Path
import threading
from typing import Any, Callable, Dict, List, Optional, Protocol
from langfuse import Langfuse

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """
    Standardized response from LLM generation.

    This model encapsulates all data returned from an LLM call,
    providing a uniform interface regardless of the underlying provider.

    Fields:
        text: The main generated text content
        thinking: Chain-of-thought reasoning (for models like o1)
        raw_response: Original response dict from the provider
        usage: Token usage statistics
        model: Model identifier that was used
        finish_reason: Why generation stopped (length, stop, etc.)
    """

    text: str = Field(description="Generated text content")
    thinking: Optional[str] = Field(
        default=None, description="Chain-of-thought reasoning (o1 models)"
    )
    raw_response: Dict[str, Any] = Field(
        default_factory=dict, description="Raw provider response"
    )
    usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Token usage (prompt_tokens, completion_tokens, total_tokens)",
    )
    model: str = Field(description="Model identifier used")
    provider: Optional[str] = Field(default=None, description="LLM provider identifier")
    finish_reason: Optional[str] = Field(
        default=None, description="Why generation stopped"
    )
    started_at: Optional[float] = Field(
        default=None, description="Unix timestamp when the API call started"
    )
    completed_at: Optional[float] = Field(
        default=None, description="Unix timestamp when the API call completed"
    )
    latency_ms: Optional[int] = Field(
        default=None, description="End-to-end latency in milliseconds"
    )

    model_config = {"arbitrary_types_allowed": True}

    def __repr__(self) -> str:
        """Concise representation."""
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"LLMResponse(model={self.model}, text='{text_preview}')"


class LLMClient(Protocol):
    """
    Protocol for LLM client implementations.

    All LLM clients (OpenAI, Anthropic, Mock, etc.) must implement this interface.
    This ensures consistent API across different providers.

    The interface is designed to be:
    - Simple: Single generate() method for most use cases
    - Flexible: Accepts **kwargs for provider-specific options
    - Type-safe: Returns structured LLMResponse objects
    - Observable: Includes usage statistics for cost tracking

    Example Implementation:
        >>> class MyLLMClient:
        ...     def generate(
        ...         self,
        ...         prompt: str,
        ...         system: str = "",
        ...         temperature: float = 0.7,
        ...         max_tokens: int = 2000,
        ...         **kwargs
        ...     ) -> LLMResponse:
        ...         # Call provider API
        ...         response = my_provider.call(prompt, system, ...)
        ...         return LLMResponse(
        ...             text=response.content,
        ...             model=response.model,
        ...             usage={...}
        ...         )

    Example Usage:
        >>> client: LLMClient = OpenAIClient(api_key="...")
        >>> response = client.generate(
        ...     prompt="Write a sorting function",
        ...     system="You are a Python expert",
        ...     temperature=0.7
        ... )
        >>> print(response.text)
        >>> print(f"Cost: {response.usage['total_tokens']} tokens")
    """

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate text from a prompt.

        This is the primary method for interacting with LLMs.
        All implementations must support these core parameters,
        though they may accept additional provider-specific options via **kwargs.

        Args:
            prompt: The user prompt/instruction
            system: System prompt to set context/role
            temperature: Sampling temperature (0.0 = deterministic, 2.0 = very random)
                - 0.0-0.3: Focused, deterministic (good for code generation)
                - 0.4-0.7: Balanced (general use)
                - 0.8-1.0: Creative (brainstorming)
                - 1.0+: Very random (experimental)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific options
                - top_p: Nucleus sampling parameter
                - frequency_penalty: Penalize token frequency
                - presence_penalty: Penalize token presence
                - stop: Stop sequences
                - seed: Random seed (for reproducibility if supported)
                - model: Override default model
                - etc.

        Returns:
            LLMResponse with generated text and metadata

        Raises:
            Exception: Implementation-specific errors (network, quota, etc.)

        Example:
            >>> response = client.generate(
            ...     prompt="Explain bubble sort",
            ...     system="You are a CS teacher",
            ...     temperature=0.5,
            ...     max_tokens=500
            ... )
            >>> print(response.text)
        """
        ...


class JSONLinesLLMRequestHook:
    """Append one JSON object per line to an `llm_requests.log` file."""

    def __init__(self, log_path: str):
        self.log_path = str(Path(log_path))
        self._lock = threading.Lock()

    def __call__(self, payload: Dict[str, Any]) -> None:
        """Persist one request log entry."""
        path = Path(self.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def __getstate__(self) -> Dict[str, Any]:
        """Drop thread lock during pickle."""
        return {"log_path": self.log_path}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore hook after pickle."""
        self.log_path = state["log_path"]
        self._lock = threading.Lock()


class BufferedLLMRequestHook:
    """Keep request log entries in memory so remote workers can return them."""

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def __call__(self, payload: Dict[str, Any]) -> None:
        """Append one request log entry to the in-memory buffer."""
        with self._lock:
            self._entries.append(dict(payload))

    def consume(self) -> List[Dict[str, Any]]:
        """Return buffered entries and clear the buffer atomically."""
        with self._lock:
            entries = list(self._entries)
            self._entries.clear()
        return entries

    def clear(self) -> None:
        """Drop all buffered entries."""
        with self._lock:
            self._entries.clear()

    def __getstate__(self) -> Dict[str, Any]:
        """Drop thread lock during pickle."""
        return {"entries": list(self._entries)}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore hook after pickle."""
        self._entries = list(state.get("entries", []))
        self._lock = threading.Lock()


# =============================================================================
# BaseLLMClient with Langfuse Template Method
# =============================================================================


class BaseLLMClient(ABC):
    """
    Base class for LLM clients with Langfuse tracking support.

    Uses template method pattern: generate() handles all instrumentation,
    subclasses implement _generate_impl() for provider-specific logic.

    This ensures that all LLM providers (OpenAI, Anthropic, etc.) automatically
    get Langfuse observability without having to implement tracing logic themselves.
    """

    def __init__(
        self,
        langfuse: Optional["Langfuse"] = None,  # type: ignore
        logger: Optional["Logger"] = None,  # type: ignore,
        **kwargs
    ):
        """
        Initialize base LLM client.

        Args:
            langfuse: Optional Langfuse client for observability
            logger: Optional logger for warnings
            **kwargs: Additional arguments for subclasses
        """
        self.langfuse = langfuse
        self.logger = logger
        self.provider = getattr(self, "provider", self.__class__.__name__.lower())
        self._request_hooks: List[Callable[[Dict[str, Any]], None]] = []

    def add_request_hook(self, hook: Callable[[Dict[str, Any]], None]) -> None:
        """Register a hook that receives one structured log entry per LLM attempt."""
        self._request_hooks.append(hook)

    def add_jsonl_request_hook(self, log_path: str) -> None:
        """Register a JSON-lines request hook unless the same path is already attached."""
        normalized = str(Path(log_path))
        for hook in self._request_hooks:
            if isinstance(hook, JSONLinesLLMRequestHook) and hook.log_path == normalized:
                return
        self._request_hooks.append(JSONLinesLLMRequestHook(normalized))

    def add_buffered_request_hook(self) -> None:
        """Register an in-memory request hook unless one is already attached."""
        for hook in self._request_hooks:
            if isinstance(hook, BufferedLLMRequestHook):
                return
        self._request_hooks.append(BufferedLLMRequestHook())

    def clear_buffered_request_entries(self) -> None:
        """Clear buffered request entries if an in-memory hook is attached."""
        for hook in self._request_hooks:
            if isinstance(hook, BufferedLLMRequestHook):
                hook.clear()

    def consume_buffered_request_entries(self) -> List[Dict[str, Any]]:
        """Return and clear buffered request entries from the in-memory hook."""
        for hook in self._request_hooks:
            if isinstance(hook, BufferedLLMRequestHook):
                return hook.consume()
        return []

    def _emit_request_hook(self, payload: Dict[str, Any]) -> None:
        """Best-effort request hook dispatch."""
        for hook in self._request_hooks:
            try:
                hook(payload)
            except Exception as exc:
                if self.logger:
                    self.logger.warning(
                        "Failed to write llm request hook",
                        hook_type=hook.__class__.__name__,
                        error=str(exc),
                    )

    def _new_request_id(self) -> str:
        """Create a shared request ID across retries for one logical LLM call."""
        return str(uuid.uuid4())

    def _build_request_log_entry(
        self,
        *,
        request_id: str,
        attempt: int,
        max_retries: int,
        status: str,
        duration_seconds: float,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        request_time: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        response_tokens: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build one `llm_requests.log` entry in the legacy-compatible shape."""
        entry: Dict[str, Any] = {
            "request_id": request_id,
            "model": model or getattr(self, "model", None),
            "api_base": api_base if api_base is not None else getattr(self, "api_base", None),
            "request_time": request_time or datetime.datetime.now().isoformat(),
            "attempt": attempt,
            "max_retries": max_retries,
            "status": status,
            "duration_seconds": duration_seconds,
        }
        if prompt_tokens is not None:
            entry["prompt_tokens"] = prompt_tokens
        if response_tokens is not None:
            entry["response_tokens"] = response_tokens
        if error_message is not None:
            entry["error_message"] = error_message
        return entry

    def _finalize_response_metadata(
        self,
        response: LLMResponse,
        *,
        started_at: float,
        completed_at: Optional[float] = None,
    ) -> LLMResponse:
        """Populate standardized timing/provider metadata on a response."""
        completed = completed_at if completed_at is not None else time.time()
        response.started_at = started_at
        response.completed_at = completed
        response.latency_ms = int((completed - started_at) * 1000)
        if not response.provider:
            response.provider = getattr(self, "provider", self.__class__.__name__.lower())
        return response

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Template method that handles Langfuse instrumentation using Langfuse v3+ API.

        Subclasses should NOT override this method. Instead, implement
        _generate_impl() to provide provider-specific generation logic.
        """
        metadata = {
            "temperature": temperature or getattr(self, "temperature", None) or 0.7,
            "max_tokens": max_tokens or getattr(self, "max_tokens", None) or 2000,
        }

        if self.langfuse:
            model_name = kwargs.get("model", None) or getattr(self, "model", None) or "unknown"

            with self.langfuse.start_as_current_observation(
                as_type="generation",
                name="LLM",
                input={
                    "system": system if system else None,
                    "prompt": prompt,
                },
                model=model_name,
                metadata=metadata,
            ) as gen_obs:
                start_time = time.time()

                try:
                    response = self._generate_impl(
                        prompt=prompt,
                        system=system,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs
                    )
                    response = self._finalize_response_metadata(
                        response,
                        started_at=start_time,
                    )

                    latency_ms = response.latency_ms or int((time.time() - start_time) * 1000)
                    metadata["latency_ms"] = latency_ms

                    if hasattr(response, "finish_reason") and response.finish_reason:
                        metadata["finish_reason"] = response.finish_reason
                    metadata["debug_trace_id"] = gen_obs.trace_id

                    usage_data = {
                        "input": response.usage.get("prompt_tokens", 0) if response.usage else 0,
                        "output": response.usage.get("completion_tokens", 0) if response.usage else 0,
                        "total": response.usage.get("total_tokens", 0) if response.usage else 0,
                    }

                    if self.logger and response.usage:
                        self.logger.debug(
                            f"LLM usage: {usage_data}",
                            raw_usage=response.usage,
                        )

                    gen_obs.update(
                        output={
                            "text": response.text,
                            "thinking": response.thinking if response.thinking else None,
                        },
                        usage_details=usage_data,
                        metadata=metadata,
                    )

                    return response

                except Exception as e:
                    latency_ms = int((time.time() - start_time) * 1000)
                    metadata["latency_ms"] = latency_ms
                    metadata["error"] = str(e)

                    gen_obs.update(
                        level="ERROR",
                        status_message=str(e),
                        metadata=metadata,
                    )
                    raise
        else:
            start_time = time.time()
            response = self._generate_impl(
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return self._finalize_response_metadata(
                response,
                started_at=start_time,
            )

    @abstractmethod
    def _generate_impl(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """Provider-specific generation implementation."""
        raise NotImplementedError("Subclasses must implement _generate_impl()")


def get_llm_max_tokens(llm_client: Any) -> Optional[int]:
    """Return the configured default max_tokens for an LLM client if available."""
    max_tokens = getattr(llm_client, "max_tokens", None)
    return int(max_tokens) if isinstance(max_tokens, int) and max_tokens > 0 else None


def get_llm_temperature(llm_client: Any) -> Optional[float]:
    """Return the configured default temperature for an LLM client if available."""
    temperature = getattr(llm_client, "temperature", None)
    return float(temperature) if isinstance(temperature, (int, float)) else None


def get_llm_timeout(llm_client: Any) -> Optional[int]:
    """Return the configured default timeout for an LLM client if available."""
    timeout = getattr(llm_client, "timeout", None)
    return int(timeout) if isinstance(timeout, int) and timeout > 0 else None


def get_llm_max_retries(llm_client: Any, default: int = 3) -> int:
    """Return the configured default retry budget for an LLM client."""
    max_retries = getattr(llm_client, "max_retries", None)
    if isinstance(max_retries, int) and max_retries > 0:
        return max_retries
    return default
