"""Fallback LLM client that tries a list of clients in order until one succeeds."""

from typing import Any, Callable, Dict, List, Optional

from famou.core.types import FatalRolloutError
from famou.infrastructure.llm.base import LLMResponse


class FallbackLLMClient:
    """
    Wraps multiple LLM clients and tries them in order on failure.

    When the primary client raises any exception, the next client in the list
    is tried, until one succeeds or all are exhausted (re-raises last exception).

    Intentionally does NOT inherit from BaseLLMClient to avoid double
    Langfuse tracing — each underlying client already handles its own
    instrumentation.

    Example:
        >>> primary = OpenAIClient(api_key="sk-primary", model="gpt-4o")
        >>> fallback = GeminiClient(api_key="AIza-...", model="gemini-2.0-flash")
        >>> client = FallbackLLMClient(clients=[primary, fallback])
        >>> response = client.generate("Hello")

    Args:
        clients: Ordered list of LLM clients. clients[0] is the primary.
        logger: Optional logger for fallback warnings.
    """

    def __init__(self, clients: List[Any], logger: Optional[Any] = None) -> None:
        """
        Initialize the fallback client with an ordered list of LLM clients.

        Args:
            clients: Ordered list of LLM clients. clients[0] is the primary;
                     subsequent entries are tried in order on failure.
            logger: Optional structlog-style logger for fallback warnings.
                    Must support ``logger.warning(msg, **kwargs)``.

        Raises:
            ValueError: If *clients* is empty.
        """
        if not clients:
            raise ValueError("FallbackLLMClient requires at least one client")
        self._clients = clients
        self._logger = logger

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Try each client in order, returning the first successful response.

        Raises:
            Exception: The last exception raised if all clients fail.
        """
        last_exc: Exception = RuntimeError("No clients available")
        for i, client in enumerate(self._clients):
            try:
                response = client.generate(
                    prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                if i > 0 and self._logger:
                    self._logger.warning(
                        "LLM primary failed, used fallback client",
                        system_only=True,
                        fallback_index=i,
                        fallback_client=type(client).__name__,
                        fallback_model=getattr(client, "model", None),
                    )
                return response
            except FatalRolloutError:
                raise  # 401 等致命错误不走 fallback，直接终止
            except Exception as exc:
                last_exc = exc
                if self._logger:
                    self._logger.warning(
                        "LLM client failed, trying next",
                        system_only=True,
                        client_index=i,
                        client_type=type(client).__name__,
                        error=str(exc),
                    )
        raise last_exc from None

    # ------------------------------------------------------------------
    # Attribute proxy — delegate to primary client for compatibility
    # with get_llm_max_tokens / get_llm_temperature helpers in base.py
    # ------------------------------------------------------------------

    @property
    def model(self) -> Optional[str]:
        """Model name of the primary client."""
        return getattr(self._clients[0], "model", None)

    @property
    def temperature(self) -> Optional[float]:
        """Default temperature of the primary client."""
        return getattr(self._clients[0], "temperature", None)

    @property
    def max_tokens(self) -> Optional[int]:
        """Default max_tokens of the primary client."""
        return getattr(self._clients[0], "max_tokens", None)

    @property
    def timeout(self) -> Optional[int]:
        """Request timeout (seconds) of the primary client."""
        return getattr(self._clients[0], "timeout", None)

    @property
    def max_retries(self) -> Optional[int]:
        """Maximum retry count of the primary client."""
        return getattr(self._clients[0], "max_retries", None)

    # ------------------------------------------------------------------
    # Request hook forwarding — propagate to all sub-clients
    # ------------------------------------------------------------------

    def add_request_hook(self, hook: Callable[[Dict[str, Any]], None]) -> None:
        """Register a request hook on all underlying clients.

        Args:
            hook: Callable that receives the request dict before it is sent.
        """
        for client in self._clients:
            if hasattr(client, "add_request_hook"):
                client.add_request_hook(hook)

    def add_jsonl_request_hook(self, log_path: str) -> None:
        """Register a JSONL logging hook on all underlying clients.

        Args:
            log_path: File path where each request will be appended as one
                      JSON line.
        """
        for client in self._clients:
            if hasattr(client, "add_jsonl_request_hook"):
                client.add_jsonl_request_hook(log_path)

    def add_buffered_request_hook(self) -> None:
        """Enable in-memory request buffering on all underlying clients.

        Buffered entries can later be collected with
        :meth:`consume_buffered_request_entries`.
        """
        for client in self._clients:
            if hasattr(client, "add_buffered_request_hook"):
                client.add_buffered_request_hook()

    def clear_buffered_request_entries(self) -> None:
        """Discard all buffered request entries on every underlying client."""
        for client in self._clients:
            if hasattr(client, "clear_buffered_request_entries"):
                client.clear_buffered_request_entries()

    def consume_buffered_request_entries(self) -> List[Dict[str, Any]]:
        """Collect and clear buffered request entries from all clients.

        Returns:
            Combined list of request entry dicts from every underlying client.
            Entries are removed from the internal buffers after being returned.
        """
        entries: List[Dict[str, Any]] = []
        for client in self._clients:
            if hasattr(client, "consume_buffered_request_entries"):
                entries.extend(client.consume_buffered_request_entries())
        return entries

    def __repr__(self) -> str:
        models = [getattr(c, "model", type(c).__name__) for c in self._clients]
        return f"FallbackLLMClient(clients={models})"
