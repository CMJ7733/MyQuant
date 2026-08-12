"""
Mock LLM client for testing.

Provides deterministic responses without making real API calls.
Useful for:
- Unit testing
- Integration testing
- Local development without API costs
- Reproducible experiments
"""

import time
from typing import Callable, List, Optional

from famou.infrastructure.llm.base import BaseLLMClient, LLMResponse


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM client that returns predefined responses.

    This client cycles through a list of predefined responses, making it
    perfect for testing without incurring API costs or network latency.

    Features:
    - Deterministic: Same input → same output sequence
    - Fast: No network calls
    - Free: No API costs
    - Configurable: Can simulate various response patterns

    Examples:
        >>> # Simple usage - returns same response every time
        >>> client = MockLLMClient(response="return 42")
        >>> resp = client.generate("write code")
        >>> resp.text
        'return 42'

        >>> # Multiple responses - cycles through them
        >>> client = MockLLMClient(responses=["code1", "code2", "code3"])
        >>> client.generate("test").text  # 'code1'
        >>> client.generate("test").text  # 'code2'
        >>> client.generate("test").text  # 'code3'
        >>> client.generate("test").text  # 'code1' (cycles)

        >>> # Dynamic responses using a function
        >>> def my_handler(prompt):
        ...     return f"Response to: {prompt[:20]}"
        >>> client = MockLLMClient(response_func=my_handler)
    """

    def __init__(
        self,
        response: Optional[str] = None,
        responses: Optional[List[str]] = None,
        response_func: Optional[Callable[[str], str]] = None,
        model: str = "mock-gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 60,
        max_retries: int = 3,
        langfuse: Optional["Langfuse"] = None,  # type: ignore
        logger: Optional["Logger"] = None,  # type: ignore
    ):
        """
        Initialize mock LLM client.

        Args:
            response: Single response to return for all calls
            responses: List of responses to cycle through
            response_func: Function that takes prompt and returns response text
            model: Model name to report in responses
            temperature: Default temperature metadata
            max_tokens: Default max_tokens metadata
            timeout: Default timeout metadata
            max_retries: Default retry budget metadata
            langfuse: Optional Langfuse client for observability (passed to BaseLLMClient)
            logger: Optional logger for warnings (passed to BaseLLMClient)

        Note:
            Priority: response_func > responses > response
        """
        # Initialize parent (BaseLLMClient) with Langfuse/logger
        super().__init__(langfuse=langfuse, logger=logger)

        self.provider = "mock"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.call_count = 0
        self.call_history: List[dict] = []

        # Set response mode
        self.response_func = response_func
        if response_func:
            self.response_mode = "function"
        elif responses:
            self.response_mode = "cycle"
            self.responses = responses
        elif response:
            self.response_mode = "single"
            self.single_response = response
        else:
            # Default: simple placeholder response
            self.response_mode = "single"
            self.single_response = "# Generated code\npass"

    def _generate_impl(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate mock response (provider-specific implementation).

        This method is called by the parent's generate() template method,
        which handles Langfuse instrumentation automatically.

        Args:
            prompt: User prompt (stored in history)
            system: System prompt (stored in history)
            temperature: Ignored (mock is deterministic)
            max_tokens: Ignored (mock returns full response)
            **kwargs: Additional args (stored in history)

        Returns:
            LLMResponse with response text
        """
        request_id = self._new_request_id()
        started_at = time.time()

        # Record this call for testing/debugging
        self.call_history.append(
            {
                "prompt": prompt,
                "system": system,
            }
        )

        # Get response text based on mode
        if self.response_mode == "function":
            response_text = self.response_func(prompt)
        elif self.response_mode == "cycle":
            response_text = self.responses[self.call_count % len(self.responses)]
        else:  # single
            response_text = self.single_response

        self.call_count += 1

        response = LLMResponse(
            text=response_text,
            thinking=None,
            raw_response={"id": f"mock-{self.call_count}", "model": self.model},
            usage={
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": len(response_text) // 4,
                "total_tokens": (len(prompt) + len(response_text)) // 4,
            },
            model=self.model,
            provider=self.provider,
            finish_reason="stop",
        )
        self._emit_request_hook(
            self._build_request_log_entry(
                request_id=request_id,
                attempt=1,
                max_retries=max(1, int(self.max_retries)),
                status="success",
                duration_seconds=time.time() - started_at,
                prompt_tokens=response.usage.get("prompt_tokens"),
                response_tokens=response.usage.get("completion_tokens"),
                api_base="mock",
            )
        )
        return response

    def reset(self) -> None:
        """Reset call count and history."""
        self.call_count = 0
        self.call_history = []

    def get_calls(self) -> List[dict]:
        """Get all call history."""
        return self.call_history.copy()

    def __repr__(self) -> str:
        """Concise representation."""
        return f"MockLLMClient(mode={self.response_mode}, calls={self.call_count})"
