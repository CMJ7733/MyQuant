"""OpenAI-compatible client.

Works against anything speaking the chat-completions API: Qianfan
(``https://qianfan.baidubce.com/v2``), a local vLLM serving the paper's
``gpt-oss-120b``, or OpenAI itself.  Retries transient failures with exponential
backoff and refuses to retry authentication errors (retrying a bad key just burns
time).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from cogalpha.llm.base import LLMClient, LLMError, LLMResponse


class OpenAIClient(LLMClient):
    """Chat-completions client with bounded retries."""

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: Optional[str] = None,
        default_temperature: float = 0.8,
        max_tokens: int = 4096,
        timeout: int = 120,
        max_retries: int = 3,
        recorder=None,
        max_calls: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        max_concurrency: int = 1,
    ) -> None:
        super().__init__(
            model=model,
            default_temperature=default_temperature,
            max_tokens=max_tokens,
            recorder=recorder,
            max_calls=max_calls,
            max_concurrency=max_concurrency,
        )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "provider 'openai' needs the openai package "
                "(pip install 'cogalpha[llm]')"
            ) from exc

        if not api_key:
            raise LLMError(
                "no API key: set llm.api_key, point llm.key_set at a key file, "
                "or export COGALPHA_API_KEY"
            )

        self.api_base = api_base
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = dict(extra_body or {})
        # Disable the SDK's own retries so ``max_retries`` here is the whole story.
        self._client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
            max_retries=0,
        )

    def _complete(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        from openai import (
            APIConnectionError,
            APIStatusError,
            AuthenticationError,
            RateLimitError,
        )
        from tenacity import (
            Retrying,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        retryer = Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=2, min=2, max=30),
            retry=retry_if_exception_type(
                (RateLimitError, APIConnectionError, APIStatusError)
            ),
            reraise=True,
        )

        try:
            for attempt in retryer:
                with attempt:
                    completion = self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **self.extra_body,
                    )
        except AuthenticationError as exc:
            raise LLMError(f"authentication failed for {self.api_base}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - surface one clear error type upward
            raise LLMError(f"LLM request failed after {self.max_retries} attempts: {exc}") from exc

        choice = completion.choices[0] if completion.choices else None
        text = (choice.message.content or "") if choice is not None else ""
        finish = getattr(choice, "finish_reason", None) if choice else None
        usage = {}
        if getattr(completion, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(completion.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(completion.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(completion.usage, "total_tokens", 0) or 0,
            }

        # A reasoning model can burn the entire output budget on its thinking trace and
        # return an empty `content` with finish_reason="length".  That is silent: the
        # parser finds no code, the generation is recorded as producing 0 alphas, and a
        # 24-generation run reports "alphas seen: 1" with no hint of the cause.
        # Observed on deepseek-v4-flash at max_tokens=4096: 95 of 99 calls came back
        # empty this way.  Raise instead, naming the fix.
        if finish == "length" and not text.strip():
            raise LLMError(
                f"model '{self.model}' hit the {max_tokens}-token output limit before "
                f"emitting any content ({usage.get('completion_tokens', '?')} tokens "
                "spent, response empty). This model writes a long reasoning trace, so "
                "the limit must cover trace + answer: raise llm.max_tokens (40960 works "
                "for this model) or use a model without an internal reasoning trace."
            )

        return LLMResponse(
            text=text,
            model=self.model,
            temperature=temperature,
            usage=usage,
            finish_reason=finish,
        )
