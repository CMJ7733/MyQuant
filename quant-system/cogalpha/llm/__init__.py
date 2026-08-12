"""LLM backends and call recording.

Two backends, one interface:

* :class:`~cogalpha.llm.openai_client.OpenAIClient` — any OpenAI-compatible
  endpoint (Qianfan, a local vLLM serving ``gpt-oss-120b``, ...).  This is the
  default; the paper serves its own model, so there is no vendor lock-in to undo.
* :class:`~cogalpha.llm.mock_client.MockLLMClient` — deterministic, offline, free.
  It returns *real, executable* alpha code drawn from a template bank, so the
  whole pipeline (checker -> fitness -> evolution) can be exercised without a
  network or a budget.

Every call, from either backend, is appended to ``llm_calls.jsonl`` by
:class:`~cogalpha.llm.recorder.CallRecorder`.  That file is the audit trail: which
agent, which guidance mode, which temperature, what prompt, what came back.
"""

from cogalpha.llm.base import LLMClient, LLMError, LLMResponse  # noqa: F401
from cogalpha.llm.factory import build_client  # noqa: F401
from cogalpha.llm.mock_client import MockLLMClient  # noqa: F401
from cogalpha.llm.recorder import CallRecorder  # noqa: F401

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "MockLLMClient",
    "CallRecorder",
    "build_client",
]
