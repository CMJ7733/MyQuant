"""LLM client interface shared by the real and mock backends."""

from __future__ import annotations

import abc
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


class LLMError(RuntimeError):
    """Raised when a generation cannot be completed after retries."""


@dataclass
class LLMResponse:
    """One completion, plus the accounting a run archive needs."""

    text: str
    model: str
    temperature: float
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    latency_ms: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens for this call, or 0 when the endpoint reported no usage."""
        return int(self.usage.get("total_tokens", 0))


class LLMClient(abc.ABC):
    """Minimal synchronous text-completion interface.

    Deliberately narrow: CogAlpha only ever needs "system + user prompt in, text
    out".  Structured output is parsed from the text by
    :mod:`cogalpha.agents.parse`, because requiring JSON mode would exclude the
    self-hosted models the paper actually uses.
    """

    def __init__(
        self,
        model: str,
        default_temperature: float = 0.8,
        max_tokens: int = 4096,
        recorder: Optional["CallRecorderProtocol"] = None,
        max_calls: Optional[int] = None,
        max_concurrency: int = 1,
    ) -> None:
        self.model = model
        self.default_temperature = default_temperature
        self.max_tokens = max_tokens
        self.recorder = recorder
        self.max_calls = max_calls
        self.max_concurrency = max(1, int(max_concurrency))
        self.n_calls = 0
        self.n_tokens = 0
        #: Guards the counters and the budget check.  ``generate_many`` runs requests
        #: on a thread pool, so ``n_calls += 1`` from several threads would lose
        #: increments and the budget ceiling would leak past its limit.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Complete ``prompt``, recording the call.

        ``tags`` is free-form provenance (agent name, guidance mode, generation)
        that lands in the JSONL record; it never reaches the model.

        Thread-safe: the budget check and the counter update are taken under a lock,
        so this may be called concurrently (see :meth:`generate_many`).
        """
        # Reserve a slot before doing any work. Checking and incrementing separately
        # would let N threads all see n_calls == max_calls - 1 and all proceed.
        with self._lock:
            if self.max_calls is not None and self.n_calls >= self.max_calls:
                raise LLMError(
                    f"LLM call budget exhausted ({self.max_calls} calls); "
                    "raise evolution.max_llm_calls or shorten the schedule"
                )
            self.n_calls += 1

        temp = self.default_temperature if temperature is None else float(temperature)
        cap = self.max_tokens if max_tokens is None else int(max_tokens)

        started = time.time()
        try:
            response = self._complete(prompt=prompt, system=system, temperature=temp, max_tokens=cap)
        except BaseException:
            # The slot was reserved above; give it back so a transport failure does
            # not silently consume budget.
            with self._lock:
                self.n_calls -= 1
            raise
        response.latency_ms = int((time.time() - started) * 1000)

        with self._lock:
            self.n_tokens += response.total_tokens
        if self.recorder is not None:
            # The recorder has its own lock and appends whole lines, so concurrent
            # writers interleave records but never a single record.
            self.recorder.record(
                prompt=prompt,
                system=system,
                response=response,
                tags=dict(tags or {}),
            )
        return response

    def generate_many(
        self,
        requests: Sequence[Dict[str, Any]],
        max_workers: Optional[int] = None,
    ) -> List[Optional[LLMResponse]]:
        """Run independent requests concurrently, preserving input order.

        Each element of ``requests`` is the keyword dict for one :meth:`generate`
        call.  Returns a list the same length, with ``None`` where that request
        raised — so the caller sees which slots failed rather than losing the
        alignment between requests and results.

        This exists because the search's expensive steps are *embarrassingly
        parallel*: the N children of a generation are bred independently, and the
        code-quality review of N alphas has no ordering constraint either.  Run
        serially against an endpoint with 36 s latency, one generation of 12 children
        costs 36 x 36 s; at concurrency 4 it costs a quarter of that.

        Concurrency is bounded by ``max_concurrency`` (from ``llm.max_concurrency``)
        rather than by the request count: endpoints rate-limit, and 96 simultaneous
        requests earns 429s that the retry logic then serialises anyway.

        A budget exhaustion (:class:`LLMError`) is returned as ``None`` like any other
        failure; the caller decides whether to stop. In-flight requests are allowed to
        finish rather than being cancelled — they have already been paid for.
        """
        if not requests:
            return []

        workers = max_workers if max_workers is not None else self.max_concurrency
        workers = max(1, min(int(workers), len(requests)))
        if workers == 1:
            # Keep the single-threaded path exactly as it was: no pool, no threads,
            # so a serial run is byte-for-byte what it used to be.
            out: List[Optional[LLMResponse]] = []
            for kwargs in requests:
                try:
                    out.append(self.generate(**kwargs))
                except LLMError:
                    out.append(None)
            return out

        results: List[Optional[LLMResponse]] = [None] * len(requests)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cogalpha-llm") as pool:
            futures = {
                pool.submit(self.generate, **kwargs): i
                for i, kwargs in enumerate(requests)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except LLMError:
                    results[index] = None
                except Exception:  # noqa: BLE001 - one bad request must not sink the batch
                    results[index] = None
        return results

    def stats(self) -> Dict[str, int]:
        """Cumulative calls and tokens for this client, for run accounting."""
        with self._lock:
            return {"calls": self.n_calls, "tokens": self.n_tokens}


    # ---------------------------------------------------------------- subclass

    @abc.abstractmethod
    def _complete(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Backend-specific completion; must not do its own recording."""


class CallRecorderProtocol:
    """Structural type for the recorder; see :mod:`cogalpha.llm.recorder`."""

    def record(
        self,
        prompt: str,
        system: Optional[str],
        response: LLMResponse,
        tags: Dict[str, Any],
    ) -> None:  # pragma: no cover - protocol
        """Persist one call. See :class:`cogalpha.llm.recorder.CallRecorder`."""
        raise NotImplementedError
