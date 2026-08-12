"""Append-only JSONL record of every LLM call.

Why this exists: the paper's own limitation section admits its alphas are not
byte-reproducible because of temperature sampling.  The next best thing is a
complete transcript — agent, mode, temperature, prompt, response, tokens — so a
run can be audited and its cost accounted for after the fact.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from cogalpha.llm.base import LLMResponse


class CallRecorder:
    """Thread-safe JSONL writer.

    Parameters
    ----------
    path:
        Destination file; parent directories are created.  ``None`` keeps records
        in memory only (used by tests that assert on call ordering).
    truncate_prompt:
        Cap on stored prompt/response characters.  Prompts carry the accumulated
        adaptive-generation feedback and get long; 20k chars keeps the archive
        readable while preserving everything that matters in practice.
    """

    def __init__(
        self,
        path: Optional[str | Path] = None,
        truncate_prompt: int = 20_000,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.truncate_prompt = truncate_prompt
        self._lock = threading.Lock()
        self._buffer: List[Dict[str, Any]] = []
        self._seq = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        prompt: str,
        system: Optional[str],
        response: LLMResponse,
        tags: Dict[str, Any],
    ) -> None:
        """Append one call to the transcript. Thread-safe; flushes immediately so a
        killed run still leaves a complete log up to the last call."""
        with self._lock:
            self._seq += 1
            entry = {
                "seq": self._seq,
                "model": response.model,
                "temperature": response.temperature,
                "system": _clip(system, self.truncate_prompt),
                "prompt": _clip(prompt, self.truncate_prompt),
                "response": _clip(response.text, self.truncate_prompt),
                "usage": response.usage,
                "finish_reason": response.finish_reason,
                "latency_ms": response.latency_ms,
                "tags": tags,
            }
            self._buffer.append(entry)
            if self.path is not None:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @property
    def entries(self) -> List[Dict[str, Any]]:
        """Snapshot of the in-memory records (tests read this instead of the file)."""
        with self._lock:
            return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)


def _clip(text: Optional[str], limit: int) -> Optional[str]:
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[:limit] + f"...[+{len(text) - limit} chars]"
