"""Helpers for persisting module input/output traces on programs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from pydantic import BaseModel

from famou.core.data import Program
from famou.infrastructure.llm.base import LLMResponse


def to_serializable(value: Any) -> Any:
    """Convert common runtime objects into JSON-serializable structures."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]
    if isinstance(value, set):
        return [to_serializable(v) for v in sorted(value, key=repr)]
    return value


def build_llm_trace(
    *,
    module_name: str,
    system: Optional[str],
    prompt: str,
    response: Optional[LLMResponse] = None,
    request_extra: Optional[Dict[str, Any]] = None,
    parsed: Optional[Any] = None,
    error: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build a normalized trace payload for a single LLM call."""
    usage = to_serializable(response.usage) if response is not None else {}
    trace: Dict[str, Any] = {
        "module": module_name,
        "request": {
            "system": system,
            "prompt": prompt,
        },
        "timing": {
            "started_at": response.started_at if response is not None else None,
            "completed_at": response.completed_at if response is not None else None,
            "latency_ms": response.latency_ms if response is not None else None,
        },
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
        "response": {
            "text": response.text if response is not None else None,
            "thinking": response.thinking if response is not None else None,
            "model": response.model if response is not None else None,
            "provider": response.provider if response is not None else None,
            "finish_reason": response.finish_reason if response is not None else None,
            "usage": usage,
            "raw_response": to_serializable(response.raw_response) if response is not None else None,
        },
    }

    if request_extra:
        trace["request"].update(to_serializable(request_extra))

    if parsed is not None:
        trace["parsed"] = to_serializable(parsed)

    if error is not None:
        trace["error"] = {
            "message": str(error),
            "type": type(error).__name__ if not isinstance(error, str) else "Error",
        }

    return trace


def build_evaluate_trace(
    *,
    module_name: str,
    request: Dict[str, Any],
    started_at: Optional[float],
    completed_at: Optional[float],
    raw_result: Optional[Any] = None,
    parsed: Optional[Any] = None,
    error: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build a normalized trace payload for a single evaluator execution."""
    latency_ms = None
    if started_at is not None and completed_at is not None:
        latency_ms = int((completed_at - started_at) * 1000)

    trace: Dict[str, Any] = {
        "module": module_name,
        "request": to_serializable(request),
        "timing": {
            "started_at": started_at,
            "completed_at": completed_at,
            "latency_ms": latency_ms,
        },
        "response": {
            "raw_result": to_serializable(raw_result),
        },
    }

    if parsed is not None:
        trace["parsed"] = to_serializable(parsed)

    if error is not None:
        trace["error"] = {
            "message": str(error),
            "type": type(error).__name__ if not isinstance(error, str) else "Error",
        }

    return trace


def ensure_trace_store(program: Program) -> Dict[str, Any]:
    """Ensure the program has a trace store in meta and return it."""
    trace_store = program.meta.get("traces")
    if not isinstance(trace_store, dict):
        trace_store = {}
        program.meta["traces"] = trace_store
    return trace_store


def set_generate_trace(program: Program, trace: Dict[str, Any]) -> None:
    """Persist the generation trace for a program."""
    trace_store = ensure_trace_store(program)
    trace_store["generate"] = to_serializable(trace)


def append_named_trace(program: Program, trace_name: str, trace: Dict[str, Any]) -> None:
    """Append a named trace entry to a program."""
    trace_store = ensure_trace_store(program)
    entries = trace_store.get(trace_name)
    if not isinstance(entries, list):
        entries = []
        trace_store[trace_name] = entries
    entries.append(to_serializable(trace))


def append_judge_trace(program: Program, trace: Dict[str, Any]) -> None:
    """Append a judge trace entry to a program."""
    append_named_trace(program, "judge", trace)


def append_evaluate_trace(program: Program, trace: Dict[str, Any]) -> None:
    """Append an evaluate trace entry to a program."""
    append_named_trace(program, "evaluate", trace)


def attach_debug_trace(
    program: Program,
    *,
    attempt: int,
    field_name: str,
    trace: Dict[str, Any],
) -> None:
    """Attach a trace payload to a specific debug attempt record."""
    debug_history = program.meta.get("debug_history")
    if not isinstance(debug_history, list):
        return

    for entry in reversed(debug_history):
        if isinstance(entry, dict) and entry.get("attempt") == attempt:
            entry[field_name] = to_serializable(trace)
            return


def snapshot_program(program: Program, *, include_code: bool = True) -> Dict[str, Any]:
    """Create a compact, serializable snapshot for debug history."""
    snapshot = {
        "id": program.id,
        "parent_id": program.parent_id,
        "generation": program.generation,
        "iteration": program.iteration,
        "language": program.language,
        "combined_score": program.combined_score,
        "validity": program.validity,
        "error_info": program.error_info,
        "metrics": to_serializable(program.metrics),
        "required_packages": to_serializable(program.required_packages),
    }
    if include_code:
        snapshot["code"] = program.code
    return snapshot


def copy_debug_history(program: Program) -> list[Dict[str, Any]]:
    """Return a deep copy of debug history stored on a program."""
    history = program.meta.get("debug_history")
    if isinstance(history, list):
        return deepcopy(to_serializable(history))
    return []
