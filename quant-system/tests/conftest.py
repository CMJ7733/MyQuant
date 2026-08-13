"""Shared monitor-reader fixtures and JSON archive factories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pytest


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    lines = [json.dumps(record) for record in records]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def generation(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "generation": 0,
        "cycle": 0,
        "agent": "AgentMarketCycle",
        "op_counts": {"generate": 7},
        "n_generated": 7,
        "n_passed_checker": 4,
        "n_qualified": 3,
        "n_elite": 2,
        "reject_counts": {"judge": 3},
        "best": {"score": 0.086, "rank_ic": 0.071},
        "percentile_cutoffs": {"qualified": 0.05, "elite": 0.07},
        "elite_mean_score": 0.072,
        "llm_calls": 1,
        "wall_seconds": 12.5,
    }
    record.update(overrides)
    return record


def call(**overrides: Any) -> dict[str, Any]:
    seq = int(overrides.pop("seq", 1))
    agent = overrides.pop("agent", "AgentMarketCycle")
    generation_number = overrides.pop("generation", 0)
    cycle = overrides.pop("cycle", 0)
    role = overrides.pop("role", "generate")
    mode = overrides.pop("mode", "mock")
    tokens = int(overrides.pop("tokens", 10))
    record: dict[str, Any] = {
        "seq": seq,
        "model": "test-model",
        "temperature": 0.3,
        "system": "system text",
        "prompt": "prompt text",
        "response": "response text",
        "usage": {
            "prompt_tokens": tokens // 2,
            "completion_tokens": tokens - tokens // 2,
            "total_tokens": tokens,
        },
        "finish_reason": "stop",
        "latency_ms": seq,
        "tags": {
            "role": role,
            "agent": agent,
            "generation": generation_number,
            "cycle": cycle,
            "mode": mode,
        },
    }
    record.update(overrides)
    return record


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    write_json(
        run / "config.json",
        {
            "evolution": {
                "agents_per_run": 0,
                "seed": 42,
                "golden_ratio_selection": True,
                "generations": 24,
            }
        },
    )
    write_jsonl(run / "generations.jsonl", [])
    write_jsonl(run / "llm_calls.jsonl", [])
    return run
