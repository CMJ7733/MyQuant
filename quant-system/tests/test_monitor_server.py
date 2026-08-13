"""HTTP API tests for the live monitor server."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from cogalpha.monitor import server as monitor_server
from conftest import call, generation, write_jsonl


def test_agent_endpoint_returns_known_agent_detail(run_dir: Path) -> None:
    response = TestClient(monitor_server.build_app(run_dir)).get(
        "/api/agent/AgentMarketCycle"
    )

    assert response.status_code == 200
    assert response.json()["name"] == "AgentMarketCycle"
    assert response.json()["selected"] is False
    assert response.json()["recent_operations"] == []


def test_agent_endpoint_returns_exact_404_for_unknown_agent(run_dir: Path) -> None:
    response = TestClient(monitor_server.build_app(run_dir)).get(
        "/api/agent/AgentDoesNotExist"
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "agent AgentDoesNotExist not found"}


def test_agent_endpoint_polls_archives_written_after_app_construction(
    run_dir: Path,
) -> None:
    client = TestClient(monitor_server.build_app(run_dir))
    write_jsonl(run_dir / "generations.jsonl", [generation(generation=3, cycle=2)])
    write_jsonl(run_dir / "llm_calls.jsonl", [call(seq=7, generation=3, cycle=2)])

    response = client.get("/api/agent/AgentMarketCycle")

    assert response.status_code == 200
    detail = response.json()
    assert detail["selected"] is True
    assert detail["current_generation"] == 3
    assert [operation["seq"] for operation in detail["recent_operations"]] == [7]


class _InstrumentedSnapshot:
    def __init__(self, reader: "_InstrumentedReader") -> None:
        self._reader = reader

    def to_dict(self) -> dict[str, str]:
        return self._reader.finish_associated_read({"source": "state"})


class _InstrumentedReader:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self._guard = threading.Lock()
        self._associated_reads = threading.Barrier(2)
        self.first_poll_started = threading.Event()
        self.active = 0
        self.max_active = 0

    def poll(self) -> _InstrumentedSnapshot:
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.first_poll_started.set()
        return _InstrumentedSnapshot(self)

    def finish_associated_read(self, payload: dict[str, str]) -> dict[str, str]:
        try:
            self._associated_reads.wait(timeout=0.5)
        except threading.BrokenBarrierError:
            pass
        with self._guard:
            self.active -= 1
        return payload

    def agent_detail(self, name: str) -> dict[str, str]:
        return self.finish_associated_read({"name": name})


def test_state_and_agent_poll_with_associated_read_are_serialized(
    run_dir: Path,
    monkeypatch: Any,
) -> None:
    reader = _InstrumentedReader()
    monkeypatch.setattr(monitor_server, "RunReader", lambda *_args, **_kwargs: reader)
    app = monitor_server.build_app(run_dir)
    state_endpoint = next(route.endpoint for route in app.routes if route.path == "/api/state")
    agent_endpoint = next(
        route.endpoint for route in app.routes if route.path == "/api/agent/{name}"
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        agent_response = executor.submit(agent_endpoint, "AgentMarketCycle")
        assert reader.first_poll_started.wait(timeout=1.0)
        state_response = executor.submit(state_endpoint)

        assert agent_response.result().status_code == 200
        assert state_response.result().status_code == 200

    assert reader.max_active == 1
