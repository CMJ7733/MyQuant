"""HTTP API tests for the monitor server, including the optional control proxy."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from monitor_artifacts import rollout, write_json
from famou.monitor import server as monitor_server
from famou.monitor.control import ControlProxy


def client(run_dir: Path, **kwargs: Any) -> TestClient:
    return TestClient(monitor_server.build_app(run_dir, **kwargs))


def test_index_serves_the_dashboard(run_dir: Path) -> None:
    response = client(run_dir).get("/")

    assert response.status_code == 200
    assert "FaQ 演化过程实时监控" in response.text


def test_state_returns_a_snapshot(run_dir: Path) -> None:
    payload = client(run_dir).get("/api/state").json()

    assert payload["experiment_id"] == "exp_demo"
    assert payload["totals"]["n_programs"] == 2
    assert payload["control"] == {"enabled": False}


def test_state_polls_artifacts_written_after_app_construction(run_dir: Path) -> None:
    api = client(run_dir)
    api.get("/api/state")
    write_json(run_dir / "results" / "rollout_2.json", rollout(rollout_id="rollout_2"))

    assert api.get("/api/state").json()["totals"]["n_rollouts"] == 2


def test_program_endpoint_returns_code_and_prompt(run_dir: Path) -> None:
    payload = client(run_dir).get("/api/program/prog_1_0").json()

    assert "def solve" in payload["code"]
    assert payload["prompt"] == "improve this"


def test_program_endpoint_404s_for_an_unknown_id(run_dir: Path) -> None:
    response = client(run_dir).get("/api/program/nope")

    assert response.status_code == 404
    assert response.json() == {"detail": "program nope not found"}


@pytest.mark.parametrize("bad", [".hidden", "..\\config", "a\\b"])
def test_program_endpoint_rejects_ids_the_handler_receives_intact(
    run_dir: Path, bad: str
) -> None:
    """A separator the router does not split on must be refused by the handler.

    A backslash survives routing as one path segment, so on Windows -- where it *is*
    the separator -- only this guard stands between the URL and the rest of the disk.
    """
    response = client(run_dir).get(f"/api/program/{bad}")

    assert response.status_code == 400
    assert response.json() == {"detail": "invalid program id"}


@pytest.mark.parametrize("bad", ["..%2F..%2Fconfig", "%2Fetc%2Fpasswd"])
def test_program_endpoint_refuses_percent_encoded_separators(run_dir: Path, bad: str) -> None:
    """Starlette decodes %2F before matching, so the route simply does not match.

    The handler is never reached, which is why this asserts refusal rather than the
    handler's own 400: what matters is that nothing outside the directory is served.
    """
    response = client(run_dir).get(f"/api/program/{bad}")

    assert response.status_code == 404
    assert "experiment" not in response.text


def test_rollout_endpoint_rejects_ids_that_could_escape_the_directory(run_dir: Path) -> None:
    assert client(run_dir).get("/api/rollout/..\\config").status_code == 400
    assert client(run_dir).get("/api/rollout/..%2Fconfig").status_code == 404


def test_rollout_endpoint_returns_the_stored_record(run_dir: Path) -> None:
    payload = client(run_dir).get("/api/rollout/rollout_1").json()

    assert payload["status"] == "success"
    assert "program" not in payload


def test_island_and_iteration_endpoints(run_dir: Path) -> None:
    api = client(run_dir)

    assert api.get("/api/island/0").json()["n_programs"] == 2
    assert api.get("/api/island/7").status_code == 404
    assert [p["id"] for p in api.get("/api/iteration/1").json()] == ["prog_1_0"]


def read_stream(app: Any, limit: int = 2) -> list[str]:
    """Consume the first ``limit`` chunks of /api/stream, then close the generator.

    The endpoint is invoked directly rather than through ``TestClient.stream``: this
    starlette/httpx combination buffers a streaming response to completion before
    returning it, so it can never open a connection to an endless generator (a finite
    one works fine). Calling the route's own coroutine exercises the same production
    generator, and closing it asserts the property that matters -- that an abandoned
    connection actually stops the polling loop instead of leaking it.
    """
    route = next(r for r in app.routes if getattr(r, "path", "") == "/api/stream")

    async def run() -> list[str]:
        response = await route.endpoint()
        chunks: list[str] = []
        iterator = response.body_iterator
        try:
            async for chunk in iterator:
                chunks.append(chunk if isinstance(chunk, str) else chunk.decode())
                if len(chunks) >= limit:
                    break
        finally:
            await iterator.aclose()
        return chunks

    return asyncio.run(run())


def test_stream_emits_a_retry_hint_then_snapshots(run_dir: Path) -> None:
    chunks = read_stream(monitor_server.build_app(run_dir, poll_interval=0.01))

    assert chunks[0].startswith("retry: ")
    assert chunks[1].startswith("data: ")
    assert json.loads(chunks[1][len("data: "):])["experiment_id"] == "exp_demo"


def test_stream_headers_defeat_proxy_buffering(run_dir: Path) -> None:
    app = monitor_server.build_app(run_dir, poll_interval=0.01)
    route = next(r for r in app.routes if getattr(r, "path", "") == "/api/stream")

    async def run() -> Any:
        response = await route.endpoint()
        await response.body_iterator.aclose()
        return response

    response = asyncio.run(run())

    assert response.media_type == "text/event-stream"
    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["cache-control"] == "no-cache"


# ------------------------------------------------------------------ run control


def test_control_endpoints_are_disabled_without_an_api_base(run_dir: Path) -> None:
    api = client(run_dir)

    for response in (api.get("/api/control"), api.post("/api/control/pause")):
        assert response.status_code == 501
        assert response.json() == {
            "detail": "run control is disabled; start with --api-base"
        }


class _RecordingProxy(ControlProxy):
    """A ControlProxy that records calls instead of reaching an api_server."""

    def __init__(self) -> None:
        super().__init__("http://api.invalid", "job1", "worker1")
        self.calls: list[tuple[str, Any]] = []

    async def check(self) -> Dict[str, Any]:
        self.calls.append(("check", None))
        return {"code": "0", "stage": "evolve", "status": "running"}

    async def act(self, action: str, iterations: Any = None) -> Dict[str, Any]:
        self.calls.append((action, iterations))
        return {"code": "0", "msg": "success"}


def test_control_state_includes_the_proxied_check(run_dir: Path) -> None:
    proxy = _RecordingProxy()

    payload = client(run_dir, control=proxy).get("/api/control").json()

    assert payload["enabled"] is True
    assert payload["worker_id"] == "worker1"
    assert payload["check"]["status"] == "running"


def test_control_action_is_forwarded(run_dir: Path) -> None:
    proxy = _RecordingProxy()

    payload = client(run_dir, control=proxy).post("/api/control/pause").json()

    assert payload == {"code": "0", "msg": "success"}
    assert proxy.calls == [("pause", None)]


def test_control_start_passes_the_iteration_count_through(run_dir: Path) -> None:
    proxy = _RecordingProxy()

    client(run_dir, control=proxy).post("/api/control/start?iterations=5")

    assert proxy.calls == [("start", 5)]


def test_unknown_control_action_is_rejected(run_dir: Path) -> None:
    proxy = _RecordingProxy()

    response = client(run_dir, control=proxy).post("/api/control/selfdestruct")

    assert response.status_code == 404
    assert proxy.calls == []


def test_state_advertises_control_when_enabled(run_dir: Path) -> None:
    payload = client(run_dir, control=_RecordingProxy()).get("/api/state").json()

    assert payload["control"]["enabled"] is True
    assert "pause" in payload["control"]["actions"]
