"""Tests for the api_server control proxy.

These exercise the real ``httpx`` call path against a local stub, because the value of
this module is entirely in how it translates ``api_server``'s conventions -- a 200 with
``code: "-1"`` meaning refusal, and a dead server meaning something else again.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from famou.monitor.control import ACTIONS, ControlProxy


class _StubTransport:
    """An httpx transport that answers from a table instead of the network."""

    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self.body = body
        self.requests: list[tuple[str, Dict[str, Any]]] = []

    def handler(self, request: Any) -> Any:
        import httpx
        import json

        self.requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(self.status, json=self.body)


def run_with_stub(proxy: ControlProxy, stub: _StubTransport, coro_factory) -> Any:
    """Run one proxy call with httpx.AsyncClient patched to use the stub."""
    import httpx

    real_client = httpx.AsyncClient

    def patched(*args: Any, **kwargs: Any) -> Any:
        kwargs["transport"] = httpx.MockTransport(stub.handler)
        return real_client(*args, **kwargs)

    httpx.AsyncClient = patched  # type: ignore[misc]
    try:
        return asyncio.run(coro_factory())
    finally:
        httpx.AsyncClient = real_client  # type: ignore[misc]


def proxy() -> ControlProxy:
    return ControlProxy("http://api.invalid:8090/", "job1", "worker1")


def test_api_base_trailing_slash_does_not_double_up() -> None:
    assert proxy().api_base == "http://api.invalid:8090"


def test_describe_lists_every_action() -> None:
    described = proxy().describe()

    assert described["enabled"] is True
    assert described["worker_id"] == "worker1"
    assert set(described["actions"]) == set(ACTIONS)


def test_check_posts_the_worker_identity() -> None:
    stub = _StubTransport(200, {"code": "0", "status": "running"})
    p = proxy()

    body = run_with_stub(p, stub, p.check)

    assert body == {"code": "0", "status": "running"}
    assert stub.requests == [("/check", {"job_id": "job1", "worker_id": "worker1"})]


@pytest.mark.parametrize("action,path", sorted(ACTIONS.items()))
def test_every_action_maps_to_its_endpoint(action: str, path: str) -> None:
    stub = _StubTransport(200, {"code": "0", "msg": "success"})
    p = proxy()

    run_with_stub(p, stub, lambda: p.act(action))

    assert stub.requests[0][0] == path


def test_start_carries_the_iteration_count_and_others_do_not() -> None:
    stub = _StubTransport(200, {"code": "0"})
    p = proxy()

    run_with_stub(p, stub, lambda: p.act("start", iterations=5))
    run_with_stub(p, stub, lambda: p.act("pause", iterations=5))

    assert stub.requests[0][1]["iterations"] == 5
    assert "iterations" not in stub.requests[1][1]


def test_a_refusal_passes_its_message_through() -> None:
    # api_server answers 200 for a refused action; the reason is only in "msg".
    stub = _StubTransport(200, {"code": "-1", "msg": "evolve is not running, status: paused"})
    p = proxy()

    body = run_with_stub(p, stub, lambda: p.act("pause"))

    assert body["code"] == "-1"
    assert "status: paused" in body["msg"]


def test_an_http_error_without_a_code_is_normalized_to_a_refusal() -> None:
    stub = _StubTransport(500, {"detail": "worker not initialized"})
    p = proxy()

    body = run_with_stub(p, stub, lambda: p.act("stop"))

    assert body["code"] == "-1"
    assert body["detail"] == "worker not initialized"


def test_an_unreachable_api_server_is_reported_not_raised() -> None:
    """A dead api_server must not surface as a bare HTTP 500 with no body."""
    p = ControlProxy("http://127.0.0.1:9", "job1", "worker1", timeout=0.5)

    body = asyncio.run(p.act("pause"))

    assert body["code"] == "-1"
    assert "无法连接 api_server" in body["msg"]


def test_an_unknown_action_is_a_key_error() -> None:
    p = proxy()

    with pytest.raises(KeyError):
        asyncio.run(p.act("selfdestruct"))
