"""Optional proxy from the dashboard to the ``api_server`` control plane.

The monitor itself is read-only by construction -- it never imports ``Evolver`` and
never writes into the experiment directory.  Run control is therefore not implemented
here; it is *forwarded* to the already-running ``api_server`` (``api_server/fm_api.py``,
port 8090 by default), which owns the task queue and the ``Evolver`` instance.

That keeps the two responsibilities separate: without ``--api-base`` the dashboard is a
pure observer that needs no other process, and with it the same page gains the buttons
that drive whichever worker was named on the command line.

Contract of the endpoints being called
--------------------------------------
Every control endpoint takes ``{"job_id": ..., "worker_id": ...}`` (``BaseRequest``),
``/evolve/start`` additionally accepts ``iterations``.  They answer HTTP 200 with a
body of ``{"code": "0"|"-1", "msg": ...}`` rather than signalling failure through the
status code, so this module passes the body through untouched and lets the UI read
``code``.  Turning ``code == "-1"`` into an HTTP error here would hide ``msg``, which
is the only place the reason ("evolve is not running, status: paused") appears.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

#: Dashboard action -> ``api_server`` path. ``start`` sends an ``EvolveRequest``;
#: the rest send a bare ``BaseRequest``.
ACTIONS: Dict[str, str] = {
    "start": "/evolve/start",
    "pause": "/evolve/pause",
    "continue": "/evolve/continue",
    "stop": "/evolve/stop",
    "cancel": "/cancel",
}


class ControlProxy:
    """Forwards dashboard actions to ``api_server``.

    Parameters
    ----------
    api_base:
        Base URL of the running ``api_server``, e.g. ``http://127.0.0.1:8090``.
    job_id, worker_id:
        Identify the task to act on. ``api_server`` keys everything by ``worker_id``,
        and rejects a request for a worker that was never ``/init``-ed.
    timeout:
        Seconds. Control calls only enqueue work or set a flag, so they return fast;
        a short timeout keeps a dead ``api_server`` from hanging the dashboard.
    """

    def __init__(
        self,
        api_base: str,
        job_id: str,
        worker_id: str,
        timeout: float = 10.0,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.job_id = job_id
        self.worker_id = worker_id
        self.timeout = timeout

    def describe(self) -> Dict[str, Any]:
        """What the dashboard needs to decide whether to render the control bar."""
        return {
            "enabled": True,
            "api_base": self.api_base,
            "job_id": self.job_id,
            "worker_id": self.worker_id,
            "actions": sorted(ACTIONS),
        }

    async def check(self) -> Dict[str, Any]:
        """Poll ``POST /check`` for stage, status, progress and diversity."""
        return await self._post("/check", {})

    async def act(self, action: str, iterations: Optional[int] = None) -> Dict[str, Any]:
        """Forward one control action. Raises ``KeyError`` for an unknown action."""
        path = ACTIONS[action]
        extra: Dict[str, Any] = {}
        if action == "start" and iterations is not None:
            extra["iterations"] = iterations
        return await self._post(path, extra)

    async def _post(self, path: str, extra: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "run control needs httpx (pip install 'famou-v2[monitor]')"
            ) from exc

        payload = {"job_id": self.job_id, "worker_id": self.worker_id, **extra}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.api_base}{path}", json=payload)
        except httpx.RequestError as exc:
            # api_server not started, already exited, or wrong --api-base. Letting this
            # propagate would surface as a bare HTTP 500 with no body, and "HTTP 500"
            # tells the operator nothing about which of those it was.
            return {
                "code": "-1",
                "msg": f"无法连接 api_server ({self.api_base}{path}): {exc}",
            }

        try:
            body = response.json()
        except ValueError:
            body = {"code": "-1", "msg": response.text}
        if not isinstance(body, dict):
            body = {"code": "-1", "msg": str(body)}
        # api_server answers 200 for a refused action, but a 4xx/5xx (worker never
        # /init-ed, server restarted) arrives with no "code" at all. Normalize so the
        # UI has exactly one field to branch on.
        body.setdefault("code", "0" if response.is_success else "-1")
        return body
