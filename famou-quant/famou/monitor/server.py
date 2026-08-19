"""HTTP + SSE server for the live dashboard.

Endpoints
---------
``GET  /``                      the single-page dashboard
``GET  /api/state``             full snapshot (JSON)
``GET  /api/stream``            Server-Sent Events, one snapshot per tick
``GET  /api/island/{id}``       one island's statistics and its programs
``GET  /api/program/{id}``      one program: code, metrics, LLM exchange, lineage
``GET  /api/rollout/{id}``      one rollout's stored record
``GET  /api/iteration/{n}``     every program created in one iteration
``GET  /api/control``           current stage/status, proxied from api_server
``POST /api/control/{action}``  start | pause | continue | stop | cancel

SSE rather than WebSocket: the traffic is one-directional, browsers reconnect
automatically after a drop, and it needs no protocol upgrade to work through a plain
SSH tunnel.

Binding
-------
Default host is ``127.0.0.1``.  This is a security decision, not a default chosen by
habit: ``/api/program/{id}`` serves the full system prompt, prompt and response for
every program, which is the prompt engineering the whole framework rests on.  Binding
to ``0.0.0.0`` publishes that to the network, so it requires an explicit flag and
prints a warning.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from famou.monitor.control import ACTIONS, ControlProxy
from famou.monitor.reader import ExperimentReader

_STATIC = Path(__file__).parent / "static"


def _safe_id(value: str) -> bool:
    """Reject anything that could escape the experiment directory.

    Program and rollout IDs arrive in the URL and are used as filenames, so a value
    containing a separator or starting with a dot is refused outright rather than
    normalized -- there is no legitimate ID that looks like that.
    """
    return bool(value) and "/" not in value and "\\" not in value and not value.startswith(".")


def build_app(
    run_dir: str | Path,
    poll_interval: float = 1.0,
    stale_after: float = 600.0,
    control: Optional[ControlProxy] = None,
):
    """Create the FastAPI application for one experiment directory.

    A single :class:`ExperimentReader` is shared by every request, so N open browser
    tabs cost one directory scan, not N. FastAPI sync handlers and the sync SSE
    iterator may run on worker threads; a lock serializes ingest with its associated
    snapshot or detail read.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "the monitor needs fastapi and uvicorn "
            "(pip install 'famou-v2[monitor]')"
        ) from exc

    reader = ExperimentReader(run_dir, stale_after=stale_after)
    reader_lock = threading.RLock()
    app = FastAPI(title="FaQ monitor", docs_url=None, redoc_url=None)

    def poll_payload() -> Dict[str, Any]:
        with reader_lock:
            state = reader.poll().to_dict()
        state["control"] = (
            control.describe() if control is not None else {"enabled": False}
        )
        return state

    @app.get("/", response_class=HTMLResponse)
    def index() -> Any:
        """Serve the dashboard."""
        path = _STATIC / "index.html"
        if not path.exists():  # pragma: no cover - packaging error
            raise HTTPException(500, "dashboard asset missing")
        return HTMLResponse(path.read_text(encoding="utf-8"))

    @app.get("/api/state")
    def state() -> Any:
        """One snapshot. Used on page load and as the SSE fallback."""
        return JSONResponse(poll_payload())

    @app.get("/api/stream")
    async def stream() -> Any:
        """Push a snapshot every ``poll_interval`` seconds.

        Sends the full snapshot rather than a diff: it is tens of kilobytes, once a
        second, to a browser usually on the same host. A diff protocol would add a
        state-reconciliation bug surface for no benefit at this scale.

        The generator is async, and the two things it waits on -- the directory scan
        and the interval -- are both cancellable. A sync generator sleeping with
        ``time.sleep`` cannot be interrupted, so every browser tab that was closed
        would leave a worker thread rescanning the run directory forever; over a long
        session that is a real leak, not a theoretical one.
        """
        import anyio

        async def events() -> AsyncIterator[str]:
            # Tell the browser how long to wait before reconnecting after a drop.
            yield f"retry: {int(poll_interval * 2000)}\n\n"
            while True:
                # Off the event loop: poll() reads files and must not block the server.
                payload = await anyio.to_thread.run_sync(poll_payload)
                yield f"data: {json.dumps(payload, default=str)}\n\n"
                await anyio.sleep(poll_interval)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                # Defeat proxy buffering, which would otherwise hold events until a
                # buffer fills and make a live view look frozen.
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get("/api/island/{island_id}")
    def island(island_id: int) -> Any:
        """One island's statistics and every program assigned to it."""
        with reader_lock:
            reader.poll()
            record = reader.island_detail(island_id)
        if record is None:
            raise HTTPException(404, f"island {island_id} not found")
        return JSONResponse(record)

    @app.get("/api/program/{program_id}")
    def program(program_id: str) -> Any:
        """One program: source, metrics, the LLM exchange that produced it, lineage."""
        if not _safe_id(program_id):
            raise HTTPException(400, "invalid program id")
        with reader_lock:
            reader.poll()
            record = reader.program_detail(program_id)
        if record is None:
            raise HTTPException(404, f"program {program_id} not found")
        return JSONResponse(record)

    @app.get("/api/rollout/{rollout_id}")
    def rollout(rollout_id: str) -> Any:
        """One rollout's stored record, minus the duplicated program dump."""
        if not _safe_id(rollout_id):
            raise HTTPException(400, "invalid rollout id")
        record = reader.rollout_detail(rollout_id)
        if record is None:
            raise HTTPException(404, f"rollout {rollout_id} not found")
        return JSONResponse(record)

    @app.get("/api/iteration/{iteration}")
    def iteration(iteration: int) -> Any:
        """Every program created in one iteration, across all islands."""
        with reader_lock:
            reader.poll()
            return JSONResponse(reader.iteration_programs(iteration))

    @app.get("/api/control")
    async def control_state() -> Any:
        """Stage, status and progress, proxied from the api_server control plane."""
        if control is None:
            raise HTTPException(501, "run control is disabled; start with --api-base")
        return JSONResponse({**control.describe(), "check": await control.check()})

    @app.post("/api/control/{action}")
    async def control_act(action: str, iterations: Optional[int] = None) -> Any:
        """Forward one control action to api_server and pass its answer through."""
        if control is None:
            raise HTTPException(501, "run control is disabled; start with --api-base")
        if action not in ACTIONS:
            raise HTTPException(404, f"unknown action {action}")
        return JSONResponse(await control.act(action, iterations=iterations))

    return app


def serve(
    run_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 8080,
    poll_interval: float = 1.0,
    stale_after: float = 600.0,
    control: Optional[ControlProxy] = None,
    open_browser: bool = False,
) -> None:
    """Run the dashboard until interrupted."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the monitor needs uvicorn (pip install 'famou-v2[monitor]')"
        ) from exc

    app = build_app(
        run_dir,
        poll_interval=poll_interval,
        stale_after=stale_after,
        control=control,
    )

    print(f"FaQ monitor  |  http://{host}:{port}", flush=True)
    if control is not None:
        print(
            f"  run control -> {control.api_base} (worker {control.worker_id})",
            flush=True,
        )
    if host not in ("127.0.0.1", "localhost"):
        print(
            "  WARNING: bound to a non-local address. /api/program serves every "
            "prompt and completion this run produced.",
            flush=True,
        )
    if open_browser:  # pragma: no cover - interactive convenience
        import webbrowser

        threading.Timer(0.8, lambda: webbrowser.open(f"http://{host}:{port}")).start()

    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)
