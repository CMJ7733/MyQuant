"""HTTP + SSE server for the live dashboard.

Endpoints
---------
``GET /``                     the single-page dashboard
``GET /api/state``            full snapshot (JSON)
``GET /api/stream``           Server-Sent Events, one snapshot per tick
``GET /api/agent/{name}``     one agent's identity, progress, and recent operations
``GET /api/call/{seq}``       one LLM call, full prompt and response
``GET /api/alpha/{id}``       one alpha: code, checks, fitness, lineage
``GET /api/generation/...``   the alphas one agent produced in one generation
``GET /api/candidate/{file}`` an archived candidate's source

SSE rather than WebSocket: the traffic is one-directional, browsers reconnect
automatically after a drop, and it needs no protocol upgrade to work through a plain
SSH tunnel.

Binding
-------
Default host is ``127.0.0.1``.  This is a security decision, not a default chosen by
habit: ``llm_calls.jsonl`` contains every prompt this system sends, which is the
prompt engineering the whole method rests on, and the detail endpoints serve it
verbatim.  Binding to ``0.0.0.0`` publishes that to the network, so it requires an
explicit flag and prints a warning.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from cogalpha.monitor.reader import RunReader

_STATIC = Path(__file__).parent / "static"


def build_app(
    run_dir: str | Path,
    poll_interval: float = 1.0,
    stale_after: float = 180.0,
):
    """Create the FastAPI application for one run directory.

    A single :class:`RunReader` is shared by every request, so N open browser tabs
    cost one file tail, not N. FastAPI sync handlers and the sync SSE iterator may run
    on worker threads; a lock serializes tail mutation with its associated snapshot or
    detail read.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "the monitor needs fastapi and uvicorn "
            "(pip install 'cogalpha[monitor]')"
        ) from exc

    reader = RunReader(run_dir, stale_after=stale_after)
    reader_lock = threading.RLock()
    app = FastAPI(title="CogAlpha monitor", docs_url=None, redoc_url=None)

    def poll_payload() -> Dict[str, Any]:
        with reader_lock:
            return reader.poll().to_dict()

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
    def stream() -> Any:
        """Push a snapshot every ``poll_interval`` seconds.

        Sends the full snapshot rather than a diff: it is ~30-60 KB, once a second,
        to a browser on the same host. A diff protocol would add a
        state-reconciliation bug surface for no benefit at this scale.
        """

        def events() -> Iterator[str]:
            # Tell the browser how long to wait before reconnecting after a drop.
            yield f"retry: {int(poll_interval * 2000)}\n\n"
            while True:
                payload = json.dumps(poll_payload(), default=str)
                yield f"data: {payload}\n\n"
                time.sleep(poll_interval)

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

    @app.get("/api/agent/{name}")
    def agent(name: str) -> Any:
        """One agent's identity, progress, and recent operations."""
        with reader_lock:
            reader.poll()
            record = reader.agent_detail(name)
        if record is None:
            raise HTTPException(404, f"agent {name} not found")
        return JSONResponse(record)

    @app.get("/api/call/{seq}")
    def call(seq: int) -> Any:
        """One LLM call with its full prompt and response."""
        record = reader.call_detail(seq)
        if record is None:
            raise HTTPException(404, f"call {seq} not found")
        return JSONResponse(record)

    @app.get("/api/alpha/{alpha_id}")
    def alpha(alpha_id: str) -> Any:
        """One alpha: source, every check, fitness, lineage."""
        record = reader.alpha_detail(alpha_id)
        if record is None:
            raise HTTPException(
                404,
                f"alpha {alpha_id} not found -- alphas.jsonl is only written when the "
                "run saves, so this is expected mid-run",
            )
        return JSONResponse(record)

    @app.get("/api/generation/{agent}/{generation}")
    def generation(agent: str, generation: int) -> Any:
        """Every alpha one agent produced in one generation, with tier and reason."""
        return JSONResponse(reader.generation_alphas(agent, generation))

    @app.get("/api/candidate/{name}")
    def candidate(name: str) -> Any:
        """An archived candidate's source, header included."""
        # Reject anything with a path separator: the name comes from the URL and must
        # not be able to escape the candidates directory.
        if "/" in name or "\\" in name or name.startswith("."):
            raise HTTPException(400, "invalid candidate name")
        path = reader.path / "candidates" / name
        if not path.exists() or path.suffix != ".py":
            raise HTTPException(404, f"candidate {name} not found")
        return JSONResponse({"file": name, "code": path.read_text(encoding="utf-8")})

    return app


def serve(
    run_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 8080,
    poll_interval: float = 1.0,
    open_browser: bool = False,
) -> None:
    """Run the dashboard until interrupted."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the monitor needs uvicorn (pip install 'cogalpha[monitor]')"
        ) from exc

    app = build_app(run_dir, poll_interval=poll_interval)

    print(f"cogalpha monitor  |  http://{host}:{port}", flush=True)
    if host not in ("127.0.0.1", "localhost"):
        print(
            "  WARNING: bound to a non-local address. The detail endpoints serve "
            "every prompt and completion this run produced.",
            flush=True,
        )
    if open_browser:  # pragma: no cover - interactive convenience
        import threading
        import webbrowser

        threading.Timer(0.8, lambda: webbrowser.open(f"http://{host}:{port}")).start()

    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)
