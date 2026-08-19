"""CLI entry point: ``python -m famou.monitor``.

Examples
--------
Follow the most recent experiment under ``famou_data/``::

    python -m famou.monitor --run famou_data/ --open

Follow one specific experiment and enable run control against a running api_server::

    python -m famou.monitor --run famou_data/circle_packing_abc123 \\
        --api-base http://127.0.0.1:8090 --job-id job1 --worker-id worker1
"""

from __future__ import annotations

import argparse
import sys

from famou.monitor.control import ControlProxy
from famou.monitor.server import serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m famou.monitor",
        description="Serve a live FaQ dashboard for a Famou experiment directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run",
        "-r",
        default="famou_data",
        help=(
            "Experiment directory, or a parent holding several -- in which case the "
            "most recently modified one is followed (default: famou_data)"
        ),
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Bind address. Local by default because /api/program serves every prompt "
            "and completion this run produced."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds between snapshots pushed over SSE (default: 1.0)",
    )
    parser.add_argument(
        "--stale-after",
        type=float,
        default=600.0,
        help=(
            "Seconds without a new artifact before the run is shown as not live "
            "(default: 600 -- a single slow LLM call was measured at 390s)"
        ),
    )
    parser.add_argument("--open", action="store_true", help="open a browser")

    control = parser.add_argument_group(
        "run control",
        "Forward start/pause/continue/stop to a running api_server. Omit --api-base "
        "to keep the dashboard strictly read-only.",
    )
    control.add_argument(
        "--api-base",
        help="Base URL of the running api_server, e.g. http://127.0.0.1:8090",
    )
    control.add_argument("--job-id", help="Job ID the api_server task was created with")
    control.add_argument("--worker-id", help="Worker ID the api_server task was created with")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    control = None
    if args.api_base:
        # A control proxy without both IDs would render buttons that answer "-1" on
        # every click, so refuse up front rather than shipping a broken bar.
        if not args.job_id or not args.worker_id:
            parser.error("--api-base also requires --job-id and --worker-id")
        control = ControlProxy(args.api_base, args.job_id, args.worker_id)

    try:
        serve(
            args.run,
            host=args.host,
            port=args.port,
            poll_interval=args.poll_interval,
            stale_after=args.stale_after,
            control=control,
            open_browser=args.open,
        )
    except FileNotFoundError as exc:
        # The directory not being there yet is the single most common mistake, and the
        # reader already explains it. One clear line beats a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nmonitor stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
