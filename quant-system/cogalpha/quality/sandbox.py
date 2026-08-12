"""Restricted execution of generated alpha code.

The paper says only that code is "executed in a restricted sandbox" (§A.3).  This
module is that sandbox, built to three requirements agreed for this
implementation: a separate process, no network, and a hard ceiling on cost.

Design
------
One forked worker evaluates a *batch* of alphas and streams a compact result per
alpha back over a queue.  Two consequences follow, both deliberate:

* The panel never crosses the process boundary.  With the ``fork`` start method
  the child inherits it copy-on-write, and only per-alpha summaries (metrics,
  NaN ratios, error text) travel back — a full factor matrix for a 300-name,
  3400-day panel is 8 MB, so returning raw values for a 96-alpha children pool
  would move ~800 MB per generation.
* A hang is attributable.  Because results stream as they finish, a parent that
  times out knows exactly which alpha was in flight, kills the process group,
  and restarts the worker on the remainder.  A single batch-level timeout would
  discard the whole generation and blame nobody.

The confinement itself:

* ``setrlimit`` on address space, CPU seconds, file size (0 — no writes at all)
  and process count (no forking out);
* ``socket`` primitives replaced with raisers, so an attempted call fails loudly
  instead of quietly reaching the network;
* the alpha runs with a restricted ``__builtins__`` and an ``__import__`` that
  honours the allow-list, so ``import os`` fails inside the sandbox even if the
  static audit were bypassed.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as queue_mod
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from cogalpha.data.panel import Panel

#: Builtins an alpha may use.  Everything else — including ``open`` and
#: ``__import__`` — is either absent or replaced.
_SAFE_BUILTINS: Tuple[str, ...] = (
    "abs", "all", "any", "bool", "callable", "dict", "divmod", "enumerate",
    "filter", "float", "format", "frozenset", "getattr", "hasattr", "int",
    "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min",
    "next", "pow", "print", "range", "repr", "reversed", "round", "set",
    "setattr", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
    "True", "False", "None", "Exception", "ValueError", "TypeError",
    "KeyError", "IndexError", "ZeroDivisionError", "ArithmeticError",
    "RuntimeError", "AttributeError", "StopIteration", "NotImplementedError",
)


@dataclass
class ExecOutcome:
    """What the sandbox learned about one alpha."""

    alpha_id: str
    ok: bool
    error: str = ""
    error_type: str = ""
    #: Payload filled by the job callback (metrics, diagnostics, probe results).
    payload: Dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Picklable payload sent back over the queue. Must stay small: it crosses
        the process boundary once per alpha."""
        return {
            "alpha_id": self.alpha_id,
            "ok": self.ok,
            "error": self.error,
            "error_type": self.error_type,
            "payload": self.payload,
            "seconds": self.seconds,
        }


class SandboxError(RuntimeError):
    """Raised only for failures of the sandbox itself, never of alpha code."""


# --------------------------------------------------------------------- child side


def _install_limits(cpu_seconds: int) -> None:
    """Apply resource limits and sever the network. Child process only.

    Note what is *not* here: an address-space cap.  ``RLIMIT_AS``/``RLIMIT_DATA``
    limit *virtual* address space, and importing numpy, pandas and qlib on a
    many-core host reserves ~5 GB of it (glibc allocates up to ``8 x ncores``
    malloc arenas, each reserving 64 MB) while resident memory is only ~450 MB.
    Any cap loose enough to admit the interpreter is far too loose to constrain a
    runaway alpha, and any cap tight enough to constrain one kills the import.
    Memory is therefore enforced by the parent's RSS watchdog
    (:class:`_MemoryWatchdog`), which measures what actually gets used.
    """
    import resource

    for res, limit in (
        (resource.RLIMIT_CPU, cpu_seconds),
        (resource.RLIMIT_FSIZE, 0),
        (resource.RLIMIT_NPROC, 64),
        (resource.RLIMIT_CORE, 0),
    ):
        try:
            hard = resource.getrlimit(res)[1]
            value = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
            resource.setrlimit(res, (value, hard))
        except (ValueError, OSError):
            # A limit we cannot tighten is covered by the parent's own accounting
            # (wall-clock timeout, RSS watchdog) rather than silently assumed.
            continue

    _disable_network()



def _disable_network() -> None:
    """Replace socket entry points with raisers."""
    import socket

    def _blocked(*_args: Any, **_kwargs: Any):
        raise PermissionError("network access is disabled inside the alpha sandbox")

    socket.socket = _blocked  # type: ignore[assignment]
    socket.create_connection = _blocked  # type: ignore[assignment]
    socket.create_server = _blocked  # type: ignore[assignment]
    socket.getaddrinfo = _blocked  # type: ignore[assignment]
    socket.gethostbyname = _blocked  # type: ignore[assignment]


def _make_namespace(allowed_imports: Sequence[str]) -> Dict[str, Any]:
    """Globals for the alpha: pre-bound libraries and a filtered importer."""
    import builtins
    import importlib

    roots = {m.split(".")[0] for m in allowed_imports}

    def guarded_import(name: str, globals_=None, locals_=None, fromlist=(), level=0):
        """Replacement for ``__import__`` enforcing the allow-list inside the sandbox.

        Second line of defence: the static audit already rejects a bad import, so
        reaching here means the audit was bypassed or a name was built dynamically.
        """
        root = name.split(".")[0]
        if level != 0:
            raise ImportError("relative imports are not allowed in the sandbox")
        if root not in roots:
            raise ImportError(
                f"import of '{name}' is blocked; allowed roots: {sorted(roots)}"
            )
        return importlib.__import__(name, globals_, locals_, fromlist, level)

    safe = {k: getattr(builtins, k) for k in _SAFE_BUILTINS if hasattr(builtins, k)}
    safe["__import__"] = guarded_import

    namespace: Dict[str, Any] = {
        "__builtins__": safe,
        "__name__": "cogalpha_sandbox",
        "np": np,
        "numpy": np,
        "pd": pd,
        "pandas": pd,
    }
    try:  # talib is optional; bind it when present so listings using it just work
        import talib

        namespace["talib"] = talib
    except ImportError:
        pass
    try:
        import math

        namespace["math"] = math
    except ImportError:  # pragma: no cover
        pass
    return namespace


def compile_alpha(
    code: str,
    name: str,
    allowed_imports: Sequence[str] = ("numpy", "pandas", "math", "scipy", "talib"),
) -> Callable[[pd.DataFrame], Any]:
    """Compile alpha source and return its function.

    Raises the compilation/lookup error unchanged: the quality checker wants the
    original message to hand to the repair agent.
    """
    namespace = _make_namespace(allowed_imports)
    exec(compile(code, f"<alpha:{name}>", "exec"), namespace)  # noqa: S102 - sandboxed
    fn = namespace.get(name)
    if fn is None:
        # The audit guarantees exactly one top-level def, but a rename earlier in the
        # pipeline can leave `name` stale. Recover by finding the single function the
        # exec defined -- filtering on __module__ excludes the pre-bound libraries.
        candidates = [
            v
            for k, v in namespace.items()
            if callable(v) and getattr(v, "__module__", None) == "cogalpha_sandbox"
        ]
        if len(candidates) != 1:
            raise NameError(f"alpha function '{name}' not found after exec")
        fn = candidates[0]
    if not callable(fn):
        raise TypeError(f"'{name}' is not callable")
    return fn


def apply_alpha(
    fn: Callable[[pd.DataFrame], Any],
    frames: Dict[str, pd.DataFrame],
    column: str,
) -> pd.DataFrame:
    """Run ``fn`` on every instrument and assemble a wide (date x instrument) frame.

    This is the loop that turns the *per-instrument* alpha contract into the
    *cross-sectional* matrix the metrics need.  It is also the hot path: called once
    per alpha per evaluation, and twice more per alpha by the leakage probe.

    Accepts a Series, a DataFrame (in which case ``column``, else the last column, is
    taken) or a 1-D ndarray, which covers every convention the paper's own listings
    use.

    A per-instrument failure is **fatal for the alpha** -- the exception propagates.
    Catching it per name and dropping that column would let a factor that works on a
    handful of stocks pass the coverage gate, which is exactly the sparse-factor
    failure the coverage check exists to catch.
    """
    columns: Dict[str, pd.Series] = {}
    for inst, frame in frames.items():
        result = fn(frame)
        if isinstance(result, pd.DataFrame):
            if column in result.columns:
                series = result[column]
            elif result.shape[1] >= 1:
                # A generated function that returns the whole frame usually appended
                # its factor last; taking the last column is the best guess.
                series = result.iloc[:, -1]
            else:
                raise ValueError("alpha returned an empty DataFrame")
        elif isinstance(result, pd.Series):
            series = result
        elif isinstance(result, np.ndarray):
            if result.ndim != 1 or len(result) != len(frame):
                raise ValueError(
                    f"alpha returned an array of shape {result.shape}, "
                    f"expected 1-D of length {len(frame)}"
                )
            series = pd.Series(result, index=frame.index)
        else:
            raise TypeError(
                f"alpha returned {type(result).__name__}, expected Series/DataFrame"
            )

        series = pd.to_numeric(series, errors="coerce")
        if len(series) != len(frame):
            series = series.reindex(frame.index)
        columns[inst] = pd.Series(
            np.asarray(series, dtype="float64"), index=frame.index
        )

    wide = pd.DataFrame(columns)
    wide.index.name = "date"
    wide.columns.name = "instrument"
    return wide.sort_index()


def _worker(
    conn_queue: "mp.Queue",
    panel: Panel,
    jobs: List[Tuple[str, str, str]],
    job_fn: Callable[..., Dict[str, Any]],
    job_kwargs: Dict[str, Any],
    cpu_seconds: int,
    allowed_imports: Sequence[str],
) -> None:  # pragma: no cover - runs in the child
    """Child entry point: evaluate each job, streaming outcomes as they finish."""
    os.setsid()
    _install_limits(cpu_seconds=cpu_seconds)


    try:
        frames = {inst: frame for inst, frame in panel.iter_instruments()}
    except Exception as exc:  # noqa: BLE001
        conn_queue.put(("__fatal__", f"failed to build instrument frames: {exc}"))
        return

    for alpha_id, name, code in jobs:
        started = time.time()
        # Announce the alpha before touching its code, so a parent that times out
        # can name the culprit even if the very first statement hangs.
        conn_queue.put(("__start__", alpha_id))
        try:
            fn = compile_alpha(code, name, allowed_imports=allowed_imports)
            values = apply_alpha(fn, frames, column=name)
            payload = job_fn(
                values=values,
                fn=fn,
                code=code,
                name=name,
                panel=panel,
                frames=frames,
                **job_kwargs,
            )
            outcome = ExecOutcome(
                alpha_id=alpha_id,
                ok=True,
                payload=payload,
                seconds=time.time() - started,
            )
        except BaseException as exc:  # noqa: BLE001 - report, never crash the batch
            outcome = ExecOutcome(
                alpha_id=alpha_id,
                ok=False,
                error=f"{type(exc).__name__}: {exc}"[:2000],
                error_type=type(exc).__name__,
                seconds=time.time() - started,
            )
        conn_queue.put(("__result__", outcome.to_dict()))

    conn_queue.put(("__done__", None))


# -------------------------------------------------------------------- parent side


class _MemoryWatchdog:
    """Polls a child's resident memory and reports when it crosses a ceiling.

    Measures RSS from ``/proc/<pid>/statm`` (summed over the process group, so a
    child that spawns is still accounted for).  RSS rather than virtual size,
    because virtual size on this interpreter is ~5 GB before any alpha runs -- see
    :func:`_install_limits`.
    """

    def __init__(self, pid: int, limit_mb: int) -> None:
        self.pid = pid
        self.limit_bytes = limit_mb * 1024 * 1024
        self.page_size = os.sysconf("SC_PAGE_SIZE")
        self.peak_bytes = 0

    def rss_bytes(self) -> int:
        """Current resident set size, or 0 once the process is gone."""
        try:
            with open(f"/proc/{self.pid}/statm", "r", encoding="ascii") as fh:
                fields = fh.read().split()
            rss = int(fields[1]) * self.page_size
        except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
            return 0
        self.peak_bytes = max(self.peak_bytes, rss)
        return rss

    def exceeded(self) -> bool:
        """True when the worker is over its RSS ceiling and should be killed."""
        return self.rss_bytes() > self.limit_bytes

    @property
    def peak_mb(self) -> float:
        """Highest RSS observed, reported in the kill message."""
        return self.peak_bytes / (1024 * 1024)


class SandboxRunner:
    """Runs alpha batches in a forked worker with per-alpha timeouts.

    Parameters
    ----------
    job_fn:
        Called in the child as ``job_fn(values=wide_frame, fn=compiled_callable,
        code=..., name=..., panel=..., frames=..., **job_kwargs)`` and must return
        a small, picklable dict.  This is where fitness metrics and leakage probes
        run, keeping heavy data in the child.
    timeout_s:
        Wall-clock budget for a single alpha.  Exceeding it kills the process
        group and the alpha is reported as a timeout.
    memory_mb:
        Resident-memory ceiling for the worker, enforced by polling rather than by
        ``setrlimit``; see :func:`_install_limits` for why.  Must exceed the
        baseline footprint of the interpreter plus the panel (~500 MB on a
        3400-day, 750-name CSI300 panel).
    """

    def __init__(
        self,
        job_fn: Callable[..., Dict[str, Any]],
        job_kwargs: Optional[Dict[str, Any]] = None,
        timeout_s: float = 60.0,
        memory_mb: int = 8192,
        allowed_imports: Sequence[str] = ("numpy", "pandas", "math", "scipy", "talib"),
        max_restarts: int = 200,
        poll_interval_s: float = 0.5,
    ) -> None:
        self.job_fn = job_fn
        self.job_kwargs = dict(job_kwargs or {})
        self.timeout_s = timeout_s
        self.memory_mb = memory_mb
        self.allowed_imports = tuple(allowed_imports)
        self.max_restarts = max_restarts
        self.poll_interval_s = poll_interval_s


    def run(
        self,
        panel: Panel,
        jobs: Sequence[Tuple[str, str, str]],
    ) -> Dict[str, ExecOutcome]:
        """Evaluate ``jobs`` (``(alpha_id, function_name, code)``) against ``panel``.

        Always returns one outcome per job, successful or not -- the caller relies on
        that, since a missing entry would be indistinguishable from a crash.

        The loop restarts the worker after each casualty and retries the remainder,
        so one pathological alpha costs one restart (~0.1 s of fork) rather than the
        whole generation.  ``max_restarts`` bounds the pathological case where every
        alpha kills the worker.
        """
        results: Dict[str, ExecOutcome] = {}
        pending = list(jobs)
        restarts = 0

        while pending and restarts <= self.max_restarts:
            done, culprit, fatal, reason = self._run_once(panel, pending)
            results.update(done)

            if fatal is not None:
                # The worker failed before running anything (bad panel, import
                # error). Retrying would fail identically, so fail the batch.
                for alpha_id, _, _ in pending:
                    if alpha_id not in results:
                        results[alpha_id] = ExecOutcome(
                            alpha_id=alpha_id,
                            ok=False,
                            error=fatal,
                            error_type="SandboxFatal",
                        )
                break

            remaining = [j for j in pending if j[0] not in results]
            if culprit is not None and culprit not in results:
                # Attribute the kill to the alpha that was in flight, and drop it
                # from the retry set -- otherwise it kills the next worker too.
                results[culprit] = ExecOutcome(
                    alpha_id=culprit,
                    ok=False,
                    error=reason or "killed by the sandbox",
                    error_type="Timeout" if reason and "budget" in reason else "Resource",
                )
                remaining = [j for j in remaining if j[0] != culprit]

            if len(remaining) == len(pending) and culprit is None:
                # No progress and nobody to blame: stop rather than spin.
                for alpha_id, _, _ in remaining:
                    results[alpha_id] = ExecOutcome(
                        alpha_id=alpha_id,
                        ok=False,
                        error="sandbox made no progress on this batch",
                        error_type="SandboxStalled",
                    )
                break

            pending = remaining
            restarts += 1

        return results

    # ------------------------------------------------------------------ internals

    def _run_once(
        self,
        panel: Panel,
        jobs: Sequence[Tuple[str, str, str]],
    ) -> Tuple[Dict[str, ExecOutcome], Optional[str], Optional[str], Optional[str]]:
        """One worker lifetime.

        Returns ``(results, killed_alpha, fatal_error, kill_reason)``.
        """
        try:
            ctx = mp.get_context("fork")
        except ValueError as exc:  # pragma: no cover - non-POSIX
            raise SandboxError(
                "the alpha sandbox requires the 'fork' start method (POSIX only)"
            ) from exc

        result_queue: "mp.Queue" = ctx.Queue()
        cpu_seconds = max(1, int(self.timeout_s * len(jobs)) + 5)
        process = ctx.Process(
            target=_worker,
            args=(
                result_queue,
                panel,
                list(jobs),
                self.job_fn,
                self.job_kwargs,
                cpu_seconds,
                self.allowed_imports,
            ),
            daemon=False,
        )
        process.start()
        watchdog = _MemoryWatchdog(process.pid, self.memory_mb) if process.pid else None

        results: Dict[str, ExecOutcome] = {}
        in_flight: Optional[str] = None
        killed: Optional[str] = None
        reason: Optional[str] = None
        fatal: Optional[str] = None
        exitcode: Optional[int] = None
        deadline = time.time() + self.timeout_s

        try:
            while True:
                # Poll on a short interval rather than blocking for the full budget:
                # the watchdog and the deadline both need to be checked while an
                # alpha is still running, not only when one finishes.
                try:
                    kind, payload = result_queue.get(timeout=self.poll_interval_s)
                except queue_mod.Empty:
                    now = time.time()
                    if watchdog is not None and watchdog.exceeded():
                        killed = in_flight
                        reason = (
                            f"resident memory reached {watchdog.peak_mb:.0f} MB, over the "
                            f"{self.memory_mb} MB ceiling"
                        )
                        break
                    if now >= deadline:
                        killed = in_flight
                        reason = f"exceeded the {self.timeout_s:.0f}s per-alpha budget"
                        break
                    if not process.is_alive():
                        # Died without sending __done__; attribution happens below.
                        break
                    continue

                # The three-message protocol: __start__ names the alpha about to run
                # (so a timeout is attributable), __result__ delivers it, __done__
                # ends the batch. The deadline resets on every message, making it a
                # per-alpha budget rather than a per-batch one.
                if kind == "__start__":
                    in_flight = payload
                    deadline = time.time() + self.timeout_s
                elif kind == "__result__":
                    outcome = ExecOutcome(**payload)
                    results[outcome.alpha_id] = outcome
                    in_flight = None
                    deadline = time.time() + self.timeout_s
                elif kind == "__fatal__":
                    fatal = str(payload)
                    break
                elif kind == "__done__":
                    break
        finally:
            exitcode = self._terminate(process)
            result_queue.close()
            result_queue.join_thread()

        if fatal is None and killed is None and exitcode not in (0, None):
            # Died without reporting: attribute it to whatever was running.
            killed = in_flight
            reason = f"worker died with exit code {exitcode} while evaluating this alpha"
            if killed is None and not results:
                fatal = f"sandbox worker exited with code {exitcode}"

        return results, killed, fatal, reason


    @staticmethod
    def _terminate(process: "mp.Process") -> Optional[int]:
        """Kill the worker's whole process group and return its exit code.

        The exit code has to be read before ``close()``, hence returning it here
        rather than letting the caller query the (by then closed) handle.
        """
        if process.pid is None:  # pragma: no cover - never started
            return None
        if process.is_alive():
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(process.pid), sig)
                except (ProcessLookupError, PermissionError):
                    break
                process.join(timeout=2.0)
                if not process.is_alive():
                    break
        process.join(timeout=2.0)
        if process.is_alive():  # pragma: no cover - should not happen after SIGKILL
            process.kill()
            process.join(timeout=2.0)
        exitcode = process.exitcode
        process.close()
        return exitcode

