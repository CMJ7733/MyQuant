"""Incremental reader: one experiment directory -> one aggregated snapshot.

What is read, and why
---------------------
A Famou run writes six kinds of artifact into ``<base_path>/<experiment_id>/``.  Two
of them are the event streams this reader follows:

``results/<rollout_id>.json``
    Written once per rollout by ``Evolver`` via ``LocalStorage.save_result``.  Files
    are write-once, so "what is new" is "which filenames I have not seen".  Each
    record carries the iteration, island, status, failed module, the LLM request log
    for that rollout, and a full dump of the program it produced.
``programs/<program_id>.json``
    Written once per program.  The authoritative source for population statistics and
    the program count, because it also contains the seed programs, which never came
    from a rollout.

and two are append-only JSON-lines files, tailed by byte offset:

``experiment.jsonl``   the structured log stream (warnings and errors surface in the UI)
``llm_requests.log``   one record per LLM attempt: model, status, duration, tokens

The remaining two are read whole, on change: ``config.yaml`` and the newest
``experiment_checkpoint_<n>.json``.

Cost
----
``programs/*.json`` and the ``program`` field inside each rollout carry the full source
code and the full LLM prompt and response.  A long run is gigabytes of that.  The
reader therefore **discards code and prompt text as it ingests**, keeping only
accounting fields, and the detail endpoints re-read the one file being asked for.
Holding it all in memory would be gigabytes for text nobody is looking at.

Tailing strategy
----------------
JSONL files are read from a remembered byte offset, and only whole lines are consumed.
A partial trailing line (the writer was mid-``write``) stays in the buffer and is
completed on the next poll -- without that, a half-written record would raise a JSON
error every second on a busy run.  Truncation or replacement of a file, and
disappearance of a directory entry we had already seen, mean a new run is reusing the
directory: the reader rewinds and rebuilds from scratch rather than mixing two runs.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Deque, Dict, Iterable, List, Optional, Set, Tuple

#: Rolling window of recent rollouts kept for the activity feed.
_RECENT_ROLLOUTS = 200

#: Rolling window of recent log lines kept for the warning stack.
_RECENT_WARNINGS = 40

#: How many programs the "Top 程序" table shows.
_TOP_PROGRAMS = 25

#: Bytes immediately before a tail offset used to distinguish append from rewrite.
_TAIL_CHECKPOINT_BYTES = 4096

#: Prevent a continuously replaced directory from keeping one request in a retry loop.
_READ_ATTEMPTS = 3

#: Longest error/message text kept in a snapshot. Full text is in the detail views.
_MESSAGE_CLIP = 400

#: Funnel stages, in pipeline order. Mirrors what a rollout actually goes through in
#: ``Evolver._update_experiment``: a rollout is dispatched, may or may not produce a
#: program, the program may or may not evaluate, and may or may not be fully valid.
FUNNEL_STAGES: Tuple[Tuple[str, str], ...] = (
    ("dispatched", "发起 Rollout"),
    ("generated", "生成程序"),
    ("evaluated", "评估出分"),
    ("valid", "有效解"),
    ("improved", "刷新最优"),
)


# =============================================================================
# Snapshot data classes
# =============================================================================


@dataclass
class IslandState:
    """Population statistics for one island.

    The metrics deliberately match ``Evolver._log_population_stats_compact`` so the
    dashboard and ``experiment.log`` never disagree about the same island.
    """

    island_id: int
    n_programs: int = 0
    n_error: int = 0
    best_score: Optional[float] = None
    best_program_id: Optional[str] = None
    avg_score: Optional[float] = None
    max_generation: int = 0
    last_iteration: Optional[int] = None
    last_event_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "island_id": self.island_id,
            "n_programs": self.n_programs,
            "n_error": self.n_error,
            "best_score": self.best_score,
            "best_program_id": self.best_program_id,
            "avg_score": self.avg_score,
            "max_generation": self.max_generation,
            "last_iteration": self.last_iteration,
            "last_event_at": self.last_event_at,
        }


@dataclass
class RunState:
    """Aggregated snapshot served to the browser."""

    run_dir: str = ""
    experiment_id: str = ""
    experiment_name: str = ""
    started_at: Optional[float] = None
    last_event_at: Optional[float] = None
    live: bool = False
    """True when an artifact was written within ``stale_after`` seconds."""

    finished: bool = False
    """True when the latest checkpoint reached ``max_iterations``."""

    config: Dict[str, Any] = field(default_factory=dict)
    current_iteration: int = 0
    max_iterations: int = 0
    progress: float = 0.0

    islands: List[Dict[str, Any]] = field(default_factory=list)
    funnel: List[Dict[str, Any]] = field(default_factory=list)
    reject_counts: List[Dict[str, Any]] = field(default_factory=list)
    totals: Dict[str, Any] = field(default_factory=dict)
    best_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    llm_by_model: List[Dict[str, Any]] = field(default_factory=list)
    recent_rollouts: List[Dict[str, Any]] = field(default_factory=list)
    top_programs: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    control: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "started_at": self.started_at,
            "last_event_at": self.last_event_at,
            "live": self.live,
            "finished": self.finished,
            "config": self.config,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "progress": self.progress,
            "islands": self.islands,
            "funnel": self.funnel,
            "reject_counts": self.reject_counts,
            "totals": self.totals,
            "best_trajectory": self.best_trajectory,
            "llm_by_model": self.llm_by_model,
            "recent_rollouts": self.recent_rollouts,
            "top_programs": self.top_programs,
            "warnings": self.warnings,
            "control": self.control,
        }


# =============================================================================
# Incremental sources
# =============================================================================


@dataclass(frozen=True)
class _Batch:
    """One incremental read, including whether the source's epoch changed."""

    records: Tuple[Dict[str, Any], ...] = ()
    epoch_changed: bool = False


class _Tail:
    """Byte-offset tail over one append-only JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self._partial = b""
        self._identity: Optional[Tuple[int, int]] = None
        self._checkpoint: Optional[bytes] = None

    @staticmethod
    def _read_checkpoint(fh: BinaryIO, offset: int) -> bytes:
        position = fh.tell()
        start = max(0, offset - _TAIL_CHECKPOINT_BYTES)
        fh.seek(start)
        checkpoint = fh.read(offset - start)
        fh.seek(position)
        return checkpoint

    def rewind(self) -> None:
        """Forget the prior file epoch and read the current file from byte zero."""
        self.offset = 0
        self._partial = b""
        self._identity = None
        self._checkpoint = None

    def poll(self) -> _Batch:
        """Read appended records and report replacement of the exact opened file."""
        try:
            fh = open(self.path, "rb")
        except OSError:
            # The file not existing yet is normal (nothing logged so far). Only a file
            # that existed and then vanished is an epoch change.
            return _Batch(epoch_changed=self._identity is not None)

        with fh:
            opened = os.fstat(fh.fileno())
            identity = (opened.st_dev, opened.st_ino)
            if self._identity is not None:
                replaced = identity != self._identity or opened.st_size < self.offset
                if not replaced and self._checkpoint is not None:
                    replaced = self._read_checkpoint(fh, self.offset) != self._checkpoint
                if replaced:
                    return _Batch(epoch_changed=True)

            fh.seek(self.offset)
            chunk = fh.read()
            next_offset = fh.tell()
            checkpoint = self._read_checkpoint(fh, next_offset)

        text = self._partial + chunk
        lines = text.split(b"\n")
        # The last element is either b"" (chunk ended on a newline) or a record the
        # writer has not finished. Either way it is carried forward.
        partial = lines.pop()

        records: List[Dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # A record we will never be able to parse. Skipping one corrupt line
                # beats stalling the whole feed on it.
                continue
            if isinstance(value, dict):
                records.append(value)

        self.offset = next_offset
        self._partial = partial
        self._identity = identity
        self._checkpoint = checkpoint
        return _Batch(records=tuple(records))


class _DirTail:
    """Incremental scan over a directory of write-once JSON files.

    ``results/`` and ``programs/`` are the framework's real event streams, but they are
    directories rather than JSONL files, so there is no byte offset to remember. What
    is remembered instead is the set of filenames already ingested. A file we had seen
    disappearing means the directory was reset for a new run, which is reported as an
    epoch change exactly like a truncated JSONL file.

    The directory is listed on **every** poll, with no mtime fast-path. That looks
    wasteful and is deliberate: a directory mtime is not a dependable change signal.
    On several filesystems this framework actually runs on (overlayfs in a container,
    a CIFS/9p-mounted volume) creating an entry does not move the parent's mtime at
    all, so a run producing a rollout a second would have gone invisible until some
    unrelated write happened to bump it -- a dashboard that silently stops updating,
    which is worse than a slow one. A ``scandir`` over even tens of thousands of names
    is a few milliseconds once a second, and only the names are touched: the JSON is
    read for new entries only.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._seen: Set[str] = set()

    def rewind(self) -> None:
        """Forget every ingested filename and rescan from empty."""
        self._seen = set()

    def poll(self) -> _Batch:
        """Return the records of files not seen before, oldest first."""
        try:
            names = {p.name for p in self.path.iterdir() if p.suffix == ".json"}
        except OSError:
            # The directory not existing yet is normal (nothing produced so far). Only
            # one that existed and then vanished is an epoch change.
            return _Batch(epoch_changed=bool(self._seen))

        if not self._seen <= names:
            return _Batch(epoch_changed=True)

        records: List[Dict[str, Any]] = []
        for name in sorted(names - self._seen):
            path = self.path / name
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                # Most likely a file being written right now. Leave it out of ``_seen``
                # so the next poll picks it up once it is complete.
                continue
            if isinstance(value, dict):
                records.append(value)
                self._seen.add(name)

        # Written-once files carry their own creation time, which orders them far more
        # reliably than the filesystem or the ID does.
        records.sort(key=lambda r: _finite(r.get("created_at")) or 0.0)
        return _Batch(records=tuple(records))


# =============================================================================
# Reader
# =============================================================================


class ExperimentReader:
    """Aggregates a Famou experiment directory into a :class:`RunState`, incrementally.

    Call :meth:`poll` on a timer. It is cheap when nothing has changed (four ``stat``
    calls) and O(new records) when something has.

    Parameters
    ----------
    run_dir:
        An experiment directory, or a parent holding several -- in which case the most
        recently modified one is followed, which is what ``--run famou_data/`` means in
        practice.
    stale_after:
        Seconds without a new artifact before the run is reported as not live.

        The binding constraint is the slowest single LLM call, because nothing at
        all is written between dispatching a rollout and its program landing. On a
        real run against a hosted endpoint whose median call was ~10s, one call
        took **390s** and still succeeded; at the 180s inherited from cogalpha the
        dashboard declared a perfectly healthy run stopped. 600s covers that with
        margin. Raise it further for a slower endpoint rather than learning to
        ignore the indicator.
    """

    def __init__(self, run_dir: str | Path, stale_after: float = 600.0) -> None:
        self.root = Path(run_dir).expanduser()
        self.stale_after = stale_after
        self.path = self._resolve(self.root)

        self._results = _DirTail(self.path / "results")
        self._programs = _DirTail(self.path / "programs")
        self._log = _Tail(self.path / "experiment.jsonl")
        self._llm = _Tail(self.path / "llm_requests.log")

        self._reset_accumulated_state()

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _resolve(root: Path) -> Path:
        """Pick the experiment directory to follow.

        A directory holding ``config.yaml`` or ``programs/`` is itself an experiment;
        otherwise the newest immediate subdirectory that looks like one is chosen, so
        ``--run famou_data/`` follows the latest experiment without having to name it.

        Resolution happens once, at construction. Re-resolving on every poll would mean
        silently jumping between experiments mid-session, which makes a dashboard you
        cannot trust. Start the monitor after the run.
        """
        if not root.exists():
            raise FileNotFoundError(
                f"experiment directory not found: {root}\n"
                "Start the run first; the monitor attaches to an existing directory."
            )
        if ExperimentReader._looks_like_experiment(root):
            return root
        try:
            candidates = [
                d for d in root.iterdir()
                if d.is_dir() and ExperimentReader._looks_like_experiment(d)
            ]
        except OSError:
            candidates = []
        if not candidates:
            raise FileNotFoundError(
                f"no experiment directory under {root}\n"
                "Expected a directory containing config.yaml or programs/. If a run is "
                "starting right now, give it a few seconds and retry."
            )
        return max(candidates, key=lambda d: d.stat().st_mtime)

    @staticmethod
    def _looks_like_experiment(path: Path) -> bool:
        return (path / "config.yaml").exists() or (path / "programs").is_dir()

    def _reset_accumulated_state(self) -> None:
        """Clear every value derived from the artifact streams."""
        self._islands: Dict[int, IslandState] = {}
        self._island_scores: Dict[int, List[float]] = {}
        self._programs_seen: Dict[str, Dict[str, Any]] = {}
        self._funnel: Counter = Counter()
        self._reject: Counter = Counter()
        self._recent: Deque[Dict[str, Any]] = deque(maxlen=_RECENT_ROLLOUTS)
        self._warnings: Deque[Dict[str, Any]] = deque(maxlen=_RECENT_WARNINGS)
        self._llm_calls: Counter = Counter()
        self._llm_failed: Counter = Counter()
        self._llm_prompt_tokens: Counter = Counter()
        self._llm_response_tokens: Counter = Counter()
        self._llm_seconds: Counter = Counter()
        #: (request_id, attempt) of every LLM attempt already counted, so the copy in
        #: results/*.json and the line in llm_requests.log are not counted twice.
        self._llm_seen: Set[Tuple[str, int]] = set()
        self._best_trajectory: List[Dict[str, Any]] = []
        self._best_score: Optional[float] = None
        self._best_program_id: Optional[str] = None
        self._n_rollouts = 0
        self._first_event_at: Optional[float] = None
        self._last_event_at: Optional[float] = None
        self._config_cache: Optional[Dict[str, Any]] = None
        self._config_mtime_ns: Optional[int] = None
        self._checkpoint_cache: Dict[str, Any] = {}
        self._checkpoint_key: Optional[Tuple[str, int]] = None

    # --------------------------------------------------------------------- poll

    def poll(self) -> RunState:
        """Consume any new artifacts and return the current snapshot."""
        sources = (self._results, self._programs, self._log, self._llm)
        for _ in range(_READ_ATTEMPTS):
            batches = [source.poll() for source in sources]
            if not any(batch.epoch_changed for batch in batches):
                break

            # One reset source invalidates every batch: a new run reusing the directory
            # would otherwise have its programs folded in on top of the old run's
            # rollouts. Rebuild the whole set from scratch before applying anything.
            for source in sources:
                source.rewind()
            self._reset_accumulated_state()
        else:
            return self._snapshot()

        results, programs, log, llm = batches
        for record in programs.records:
            self._apply_program(record)
        for record in results.records:
            self._apply_rollout(record)
        for record in llm.records:
            self._apply_llm_request(record)
        for record in log.records:
            self._apply_log(record)
        return self._snapshot()

    # ------------------------------------------------------------------- ingest

    def _apply_program(self, record: Dict[str, Any]) -> None:
        """Fold one ``programs/*.json`` file into the population statistics.

        Only accounting fields are kept. ``code``, ``prompt``, ``response`` and
        ``thinking`` are dropped here and re-read by :meth:`program_detail` on demand.
        """
        program_id = record.get("id")
        if not isinstance(program_id, str) or program_id in self._programs_seen:
            return

        digest = _program_digest(record)
        self._programs_seen[program_id] = digest
        self._touch(digest.get("created_at"))

        island_id = digest["island_id"]
        if island_id is None:
            return
        island = self._islands.get(island_id)
        if island is None:
            island = IslandState(island_id=island_id)
            self._islands[island_id] = island
            self._island_scores[island_id] = []

        island.n_programs += 1
        island.max_generation = max(island.max_generation, digest["generation"] or 0)
        if digest["is_buggy"]:
            island.n_error += 1

        score = digest["combined_score"]
        if score is not None:
            self._island_scores[island_id].append(score)
            scores = self._island_scores[island_id]
            island.avg_score = sum(scores) / len(scores)
            if island.best_score is None or score > island.best_score:
                island.best_score = score
                island.best_program_id = program_id

    def _apply_rollout(self, record: Dict[str, Any]) -> None:
        """Fold one ``results/*.json`` file into the funnel and the activity feed."""
        self._n_rollouts += 1
        self._funnel["dispatched"] += 1

        program = record.get("program") if isinstance(record.get("program"), dict) else None
        status = str(record.get("status") or "")
        island_id = _as_int(record.get("island_id"))
        iteration = _as_int(record.get("iteration")) or 0
        created_at = _finite(record.get("created_at"))
        completed_at = _finite(record.get("completed_at"))

        self._touch(completed_at or created_at)

        score: Optional[float] = None
        program_id: Optional[str] = None
        if program is not None:
            self._funnel["generated"] += 1
            program_id = program.get("id") if isinstance(program.get("id"), str) else None
            score = _finite(program.get("combined_score"))
            validity = _finite(program.get("validity"))
            if score is not None:
                self._funnel["evaluated"] += 1
            # ``Program.is_buggy`` is ``validity < 1.0``; an unset validity means the
            # evaluator never reported one, which is not evidence of a bug.
            if validity is None or validity >= 1.0:
                self._funnel["valid"] += 1
            if score is not None and (self._best_score is None or score > self._best_score):
                self._best_score = score
                self._best_program_id = program_id
                self._funnel["improved"] += 1
                self._best_trajectory.append(
                    {
                        "iteration": iteration,
                        "best_score": score,
                        "program_id": program_id,
                        "at": completed_at or created_at,
                    }
                )

        failed_module = record.get("failed_module")
        if status != "success":
            self._reject[str(failed_module or status or "unknown")] += 1

        if island_id is not None:
            island = self._islands.get(island_id)
            if island is not None:
                island.last_iteration = iteration
                island.last_event_at = completed_at or created_at

        seconds: Optional[float] = None
        if isinstance(record.get("stats"), dict):
            seconds = _finite(record["stats"].get("execution_time"))
        if seconds is None and created_at is not None and completed_at is not None:
            seconds = max(0.0, completed_at - created_at)

        self._recent.append(
            {
                "rollout_id": record.get("rollout_id"),
                "rollout_name": record.get("rollout_name"),
                "iteration": iteration,
                "island_id": island_id,
                "status": status,
                "failed_module": failed_module,
                "error_message": _clip(record.get("error_message")),
                "program_id": program_id,
                "score": score,
                "seconds": round(seconds, 2) if seconds is not None else None,
                "at": completed_at or created_at,
            }
        )

        # A rollout carries the LLM attempts made on its behalf, which is the only
        # attribution available when the run used a remote backend: the worker's
        # ``llm_requests.log`` lives on the worker, not here.
        for entry in record.get("llm_request_logs") or []:
            if isinstance(entry, dict):
                self._apply_llm_request(entry)

    def _apply_llm_request(self, record: Dict[str, Any]) -> None:
        """Fold one LLM attempt into the per-model cost table.

        The same attempt reaches this method twice. ``Evolver._append_rollout_llm_request_logs``
        appends a rollout's buffered entries into ``llm_requests.log`` -- the very file
        this reader also tails -- so an entry is present both inside ``results/*.json``
        and as a line in the log. Both are read because neither alone is complete: on a
        Ray backend the worker's direct writes never reach the driver's log, and on a
        threadpool backend the rollout's copy is empty. Deduplication is therefore by
        ``(request_id, attempt)``, which ``BaseLLMClient`` assigns per logical call and
        per retry.

        There is no role/module field on an entry (see
        ``BaseLLMClient._build_request_log_entry``), so cost is aggregated by model.
        """
        request_id = record.get("request_id")
        if isinstance(request_id, str) and request_id:
            key = (request_id, _as_int(record.get("attempt")) or 0)
            if key in self._llm_seen:
                return
            self._llm_seen.add(key)

        # A completed LLM call is proof the run is alive, and during a slow one it is
        # the ONLY proof: nothing else is written between dispatching a rollout and
        # its program landing. Measured on a real run, a single call reached 390s
        # against an endpoint whose median was ~10s, which is long enough that
        # ignoring this signal made a healthy run look stopped.
        #
        # ``request_time`` is an ISO string stamped when the attempt finished (see
        # ``_build_request_log_entry``), not the epoch float ``_touch`` takes.
        self._touch(_iso_to_epoch(record.get("request_time")))

        model = str(record.get("model") or "unknown")
        self._llm_calls[model] += 1
        if str(record.get("status") or "").lower() not in {"success", "ok", "200"}:
            self._llm_failed[model] += 1

        prompt_tokens = _as_int(record.get("prompt_tokens"))
        response_tokens = _as_int(record.get("response_tokens"))
        if prompt_tokens:
            self._llm_prompt_tokens[model] += prompt_tokens
        if response_tokens:
            self._llm_response_tokens[model] += response_tokens
        duration = _finite(record.get("duration_seconds"))
        if duration is not None:
            self._llm_seconds[model] += duration

    def _apply_log(self, record: Dict[str, Any]) -> None:
        """Keep warnings and errors from ``experiment.jsonl`` for the warning stack."""
        self._touch(_finite(record.get("timestamp")))
        level = str(record.get("level") or "").upper()
        if level not in {"WARNING", "ERROR"}:
            return
        self._warnings.append(
            {
                "level": level,
                "message": _clip(record.get("message")),
                "at": _finite(record.get("timestamp")),
            }
        )

    def _touch(self, when: Optional[float]) -> None:
        """Track the first and last moment this run produced anything."""
        if when is None:
            return
        if self._first_event_at is None or when < self._first_event_at:
            self._first_event_at = when
        if self._last_event_at is None or when > self._last_event_at:
            self._last_event_at = when

    # ----------------------------------------------------------------- snapshot

    def _snapshot(self) -> RunState:
        config = self._read_config()
        checkpoint = self._read_checkpoint()

        max_iterations = _as_int(config.get("max_iterations")) or 0
        current_iteration = _as_int(checkpoint.get("current_iteration")) or 0

        # The checkpoint is the framework's own verdict on which program is best, so it
        # wins over the reader's running maximum whenever one has been written.
        best_score = _finite(checkpoint.get("best_program_score"))
        best_program_id = checkpoint.get("best_program_id") or None
        if best_score is None:
            best_score = self._best_score
            best_program_id = self._best_program_id

        last_event_at = self._last_event_at
        live = (
            last_event_at is not None
            and (time.time() - last_event_at) < self.stale_after
        )
        finished = max_iterations > 0 and current_iteration >= max_iterations

        return RunState(
            run_dir=str(self.path),
            experiment_id=self.path.name,
            experiment_name=str(config.get("name") or self.path.name),
            started_at=self._first_event_at,
            last_event_at=last_event_at,
            live=live,
            finished=finished,
            config=config,
            current_iteration=current_iteration,
            max_iterations=max_iterations,
            progress=(current_iteration / max_iterations) if max_iterations > 0 else 0.0,
            islands=[
                self._islands[key].to_dict() for key in sorted(self._islands)
            ],
            funnel=[
                {"key": key, "label": label, "count": self._funnel.get(key, 0)}
                for key, label in FUNNEL_STAGES
            ],
            reject_counts=[
                {"reason": reason, "count": count}
                for reason, count in self._reject.most_common()
            ],
            totals=self._totals(best_score, best_program_id),
            best_trajectory=list(self._best_trajectory),
            llm_by_model=self._llm_by_model(),
            recent_rollouts=list(reversed(self._recent)),
            top_programs=self._top_programs(),
            warnings=list(reversed(self._warnings)),
            control={},
        )

    def _totals(
        self, best_score: Optional[float], best_program_id: Optional[str]
    ) -> Dict[str, Any]:
        elapsed = None
        if self._first_event_at is not None and self._last_event_at is not None:
            elapsed = round(self._last_event_at - self._first_event_at, 1)
        return {
            "n_rollouts": self._n_rollouts,
            "n_programs": len(self._programs_seen),
            "n_islands": len(self._islands),
            "best_score": best_score,
            "best_program_id": best_program_id,
            "llm_calls": sum(self._llm_calls.values()),
            "prompt_tokens": sum(self._llm_prompt_tokens.values()),
            "response_tokens": sum(self._llm_response_tokens.values()),
            "elapsed": elapsed,
        }

    def _llm_by_model(self) -> List[Dict[str, Any]]:
        rows = []
        for model, calls in self._llm_calls.most_common():
            rows.append(
                {
                    "model": model,
                    "calls": calls,
                    "n_failed": self._llm_failed.get(model, 0),
                    "prompt_tokens": self._llm_prompt_tokens.get(model, 0),
                    "response_tokens": self._llm_response_tokens.get(model, 0),
                    "avg_seconds": round(self._llm_seconds.get(model, 0.0) / calls, 2)
                    if calls
                    else None,
                }
            )
        return rows

    def _top_programs(self) -> List[Dict[str, Any]]:
        scored = [
            digest for digest in self._programs_seen.values()
            if digest["combined_score"] is not None
        ]
        scored.sort(key=lambda d: d["combined_score"], reverse=True)
        return scored[:_TOP_PROGRAMS]

    # ------------------------------------------------------------ whole-file reads

    def _read_config(self) -> Dict[str, Any]:
        """Read and digest ``config.yaml``, re-parsing only when it changes."""
        path = self.path / "config.yaml"
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            return self._config_cache or {}
        if self._config_cache is not None and self._config_mtime_ns == mtime_ns:
            return self._config_cache

        try:
            import yaml

            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001 - a missing/odd config must not break the UI
            raw = {}

        self._config_cache = _config_digest(raw)
        self._config_mtime_ns = mtime_ns
        return self._config_cache

    def _read_checkpoint(self) -> Dict[str, Any]:
        """Read the highest-numbered ``experiment_checkpoint_<n>.json``."""
        newest = self._newest_checkpoint()
        if newest is None:
            return self._checkpoint_cache

        try:
            key = (newest.name, newest.stat().st_mtime_ns)
        except OSError:
            return self._checkpoint_cache
        if self._checkpoint_key == key:
            return self._checkpoint_cache

        try:
            raw = json.loads(newest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            # Being written right now. Keep the previous one and retry next poll.
            return self._checkpoint_cache
        if not isinstance(raw, dict):
            return self._checkpoint_cache

        self._checkpoint_cache = {
            "current_iteration": raw.get("current_iteration"),
            "best_program_id": raw.get("best_program_id"),
            "best_program_score": raw.get("best_program_score"),
            "checkpoint_file": newest.name,
        }
        self._checkpoint_key = key
        return self._checkpoint_cache

    def _newest_checkpoint(self) -> Optional[Path]:
        best: Optional[Tuple[int, Path]] = None
        try:
            entries: Iterable[Path] = self.path.glob("experiment_checkpoint_*.json")
        except OSError:
            return None
        for path in entries:
            suffix = path.stem.rsplit("_", 1)[-1]
            if not suffix.isdigit():
                continue
            number = int(suffix)
            if best is None or number > best[0]:
                best = (number, path)
        return best[1] if best else None

    # ------------------------------------------------------------------- details

    def island_detail(self, island_id: int) -> Optional[Dict[str, Any]]:
        """One island's statistics plus every program assigned to it."""
        island = self._islands.get(island_id)
        if island is None:
            return None
        members = [
            digest for digest in self._programs_seen.values()
            if digest["island_id"] == island_id
        ]
        members.sort(key=lambda d: (d["iteration"] or 0, d["id"]))
        detail = island.to_dict()
        detail["programs"] = members
        return detail

    def program_detail(self, program_id: str) -> Optional[Dict[str, Any]]:
        """One program: source, metrics, the LLM exchange that produced it, lineage.

        The heavy text fields were dropped at ingest, so this re-reads the file.
        """
        record = self._read_program_file(program_id)
        if record is None:
            return None

        detail = _program_digest(record)
        detail.update(
            {
                "code": record.get("code") or self._read_program_code(record),
                "system_prompt": record.get("system_prompt"),
                "prompt": record.get("prompt"),
                "response": record.get("response"),
                "thinking": record.get("thinking"),
                "error_info": record.get("error_info"),
                "language": record.get("language"),
                "meta": record.get("meta") or {},
                "lineage": self._lineage(program_id),
            }
        )
        return detail

    def rollout_detail(self, rollout_id: str) -> Optional[Dict[str, Any]]:
        """One rollout's stored record, minus the embedded program dump.

        The program is served by ``program_detail`` instead of being duplicated here;
        the dump inside a rollout is the same object and is the bulk of the file.
        """
        path = self.path / "results" / f"{rollout_id}.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None
        if not isinstance(record, dict):
            return None

        program = record.pop("program", None)
        record["generated_program_id"] = (
            program.get("id") if isinstance(program, dict) else record.get("generated_program_id")
        )
        return record

    def iteration_programs(self, iteration: int) -> List[Dict[str, Any]]:
        """Every program created in one iteration, across all islands."""
        members = [
            digest for digest in self._programs_seen.values()
            if digest["iteration"] == iteration
        ]
        members.sort(key=lambda d: (d["island_id"] if d["island_id"] is not None else -1, d["id"]))
        return members

    def _lineage(self, program_id: str) -> List[Dict[str, Any]]:
        """Walk ``parent_id`` up to the seed, newest first.

        Guarded against a cycle: a corrupted archive must not hang a request.
        """
        chain: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        current: Optional[str] = program_id
        while current and current not in seen:
            seen.add(current)
            digest = self._programs_seen.get(current)
            if digest is None:
                record = self._read_program_file(current)
                if record is None:
                    break
                digest = _program_digest(record)
            chain.append(digest)
            current = digest.get("parent_id")
        return chain

    def _read_program_file(self, program_id: str) -> Optional[Dict[str, Any]]:
        path = self.path / "programs" / f"{program_id}.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None
        return record if isinstance(record, dict) else None

    def _read_program_code(self, record: Dict[str, Any]) -> Optional[str]:
        """Read the sibling source file when the JSON has no inline ``code``.

        ``LocalStorage.save_program`` writes the source next to the JSON using the
        program's own ``file_extension`` (``.h`` for a C++ header, and so on), so the
        extension is taken from the record rather than assumed to be ``.py``.
        """
        program_id = record.get("id")
        if not isinstance(program_id, str):
            return None
        extension = record.get("file_extension") or _LANGUAGE_EXTENSIONS.get(
            str(record.get("language") or "python").lower(), ".py"
        )
        path = self.path / "programs" / f"{program_id}{extension}"
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None


# =============================================================================
# Record helpers
# =============================================================================

#: Mirrors ``LocalStorage.LANGUAGE_EXTENSIONS`` for the fallback path only; a program
#: that recorded its own ``file_extension`` never reaches this table.
_LANGUAGE_EXTENSIONS: Dict[str, str] = {
    "python": ".py",
    "cpp": ".cpp",
    "c": ".c",
    "java": ".java",
    "rust": ".rs",
    "go": ".go",
}


def _program_digest(record: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a stored program to the fields a snapshot needs.

    Everything heavy -- source, prompt, response, thinking, feature vector -- is left
    behind here on purpose; see the module docstring.
    """
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    island_id = _as_int(meta.get("island_id"))
    if island_id is None:
        island_id = _as_int(meta.get("assigned_island"))
    validity = _finite(record.get("validity"))
    return {
        "id": record.get("id"),
        "generation": _as_int(record.get("generation")) or 0,
        "iteration": _as_int(record.get("iteration")) or 0,
        "island_id": island_id,
        "parent_id": record.get("parent_id"),
        "combined_score": _finite(record.get("combined_score")),
        "validity": validity,
        "is_buggy": validity is not None and validity < 1.0,
        "metrics": record.get("metrics") if isinstance(record.get("metrics"), dict) else {},
        "error": _clip(record.get("error_info")),
        "created_at": _finite(record.get("created_at")),
    }


def _config_digest(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the handful of settings the header shows out of the saved config."""
    experiment = raw.get("experiment") if isinstance(raw.get("experiment"), dict) else {}
    island = experiment.get("island") if isinstance(experiment.get("island"), dict) else {}
    infra = raw.get("infrastructure") if isinstance(raw.get("infrastructure"), dict) else {}
    llm = infra.get("llm") if isinstance(infra.get("llm"), dict) else {}
    backend = infra.get("backend") if isinstance(infra.get("backend"), dict) else {}
    return {
        "name": experiment.get("name"),
        "max_iterations": experiment.get("max_iterations"),
        "population_size": island.get("population_size"),
        "num_islands": island.get("num_islands"),
        "strategy": experiment.get("strategy"),
        "language": experiment.get("language"),
        "task_description": _clip(experiment.get("task_description"), 1200),
        "llm_provider": llm.get("provider"),
        "llm_model": llm.get("model"),
        "backend": backend.get("mode"),
    }


def _finite(value: Any) -> Optional[float]:
    """Return a float only when the value is a real, finite number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clip(value: Any, limit: int = _MESSAGE_CLIP) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _iso_to_epoch(value: Any) -> Optional[float]:
    """Parse an ISO-8601 timestamp into epoch seconds, or None.

    ``llm_requests.log`` stamps times as ISO strings while every other stream uses
    epoch floats, so this is the one place a conversion is needed.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        import datetime

        return datetime.datetime.fromisoformat(value).timestamp()
    except (ValueError, OSError, OverflowError):
        return None
