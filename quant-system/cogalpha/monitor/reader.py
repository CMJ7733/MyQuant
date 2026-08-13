"""Incremental reader: two JSONL streams -> one aggregated snapshot.

Tailing strategy
----------------
Each file is read from a remembered byte offset, and only whole lines are consumed.
A partial trailing line (the writer was mid-``write`` when we read) is left in the
buffer and completed on the next poll — without that, a half-written record would
raise a JSON error every second on a busy run.

Truncation or replacement of a file (a new run reusing the directory) is detected by
the file shrinking below the remembered offset, which resets the reader.

Cost
----
``llm_calls.jsonl`` carries full prompts and responses, so it is the large file: a
full paper-spec run reaches a few GB.  The reader therefore **discards prompt and
response text** as it aggregates, keeping only accounting fields, and the API serves
the text on demand by seeking the one line asked for.  Holding it all in memory would
be gigabytes for information nobody is looking at.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Deque, Dict, List, Optional, Tuple

#: Rolling window of recent calls kept for the activity feed.
_RECENT_CALLS = 60

#: Rolling window of recent calls kept for each agent detail view.
_AGENT_RECENT_CALLS = 200

#: Bytes immediately before a tail offset used to distinguish append from rewrite.
_TAIL_CHECKPOINT_BYTES = 4096

#: Prevent a continuously replaced archive from keeping one request in a retry loop.
_TAIL_READ_ATTEMPTS = 3

#: Checker stages in pipeline order. The funnel is drawn in this order, and a stage
#: absent from a run's rejection counts simply contributes no drop.
STAGE_ORDER: Tuple[str, ...] = (
    "code_quality",
    "code_repair",
    "judge",
    "logic_improvement",
    "static_audit",
    "execute",
    "numeric_stability",
    "leakage_unit_test",
)

#: Roles a call can have, in the order they appear in a generation.
ROLE_ORDER: Tuple[str, ...] = (
    "generate",
    "mutate",
    "crossover",
    "code_quality",
    "repair",
    "judge",
    "improve",
    "analyse",
)


@dataclass
class AgentState:
    """Everything known about one of the 21 agents in this run."""

    name: str
    level: int
    selected: bool = False
    """False for the 8 agents the golden-ratio selection did not pick (§B.8)."""

    status: str = "queued"
    """``queued`` | ``running`` | ``done``."""

    generations: int = 0
    n_generated: int = 0
    n_passed: int = 0
    n_qualified: int = 0
    n_elite: int = 0
    best_score: Optional[float] = None
    best_rank_ic: Optional[float] = None
    llm_calls: int = 0
    seconds: float = 0.0
    stopped_early: Optional[str] = None
    elite_trajectory: List[Optional[float]] = field(default_factory=list)
    current_generation: Optional[int] = None
    current_cycle: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level,
            "selected": self.selected,
            "status": self.status,
            "generations": self.generations,
            "n_generated": self.n_generated,
            "n_passed": self.n_passed,
            "n_qualified": self.n_qualified,
            "n_elite": self.n_elite,
            "best_score": self.best_score,
            "best_rank_ic": self.best_rank_ic,
            "llm_calls": self.llm_calls,
            "seconds": round(self.seconds, 1),
            "stopped_early": self.stopped_early,
            "elite_trajectory": self.elite_trajectory,
        }


@dataclass
class RunState:
    """Aggregated snapshot served to the browser."""

    run_dir: str = ""
    run_name: str = ""
    started_at: Optional[float] = None
    last_event_at: Optional[float] = None
    live: bool = False
    """True when a record arrived within ``stale_after`` seconds."""

    finished: bool = False
    """True when ``summary.json`` exists -- the run saved itself and ended."""

    config: Dict[str, Any] = field(default_factory=dict)
    panel: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    agents: List[Dict[str, Any]] = field(default_factory=list)
    current_agent: Optional[str] = None
    current_generation: Optional[int] = None
    current_cycle: Optional[int] = None
    agents_done: int = 0
    agents_total: int = 0

    generations_seen: int = 0
    generations_planned: int = 0
    funnel: List[Dict[str, Any]] = field(default_factory=list)
    latest_generation: Dict[str, Any] = field(default_factory=dict)
    funnel_generation: Dict[str, Any] = field(default_factory=dict)
    """Set only when the funnel is showing an *earlier* generation than the latest,
    because the latest produced nothing. Lets the UI say so instead of implying the
    funnel is current."""

    totals: Dict[str, Any] = field(default_factory=dict)
    calls_by_role: List[Dict[str, Any]] = field(default_factory=list)
    recent_calls: List[Dict[str, Any]] = field(default_factory=list)
    elite_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    token_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    tier_counts: Dict[str, int] = field(default_factory=dict)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    plateau: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "run_name": self.run_name,
            "started_at": self.started_at,
            "last_event_at": self.last_event_at,
            "live": self.live,
            "finished": self.finished,
            "config": self.config,
            "panel": self.panel,
            "summary": self.summary,
            "agents": self.agents,
            "current_agent": self.current_agent,
            "current_generation": self.current_generation,
            "current_cycle": self.current_cycle,
            "agents_done": self.agents_done,
            "agents_total": self.agents_total,
            "generations_seen": self.generations_seen,
            "generations_planned": self.generations_planned,
            "funnel": self.funnel,
            "latest_generation": self.latest_generation,
            "funnel_generation": self.funnel_generation,
            "totals": self.totals,
            "calls_by_role": self.calls_by_role,
            "recent_calls": self.recent_calls,
            "elite_trajectory": self.elite_trajectory,
            "token_trajectory": self.token_trajectory,
            "tier_counts": self.tier_counts,
            "candidates": self.candidates,
            "plateau": self.plateau,
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class _TailBatch:
    """One tail read, including whether its file epoch changed."""

    records: Tuple[Dict[str, Any], ...] = ()
    epoch_changed: bool = False


class _Tail:
    """Byte-offset tail over one append-only JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self._partial = b""
        self._identity: Optional[Tuple[int, int]] = None
        self._mtime_ns: Optional[int] = None
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
        self._mtime_ns = None
        self._checkpoint = None

    def poll(self) -> _TailBatch:
        """Read appended records and report replacement of the exact opened file."""
        try:
            fh = open(self.path, "rb")
        except OSError:
            return _TailBatch(epoch_changed=self._identity is not None)

        with fh:
            opened = os.fstat(fh.fileno())
            identity = (opened.st_dev, opened.st_ino)
            if self._identity is not None:
                changed = identity != self._identity or opened.st_size < self.offset
                if not changed and self._checkpoint is not None:
                    changed = self._read_checkpoint(fh, self.offset) != self._checkpoint
                if not changed and opened.st_size == self.offset:
                    changed = opened.st_mtime_ns != self._mtime_ns
                if changed:
                    return _TailBatch(epoch_changed=True)

            fh.seek(self.offset)
            chunk = fh.read()
            next_offset = fh.tell()
            after_read = os.fstat(fh.fileno())
            checkpoint = self._read_checkpoint(fh, next_offset)

        try:
            current = self.path.stat()
        except OSError:
            return _TailBatch(epoch_changed=True)
        if (current.st_dev, current.st_ino) != identity or current.st_size < next_offset:
            return _TailBatch(epoch_changed=True)
        if (
            current.st_size == after_read.st_size
            and current.st_mtime_ns != after_read.st_mtime_ns
        ):
            return _TailBatch(epoch_changed=True)

        expected_checkpoint = ((self._checkpoint or b"") + chunk)[-_TAIL_CHECKPOINT_BYTES:]
        if checkpoint != expected_checkpoint:
            return _TailBatch(epoch_changed=True)

        text = self._partial + chunk
        lines = text.split(b"\n")
        # The last element is either "" (chunk ended on a newline) or a partial
        # record the writer has not finished. Either way it is carried forward.
        partial = lines.pop()

        out: List[Dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # A record we will never be able to parse; skipping it is better
                # than stalling the whole feed on one corrupt line.
                continue

        self.offset = next_offset
        self._partial = partial
        self._identity = identity
        self._mtime_ns = after_read.st_mtime_ns
        self._checkpoint = checkpoint
        return _TailBatch(records=tuple(out))


class RunReader:
    """Aggregates a run directory into a :class:`RunState`, incrementally.

    Call :meth:`poll` on a timer.  It is cheap when nothing has changed (two
    ``stat`` calls) and O(new records) when something has.

    Parameters
    ----------
    run_dir:
        A run archive directory, or a parent containing several -- in which case the
        most recently modified one is followed, which is what "monitor whatever is
        running" means in practice.
    stale_after:
        Seconds without a new record before the run is reported as not live.  A
        generation on real data takes 30-60 s, so the default is generous; a shorter
        value would flap.
    """

    def __init__(self, run_dir: str | Path, stale_after: float = 180.0) -> None:
        self.root = Path(run_dir).expanduser()
        self.stale_after = stale_after
        self.path = self._resolve(self.root)

        self._gen_tail = _Tail(self.path / "generations.jsonl")
        self._call_tail = _Tail(self.path / "llm_calls.jsonl")

        self._reset_accumulated_state()

    # ------------------------------------------------------------------ helpers

    def _reset_accumulated_state(self) -> None:
        """Clear every value derived from the two stream files."""
        self._agents: Dict[str, AgentState] = {}
        self._agent_order: List[str] = []
        self._generations: List[Dict[str, Any]] = []
        self._role_counts: Counter = Counter()
        self._role_tokens: Counter = Counter()
        self._role_latency: Counter = Counter()
        self._recent: Deque[Dict[str, Any]] = deque(maxlen=_RECENT_CALLS)
        self._agent_recent: Dict[str, Deque[Dict[str, Any]]] = {}
        self._agent_call_counts: Counter = Counter()
        self._agent_token_counts: Counter = Counter()
        self._agent_latency_totals: Counter = Counter()
        self._n_calls = 0
        self._n_tokens = 0
        self._last_event_at: Optional[float] = None
        self._first_event_at: Optional[float] = None
        self._token_traj: List[Dict[str, Any]] = []
        #: seq -> byte offset, so a single call's full text can be fetched later
        #: without holding every prompt in memory.
        self._call_index: Dict[int, int] = {}
        self._config_cache: Optional[Dict[str, Any]] = None

        self._init_agents()

    @staticmethod
    def _resolve(root: Path) -> Path:
        """Pick the run directory to follow.

        A directory holding ``generations.jsonl`` is itself a run; otherwise the
        newest immediate subdirectory that looks like one is chosen, so
        ``--run runs/`` follows the latest run without having to name it.

        Resolution happens once, at construction. Following a *future* run would mean
        re-resolving on every poll and silently jumping between runs mid-session,
        which makes a dashboard you cannot trust. Start the monitor after the run.
        """
        if not root.exists():
            raise FileNotFoundError(
                f"run directory not found: {root}\n"
                "Start the search first; the monitor attaches to an existing archive."
            )
        if (root / "generations.jsonl").exists() or (root / "config.json").exists():
            return root
        candidates = [
            d for d in root.iterdir()
            if d.is_dir() and ((d / "generations.jsonl").exists() or (d / "config.json").exists())
        ]
        if not candidates:
            raise FileNotFoundError(
                f"no run archive under {root}\n"
                "Expected a directory containing generations.jsonl or config.json. "
                "If a search is starting right now, give it a few seconds and retry."
            )
        return max(candidates, key=lambda d: d.stat().st_mtime)

    def _init_agents(self) -> None:
        """Seed all 21 agents as queued, marking which ones this run selected.

        Reading the selection from the config (rather than waiting for each agent to
        appear in the stream) is what lets the matrix show the full hierarchy from the
        first second, with the 8 unselected agents distinguishable from the ones that
        simply have not started yet.
        """
        from cogalpha.agents.hierarchy import HIERARCHY, select_agents

        for spec in HIERARCHY:
            self._agents[spec.name] = AgentState(name=spec.name, level=spec.level)

        cfg = self._read_json("config.json")
        ev = (cfg or {}).get("evolution", {})
        try:
            chosen = select_agents(
                int(ev.get("agents_per_run", 13)),
                seed=int(ev.get("seed", 42)),
                use_golden_ratio=bool(ev.get("golden_ratio_selection", True)),
            )
        except Exception:  # noqa: BLE001 - a missing/odd config must not break the UI
            chosen = []
        for spec in chosen:
            if spec.name in self._agents:
                self._agents[spec.name].selected = True
            self._agent_order.append(spec.name)

    def _read_json(self, name: str) -> Dict[str, Any]:
        path = self.path / name
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    # --------------------------------------------------------------------- poll

    def poll(self) -> RunState:
        """Consume any new records and return the current snapshot."""
        for _ in range(_TAIL_READ_ATTEMPTS):
            generations = self._gen_tail.poll()
            calls = self._call_tail.poll()
            if not (generations.epoch_changed or calls.epoch_changed):
                break

            # One changed stream invalidates both batches. Rebuild the pair from
            # byte zero before applying either, so two run epochs cannot mix.
            self._gen_tail.rewind()
            self._call_tail.rewind()
            self._reset_accumulated_state()
        else:
            return self._snapshot()

        for record in generations.records:
            self._apply_generation(record)
        for record in calls.records:
            self._apply_call(record)
        return self._snapshot()

    def _apply_generation(self, record: Dict[str, Any]) -> None:
        """Fold one generation record into the per-agent and run-level state."""
        self._generations.append(record)
        name = str(record.get("agent", "?"))
        agent = self._agents.get(name)
        if agent is None:
            # An agent name not in the hierarchy: a hand-edited run, or a renamed
            # agent. Show it rather than dropping the data.
            agent = AgentState(name=name, level=0, selected=True)
            self._agents[name] = agent
            self._agent_order.append(name)

        # The first record for an agent normally means the previous one finished.
        # A record for an already-completed agent can arrive late from the other
        # tailed stream, so aggregate it without reversing a newer transition.
        if agent.status != "done":
            for other in self._agents.values():
                if other.status == "running" and other.name != name:
                    other.status = "done"
            agent.status = "running"
        agent.selected = True
        agent.current_generation = record.get("generation")
        agent.current_cycle = record.get("cycle")

        agent.generations += 1
        agent.n_generated += int(record.get("n_generated", 0) or 0)
        agent.n_passed += int(record.get("n_passed_checker", 0) or 0)
        agent.n_qualified += int(record.get("n_qualified", 0) or 0)
        agent.n_elite += int(record.get("n_elite", 0) or 0)
        agent.llm_calls += int(record.get("llm_calls", 0) or 0)
        agent.seconds += float(record.get("wall_seconds", 0.0) or 0.0)

        score = record.get("elite_mean_score")
        agent.elite_trajectory.append(_finite(score))

        best = record.get("best") or {}
        best_score = _finite(best.get("score"))
        if best_score is not None and (agent.best_score is None or best_score > agent.best_score):
            agent.best_score = best_score
            agent.best_rank_ic = _finite(best.get("rank_ic"))

        self._last_event_at = time.time()
        if self._first_event_at is None:
            self._first_event_at = self._last_event_at

    def _apply_call(self, record: Dict[str, Any]) -> None:
        """Fold one LLM call into the accounting. Prompt/response text is dropped."""
        from cogalpha.agents.hierarchy import BY_NAME

        tags = record.get("tags") or {}
        role = str(tags.get("role", "?"))
        tokens = int((record.get("usage") or {}).get("total_tokens", 0) or 0)
        latency = int(record.get("latency_ms", 0) or 0)

        self._role_counts[role] += 1
        self._role_tokens[role] += tokens
        self._role_latency[role] += latency
        self._n_calls += 1
        self._n_tokens += tokens

        seq = record.get("seq")
        if isinstance(seq, int):
            self._call_index[seq] = seq

        # The activity feeds: enough to see what is happening, without the text.
        agent_name = tags.get("agent")
        digest = {
            "seq": seq,
            "role": role,
            "agent": agent_name,
            "generation": tags.get("generation"),
            "cycle": tags.get("cycle"),
            "mode": tags.get("mode"),
            "temperature": record.get("temperature"),
            "model": record.get("model"),
            "tokens": tokens,
            "latency_ms": record.get("latency_ms"),
            "chars": len(record.get("response") or ""),
        }
        self._recent.append(digest)
        if isinstance(agent_name, str) and agent_name in BY_NAME:
            agent = self._agents[agent_name]
            if agent.status != "done":
                for other in self._agents.values():
                    if other.status == "running" and other.name != agent_name:
                        other.status = "done"
                agent.status = "running"
                agent.selected = True
                call_generation = tags.get("generation")
                advances_generation = (
                    agent.current_generation is None
                    or (
                        isinstance(call_generation, int)
                        and isinstance(agent.current_generation, int)
                        and call_generation > agent.current_generation
                    )
                )
                if "generation" in tags and advances_generation:
                    agent.current_generation = call_generation
                    if "cycle" in tags:
                        agent.current_cycle = tags.get("cycle")
                elif agent.current_generation is None and "cycle" in tags:
                    agent.current_cycle = tags.get("cycle")
            recent = self._agent_recent.setdefault(
                agent_name, deque(maxlen=_AGENT_RECENT_CALLS)
            )
            recent.append(digest)
            self._agent_call_counts[agent_name] += 1
            self._agent_token_counts[agent_name] += tokens
            self._agent_latency_totals[agent_name] += latency

        # One trajectory point per 25 calls keeps the chart readable over a 100k-call
        # run without downsampling logic in the browser.
        if self._n_calls % 25 == 0:
            self._token_traj.append({"calls": self._n_calls, "tokens": self._n_tokens})

        self._last_event_at = time.time()
        if self._first_event_at is None:
            self._first_event_at = self._last_event_at

    # ----------------------------------------------------------------- snapshot

    def _snapshot(self) -> RunState:
        """Build the state object served to the browser."""
        cfg = self._config_cache
        if cfg is None:
            cfg = self._read_json("config.json")
            # Cache only once it exists: `search` writes it early, but a monitor
            # attached first would otherwise cache the empty dict forever.
            if cfg:
                self._config_cache = cfg

        summary = self._read_json("summary.json")
        panel = self._read_json("panel.json")

        ev = cfg.get("evolution", {}) if cfg else {}
        planned_per_agent = int(ev.get("generations", 0) or 0)
        agents_total = int(ev.get("agents_per_run", 0) or 0) or sum(
            1 for a in self._agents.values() if a.selected
        )

        latest = self._generations[-1] if self._generations else {}
        # The funnel needs a generation that actually produced something: an empty
        # generation (model returned prose, or everything was a duplicate) is a normal
        # outcome but would draw an all-zero funnel and hide the last real one.
        latest_nonempty = next(
            (r for r in reversed(self._generations) if int(r.get("n_generated", 0) or 0) > 0),
            latest,
        )
        # `finished` comes from summary.json, which `save_run` writes last: its
        # presence is the one unambiguous signal that the search ended normally.
        finished = bool(summary)

        if finished:
            for agent in self._agents.values():
                if agent.status == "running":
                    agent.status = "done"

        for name, reason in (summary.get("stopped_early") or {}).items():
            if name in self._agents:
                self._agents[name].stopped_early = str(reason)

        done = sum(1 for a in self._agents.values() if a.status == "done")
        running = next((a.name for a in self._agents.values() if a.status == "running"), None)
        active = self._agents.get(running) if running is not None else None

        return RunState(
            run_dir=str(self.path),
            run_name=self.path.name,
            started_at=self._first_event_at,
            last_event_at=self._last_event_at,
            live=(
                not finished
                and self._last_event_at is not None
                and (time.time() - self._last_event_at) < self.stale_after
            ),
            finished=finished,
            config=_config_digest(cfg),
            panel=panel,
            summary=summary,
            agents=[
                {
                    **self._agents[name].to_dict(),
                    "llm_calls": max(
                        self._agents[name].llm_calls,
                        self._agent_call_counts[name],
                    ),
                }
                for name in self._ordered_agent_names()
            ],
            current_agent=running,
            current_generation=(
                active.current_generation if active is not None else latest.get("generation")
            ),
            current_cycle=active.current_cycle if active is not None else latest.get("cycle"),
            agents_done=done,
            agents_total=max(agents_total, done + (1 if running else 0)),
            generations_seen=len(self._generations),
            generations_planned=planned_per_agent * max(agents_total, 1),
            funnel=_funnel(latest_nonempty),
            latest_generation=_generation_digest(latest),
            funnel_generation=_generation_digest(latest_nonempty)
            if latest_nonempty is not latest
            else {},
            totals=self._totals(summary),
            calls_by_role=self._calls_by_role(),
            recent_calls=[dict(item) for item in reversed(self._recent)],
            elite_trajectory=self._elite_trajectory(),
            token_trajectory=self._token_traj[-400:],
            tier_counts=dict(summary.get("tiers") or {}) or self._tier_estimate(),
            candidates=self._candidates(),
            plateau=self._plateau(ev),
            warnings=self._warnings(latest, ev),
        )

    def _ordered_agent_names(self) -> List[str]:
        """Selected agents in run order first, then the unselected, grouped by level.

        Run order matters for the matrix: it is the order they will execute in, so the
        "next up" agent is visually adjacent to the running one.
        """
        ordered = [n for n in self._agent_order if n in self._agents]
        rest = sorted(
            (n for n in self._agents if n not in ordered),
            key=lambda n: (self._agents[n].level, n),
        )
        return ordered + rest

    def _totals(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Run-level counters, preferring the archive's own numbers when present."""
        elapsed = (
            (self._last_event_at - self._first_event_at)
            if (self._first_event_at and self._last_event_at)
            else 0.0
        )
        return {
            "llm_calls": int(summary.get("llm_calls") or self._n_calls),
            "llm_tokens": int(summary.get("llm_tokens") or self._n_tokens),
            "alphas_seen": summary.get("alphas_seen"),
            "unique_structures": summary.get("unique_structures"),
            "duplicates_reused": summary.get("duplicates_reused"),
            "candidates": summary.get("candidates"),
            "elapsed_seconds": round(float(summary.get("wall_seconds") or elapsed), 1),
            "tokens_per_call": round(self._n_tokens / self._n_calls, 1) if self._n_calls else 0.0,
        }

    def _calls_by_role(self) -> List[Dict[str, Any]]:
        """Per-role call counts, tokens and mean latency, in pipeline order.

        This is the cost breakdown: on a mock run ``mutate`` alone is ~60% of tokens,
        which is the first thing to look at when a budget disappears faster than
        expected.
        """
        out: List[Dict[str, Any]] = []
        roles = [r for r in ROLE_ORDER if r in self._role_counts]
        roles += [r for r in sorted(self._role_counts) if r not in ROLE_ORDER]
        for role in roles:
            calls = self._role_counts[role]
            out.append(
                {
                    "role": role,
                    "calls": calls,
                    "tokens": self._role_tokens[role],
                    "mean_latency_ms": round(self._role_latency[role] / calls, 1) if calls else 0.0,
                    "token_share": round(self._role_tokens[role] / self._n_tokens, 4)
                    if self._n_tokens
                    else 0.0,
                }
            )
        return out

    def _elite_trajectory(self) -> List[Dict[str, Any]]:
        """Elite mean score per generation, with agent boundaries marked.

        Boundaries matter for reading the chart: each agent restarts at generation 0,
        so a drop at a boundary is a new search starting, not the search degrading.
        """
        out: List[Dict[str, Any]] = []
        prev_agent: Optional[str] = None
        for i, record in enumerate(self._generations):
            agent = str(record.get("agent", "?"))
            out.append(
                {
                    "i": i,
                    "agent": agent,
                    "generation": record.get("generation"),
                    "cycle": record.get("cycle"),
                    "score": _finite(record.get("elite_mean_score")),
                    "n_elite": int(record.get("n_elite", 0) or 0),
                    "boundary": agent != prev_agent,
                }
            )
            prev_agent = agent
        return out[-600:]

    def _tier_estimate(self) -> Dict[str, int]:
        """Approximate tier counts before ``summary.json`` exists.

        Only qualified/elite and the per-stage rejections are knowable from the
        generation stream; ``plain`` is not, so this is explicitly an estimate that
        ``summary.json`` supersedes once the run saves.
        """
        counts: Counter = Counter()
        for record in self._generations:
            counts["qualified"] += int(record.get("n_qualified", 0) or 0)
            counts["elite"] += int(record.get("n_elite", 0) or 0)
            for stage, n in (record.get("reject_counts") or {}).items():
                counts[f"rejected:{stage}"] += int(n or 0)
        return dict(counts)

    def _candidates(self) -> List[Dict[str, Any]]:
        """The archived candidate files, parsed from their provenance headers.

        Only present after the run saved itself; during a run the best-so-far is
        visible per agent instead.
        """
        directory = self.path / "candidates"
        if not directory.exists():
            return []
        out: List[Dict[str, Any]] = []
        for path in sorted(directory.glob("*.py")):
            header: Dict[str, Any] = {"file": path.name}
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.startswith("#"):
                    break
                text = line.lstrip("# ").strip()
                if text.startswith("IC="):
                    for part in text.split():
                        if "=" in part:
                            key, _, value = part.partition("=")
                            header[key.lower()] = _finite(value)
                elif text.startswith("origin:"):
                    header["origin"] = text[len("origin:") :].strip()
                elif text.startswith("tier:"):
                    header["tier"] = text[len("tier:") :].strip()
            out.append(header)
        return out

    def _plateau(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        """How close the running agent is to the §B.4 early stop.

        Reproduces :class:`~cogalpha.evolution.pool.PlateauStopper` on the archived
        trajectory rather than importing it, because the monitor deliberately does not
        depend on the search package's runtime objects. The rule is small enough that
        duplicating it is cheaper than the coupling.
        """
        window = int(ev.get("plateau_window", 3) or 3)
        delta_limit = float(ev.get("plateau_delta", 0.001) or 0.001)
        running = next((a for a in self._agents.values() if a.status == "running"), None)
        if running is None:
            return {"window": window, "delta_limit": delta_limit, "armed": False}

        history = running.elite_trajectory
        armed = len(history) >= 2 * window
        delta: Optional[float] = None
        if armed:
            recent = [v for v in history[-window:] if v is not None]
            previous = [v for v in history[-2 * window : -window] if v is not None]
            if recent and previous:
                delta = sum(recent) / len(recent) - sum(previous) / len(previous)
        return {
            "window": window,
            "delta_limit": delta_limit,
            "armed": armed,
            "observations": len(history),
            "delta": delta,
            "would_stop": bool(delta is not None and delta <= delta_limit),
        }

    def _warnings(self, latest: Dict[str, Any], ev: Dict[str, Any]) -> List[str]:
        """Conditions worth interrupting a run for.

        Deliberately few: a monitor that cries wolf gets ignored. Each of these means
        the run is producing nothing useful and will keep doing so.
        """
        out: List[str] = []
        if not self._generations:
            return out

        # One stage swallowing most of the output is a systematic failure, not
        # attrition -- the same 35% threshold the `report` command uses.
        rejects: Counter = Counter()
        total = 0
        for record in self._generations[-10:]:
            for stage, n in (record.get("reject_counts") or {}).items():
                rejects[stage] += int(n or 0)
            total += int(record.get("n_generated", 0) or 0)
        if total:
            stage, n = (rejects.most_common(1) or [(None, 0)])[0]
            if stage and n / total > 0.35:
                out.append(
                    f"{n / total:.0%} of the last 10 generations' alphas died at "
                    f"'{stage}' -- systematic, not attrition"
                )

        # Duplicates dominating means the generator has run out of ideas.
        dup = sum(
            int((r.get("op_counts") or {}).get("duplicates_dropped", 0) or 0)
            for r in self._generations[-5:]
        )
        made = sum(int(r.get("n_generated", 0) or 0) for r in self._generations[-5:])
        if made and dup > made * 3:
            out.append(
                f"{dup} duplicates against {made} new alphas over the last 5 "
                "generations -- the generator is repeating itself"
            )

        # No elites at all, late in a run, means the gate is unreachable.
        if len(self._generations) >= 8:
            elites = sum(int(r.get("n_elite", 0) or 0) for r in self._generations)
            if elites == 0:
                out.append(
                    "no alpha has cleared the elite gate yet -- check the funnel "
                    "before loosening thresholds"
                )

        budget = ev.get("max_llm_calls")
        if budget:
            used = self._n_calls / float(budget)
            if used > 0.9:
                out.append(f"{used:.0%} of the {budget}-call budget consumed")
        return out

    # ------------------------------------------------------------- detail lookup

    def agent_detail(self, name: str) -> Optional[Dict[str, Any]]:
        """Return static identity and live aggregates for one hierarchy agent."""
        from cogalpha.agents.hierarchy import BY_NAME

        spec = BY_NAME.get(name)
        if spec is None:
            return None

        agent = self._agents[name]
        records = [record for record in self._generations if record.get("agent") == name]
        observed_calls = self._agent_call_counts[name]
        calls = max(observed_calls, agent.llm_calls)
        return {
            "name": name,
            "display_name": name.removeprefix("Agent"),
            "level": spec.level,
            "layer": spec.layer,
            "focus": spec.focus,
            "probe": spec.probe,
            "selected": agent.selected,
            "status": agent.status,
            "current_generation": agent.current_generation,
            "current_cycle": agent.current_cycle,
            "summary": {
                "generations": agent.generations,
                "generated": agent.n_generated,
                "passed": agent.n_passed,
                "qualified": agent.n_qualified,
                "elite": agent.n_elite,
                "best_score": agent.best_score,
                "best_rank_ic": agent.best_rank_ic,
                "llm_calls": calls,
                "llm_tokens": self._agent_token_counts[name],
                "mean_latency_ms": round(
                    self._agent_latency_totals[name] / observed_calls, 1
                )
                if observed_calls
                else 0.0,
                "seconds": round(agent.seconds, 1),
                "stopped_early": agent.stopped_early,
            },
            "trajectory": [
                {
                    "generation": record.get("generation"),
                    "cycle": record.get("cycle"),
                    "score": _finite(record.get("elite_mean_score")),
                    "elite": int(record.get("n_elite", 0) or 0),
                }
                for record in records
            ],
            "generations": [_agent_generation_digest(record) for record in records],
            "recent_operations": [
                dict(digest) for digest in reversed(self._agent_recent.get(name, ()))
            ],
        }

    def call_detail(self, seq: int) -> Optional[Dict[str, Any]]:
        """Fetch one call's full prompt and response by sequence number.

        Scans the file rather than keeping an offset index: this runs only when a
        human clicks a row, so a linear scan is acceptable and costs no memory during
        the run -- which is the whole reason prompts are dropped during aggregation.
        """
        path = self.path / "llm_calls.jsonl"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or f'"seq": {seq}' not in line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("seq") == seq:
                    return record
        return None

    def alpha_detail(self, alpha_id: str) -> Optional[Dict[str, Any]]:
        """Fetch one alpha's record -- code, checks, fitness, lineage.

        ``alphas.jsonl`` is only written when the run saves, so this returns None
        during a run and the UI falls back to the per-generation summary.
        """
        path = self.path / "alphas.jsonl"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if alpha_id not in line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("alpha_id") == alpha_id:
                    return record
        return None

    def generation_alphas(self, agent: str, generation: int) -> List[Dict[str, Any]]:
        """Alphas produced by one agent in one generation, from the saved archive."""
        path = self.path / "alphas.jsonl"
        if not path.exists():
            return []
        out: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lineage = record.get("lineage") or {}
                if lineage.get("agent") != agent or lineage.get("generation") != generation:
                    continue
                out.append(
                    {
                        "alpha_id": record.get("alpha_id"),
                        "name": record.get("name"),
                        "tier": record.get("tier"),
                        "rejected_at": record.get("rejected_at"),
                        "reject_reason": record.get("reject_reason"),
                        "op": lineage.get("op"),
                        "fitness": record.get("fitness"),
                    }
                )
        return out


# ----------------------------------------------------------------------- helpers


def _finite(value: Any) -> Optional[float]:
    """Coerce to a JSON-safe float, mapping NaN/inf/garbage to None."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def _funnel(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Stage-by-stage survivor counts for the latest generation.

    Reconstructed by walking the checker's stage order and subtracting each stage's
    rejections from the running total. Stages that rejected nothing are still emitted,
    so the funnel keeps a constant shape between generations and the eye can compare
    them without re-reading the labels.
    """
    if not record:
        return []
    proposed = int(record.get("n_generated", 0) or 0)
    rejects = {str(k): int(v or 0) for k, v in (record.get("reject_counts") or {}).items()}

    stages: List[Dict[str, Any]] = [{"stage": "proposed", "survivors": proposed, "dropped": 0}]
    survivors = proposed
    for stage in STAGE_ORDER:
        dropped = rejects.get(stage, 0)
        survivors -= dropped
        stages.append({"stage": stage, "survivors": max(survivors, 0), "dropped": dropped})

    stages.append(
        {"stage": "qualified", "survivors": int(record.get("n_qualified", 0) or 0), "dropped": 0}
    )
    stages.append(
        {"stage": "elite", "survivors": int(record.get("n_elite", 0) or 0), "dropped": 0}
    )
    return stages


def _generation_digest(record: Dict[str, Any]) -> Dict[str, Any]:
    """The subset of a generation record the dashboard header needs."""
    if not record:
        return {}
    best = record.get("best") or {}
    return {
        "agent": record.get("agent"),
        "generation": record.get("generation"),
        "cycle": record.get("cycle"),
        "n_generated": record.get("n_generated"),
        "n_passed_checker": record.get("n_passed_checker"),
        "n_qualified": record.get("n_qualified"),
        "n_elite": record.get("n_elite"),
        "op_counts": record.get("op_counts") or {},
        "reject_counts": record.get("reject_counts") or {},
        "wall_seconds": record.get("wall_seconds"),
        "llm_calls": record.get("llm_calls"),
        "elite_mean_score": _finite(record.get("elite_mean_score")),
        "percentile_cutoffs": {
            k: _finite(v) for k, v in (record.get("percentile_cutoffs") or {}).items()
        },
        "best": {
            "name": best.get("name"),
            "alpha_id": best.get("alpha_id"),
            "score": _finite(best.get("score")),
            "ic": _finite(best.get("ic")),
            "rank_ic": _finite(best.get("rank_ic")),
            "icir": _finite(best.get("icir")),
            "rank_icir": _finite(best.get("rank_icir")),
            "mi": _finite(best.get("mi")),
        },
    }


def _agent_generation_digest(record: Dict[str, Any]) -> Dict[str, Any]:
    """The compact generation record used in an agent detail view."""
    return {
        "generation": record.get("generation"),
        "cycle": record.get("cycle"),
        "generated": int(record.get("n_generated", 0) or 0),
        "passed": int(record.get("n_passed_checker", 0) or 0),
        "qualified": int(record.get("n_qualified", 0) or 0),
        "elite": int(record.get("n_elite", 0) or 0),
        "elite_mean_score": _finite(record.get("elite_mean_score")),
        "best": dict(record.get("best") or {}),
        "reject_counts": dict(record.get("reject_counts") or {}),
        "op_counts": dict(record.get("op_counts") or {}),
        "llm_calls": int(record.get("llm_calls", 0) or 0),
        "wall_seconds": round(float(record.get("wall_seconds", 0.0) or 0.0), 1),
    }


def _config_digest(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The config fields the dashboard displays.

    A digest rather than the whole config: the full object is large, and while the
    api_key is already redacted at write time there is no reason to ship
    credential-adjacent fields to a browser at all.
    """
    if not cfg:
        return {}
    data = cfg.get("data", {})
    ev = cfg.get("evolution", {})
    fit = cfg.get("fitness", {})
    llm = cfg.get("llm", {})
    return {
        "market": data.get("market"),
        "provider": data.get("provider"),
        "horizon": data.get("horizon"),
        "fit_split": data.get("fit_split"),
        "agents_per_run": ev.get("agents_per_run"),
        "generations": ev.get("generations"),
        "sub_cycles": ev.get("sub_cycles"),
        "parent_pool_size": ev.get("parent_pool_size"),
        "children_multiplier": ev.get("children_multiplier"),
        "max_llm_calls": ev.get("max_llm_calls"),
        "qualified_percentile": fit.get("qualified_percentile"),
        "elite_percentile": fit.get("elite_percentile"),
        "llm_provider": llm.get("provider"),
        "llm_model": llm.get("model"),
    }
