"""Run archive: everything needed to audit or re-analyse a search without re-running it.

Layout of ``runs/<timestamp>-<name>/``::

    config.json          resolved configuration, secrets stripped
    panel.json           data provenance: provider, window, shape -- never the data
    generations.jsonl     one record per generation
    alphas.jsonl          every alpha, including rejected ones and why
    llm_calls.jsonl       written directly by the CallRecorder during the run
    candidates/*.py       the top candidates as runnable files, with lineage headers
    summary.json          run totals
    report.md             human-readable report

Two things are deliberately *not* stored: the panel itself (provenance is enough to
rebuild it, and the file would dwarf everything else) and API keys.

Rejected alphas are kept because they carry most of the diagnostic value: a run
that rejects 80% of its output at the leakage stage has a prompt problem, and that
is invisible if only the survivors are recorded.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from cogalpha.config import CogAlphaConfig
from cogalpha.evolution.loop import SearchResult
from cogalpha.fitness.thresholds import combined_score
from cogalpha.types import Alpha, GenerationRecord

#: Config keys whose values must never be written to disk.
_SECRET_KEYS = ("api_key", "key_set")


class RunArchive:
    """Creates and writes a run directory."""

    def __init__(
        self,
        out_dir: str | Path,
        run_name: Optional[str] = None,
        create: bool = True,
    ) -> None:
        base = Path(out_dir).expanduser()
        if run_name is None:
            run_name = time.strftime("%Y%m%d-%H%M%S")
        else:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            run_name = f"{stamp}-{_slug(run_name)}"
        self.path = base / run_name
        if create:
            (self.path / "candidates").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------- writers

    @property
    def llm_log_path(self) -> Path:
        """Where the :class:`~cogalpha.llm.recorder.CallRecorder` should write."""
        return self.path / "llm_calls.jsonl"

    def write_config(self, cfg: CogAlphaConfig) -> None:
        """Write the fully resolved config, with secret-bearing keys redacted."""
        payload = _scrub(cfg.to_dict())
        _write_json(self.path / "config.json", payload)

    def write_panel(self, description: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
        """Write data provenance only -- shape, window, provider. Never the data."""
        payload = dict(description)
        if extra:
            payload.update(extra)
        _write_json(self.path / "panel.json", payload)

    def write_generation(self, record: GenerationRecord) -> None:
        """Append one generation record.

        Called per generation rather than at the end so a run killed at hour six
        still leaves a readable trace.
        """
        _append_jsonl(self.path / "generations.jsonl", record.to_dict())

    def write_alphas(self, alphas: Iterable[Alpha]) -> int:
        """Append every alpha, rejects included, and return how many were written."""
        path = self.path / "alphas.jsonl"
        count = 0
        with open(path, "a", encoding="utf-8") as fh:
            for alpha in alphas:
                fh.write(json.dumps(alpha.to_dict(), ensure_ascii=False, default=_default) + "\n")
                count += 1
        return count

    def write_candidates(self, candidates: Sequence[Alpha], use_abs_ic: bool = True) -> List[Path]:
        """Write each candidate as a runnable ``.py`` with a provenance header."""
        out: List[Path] = []
        for rank, alpha in enumerate(candidates, start=1):
            header = _candidate_header(alpha, rank, use_abs_ic)
            path = self.path / "candidates" / f"{rank:02d}_{_slug(alpha.name)}.py"
            path.write_text(header + alpha.code.rstrip() + "\n", encoding="utf-8")
            out.append(path)
        return out

    def write_summary(self, result: SearchResult, extra: Optional[Dict[str, Any]] = None) -> None:
        """Write run totals: tier counts, LLM accounting, early stops."""
        payload = result.summary()
        if extra:
            payload.update(extra)
        _write_json(self.path / "summary.json", payload)

    def write_report(
        self,
        result: SearchResult,
        cfg: CogAlphaConfig,
        panel_description: Dict[str, Any],
    ) -> Path:
        """Render the human-readable report and return its path."""
        path = self.path / "report.md"
        path.write_text(
            _render_report(result, cfg, panel_description), encoding="utf-8"
        )
        return path

    def save_run(
        self,
        result: SearchResult,
        cfg: CogAlphaConfig,
        panel_description: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Write everything that is not already streamed during the run."""
        self.write_config(cfg)
        self.write_panel(panel_description)
        n_alphas = self.write_alphas(result.all_alphas)
        files = self.write_candidates(result.candidates, cfg.fitness.use_abs_ic)
        self.write_summary(result, extra={"alphas_written": n_alphas})
        report = self.write_report(result, cfg, panel_description)
        return {
            "path": str(self.path),
            "alphas_written": n_alphas,
            "candidate_files": [str(p) for p in files],
            "report": str(report),
        }

    # ------------------------------------------------------------------- readers

    @classmethod
    def load(cls, path: str | Path) -> "LoadedRun":
        """Open an existing run directory for reading. See :class:`LoadedRun`."""
        return LoadedRun(Path(path).expanduser())


class LoadedRun:
    """Read-only view of an archived run, for post-hoc analysis."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"run directory not found: {path}")
        self.path = path

    @property
    def config(self) -> Dict[str, Any]:
        """The run's resolved config (secrets already redacted at write time)."""
        return _read_json(self.path / "config.json")

    @property
    def panel(self) -> Dict[str, Any]:
        """Data provenance: provider, market, window, shape."""
        return _read_json(self.path / "panel.json")

    @property
    def summary(self) -> Dict[str, Any]:
        """Run totals. Empty dict if the run was killed before saving."""
        return _read_json(self.path / "summary.json")

    def generations(self) -> List[Dict[str, Any]]:
        """One record per generation, in order. Written during the run, so present
        even for a run that never finished."""
        return _read_jsonl(self.path / "generations.jsonl")

    def alphas(self) -> List[Dict[str, Any]]:
        """Every alpha the run produced, including rejects with their reasons."""
        return _read_jsonl(self.path / "alphas.jsonl")

    def llm_calls(self) -> List[Dict[str, Any]]:
        """The full transcript. Filter on ``tags.role`` to isolate one stage."""
        return _read_jsonl(self.path / "llm_calls.jsonl")

    def candidate_files(self) -> List[Path]:
        """Archived candidate ``.py`` paths, in rank order (01_ is the best)."""
        return sorted((self.path / "candidates").glob("*.py"))

    def candidate_codes(self) -> Dict[str, str]:
        """Candidate name -> source, for feeding :mod:`cogalpha.compose`."""
        out: Dict[str, str] = {}
        for path in self.candidate_files():
            out[path.stem] = path.read_text(encoding="utf-8")
        return out


# ----------------------------------------------------------------------- helpers


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")[:60] or "run"


def _default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _scrub(payload: Any) -> Any:
    """Recursively blank out secret-bearing keys."""
    if isinstance(payload, dict):
        out: Dict[str, Any] = {}
        for key, value in payload.items():
            if key in _SECRET_KEYS and value:
                out[key] = "<redacted>"
            else:
                out[key] = _scrub(value)
        return out
    if isinstance(payload, (list, tuple)):
        return [_scrub(v) for v in payload]
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_default),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, default=_default) + "\n")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _candidate_header(alpha: Alpha, rank: int, use_abs_ic: bool) -> str:
    """Provenance comment block: metrics, lineage, and the checks it passed."""
    f = alpha.fitness
    lines = [
        f"# CogAlpha candidate #{rank}: {alpha.name}",
        f"# alpha_id: {alpha.alpha_id}   (hash of the canonical source)",
    ]
    if f is not None:
        lines += [
            f"# IC={f.ic:.6f}  ICIR={f.icir:.4f}  RankIC={f.rank_ic:.6f}  "
            f"RankICIR={f.rank_icir:.4f}  MI={f.mi:.6f}",
            f"# combined score={combined_score(f, use_abs_ic):.6f}  "
            f"days={f.n_days}  nan_ratio={f.nan_ratio:.4f}",
        ]
        if f.aer is not None or f.ir is not None:
            lines.append(f"# AER={f.aer}  IR={f.ir}")
    lin = alpha.lineage
    lines += [
        f"# tier: {alpha.tier.value}",
        f"# origin: {lin.op.value} by {lin.agent} (level {lin.level}, "
        f"guidance {lin.guidance_mode})",
        f"# generation {lin.generation}, sub-cycle {lin.cycle}",
    ]
    if lin.parents:
        lines.append(f"# parents: {', '.join(lin.parents)}")
    if lin.repair_rounds or lin.improve_rounds:
        lines.append(
            f"# revisions: {lin.repair_rounds} repair, {lin.improve_rounds} improvement"
        )
    passed = [c.stage.value for c in alpha.checks if c.passed]
    lines.append(f"# checks passed: {', '.join(passed) or 'none recorded'}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _render_report(
    result: SearchResult,
    cfg: CogAlphaConfig,
    panel: Dict[str, Any],
) -> str:
    s = result.summary()
    lines: List[str] = [
        "# CogAlpha run report",
        "",
        "## Data",
        "",
        f"- provider: `{panel.get('provider', '?')}`  market: `{panel.get('market', '?')}`",
        f"- window: {panel.get('start')} .. {panel.get('end')}  "
        f"({panel.get('days')} days, {panel.get('instruments')} instruments)",
        f"- fitness split: `{cfg.data.fit_split}`  horizon: {cfg.data.horizon} days",
        "",
        "## Search",
        "",
        f"- agents: {cfg.evolution.agents_per_run} of 21  "
        f"schedule: {cfg.evolution.generations} generations "
        f"in {cfg.evolution.sub_cycles} sub-cycles",
        f"- parent pool {cfg.evolution.parent_pool_size}, "
        f"children {cfg.evolution.parent_pool_size * cfg.evolution.children_multiplier}",
        f"- tier gates: qualified p{cfg.fitness.qualified_percentile:g}, "
        f"elite p{cfg.fitness.elite_percentile:g}",
        f"- generations run: {s['generations_run']}",
        f"- alphas seen: {s['alphas_seen']} ({s['unique_structures']} unique structures)",
        f"- LLM: {s['llm_calls']} calls, {s['llm_tokens']:,} tokens",
        f"- wall time: {s['wall_seconds']}s",
        "",
        "### Outcome by tier",
        "",
    ]
    for key, count in sorted(s["tiers"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{key}`: {count}")

    if s["stopped_early"]:
        lines += ["", "### Early stops", ""]
        for agent, reason in s["stopped_early"].items():
            lines.append(f"- **{agent}**: {reason}")

    lines += ["", f"## Candidates ({len(result.candidates)})", ""]
    if not result.candidates:
        lines.append(
            "_No alpha cleared the elite gate. Check `alphas.jsonl` for the "
            "rejection breakdown before adjusting thresholds._"
        )
    else:
        lines += [
            "| # | name | IC | ICIR | RankIC | RankICIR | MI | origin |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for i, alpha in enumerate(result.candidates, start=1):
            f = alpha.fitness
            if f is None:
                continue
            lines.append(
                f"| {i} | `{alpha.name}` | {f.ic:+.4f} | {f.icir:+.3f} | "
                f"{f.rank_ic:+.4f} | {f.rank_icir:+.3f} | {f.mi:.4f} | "
                f"{alpha.lineage.op.value} / {alpha.lineage.agent} |"
            )

        lines += ["", "### Rationale and code", ""]
        for i, alpha in enumerate(result.candidates, start=1):
            lines += [
                f"#### {i}. `{alpha.name}`",
                "",
                alpha.rationale or "_no rationale recorded_",
                "",
                "```python",
                alpha.code.rstrip(),
                "```",
                "",
            ]

    lines += [
        "## Reproducibility",
        "",
        "- `llm_calls.jsonl` holds every prompt and completion, tagged by role,",
        "  agent, guidance mode, temperature and generation.",
        "- `alphas.jsonl` holds every alpha including rejects, with the stage and",
        "  reason for each rejection.",
        "- `generations.jsonl` holds per-generation counts and percentile cutoffs,",
        "  which is enough to re-derive the ablation and threshold-sensitivity",
        "  analyses without re-running the search.",
        "",
        "Note that LLM sampling makes an exact re-run impossible by construction;",
        "the transcript is what makes a run auditable instead.",
        "",
    ]
    return "\n".join(lines)
