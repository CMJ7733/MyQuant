"""Multi-Agent Quality Checker (§3.3, Appendix A.3).

The pipeline, in the order Figure 1 draws it:

    Code Quality Agent
        -> Code Repair Agent (loop, bounded)
    Judge Agent
        -> Logic Improvement Agent (loop, bounded)
    Static audit
    Execute + numerical stability
    Temporal leakage unit test
    -> qualified code

The two LLM review stages are bounded loops: "codes that cannot be repaired or
improved after several attempts are discarded".  The three deterministic stages
that follow are not loops — they are gates.

Ordering note: the static audit sits *after* the LLM stages rather than first,
because a repair agent can fix a bad import or a missing docstring, and rejecting
before giving it the chance would discard alphas the paper's design intends to
recover.  Anything the audit still rejects is unrunnable or unsafe, and no further
LLM call is spent on it.

The execute / stability / leakage trio is delegated to
:class:`~cogalpha.fitness.evaluate.FitnessEvaluator`, which visits the sandbox once
and returns all three verdicts plus the metrics.  Splitting them into separate
sandbox visits would triple the dominant cost of a generation for no additional
information.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from cogalpha.agents.parse import parse_alphas, parse_verdict, rename_function
from cogalpha.config import QualityConfig
from cogalpha.llm.base import LLMClient, LLMError
from cogalpha.prompts import (
    ALPHA_CONTRACT,
    CODE_QUALITY_PROMPT,
    IMPROVE_PROMPT,
    JUDGE_PROMPT,
    REPAIR_PROMPT,
    SYSTEM_PROMPT,
)
from cogalpha.quality.audit import audit_code
from cogalpha.types import Alpha, CheckReport, CheckStage, Lineage

if TYPE_CHECKING:
    # Deferred: ``cogalpha.fitness.evaluate`` imports the deterministic gates from
    # this package, so a module-level import here would close the cycle.
    from cogalpha.fitness.evaluate import EvalOutcome, FitnessEvaluator



@dataclass
class CheckerStats:
    """Per-generation tally of what the checker did.

    Mutated from the review worker threads, so every counter update goes through
    ``_lock``. Without it, ``+= 1`` from two threads loses increments and the
    archived call counts drift below the transcript's real length.
    """

    n_in: int = 0
    n_out: int = 0
    rejected: Dict[str, int] = field(default_factory=dict)
    repair_calls: int = 0
    improve_calls: int = 0
    review_calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def reject(self, stage: CheckStage) -> None:
        """Increment the per-stage rejection tally."""
        with self._lock:
            self.rejected[stage.value] = self.rejected.get(stage.value, 0) + 1

    def bump(self, field_name: str, n: int = 1) -> None:
        """Increment one counter under the lock."""
        with self._lock:
            setattr(self, field_name, getattr(self, field_name) + n)

    def to_dict(self) -> Dict[str, object]:
        """Counts for the generation record; ``n_in - n_out`` is the attrition."""
        return {
            "n_in": self.n_in,
            "n_out": self.n_out,
            "rejected": dict(self.rejected),
            "repair_calls": self.repair_calls,
            "improve_calls": self.improve_calls,
            "review_calls": self.review_calls,
        }


class QualityChecker:
    """Runs the checker pipeline over a batch of freshly generated alphas."""

    def __init__(
        self,
        llm: LLMClient,
        cfg: QualityConfig,
        evaluator: "FitnessEvaluator",
        temperature: float = 0.8,
    ) -> None:
        self.llm = llm
        self.cfg = cfg
        self.evaluator = evaluator
        #: Quality-checker agents run at a fixed temperature (§4.1), unlike the
        #: task and evolution agents which sample it.
        self.temperature = temperature

    # --------------------------------------------------------------------- main

    def check(
        self,
        alphas: Sequence[Alpha],
        generation: int = 0,
        agent: Optional[str] = None,
    ) -> Tuple[List[Alpha], List[Alpha], CheckerStats]:
        """Filter ``alphas``.

        Returns ``(passed, rejected, stats)``.  ``passed`` alphas carry a
        :class:`~cogalpha.types.Fitness`; ``rejected`` ones carry
        ``rejected_at``/``reject_reason`` and are kept so the archive records why
        each died and Adaptive Generation can learn from them.

        Shape of the work: stages 1-5 run **per alpha** (each is a chain of LLM calls
        on one candidate), stages 6-8 run **per batch** in a single sandbox visit.
        That split is why the loop below is structured as "filter, then evaluate the
        survivors together" rather than one pass per alpha.
        """
        stats = CheckerStats(n_in=len(alphas))
        surviving: List[Alpha] = []
        rejected: List[Alpha] = []

        # --- stage 1-5: the LLM review loops, per alpha -----------------------
        # Each alpha's chain (quality -> repair -> judge -> improve -> audit) is
        # sequential within itself but independent of every other alpha's, so the
        # alphas are reviewed concurrently at `llm.max_concurrency`. On an endpoint
        # with 36 s latency this is the difference between 24 and 6 serial waits for a
        # 12-alpha generation.
        #
        # `CheckerStats` is mutated from the worker threads, hence the lock inside it.
        # Results are collected in input order afterwards so the archive does not
        # depend on which thread finished first.
        reviewed = self._review_all(alphas, stats, generation, agent)

        for current in reviewed:
            if current.rejected_at is not None:
                rejected.append(current)
                stats.reject(current.rejected_at)
                continue
            surviving.append(current)

        # --- stage 6-8: one sandbox visit for execute + stability + leakage ---
        # The evaluator returns all three verdicts plus the metrics from a single
        # factor computation; `_apply_outcome` unpacks them into three CheckReports.
        if surviving:
            outcomes = self.evaluator.evaluate(surviving)
            passed: List[Alpha] = []
            for alpha in surviving:
                outcome = outcomes.get(alpha.alpha_id)
                if self._apply_outcome(alpha, outcome):
                    passed.append(alpha)
                else:
                    rejected.append(alpha)
                    if alpha.rejected_at is not None:
                        stats.reject(alpha.rejected_at)
            surviving = passed

        stats.n_out = len(surviving)
        return surviving, rejected, stats

    # ------------------------------------------------------- stage 1-5, per alpha

    def _review_all(
        self,
        alphas: Sequence[Alpha],
        stats: CheckerStats,
        generation: int,
        agent: Optional[str],
    ) -> List[Alpha]:
        """Run the review chain over every alpha, concurrently, in input order.

        Returns one Alpha per input — the surviving revision, or one marked rejected.
        Order is preserved so the archive is independent of thread scheduling.
        """
        workers = max(1, min(getattr(self.llm, "max_concurrency", 1), len(alphas)))
        if workers == 1 or len(alphas) <= 1:
            return [self._review_one(a, stats, generation, agent) for a in alphas]

        results: List[Optional[Alpha]] = [None] * len(alphas)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cogalpha-check") as pool:
            futures = {
                pool.submit(self._review_one, alpha, stats, generation, agent): i
                for i, alpha in enumerate(alphas)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except LLMError:
                    # Budget exhaustion mid-batch: mark this one rejected and let the
                    # rest of the batch finish. The loop stops the run at the next
                    # LLM call it makes itself.
                    alpha = alphas[index]
                    alpha.reject(CheckStage.CODE_QUALITY, "llm budget exhausted during review")
                    results[index] = alpha
        return [r if r is not None else alphas[i] for i, r in enumerate(results)]

    def _review_one(
        self,
        alpha: Alpha,
        stats: CheckerStats,
        generation: int,
        agent: Optional[str],
    ) -> Alpha:
        """One alpha's full review chain: quality -> repair -> judge -> improve -> audit.

        Note `current = ...` rather than mutating `alpha`: a repair or improvement
        round returns a *new* Alpha carrying the revised code, so the variable has to
        be rebound. Returning `alpha` instead of `current` would archive the
        pre-repair version.
        """
        current = alpha
        if self.cfg.enable_llm_repair:
            current = self._code_quality_loop(current, stats, generation, agent)
        if current.rejected_at is not None:
            return current

        if self.cfg.enable_judge:
            current = self._judge_loop(current, stats, generation, agent)
        if current.rejected_at is not None:
            return current

        # --- stage 5: static audit -------------------------------------------
        # Deliberately after the LLM stages: a repair agent can fix a bad import or a
        # missing docstring, and auditing first would discard alphas the method intends
        # to recover. `require_docstring=False` because the judge already assesses
        # rationale, and a missing docstring is not a safety problem -- only an
        # interpretability one.
        audit = audit_code(
            current.code,
            allowed_imports=self.cfg.allowed_imports,
            require_docstring=False,
        )
        current.add_check(
            CheckReport(
                stage=CheckStage.STATIC_AUDIT,
                passed=audit.ok,
                detail=audit.detail,
                payload={"imports": audit.imports},
            )
        )
        if not audit.ok:
            current.reject(CheckStage.STATIC_AUDIT, audit.detail)
            return current

        # Keep the function name and the alpha's name in step: repair and improvement
        # rounds routinely rename the function, and the sandbox looks up the callable
        # by `name`.
        if audit.function_name and audit.function_name != current.name:
            current.name = audit.function_name
        return current

    # ----------------------------------------------------------------- stage 1-2

    def _code_quality_loop(
        self,
        alpha: Alpha,
        stats: CheckerStats,
        generation: int,
        agent: Optional[str] = None,
    ) -> Alpha:
        """Code Quality Agent, with the Code Repair Agent as its recovery path.

        Bounded loop: review, and on failure repair and review again, up to
        ``max_repair_rounds`` repairs.  Returns the surviving (possibly revised)
        alpha, or one marked rejected if it could not be repaired -- "codes that
        cannot be repaired ... after several attempts are discarded" (§A.3).

        The ``+ 1`` in the range is the initial review: N repairs need N+1 reviews.
        """
        current = alpha
        for attempt in range(self.cfg.max_repair_rounds + 1):

            prompt = CODE_QUALITY_PROMPT.format(code=current.code)
            try:
                response = self.llm.generate(
                    prompt=prompt,
                    system=SYSTEM_PROMPT,
                    temperature=self.temperature,
                    tags={
                        "role": "code_quality",
                        "agent": agent,
                        "alpha": current.alpha_id,
                        "generation": generation,
                        "attempt": attempt,
                    },
                )
            except LLMError:
                raise
            stats.bump('review_calls')

            ok, detail = parse_verdict(response.text, pass_token="PASS")
            current.add_check(
                CheckReport(
                    stage=CheckStage.CODE_QUALITY,
                    passed=ok,
                    detail=detail,
                    payload={"attempt": attempt},
                )
            )
            if ok:
                return current

            if attempt >= self.cfg.max_repair_rounds:
                current.reject(
                    CheckStage.CODE_QUALITY,
                    f"unrepaired after {self.cfg.max_repair_rounds} attempts: {detail}",
                )
                return current

            repaired = self._repair(current, detail, stats, generation, attempt, agent)
            if repaired is None:
                current.reject(
                    CheckStage.CODE_REPAIR,
                    "repair agent returned no usable code",
                )
                return current
            current = repaired
        return current

    def _repair(
        self,
        alpha: Alpha,
        issues: str,
        stats: CheckerStats,
        generation: int,
        attempt: int,
        agent: Optional[str] = None,
    ) -> Optional[Alpha]:
        """Code Repair Agent: one attempt at fixing the reported defects.

        Returns ``None`` when the response contained no parseable function, which the
        caller treats as an unrecoverable repair.
        """
        prompt = REPAIR_PROMPT.format(
            issues=issues or "unspecified issues",
            code=alpha.code,
            contract=ALPHA_CONTRACT,
        )
        response = self.llm.generate(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            temperature=self.temperature,
            tags={
                "role": "repair",
                "agent": agent,
                "alpha": alpha.alpha_id,
                "generation": generation,
                "attempt": attempt,
            },
        )
        stats.bump('repair_calls')
        return self._adopt(alpha, response.text, stage=CheckStage.CODE_REPAIR)

    # ----------------------------------------------------------------- stage 3-4

    def _judge_loop(
        self,
        alpha: Alpha,
        stats: CheckerStats,
        generation: int,
        agent: Optional[str] = None,
    ) -> Alpha:
        """Judge Agent, with the Logic Improvement Agent as its recovery path.

        Same bounded-loop shape as :meth:`_code_quality_loop`, but a different
        question: that stage asks "will this run?", this one asks "is it logically
        consistent, technically correct and economically meaningful?" (§A.3).  A
        flawless implementation of an arbitrary column combination fails here.
        """
        current = alpha
        for attempt in range(self.cfg.max_improve_rounds + 1):
            prompt = JUDGE_PROMPT.format(code=current.code)
            response = self.llm.generate(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                temperature=self.temperature,
                tags={
                    "role": "judge",
                    "agent": agent,
                    "alpha": current.alpha_id,
                    "generation": generation,
                    "attempt": attempt,
                },
            )
            stats.bump('review_calls')

            ok, detail = parse_verdict(response.text, pass_token="PASS")
            current.add_check(
                CheckReport(
                    stage=CheckStage.JUDGE,
                    passed=ok,
                    detail=detail,
                    payload={"attempt": attempt},
                )
            )
            if ok:
                return current

            if attempt >= self.cfg.max_improve_rounds:
                current.reject(
                    CheckStage.JUDGE,
                    f"not improved after {self.cfg.max_improve_rounds} attempts: {detail}",
                )
                return current

            improved = self._improve(current, detail, stats, generation, attempt, agent)
            if improved is None:
                current.reject(
                    CheckStage.LOGIC_IMPROVEMENT,
                    "logic improvement agent returned no usable code",
                )
                return current
            current = improved
        return current

    def _improve(
        self,
        alpha: Alpha,
        assessment: str,
        stats: CheckerStats,
        generation: int,
        attempt: int,
        agent: Optional[str] = None,
    ) -> Optional[Alpha]:
        """Logic Improvement Agent: restructure to address the judge's assessment.

        Distinct from repair: the code already runs. This changes *what it computes*
        while preserving the modelling intent, so a rewrite that abandons the original
        hypothesis is a prompt failure, not a success.
        """
        prompt = IMPROVE_PROMPT.format(
            assessment=assessment or "unspecified assessment",
            code=alpha.code,
            contract=ALPHA_CONTRACT,
        )
        response = self.llm.generate(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            temperature=self.temperature,
            tags={
                "role": "improve",
                "agent": agent,
                "alpha": alpha.alpha_id,
                "generation": generation,
                "attempt": attempt,
            },
        )
        stats.bump('improve_calls')
        return self._adopt(alpha, response.text, stage=CheckStage.LOGIC_IMPROVEMENT)

    # -------------------------------------------------------------------- shared

    def _adopt(
        self,
        parent: Alpha,
        text: str,
        stage: CheckStage,
    ) -> Optional[Alpha]:
        """Replace ``parent``'s code with the revised version from ``text``.

        Lineage is preserved and the round counter bumped: a repaired alpha is the
        *same* candidate at a later revision, not a new individual. Treating it as
        new would double-count generation volume and break the audit trail from an
        archived candidate back to the agent that proposed it.
        """
        candidates = parse_alphas(text, parent.lineage, max_alphas=1)
        if not candidates:
            return None
        revised = candidates[0]

        # Keep the original name so downstream column lookups stay valid.
        code = rename_function(revised.code, parent.name)

        lineage = Lineage(**parent.lineage.to_dict())
        lineage.op = parent.lineage.op
        if stage is CheckStage.CODE_REPAIR:
            lineage.repair_rounds = parent.lineage.repair_rounds + 1
        else:
            lineage.improve_rounds = parent.lineage.improve_rounds + 1

        out = Alpha(
            code=code,
            name=parent.name,
            rationale=revised.rationale or parent.rationale,
            lineage=lineage,
            checks=list(parent.checks),
            meta={**parent.meta, "revised_by": stage.value},
        )
        out.add_check(
            CheckReport(stage=stage, passed=True, detail="revision adopted")
        )
        return out

    def _apply_outcome(self, alpha: Alpha, outcome: Optional["EvalOutcome"]) -> bool:
        """Translate one sandbox visit into the execute/stability/leakage verdicts.

        Returns True when the alpha survives all three and carries a ``Fitness``.

        Records **all three** CheckReports before deciding, so the archive shows the
        full picture even for an alpha that failed the first of them. Rejection is
        then attributed leakage-first: an alpha that is both leaky and unstable is
        recorded under the more serious cause, which keeps the rejection histogram
        interpretable.
        """
        if outcome is None:
            alpha.reject(CheckStage.EXECUTE, "sandbox returned no result")
            return False

        if not outcome.ok:
            alpha.add_check(
                CheckReport(
                    stage=CheckStage.EXECUTE,
                    passed=False,
                    detail=outcome.error,
                    payload={"error_type": outcome.error_type},
                )
            )
            alpha.reject(CheckStage.EXECUTE, outcome.error)
            return False

        alpha.add_check(
            CheckReport(
                stage=CheckStage.EXECUTE,
                passed=True,
                detail=f"executed in {outcome.seconds:.2f}s",
            )
        )

        numeric = outcome.numeric or {}
        numeric_ok = bool(numeric.get("ok", False))
        alpha.add_check(
            CheckReport(
                stage=CheckStage.NUMERIC_STABILITY,
                passed=numeric_ok,
                detail="; ".join(numeric.get("issues", [])) or "numerically stable",
                payload={
                    k: numeric.get(k)
                    for k in (
                        "nan_ratio",
                        "coverage",
                        "mean_distinct_per_day",
                        "mean_tie_ratio",
                        "n_days",
                    )
                },
            )
        )

        leak = outcome.leakage or {}
        leaked = bool(leak.get("leaked", False))
        deterministic = bool(leak.get("deterministic", True))
        leak_ok = (not leaked) and deterministic
        alpha.add_check(
            CheckReport(
                stage=CheckStage.LEAKAGE_UNIT_TEST,
                passed=leak_ok,
                detail="; ".join(leak.get("findings", [])) or "no leakage detected",
                payload={
                    "probe_ran": leak.get("probe_ran"),
                    "max_abs_diff": leak.get("max_abs_diff"),
                    "n_diff_cells": leak.get("n_diff_cells"),
                    "deterministic": deterministic,
                },
            )
        )

        # Leakage is reported before stability so a leaky-and-unstable alpha is
        # recorded under the more serious cause.
        if not leak_ok:
            alpha.reject(
                CheckStage.LEAKAGE_UNIT_TEST,
                "; ".join(leak.get("findings", [])) or "leakage detected",
            )
            return False
        if not numeric_ok:
            alpha.reject(
                CheckStage.NUMERIC_STABILITY,
                "; ".join(numeric.get("issues", [])) or "numerically unstable",
            )
            return False
        if outcome.fitness is None:
            alpha.reject(CheckStage.EXECUTE, "no metrics produced")
            return False

        alpha.fitness = outcome.fitness
        alpha.meta["eval_seconds"] = outcome.seconds
        if outcome.backtest:
            alpha.meta["backtest"] = outcome.backtest
        return True
