"""Thinking Evolution operators (§3.6).

Two agents, three operators.  The Mutation Agent makes a small deliberate change
to one alpha's logic; the Crossover Agent recombines two.  The three operators the
paper conducts are ``mutation`` alone, ``crossover`` alone, and ``crossover``
followed by ``mutation``.

What makes this different from a genetic algorithm is what the operator *is*: not
a random perturbation of a parameter vector but an LLM reading code and rewriting
it, so the child stays semantically coherent and its rationale is auditable.  The
cost is that an operator can fail — the model may return prose, or code that does
not parse — which is a normal outcome here and is counted rather than raised.
"""

from __future__ import annotations

import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from cogalpha.agents.parse import parse_alphas
from cogalpha.config import EvolutionConfig
from cogalpha.llm.base import LLMClient, LLMError
from cogalpha.prompts import (
    ALPHA_CONTRACT,
    CROSSOVER_PROMPT,
    MUTATE_PROMPT,
    SYSTEM_PROMPT,
)
from cogalpha.types import Alpha, EvolutionOp, Fitness, Lineage

#: The three evolution types of §3.6.
OPERATORS: Tuple[str, ...] = ("mutation", "crossover", "crossover_then_mutation")

_OP_TO_ENUM = {
    "mutation": EvolutionOp.MUTATION,
    "crossover": EvolutionOp.CROSSOVER,
    "crossover_then_mutation": EvolutionOp.CROSSOVER_THEN_MUTATION,
}


@dataclass
class EvolutionStats:
    """Per-generation tally of what the evolution agents produced."""

    attempts: Dict[str, int] = field(default_factory=dict)
    produced: Dict[str, int] = field(default_factory=dict)
    failures: Dict[str, int] = field(default_factory=dict)
    duplicates: int = 0
    llm_calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def attempt(self, op: str) -> None:
        """Count one operator invocation, successful or not."""
        with self._lock:
            self.attempts[op] = self.attempts.get(op, 0) + 1

    def success(self, op: str) -> None:
        """Count one child that survived parsing and de-duplication."""
        with self._lock:
            self.produced[op] = self.produced.get(op, 0) + 1

    def failure(self, op: str) -> None:
        """Count one invocation that yielded no usable code."""
        with self._lock:
            self.failures[op] = self.failures.get(op, 0) + 1

    def to_dict(self) -> Dict[str, object]:
        """Per-operator yield: attempts vs produced vs failures, plus duplicates."""
        return {
            "attempts": dict(self.attempts),
            "produced": dict(self.produced),
            "failures": dict(self.failures),
            "duplicates": self.duplicates,
            "llm_calls": self.llm_calls,
        }


def format_metrics(fitness: Optional[Fitness]) -> str:
    """One-line metric summary for a prompt.

    Parents are shown their own scores so the model can reason about *why* it is
    being asked to change something — a mutation prompt without the parent's
    numbers is just a request to paraphrase code.
    """
    if fitness is None:
        return "not yet evaluated"
    def fmt(value: float, digits: int = 4) -> str:
        return "n/a" if value != value else f"{value:.{digits}f}"

    return (
        f"IC={fmt(fitness.ic)} ICIR={fmt(fitness.icir, 3)} "
        f"RankIC={fmt(fitness.rank_ic)} RankICIR={fmt(fitness.rank_icir, 3)} "
        f"MI={fmt(fitness.mi)}"
    )


class ThinkingEvolution:
    """Mutation and Crossover agents, plus operator sampling.

    Parameters
    ----------
    llm:
        Shared client; evolution agents sample their temperature per call from
        ``task_temperatures`` (§4.1), like the task-specific agents.
    """

    def __init__(
        self,
        llm: LLMClient,
        cfg: EvolutionConfig,
        task_temperatures: Sequence[float] = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2),
        seed: int = 42,
    ) -> None:
        self.llm = llm
        self.cfg = cfg
        self.task_temperatures = tuple(task_temperatures)
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ sampling

    def sample_operator(self) -> str:
        """Draw one of the three operators with the configured weights."""
        weights = [max(self.cfg.op_weights.get(op, 0.0), 0.0) for op in OPERATORS]
        if sum(weights) <= 0:
            return "mutation"
        return self.rng.choices(list(OPERATORS), weights=weights, k=1)[0]

    def _pick_parents(self, pool: Sequence[Alpha], n: int) -> List[Alpha]:
        """Sample ``n`` distinct parents, preferring the better ones.

        The paper does not specify parent selection beyond "all qualified alphas
        are evolved".  A uniform draw over the parent pool wastes calls on its
        weakest members, so this uses rank-based weights (1/rank on the pool's
        existing order, which :func:`~cogalpha.fitness.thresholds.assign_tiers`
        leaves sorted by score) — mild pressure that still reaches the tail.
        """
        if not pool:
            return []
        if len(pool) <= n:
            return list(pool)
        weights = [1.0 / (i + 1) for i in range(len(pool))]
        picked: List[Alpha] = []
        remaining = list(pool)
        remaining_w = list(weights)
        while remaining and len(picked) < n:
            choice = self.rng.choices(range(len(remaining)), weights=remaining_w, k=1)[0]
            picked.append(remaining.pop(choice))
            remaining_w.pop(choice)
        return picked

    # ----------------------------------------------------------------- operators

    def mutate(
        self,
        parent: Alpha,
        generation: int,
        cycle: int,
        feedback: str = "",
        agent: Optional[str] = None,
    ) -> Optional[Alpha]:
        """Mutation Agent: one small deliberate change to ``parent``."""
        prompt = MUTATE_PROMPT.format(
            parent_metrics=format_metrics(parent.fitness),
            parent_code=parent.code,
            feedback=_feedback_block(feedback),
            contract=ALPHA_CONTRACT,
        )
        response = self._call(
            prompt,
            tags={
                "role": "mutate",
                # `agent` is what lets a monitor attribute this call: the generation
                # counter restarts at 0 for every agent, so it is not a unique key.
                "agent": agent or parent.lineage.agent,
                "parent": parent.alpha_id,
                "generation": generation,
                "cycle": cycle,
            },
        )
        lineage = Lineage(
            op=EvolutionOp.MUTATION,
            parents=[parent.alpha_id],
            agent=parent.lineage.agent,
            level=parent.lineage.level,
            guidance_mode=parent.lineage.guidance_mode,
            generation=generation,
            cycle=cycle,
        )
        return _first_alpha(response.text, lineage)

    def crossover(
        self,
        parent_a: Alpha,
        parent_b: Alpha,
        generation: int,
        cycle: int,
        feedback: str = "",
        agent: Optional[str] = None,
    ) -> Optional[Alpha]:
        """Crossover Agent: recombine the logic of two parents."""
        prompt = CROSSOVER_PROMPT.format(
            parent_a_metrics=format_metrics(parent_a.fitness),
            parent_a_code=parent_a.code,
            parent_b_metrics=format_metrics(parent_b.fitness),
            parent_b_code=parent_b.code,
            feedback=_feedback_block(feedback),
            contract=ALPHA_CONTRACT,
        )
        response = self._call(
            prompt,
            tags={
                "role": "crossover",
                "agent": agent or parent_a.lineage.agent,
                "parents": [parent_a.alpha_id, parent_b.alpha_id],
                "generation": generation,
                "cycle": cycle,
            },
        )
        # A crossover child inherits both parents' agent labels only when they
        # agree; otherwise the level is genuinely mixed and recorded as such.
        same_agent = parent_a.lineage.agent == parent_b.lineage.agent
        lineage = Lineage(
            op=EvolutionOp.CROSSOVER,
            parents=[parent_a.alpha_id, parent_b.alpha_id],
            agent=parent_a.lineage.agent if same_agent else "crossbred",
            level=parent_a.lineage.level if same_agent else None,
            generation=generation,
            cycle=cycle,
        )
        return _first_alpha(response.text, lineage)

    # -------------------------------------------------------------------- driver

    def breed(
        self,
        parents: Sequence[Alpha],
        n_children: int,
        generation: int,
        cycle: int,
        feedback: str = "",
        seen_ids: Optional[set] = None,
        agent: Optional[str] = None,
    ) -> Tuple[List[Alpha], EvolutionStats]:
        """Produce up to ``n_children`` offspring from ``parents``.

        Returns the children and a tally.  Fewer than ``n_children`` is normal:
        operators fail, and de-duplication drops children whose code is identical
        to something already seen (which is how the "structural diversity" claim
        stays measurable rather than rhetorical).

        Cost note: one child is 1-2 LLM calls (crossover_then_mutation is 2), so a
        96-child generation is ~130 calls. This is the single most expensive function
        in the system; ``stats.llm_calls`` reports what it actually spent.
        """
        stats = EvolutionStats()
        children: List[Alpha] = []
        if not parents:
            return children, stats

        # Local copy: children are checked against the run-wide set *and* against
        # each other within this generation, but the caller's set is only updated
        # after the checker runs.
        seen = set(seen_ids or ())
        calls_before = self.llm.n_calls
        # Bound the attempts so a run cannot spin when every operator is failing.
        # 3x gives room for a ~65% failure/duplicate rate before the generation comes
        # back short, which is roughly what an exhausted parent pool produces.
        max_attempts = n_children * 3

        workers = max(1, min(getattr(self.llm, "max_concurrency", 1), n_children))
        if workers > 1:
            return self._breed_concurrent(
                parents, n_children, generation, cycle, feedback, seen, stats,
                agent, workers, max_attempts, calls_before,
            )

        for _ in range(max_attempts):
            if len(children) >= n_children:
                break
            op = self.sample_operator()
            stats.attempt(op)

            try:
                child = self._apply(op, parents, generation, cycle, feedback, agent)
            except LLMError:
                # Budget exhaustion or a hard transport failure: stop breeding and
                # let the caller decide whether the run continues.
                stats.failure(op)
                break

            if child is None:
                # The model returned prose, or code that would not parse. Normal;
                # counted so the archive shows the operator's real yield.
                stats.failure(op)
                continue

            if self.cfg.dedup and child.alpha_id in seen:
                stats.duplicates += 1
                continue

            seen.add(child.alpha_id)
            child.lineage.op = _OP_TO_ENUM[op]
            child.meta["operator"] = op
            children.append(child)
            stats.success(op)

        stats.llm_calls = self.llm.n_calls - calls_before
        return children, stats

    def _breed_concurrent(
        self,
        parents: Sequence[Alpha],
        n_children: int,
        generation: int,
        cycle: int,
        feedback: str,
        seen: set,
        stats: EvolutionStats,
        agent: Optional[str],
        workers: int,
        max_attempts: int,
        calls_before: int,
    ) -> Tuple[List[Alpha], EvolutionStats]:
        """Breed in waves of ``workers`` concurrent operator calls.

        Waves rather than one big pool, for two reasons:

        * **de-duplication needs a checkpoint.** Two threads mutating one ``seen`` set
          would need a lock around every membership test, and would still let both
          produce the same child before either records it. Collecting a wave and then
          de-duplicating it serially keeps the semantics identical to the serial path.
        * **the target is a count, not a list.** Breeding stops as soon as
          ``n_children`` survive; a single pool of ``max_attempts`` futures would pay
          for every attempt even after the quota was met.

        Operator sampling still happens on this thread, so the RNG sequence — and
        therefore the mix of mutation / crossover / crossover_then_mutation — is
        unchanged by concurrency.
        """
        children: List[Alpha] = []
        attempts = 0

        while len(children) < n_children and attempts < max_attempts:
            wave = min(workers, n_children - len(children), max_attempts - attempts)
            ops = [self.sample_operator() for _ in range(wave)]
            for op in ops:
                stats.attempt(op)
            attempts += wave

            produced: List[Optional[Alpha]] = [None] * wave
            budget_hit = False
            with ThreadPoolExecutor(max_workers=wave, thread_name_prefix="cogalpha-breed") as pool:
                futures = {
                    pool.submit(
                        self._apply, op, parents, generation, cycle, feedback, agent
                    ): i
                    for i, op in enumerate(ops)
                }
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        produced[i] = future.result()
                    except LLMError:
                        budget_hit = True
                    except Exception:  # noqa: BLE001 - one bad operator, not the batch
                        produced[i] = None

            # De-duplicate the wave serially, in submission order, so the surviving
            # set does not depend on which thread finished first.
            for op, child in zip(ops, produced):
                if child is None:
                    stats.failure(op)
                    continue
                if self.cfg.dedup and child.alpha_id in seen:
                    stats.duplicates += 1
                    continue
                seen.add(child.alpha_id)
                child.lineage.op = _OP_TO_ENUM[op]
                child.meta["operator"] = op
                children.append(child)
                stats.success(op)

            if budget_hit:
                # Whatever this wave produced is kept; the caller stops the run at its
                # next LLM call.
                break

        stats.llm_calls = self.llm.n_calls - calls_before
        return children, stats

    def _apply(
        self,
        op: str,
        parents: Sequence[Alpha],
        generation: int,
        cycle: int,
        feedback: str,
        agent: Optional[str] = None,
    ) -> Optional[Alpha]:
        """Dispatch one operator. Returns ``None`` when the LLM produced nothing usable.

        Note the shared code path: ``crossover_then_mutation`` is literally a
        crossover whose child is then mutated, so the two operators cannot drift apart.
        """
        if op == "mutation":
            picked = self._pick_parents(parents, 1)
            return self.mutate(picked[0], generation, cycle, feedback, agent)

        picked = self._pick_parents(parents, 2)
        if len(picked) < 2:
            # A single-member pool cannot cross; fall back to mutation rather than
            # skipping the slot, so a small pool still makes progress.
            return self.mutate(picked[0], generation, cycle, feedback, agent) if picked else None

        child = self.crossover(picked[0], picked[1], generation, cycle, feedback, agent)
        if child is None or op == "crossover":
            return child

        # crossover_then_mutation: mutate the freshly bred child.
        mutated = self.mutate(child, generation, cycle, feedback, agent)
        if mutated is None:
            # The crossover succeeded; keep it rather than discarding both calls.
            return child
        # Re-point the lineage at the *grandparents*: the intermediate crossover child
        # was never evaluated or archived, so recording it as a parent would leave a
        # dangling id in the provenance chain.
        mutated.lineage.parents = list(child.lineage.parents)
        return mutated

    # -------------------------------------------------------------------- helper

    def _call(self, prompt: str, tags: Dict[str, object]):
        return self.llm.generate(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            temperature=self.rng.choice(self.task_temperatures),
            tags=tags,
        )


def _first_alpha(text: str, lineage: Lineage) -> Optional[Alpha]:
    alphas = parse_alphas(text, lineage, max_alphas=1)
    return alphas[0] if alphas else None


def _feedback_block(feedback: str) -> str:
    if not feedback.strip():
        return ""
    return (
        "LESSONS FROM THE PREVIOUS GENERATION\n"
        f"{feedback.strip()}\n\n"
        "Use these lessons: pursue what worked, avoid what did not.\n"
    )
