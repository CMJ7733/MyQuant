"""The working stream of Figure 1: the search loop.

Schedule, per §B.4 and §B.8:

* 13 of the 21 agents are selected per run (golden ratio);
* each selected agent leads a complete evolutionary cycle of 24 generations, split
  into 3 inner sub-cycles of 8;
* generation 0 of each sub-cycle is produced by the task-specific agent; later
  generations by Thinking Evolution over the parent pool;
* every 2 generations, freshly generated agent alphas are filtered and injected
  into the parent pool;
* the previous generation's top two elites are always carried forward;
* qualified alphas form the next parent pool, elites go to the candidate pool;
* a plateau in the elite trajectory ends that agent's run early.

One structural decision worth stating: **the sub-cycle boundary resets the parent
pool but not the candidate pool.**  The paper says each agent "initiates the
evolutionary search 3 times", which only means something if a sub-cycle starts from
fresh task-generated alphas rather than continuing the previous population — a
restart is the mechanism by which the paper escapes a converged local optimum.
Candidates found before the restart are kept, because they are the output.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from cogalpha.agents.generator import AlphaGenerator
from cogalpha.agents.hierarchy import AgentSpec, select_agents
from cogalpha.config import CogAlphaConfig
from cogalpha.evolution.adaptive import AdaptiveGeneration, Feedback
from cogalpha.evolution.operators import ThinkingEvolution
from cogalpha.evolution.pool import (
    CandidatePool,
    PlateauStopper,
    build_parent_pool,
    generation_elite_score,
)
from cogalpha.fitness.evaluate import FitnessEvaluator
from cogalpha.fitness.thresholds import assign_tiers, combined_score
from cogalpha.llm.base import LLMClient, LLMError
from cogalpha.quality.checker import QualityChecker
from cogalpha.types import Alpha, AlphaTier, EvolutionOp, GenerationRecord


@dataclass
class SearchResult:
    """Everything a run produced."""

    candidates: List[Alpha] = field(default_factory=list)
    records: List[GenerationRecord] = field(default_factory=list)
    all_alphas: List[Alpha] = field(default_factory=list)
    stopped_early: Dict[str, str] = field(default_factory=dict)
    llm_calls: int = 0
    llm_tokens: int = 0
    wall_seconds: float = 0.0
    unique_structures: int = 0
    duplicates_reused: int = 0

    def summary(self) -> Dict[str, object]:
        """Run totals for ``summary.json`` and the CLI's closing lines."""
        tiers: Dict[str, int] = {}
        for alpha in self.all_alphas:
            key = alpha.tier.value if alpha.rejected_at is None else f"rejected:{alpha.rejected_at.value}"
            tiers[key] = tiers.get(key, 0) + 1
        return {
            "candidates": len(self.candidates),
            "generations_run": len(self.records),
            "alphas_seen": len(self.all_alphas),
            "unique_structures": self.unique_structures,
            "duplicates_reused": self.duplicates_reused,
            "tiers": tiers,
            "stopped_early": dict(self.stopped_early),
            "llm_calls": self.llm_calls,
            "llm_tokens": self.llm_tokens,
            "wall_seconds": round(self.wall_seconds, 1),
        }



ProgressFn = Callable[[GenerationRecord], None]


class CogAlphaSearch:
    """Drives the working stream for one run."""

    def __init__(
        self,
        cfg: CogAlphaConfig,
        llm: LLMClient,
        evaluator: FitnessEvaluator,
        on_generation: Optional[ProgressFn] = None,
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.evaluator = evaluator
        self.on_generation = on_generation

        # Four collaborators, all sharing one LLM client so that call counting and
        # the budget ceiling are global rather than per-component. Their RNG seeds
        # are offset from one base so a run is reproducible in everything except the
        # model's own sampling.
        self.generator = AlphaGenerator(
            llm,
            task_temperatures=cfg.llm.task_temperatures,
            seed=cfg.evolution.seed,
        )
        self.checker = QualityChecker(
            llm,
            cfg.quality,
            evaluator,
            # Checker agents run at a fixed temperature (§4.1), unlike the task and
            # evolution agents which sample it per call.
            temperature=cfg.llm.checker_temperature,
        )
        self.evolution = ThinkingEvolution(
            llm,
            cfg.evolution,
            task_temperatures=cfg.llm.task_temperatures,
            seed=cfg.evolution.seed + 1,
        )
        self.adaptive = AdaptiveGeneration(
            llm,
            cfg.evolution,
            cfg.fitness,
            temperature=cfg.llm.checker_temperature,
            seed=cfg.evolution.seed + 2,
        )

    # ------------------------------------------------------------------- driver

    def run(self) -> SearchResult:
        """Run every selected agent in turn and collect the candidate pool.

        Agents are independent by design: they share only ``seen_ids`` (so no two
        agents claim the same structure) and the candidate pool. The paper does not
        share knowledge between agents either — ``reflection.md`` names a shared
        factor memory as its first extension idea.
        """
        started = time.time()
        ev = self.cfg.evolution
        result = SearchResult()
        pool = CandidatePool()
        #: Canonical ids of every structure proposed in this run, across all agents.
        seen_ids: set = set()

        agents = select_agents(
            ev.agents_per_run,
            seed=ev.seed,
            use_golden_ratio=ev.golden_ratio_selection,
        )

        for agent in agents:
            try:
                self._run_agent(agent, pool, seen_ids, result)
            except LLMError as exc:
                # The budget is a hard ceiling; report where it ran out rather than
                # letting a partially-searched agent look like a completed one.
                result.stopped_early[agent.name] = f"llm budget/transport: {exc}"
                break

        result.candidates = pool.top(self.cfg.run.top_candidates)
        result.unique_structures = len({a.alpha_id for a in result.all_alphas})
        result.duplicates_reused = sum(
            int(r.op_counts.get("duplicates_dropped", 0)) for r in result.records
        )
        result.llm_calls = self.llm.n_calls
        result.llm_tokens = self.llm.n_tokens
        result.wall_seconds = time.time() - started
        return result

    # -------------------------------------------------------------- one agent

    def _run_agent(
        self,
        agent: AgentSpec,
        pool: CandidatePool,
        seen_ids: set,
        result: SearchResult,
    ) -> None:
        """One agent's complete evolutionary cycle.

        Mutates ``pool``, ``seen_ids`` and ``result`` in place; returns early when
        the plateau rule fires.  ``seen_ids`` is shared across *all* agents so two
        agents cannot both claim the same structure.
        """
        ev = self.cfg.evolution
        gens_per_cycle = max(1, ev.generations // max(ev.sub_cycles, 1))
        n_children = ev.parent_pool_size * ev.children_multiplier
        stopper = PlateauStopper(ev.plateau_window, ev.plateau_delta)

        parents: List[Alpha] = []
        prev_elites: List[Alpha] = []
        feedback = Feedback()
        # Counts across sub-cycles, so `generation` is unique within this agent's run
        # while `cycle` says which restart it belongs to.
        generation = 0

        for cycle in range(ev.sub_cycles):
            # A sub-cycle is a restart: the population is re-seeded from the
            # task-specific agent, which is what makes three searches three
            # searches rather than one long one.  Note what is *not* reset:
            # `pool` (the output), `prev_elites` (elitism survives the restart) and
            # `stopper` (the plateau rule looks at the whole agent trajectory).
            parents = []

            for step in range(gens_per_cycle):
                gen_started = time.time()
                calls_before = self.llm.n_calls

                # --- propose ------------------------------------------------
                raw, op_counts = self._propose(
                    agent=agent,
                    parents=parents,
                    step=step,
                    generation=generation,
                    cycle=cycle,
                    feedback=feedback.text,
                    n_children=n_children,
                    seen_ids=seen_ids,
                )

                if not raw:
                    # An empty generation is a normal outcome (the model returned
                    # prose, or everything was a duplicate). Record it so the gap is
                    # visible in the archive, then move on -- do not retry, or a
                    # played-out agent burns the whole budget.
                    record = GenerationRecord(
                        generation=generation,
                        cycle=cycle,
                        agent=agent.name,
                        op_counts=op_counts,
                        n_generated=0,
                        llm_calls=self.llm.n_calls - calls_before,
                        wall_seconds=time.time() - gen_started,
                    )
                    self._emit(record, result)
                    generation += 1
                    continue

                # Claim these structures before evaluating: an alpha that fails the
                # checker should still not be re-proposed later.
                for alpha in raw:
                    seen_ids.add(alpha.alpha_id)

                # --- filter and score ---------------------------------------
                passed, rejected, checker_stats = self.checker.check(
                    raw, generation=generation, agent=agent.name
                )
                tiers = assign_tiers(passed, self.cfg.fitness)

                # --- update the two pools -----------------------------------
                # Elites go to the output; qualified (which includes elites) form
                # the next parent pool, with last generation's top-2 carried in.
                pool.add(tiers.elite, use_abs_ic=self.cfg.fitness.use_abs_ic)
                parents = build_parent_pool(
                    tiers.qualified, prev_elites, ev, self.cfg.fitness
                )
                # Only overwrite the elitism carry-set when this generation produced
                # elites; otherwise an unlucky generation would erase the memory of
                # the best solution found so far.
                if tiers.elite:
                    prev_elites = tiers.elite

                # --- Adaptive Generation feedback for the next prompt --------
                feedback = self.adaptive.build(
                    scored=passed,
                    rejected=rejected,
                    generation=generation,
                    agent=agent.name,
                )

                # Rejected alphas are retained deliberately: the rejection breakdown
                # is the most useful diagnostic a run produces.
                result.all_alphas.extend(passed)
                result.all_alphas.extend(rejected)

                gen_score = generation_elite_score(tiers.elite, self.cfg.fitness)
                stopper.observe(gen_score)

                best = tiers.qualified[0] if tiers.qualified else None
                record = GenerationRecord(
                    generation=generation,
                    cycle=cycle,
                    agent=agent.name,
                    op_counts=op_counts,
                    n_generated=len(raw),
                    n_passed_checker=len(passed),
                    n_qualified=len(tiers.qualified),
                    n_elite=len(tiers.elite),
                    reject_counts=dict(checker_stats.rejected),
                    best=(
                        {
                            "alpha_id": best.alpha_id,
                            "name": best.name,
                            "score": combined_score(best.fitness, self.cfg.fitness.use_abs_ic),
                            **(best.fitness.to_dict() if best.fitness else {}),
                        }
                        if best is not None
                        else {}
                    ),
                    # Recording the cutoffs is what lets the §4.7 threshold study be
                    # re-derived from the archive without re-running the search.
                    percentile_cutoffs={k: float(v) for k, v in tiers.cutoffs.items()},
                    elite_mean_score=gen_score,
                    llm_calls=self.llm.n_calls - calls_before,
                    wall_seconds=time.time() - gen_started,
                )
                self._emit(record, result)
                generation += 1

                # Checked after emitting, so the generation that triggered the stop
                # is in the archive.
                if stopper.should_stop():
                    result.stopped_early[agent.name] = stopper.reason()
                    return

    # ------------------------------------------------------------- proposal step

    def _propose(
        self,
        agent: AgentSpec,
        parents: Sequence[Alpha],
        step: int,
        generation: int,
        cycle: int,
        feedback: str,
        n_children: int,
        seen_ids: set,
    ) -> tuple[List[Alpha], Dict[str, int]]:
        """Produce this generation's raw alphas.

        Three cases, in the order the schedule dictates:

        1. the first generation of a sub-cycle, or an empty parent pool, comes
           entirely from the task-specific agent;
        2. an injection generation (every ``inject_every``) mixes fresh agent
           alphas into the evolved children — this is §B.4's "every 2 generations,
           new alphas generated by the task-specific agents are filtered and
           injected into the parent pool";
        3. otherwise the generation is pure Thinking Evolution.

        ``op_counts`` is tallied **after** de-duplication, so the record says how
        many alphas each operator actually contributed to this generation rather
        than how many it emitted -- the difference is exactly the duplicate count,
        which is reported separately.
        """
        ev = self.cfg.evolution

        if step == 0 or not parents:
            # A sub-cycle restart re-seeds with a full initial pool, not a handful:
            # the point of restarting is to give Thinking Evolution a fresh
            # population of ``parent_pool_size`` candidates to select from, and a
            # 6-alpha seed cannot fill a 32-slot parent pool.
            n = ev.initial_pool_size if step == 0 else ev.alphas_per_agent
            raw = self._generate(agent, n, generation, cycle, feedback, seen_ids)
            kept = _dedup(raw, seen_ids, ev.dedup)
            counts = {EvolutionOp.HIERARCHY.value: len(kept)}
            if len(kept) < len(raw):
                counts["duplicates_dropped"] = len(raw) - len(kept)
            return kept, counts

        children, stats = self.evolution.breed(
            parents=parents,
            n_children=n_children,
            generation=generation,
            cycle=cycle,
            feedback=feedback,
            seen_ids=seen_ids,
            agent=agent.name,
        )

        raw = list(children)
        n_injected = 0
        if ev.inject_every > 0 and generation % ev.inject_every == 0:
            injected = self._generate(
                agent, ev.alphas_per_agent, generation, cycle, feedback, seen_ids
            )
            n_injected = len(injected)
            raw.extend(injected)

        kept = _dedup(raw, seen_ids, ev.dedup)
        kept_ids = {a.alpha_id for a in kept}

        counts: Dict[str, int] = {}
        for child in children:
            if child.alpha_id in kept_ids:
                op = str(child.meta.get("operator", child.lineage.op.value))
                counts[op] = counts.get(op, 0) + 1
        n_hierarchy = len(kept) - sum(counts.values())
        if n_hierarchy > 0:
            counts[EvolutionOp.HIERARCHY.value] = n_hierarchy

        dropped = stats.duplicates + (len(raw) - len(kept))
        if dropped:
            counts["duplicates_dropped"] = dropped
        failed = sum(stats.failures.values())
        if failed:
            counts["operator_failures"] = failed
        if n_injected:
            counts["injected"] = n_injected

        return kept, counts

    def _generate(
        self,
        agent: AgentSpec,
        n_wanted: int,
        generation: int,
        cycle: int,
        feedback: str,
        seen_ids: Optional[set] = None,
    ) -> List[Alpha]:
        """Call the task-specific agent until it has produced ``n_wanted`` *new* alphas.

        A single call is capped at ``alphas_per_agent`` ("approximately 5-6 alpha
        factors", §B.8) because asking one completion for 80 alphas within a
        4096-token budget yields 80 one-liners.  The initial pool of 80 is
        therefore assembled from several calls, each with a freshly sampled
        guidance mode and temperature — which is also how the five paraphrasing
        modes get exercised within one generation.

        Duplicates are counted against the target rather than the budget: an agent
        that has exhausted its ideas re-proposes the same expressions, and counting
        those as delivered leaves a sub-cycle restart with an empty pool.  Observed
        on a synthetic run before this fix: an agent's second sub-cycle produced 0-2
        new alphas per generation while 5 of every 5 were dropped as duplicates, and
        the whole sub-cycle did nothing.  The call ceiling stops the retry loop from
        chasing an agent that genuinely has nothing left.
        """
        per_call = max(1, self.cfg.evolution.alphas_per_agent)
        seen = set(seen_ids or ())
        fresh: List[Alpha] = []
        # Allow a few extra calls to absorb duplicates, but not unboundedly.
        max_calls = -(-n_wanted // per_call) + 3
        # Issue the calls a wave at a time: they are independent (each samples its own
        # guidance mode and temperature), so a 12-alpha seed generation is 2 calls that
        # can run together rather than one after the other. Waves rather than one pool
        # so the duplicate check still happens at a serial checkpoint, and so a model
        # that has stopped returning anything usable stops costing money after one wave
        # instead of after `max_calls`.
        workers = max(1, min(getattr(self.llm, "max_concurrency", 1), max_calls))
        issued = 0

        while len(fresh) < n_wanted and issued < max_calls:
            wave = max(1, min(workers, max_calls - issued))
            issued += wave
            batches = [
                self.generator.generate(
                    agent=agent,
                    count=per_call,
                    generation=generation,
                    cycle=cycle,
                    feedback=feedback,
                    allowed_modes=self.cfg.evolution.guidance_modes,
                )
                for _ in range(wave)
            ] if wave == 1 else self._generate_wave(
                agent, per_call, generation, cycle, feedback, wave
            )

            produced = 0
            for batch in batches:
                produced += len(batch)
                for alpha in batch:
                    if alpha.alpha_id in seen:
                        continue
                    seen.add(alpha.alpha_id)
                    fresh.append(alpha)
            if produced == 0:
                # The model returned nothing usable across a whole wave; stop paying
                # for it.
                break
        return fresh[:n_wanted]

    def _generate_wave(
        self,
        agent: AgentSpec,
        per_call: int,
        generation: int,
        cycle: int,
        feedback: str,
        wave: int,
    ) -> List[List[Alpha]]:
        """Issue ``wave`` independent generation calls concurrently.

        Guidance mode and temperature are sampled inside
        :meth:`AlphaGenerator.generate`, which uses its own ``random.Random`` — that
        is not thread-safe for reproducibility purposes, so the *sequence* of modes
        differs from a serial run. The distribution does not, and the modes are drawn
        uniformly by design (§3.2), so this changes nothing that matters.
        """
        out: List[List[Alpha]] = [[] for _ in range(wave)]
        with ThreadPoolExecutor(max_workers=wave, thread_name_prefix="cogalpha-gen") as pool:
            futures = {
                pool.submit(
                    self.generator.generate,
                    agent=agent,
                    count=per_call,
                    generation=generation,
                    cycle=cycle,
                    feedback=feedback,
                    allowed_modes=self.cfg.evolution.guidance_modes,
                ): i
                for i in range(wave)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    out[i] = future.result()
                except LLMError:
                    # Budget exhaustion: return what the wave produced and let the
                    # caller's next call raise.
                    out[i] = []
                except Exception:  # noqa: BLE001 - one bad response, not the wave
                    out[i] = []
        return out

    def _emit(self, record: GenerationRecord, result: SearchResult) -> None:
        result.records.append(record)
        if self.on_generation is not None:
            self.on_generation(record)


def _dedup(alphas: Sequence[Alpha], seen: set, enabled: bool) -> List[Alpha]:
    """Drop alphas whose exact source has already been proposed in this run."""
    if not enabled:
        return list(alphas)
    out: List[Alpha] = []
    local: set = set()
    for alpha in alphas:
        if alpha.alpha_id in seen or alpha.alpha_id in local:
            continue
        local.add(alpha.alpha_id)
        out.append(alpha)
    return out
