"""Freeze protocol and the one-shot final test.

The last節 of the architecture, and the only one whose output has no return
edge. Everything else in this package feeds back into the agent; a
``PaperResult`` must not, or the final split silently becomes a third dev
set and every number in the paper stops meaning what it claims.

Two objects enforce that:

- ``FreezeManifest`` records exactly what was locked before the final run —
  policy version, prompts, expert registry, certified snapshot, seeds, the
  TaskSpec and the data contract. Producing it ENDS the search: the freeze
  is written to the StateStore and, once present, the reliability strategy
  refuses to make further decisions for that episode.
- ``FinalTestService`` is the only component allowed to touch final_test
  data. It spends the per-episode final query (default: exactly one),
  returns a PaperResult, and deliberately provides no path back into the
  archives, the trajectory store, the reward builder or the observation.

The one-way property is structural, not documentary: ``PaperResult`` is
never accepted as an argument anywhere in this package, and the service
takes no reference to any archive or store it could write to.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from famou.core.state import StateStore
from famou.reliability.archives import CertifiedArchive
from famou.reliability.budget import BudgetExhausted, BudgetLedger
from famou.reliability.types import (
    EvaluationCost,
    FrozenSplitManifest,
    PaperResult,
    TaskSpec,
)


_FREEZE_PATH = ("reliability", "freeze")
_FINAL_RUN_PATH = ("reliability", "final_test_runs")


class ExperimentNotFrozen(RuntimeError):
    """Final test attempted before the experiment was frozen."""


class FinalTestAlreadyRun(RuntimeError):
    """The one-shot final evaluation was attempted twice."""


class SearchClosed(RuntimeError):
    """A decision was attempted after the freeze."""


class FreezeManifest(BaseModel):
    """Immutable record of everything locked before the final evaluation."""

    freeze_id: str
    episode_id: str
    frozen_at: float
    protocol_version: str
    data_contract_hash: str
    task_spec_hash: str
    protected_hash: str = Field(
        description="hash of the frozen portfolio/cost/backtest half of TaskSpec"
    )
    policy_version: str
    prompts_hash: str = Field(default="", description="hash of prompt templates in use")
    expert_registry: List[str] = Field(
        default_factory=list, description="proposal experts available during search"
    )
    certified_snapshot: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Certified Archive contents at freeze time"
    )
    seeds: List[int] = Field(default_factory=list)
    search_state_version: int = 0
    total_cost: EvaluationCost = Field(default_factory=EvaluationCost)
    notes: str = ""

    def compute_hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def freeze_experiment(
    *,
    state_store: StateStore,
    manifest: FrozenSplitManifest,
    task_spec: TaskSpec,
    certified_archive: CertifiedArchive,
    policy_version: str,
    expert_registry: Optional[List[str]] = None,
    prompts_hash: str = "",
    seeds: Optional[List[int]] = None,
    search_state_version: int = 0,
    total_cost: Optional[EvaluationCost] = None,
    notes: str = "",
) -> FreezeManifest:
    """Close the search and snapshot everything the final run depends on.

    Idempotent per episode: calling it twice returns the existing freeze
    rather than silently re-snapshotting a mutated archive.
    """
    existing = state_store.get(*_FREEZE_PATH, manifest.episode_id, default=None)
    if existing is not None:
        return FreezeManifest.model_validate(existing)

    freeze = FreezeManifest(
        freeze_id=f"freeze_{uuid.uuid4().hex[:12]}",
        episode_id=manifest.episode_id,
        frozen_at=time.time(),
        protocol_version=manifest.protocol_version,
        data_contract_hash=manifest.compute_hash(),
        task_spec_hash=task_spec.compute_hash(),
        protected_hash=task_spec.protected_hash(),
        policy_version=policy_version,
        prompts_hash=prompts_hash,
        expert_registry=sorted(expert_registry or []),
        certified_snapshot=dict(certified_archive.members()),
        seeds=list(seeds or []),
        search_state_version=search_state_version,
        total_cost=total_cost or EvaluationCost(),
        notes=notes,
    )
    state_store.set(
        *_FREEZE_PATH, manifest.episode_id, value=freeze.model_dump(mode="json")
    )
    return freeze


def get_freeze(state_store: StateStore, episode_id: str) -> Optional[FreezeManifest]:
    data = state_store.get(*_FREEZE_PATH, episode_id, default=None)
    return FreezeManifest.model_validate(data) if data else None


def is_frozen(state_store: StateStore, episode_id: str) -> bool:
    return state_store.get(*_FREEZE_PATH, episode_id, default=None) is not None


class FinalTestService:
    """One-shot evaluation on ``final_test``. Results never flow back.

    ``final_eval_fn`` is the only hook and, like the sealed gate, must live
    where the final data does::

        final_eval_fn(candidate_code, manifest, seeds) -> {
            "rank_ic": float, "icir": float, "sharpe": float,
            "max_drawdown": float, "turnover": float,
            "regime_stability": {regime: float},
            "multi_seed_ci": {metric: [low, high]},
        }

    Note what this class does NOT hold: no SearchArchive, no
    CertifiedArchive, no TrajectoryStore, no ObservationBuilder. It has
    nothing to write the result into even if someone later tried.
    """

    def __init__(
        self,
        manifest: FrozenSplitManifest,
        final_eval_fn: Callable[..., Dict[str, Any]],
        *,
        state_store: StateStore,
        ledger: BudgetLedger,
    ):
        self.manifest = manifest
        self._eval_fn = final_eval_fn
        self._store = state_store
        self._ledger = ledger

    def run(
        self,
        *,
        candidate_id: str,
        candidate_code: str,
        seeds: Optional[List[int]] = None,
    ) -> PaperResult:
        """Evaluate ONE frozen candidate on final_test, exactly once."""
        episode = self.manifest.episode_id
        freeze = get_freeze(self._store, episode)
        if freeze is None:
            raise ExperimentNotFrozen(
                f"episode {episode} is not frozen; call freeze_experiment() "
                "before touching final_test"
            )
        if freeze.data_contract_hash != self.manifest.compute_hash():
            raise ExperimentNotFrozen(
                "data contract changed after the freeze; the final test would "
                "not measure the protocol the experiment was run under"
            )

        previous = self._store.get(*_FINAL_RUN_PATH, episode, default=None)
        if previous is not None:
            raise FinalTestAlreadyRun(
                f"final_test for episode {episode} already ran "
                f"(candidate={previous.get('candidate_id')}). It is one-shot by "
                "protocol: a second look would make it a validation set."
            )

        # Spend the per-episode final query. Charged BEFORE evaluation so a
        # crashed run still consumes it — otherwise retries are free looks.
        state = self._ledger.snapshot()
        budget = state.final_queries.get(episode)
        if budget is None or budget.remaining < 1:
            raise BudgetExhausted(f"final_queries[{episode}]", 1.0,
                                  budget.remaining if budget else 0.0)
        self._store.set(
            *_FINAL_RUN_PATH, episode,
            value={
                "candidate_id": candidate_id,
                "freeze_id": freeze.freeze_id,
                "ran_at": time.time(),
            },
        )
        self._ledger.charge_final_query(episode)

        seed_list = list(seeds or freeze.seeds or [101, 202, 303])
        raw = self._eval_fn(candidate_code, self.manifest, seed_list)

        return PaperResult(
            candidate_id=candidate_id,
            episode_id=episode,
            rank_ic=float(raw.get("rank_ic", 0.0)),
            icir=float(raw.get("icir", 0.0)),
            sharpe=float(raw.get("sharpe", 0.0)),
            max_drawdown=float(raw.get("max_drawdown", 0.0)),
            turnover=float(raw.get("turnover", 0.0)),
            regime_stability=dict(raw.get("regime_stability", {})),
            multi_seed_ci={k: list(v) for k, v in (raw.get("multi_seed_ci") or {}).items()},
            total_compute_cost=freeze.total_cost,
            query_costs={
                "visible": int(state.visible_queries.spent),
                "sealed": int(
                    state.sealed_queries[episode].spent
                    if episode in state.sealed_queries else 0
                ),
                "final": 1,
            },
        )

    def has_run(self) -> bool:
        return (
            self._store.get(*_FINAL_RUN_PATH, self.manifest.episode_id, default=None)
            is not None
        )
