"""Atomic budget ledger.

Lives inside the StateStore so it checkpoints with the experiment, but all
mutations go through ``BudgetLedger`` which is thread-safe and atomic
(check-then-charge is a single lock acquisition, so concurrent workers can
never overspend).

Scope rules (design review, frozen):
- gpu/wall/llm/visible budgets: experiment-global (research cost).
- sealed/final query budgets: PER-EPISODE (statistical-evidence budget;
  sharing across episodes would enable cross-episode query arbitrage).
"""

from __future__ import annotations

import threading
import uuid
from typing import Dict, Optional

from famou.core.state import StateStore
from famou.reliability.types import (
    BudgetLedgerState,
    EvaluationCost,
    ResourceBudget,
)


class BudgetExhausted(RuntimeError):
    """Raised when a charge would exceed a budget limit."""

    def __init__(self, resource: str, requested: float, remaining: float):
        self.resource = resource
        self.requested = requested
        self.remaining = remaining
        super().__init__(
            f"Budget exhausted for {resource}: requested={requested}, "
            f"remaining={remaining}"
        )


_STATE_PATH = ("reliability", "budget")


class BudgetLedger:
    """Thread-safe atomic ledger backed by StateStore global state."""

    def __init__(self, state_store: StateStore):
        self._store = state_store
        self._lock = threading.Lock()
        if self._store.get(*_STATE_PATH, default=None) is None:
            self._store.set(*_STATE_PATH, value=BudgetLedgerState().model_dump(mode="json"))

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> BudgetLedgerState:
        return BudgetLedgerState.model_validate(self._store.get(*_STATE_PATH))

    def _save(self, state: BudgetLedgerState) -> None:
        self._store.set(*_STATE_PATH, value=state.model_dump(mode="json"))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def snapshot(self) -> BudgetLedgerState:
        with self._lock:
            return self._load()

    def configure_episode(
        self,
        episode_id: str,
        *,
        sealed_limit: int,
        final_limit: int = 1,
    ) -> None:
        """Register per-episode sealed/final budgets (idempotent)."""
        with self._lock:
            state = self._load()
            if episode_id not in state.sealed_queries:
                state.sealed_queries[episode_id] = ResourceBudget(limit=float(sealed_limit))
            if episode_id not in state.final_queries:
                state.final_queries[episode_id] = ResourceBudget(limit=float(final_limit))
            self._save(state)

    def set_global_limit(self, resource: str, limit: float) -> None:
        """Set a global budget limit (gpu_seconds/wall_seconds/llm_tokens/visible_queries)."""
        with self._lock:
            state = self._load()
            getattr(state, resource).limit = limit
            self._save(state)

    def charge(self, cost: EvaluationCost, *, episode_id: Optional[str] = None) -> None:
        """Atomically charge an evaluation cost. Raises BudgetExhausted if over.

        Check-and-charge happen under one lock so concurrent charges are safe.
        """
        with self._lock:
            state = self._load()
            checks = [
                ("gpu_seconds", state.gpu_seconds, cost.gpu_seconds),
                ("wall_seconds", state.wall_seconds, cost.wall_seconds),
                ("llm_tokens", state.llm_tokens, float(cost.llm_tokens)),
                ("visible_queries", state.visible_queries, float(cost.visible_query_count)),
            ]
            for name, budget, amount in checks:
                if amount > budget.remaining:
                    raise BudgetExhausted(name, amount, budget.remaining)

            if cost.sealed_query_count > 0:
                if episode_id is None:
                    raise ValueError("sealed charges require episode_id")
                budget = state.sealed_queries.get(episode_id)
                if budget is None or cost.sealed_query_count > budget.remaining:
                    raise BudgetExhausted(
                        f"sealed_queries[{episode_id}]",
                        float(cost.sealed_query_count),
                        budget.remaining if budget else 0.0,
                    )

            # All checks passed — apply
            state.gpu_seconds.spent += cost.gpu_seconds
            state.wall_seconds.spent += cost.wall_seconds
            state.llm_tokens.spent += float(cost.llm_tokens)
            state.visible_queries.spent += float(cost.visible_query_count)
            if cost.sealed_query_count > 0 and episode_id is not None:
                state.sealed_queries[episode_id].spent += float(cost.sealed_query_count)
            self._save(state)

    def issue_gate_token(self, episode_id: str) -> str:
        """Atomically spend ONE sealed query and return a one-time token.

        The token is later consumed by BudgetedGate so that a GateRequest
        cannot be replayed to farm extra sealed evaluations.
        """
        with self._lock:
            state = self._load()
            budget = state.sealed_queries.get(episode_id)
            if budget is None:
                raise BudgetExhausted(f"sealed_queries[{episode_id}]", 1.0, 0.0)
            if budget.remaining < 1:
                raise BudgetExhausted(
                    f"sealed_queries[{episode_id}]", 1.0, budget.remaining
                )
            budget.spent += 1.0
            self._save(state)

        token = f"gate_{episode_id}_{uuid.uuid4().hex[:16]}"
        self._store.set("reliability", "gate_tokens", token, value="issued")
        return token

    def consume_gate_token(self, token: str) -> bool:
        """Consume a one-time gate token. Returns False if invalid/replayed."""
        with self._lock:
            status = self._store.get("reliability", "gate_tokens", token, default=None)
            if status != "issued":
                return False
            self._store.set("reliability", "gate_tokens", token, value="consumed")
            return True

    def token_status(self, token: str) -> Optional[str]:
        """Lifecycle state of a gate token: 'issued' | 'consumed' | None.

        Read by CertifiedAdmission: a verdict whose token was never consumed
        did not come from the real gate, so it must not admit anything.
        """
        with self._lock:
            return self._store.get("reliability", "gate_tokens", token, default=None)

    def charge_final_query(self, episode_id: str) -> None:
        """Spend the one-shot final-test query for an episode.

        Separate from ``charge()`` because the final budget must never be
        reachable from the ordinary evaluation path: only FinalTestService
        calls this, and only after the experiment has been frozen.
        """
        with self._lock:
            state = self._load()
            budget = state.final_queries.get(episode_id)
            if budget is None or budget.remaining < 1:
                raise BudgetExhausted(
                    f"final_queries[{episode_id}]",
                    1.0,
                    budget.remaining if budget else 0.0,
                )
            budget.spent += 1.0
            self._save(state)

    def remaining(self, episode_id: Optional[str] = None) -> Dict[str, float]:
        """Compact remaining-budget view for the AgentObservation."""
        with self._lock:
            state = self._load()
            out: Dict[str, float] = {
                "gpu_seconds": state.gpu_seconds.remaining,
                "wall_seconds": state.wall_seconds.remaining,
                "llm_tokens": state.llm_tokens.remaining,
                "visible_queries": state.visible_queries.remaining,
            }
            if episode_id is not None:
                sealed = state.sealed_queries.get(episode_id)
                final = state.final_queries.get(episode_id)
                out["sealed_queries"] = sealed.remaining if sealed else 0.0
                out["final_queries"] = final.remaining if final else 0.0
            return out
