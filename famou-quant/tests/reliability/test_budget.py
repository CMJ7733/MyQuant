"""BudgetLedger: atomicity, per-episode sealed budgets, one-time gate tokens."""

from __future__ import annotations

import threading

import pytest

from famou.reliability.budget import BudgetExhausted, BudgetLedger
from famou.reliability.types import EvaluationCost


class TestGlobalBudgets:
    def test_charge_within_limit(self, ledger):
        ledger.set_global_limit("visible_queries", 10)
        ledger.charge(EvaluationCost(visible_query_count=3))
        assert ledger.remaining()["visible_queries"] == 7

    def test_charge_over_limit_raises(self, ledger):
        ledger.set_global_limit("visible_queries", 2)
        with pytest.raises(BudgetExhausted):
            ledger.charge(EvaluationCost(visible_query_count=3))
        # failed charge must not partially apply
        assert ledger.remaining()["visible_queries"] == 2

    def test_check_and_charge_atomic_under_concurrency(self, ledger):
        ledger.set_global_limit("visible_queries", 10)
        errors = []

        def worker():
            for _ in range(5):
                try:
                    ledger.charge(EvaluationCost(visible_query_count=1))
                except BudgetExhausted:
                    errors.append("exhausted")

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 20 attempts against a limit of 10: exactly 10 succeed, 10 raise,
        # and spent never exceeds the limit.
        assert len(errors) == 10
        assert ledger.snapshot().visible_queries.spent == 10


class TestPerEpisodeSealedBudget:
    def test_sealed_charge_requires_episode(self, ledger):
        with pytest.raises(ValueError):
            ledger.charge(EvaluationCost(sealed_query_count=1))

    def test_sealed_budget_isolated_per_episode(self, ledger):
        ledger.configure_episode("E2", sealed_limit=1)
        ledger.charge(EvaluationCost(sealed_query_count=1), episode_id="E1")
        # E2 untouched
        assert ledger.remaining(episode_id="E2")["sealed_queries"] == 1
        assert ledger.remaining(episode_id="E1")["sealed_queries"] == 2

    def test_sealed_exhaustion(self, ledger):
        ledger.configure_episode("E9", sealed_limit=1)
        ledger.charge(EvaluationCost(sealed_query_count=1), episode_id="E9")
        with pytest.raises(BudgetExhausted):
            ledger.charge(EvaluationCost(sealed_query_count=1), episode_id="E9")


class TestGateTokens:
    def test_token_issuance_spends_query(self, ledger):
        token = ledger.issue_gate_token("E1")
        assert token.startswith("gate_E1_")
        assert ledger.remaining(episode_id="E1")["sealed_queries"] == 2

    def test_token_is_one_time(self, ledger):
        token = ledger.issue_gate_token("E1")
        assert ledger.consume_gate_token(token) is True
        assert ledger.consume_gate_token(token) is False  # replay rejected

    def test_unknown_token_rejected(self, ledger):
        assert ledger.consume_gate_token("gate_E1_doesnotexist") is False

    def test_issuance_blocked_when_exhausted(self, ledger):
        ledger.configure_episode("E5", sealed_limit=0)
        with pytest.raises(BudgetExhausted):
            ledger.issue_gate_token("E5")
