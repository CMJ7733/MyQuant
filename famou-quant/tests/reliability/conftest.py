"""Shared fixtures for reliability-layer tests."""

from __future__ import annotations

import pytest

from famou.core.state import StateStore
from famou.reliability.budget import BudgetLedger
from famou.reliability.types import (
    FrozenSplitManifest,
    SplitRange,
)


@pytest.fixture
def manifest() -> FrozenSplitManifest:
    """A small frozen manifest mirroring Protocol B E1 (development)."""
    return FrozenSplitManifest(
        protocol_version="protocol_b_v2",
        episode_id="E1",
        episode_role="development",
        train=SplitRange(start="2008-01-01", end="2013-12-31"),
        visible_dev=SplitRange(start="2014-01-01", end="2014-12-31"),
        sealed_promotion=SplitRange(start="2015-01-01", end="2015-12-31"),
        final_test=SplitRange(start="2016-01-01", end="2016-12-31"),
    )


@pytest.fixture
def state_store() -> StateStore:
    return StateStore()


@pytest.fixture
def ledger(state_store) -> BudgetLedger:
    ledger = BudgetLedger(state_store)
    ledger.configure_episode("E1", sealed_limit=3, final_limit=1)
    return ledger
