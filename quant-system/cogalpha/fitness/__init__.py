"""Fitness evaluation: five metrics, tiering, and the ranking backtest (§3.4, B.2, B.3)."""

from cogalpha.fitness.metrics import (  # noqa: F401
    ICSeries,
    align,
    ic_series,
    label_to_wide,
    mutual_information,
)
from cogalpha.fitness.thresholds import (  # noqa: F401
    TIER_METRICS,
    TierOutcome,
    assign_tiers,
    combined_score,
    percentile_cutoffs,
    tier_values,
    worst_invalid,
)
from cogalpha.fitness.backtest import BacktestResult, run_backtest  # noqa: F401
from cogalpha.fitness.evaluate import (  # noqa: F401
    EvalOutcome,
    FitnessEvaluator,
    evaluation_job,
)

__all__ = [
    "ICSeries",
    "align",
    "ic_series",
    "label_to_wide",
    "mutual_information",
    "TIER_METRICS",
    "TierOutcome",
    "assign_tiers",
    "combined_score",
    "percentile_cutoffs",
    "tier_values",
    "worst_invalid",
    "BacktestResult",
    "run_backtest",
    "EvalOutcome",
    "FitnessEvaluator",
    "evaluation_job",
]
