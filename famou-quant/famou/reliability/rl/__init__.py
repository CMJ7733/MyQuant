"""Agentic RL: encode, train, and serve the meta-policy.

What is being learned here is NOT a stock-prediction model. It is the search
policy: given the state of the archives and the remaining budget, which expert
to invoke, on which family, at what fidelity, how wide to fan out, and whether
to spend a sealed query. The candidates themselves are still produced by the
proposal experts.

Layout:
    encoding  ObservationEncoder / ActionCodec — the state and action spaces
    trainers  BehaviorCloning (stage A) and AdvantageWeightedRegression (B)
    policy    LearnedMetaPolicy — drops into the ``meta_policy`` slot

Staging follows the design: clone the heuristic first so online exploration
never starts from a random policy, then improve offline against delayed
rewards. Budget-constrained on-policy RL is stage C and is deliberately not
implemented here — it needs live rollouts to be meaningful, and the offline
stages have to demonstrably work first.

Versioning: ``ENCODING_VERSION`` is stamped into every PolicyCheckpoint. The
observation space is derived from AgentObservation and FailureKind, so a
change to either invalidates old checkpoints; loading one with a mismatched
version raises rather than silently feeding a policy garbage features.
"""

from __future__ import annotations

from famou.reliability.rl.encoding import (
    ENCODING_VERSION,
    ActionCodec,
    ObservationEncoder,
)
from famou.reliability.rl.policy import LearnedMetaPolicy
from famou.reliability.rl.trainer import (
    AdvantageWeightedRegression,
    BehaviorCloning,
    TrainingReport,
    build_dataset,
)

__all__ = [
    "ENCODING_VERSION",
    "ActionCodec",
    "ObservationEncoder",
    "LearnedMetaPolicy",
    "BehaviorCloning",
    "AdvantageWeightedRegression",
    "TrainingReport",
    "build_dataset",
]
