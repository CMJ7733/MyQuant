"""
Core data models and protocols for Famou 2.0.

Sub-modules:
- data: Data models (Program, Context, RolloutResult, Experiment, SelectionData)
- protocol: Module protocol and pipeline (Module, Rollout, WorkBatch, Strategy, RequiresLLM, RequiresEnv)
- state: Thread-safe state store for stateful strategies (StateStore)
- types: Enums (Language, RolloutStatus)

Import from sub-modules directly:
    from famou.core.data import Program, Context, SelectionData
    from famou.core.protocol import Module, Rollout, WorkBatch, Strategy
    from famou.core.state import StateStore
    from famou.core.types import Language

Key Concepts:
- WorkBatch: Batch of rollouts for concurrent execution (new in v2.1)
- Strategy Protocol: Dynamic decision-making interface with forward() method (new in v2.1)
- Static Strategy: Legacy dataclass in famou.strategies._registry (backward compatible)

Note: StateStore is now accessible via context.state (no RequiresState protocol needed)
"""
