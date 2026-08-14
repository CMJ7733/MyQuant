"""Core type definitions for the reliability layer.

All models are Pydantic v2 and JSON-serializable so they can be persisted
into the Trajectory Store and replayed by the RL trainer.

The two most important rules encoded here:

- ``GateVerdict`` deliberately carries NO numeric performance fields. The
  sealed evaluator must never leak RankIC/Sharpe/margin magnitudes back to
  the agent — otherwise repeated querying turns the sealed split into
  another visible-dev.
- ``FrozenSplitManifest.compute_hash()`` is the data-contract anchor: any
  drift in split ranges / label / embargo invalidates comparisons.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# =============================================================================
# Enums
# =============================================================================


class Fidelity(int, Enum):
    """Evaluation fidelity ladder. Higher = more expensive = stronger evidence."""

    F0_STATIC = 0
    F1_CHEAP = 1
    F2_FULL = 2


class ExpertKind(str, Enum):
    EXPLORE = "explore"
    MUTATE = "mutate"
    CROSSOVER = "crossover"
    DEBUG = "debug"
    LOCAL_HPO = "local_hpo"
    FUSION = "fusion"


class GateVerdictKind(str, Enum):
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"


class GateReasonCode(str, Enum):
    # PROMOTE reasons
    ROBUST_IMPROVEMENT = "ROBUST_IMPROVEMENT"
    # REJECT reasons
    NO_IMPROVEMENT = "NO_IMPROVEMENT"
    UNSTABLE_ACROSS_SEEDS = "UNSTABLE_ACROSS_SEEDS"
    REGIME_FRAGILE = "REGIME_FRAGILE"
    COST_UNACCEPTABLE = "COST_UNACCEPTABLE"
    PROTOCOL_VIOLATION = "PROTOCOL_VIOLATION"
    # INCONCLUSIVE reasons
    INSUFFICIENT_POWER = "INSUFFICIENT_POWER"
    EVALUATION_ERROR = "EVALUATION_ERROR"


class MarginBand(str, Enum):
    """Coarse effect-size band returned by the sealed gate.

    The gate quantizes the true margin into three bands so the policy can
    learn "barely passed" vs "clearly passed" without ever seeing a number.
    """

    CLEAR_PASS = "clear_pass"
    MARGINAL = "marginal"
    CLEAR_FAIL = "clear_fail"
    UNKNOWN = "unknown"


class GateQueryStatus(str, Enum):
    """Lifecycle of a gate query token. One token = at most one evaluation."""

    ISSUED = "issued"
    CONSUMED = "consumed"
    EXPIRED = "expired"


# =============================================================================
# Frozen protocol
# =============================================================================


class SplitRange(BaseModel):
    start: str
    end: str
    n_trading_days: Optional[int] = None


class FrozenSplitManifest(BaseModel):
    """Frozen data/protocol contract for ONE episode (e.g. E1 or E11).

    Lives outside the famou config system because it must be frozen before
    the experiment and hashed into every GateRequest / EvidenceVector.
    """

    protocol_version: str = Field(description="e.g. 'protocol_b_v2'")
    episode_id: str = Field(description="e.g. 'E11'")
    episode_role: str = Field(
        default="evaluation",
        description="development | evaluation | evaluation_post_cutoff",
    )
    market: str = "csi300"
    benchmark: str = "SH000300"
    train: SplitRange
    visible_dev: SplitRange
    sealed_promotion: SplitRange
    final_test: SplitRange
    label_expression: str = "Ref($close, -2) / Ref($close, -1) - 1"
    label_norm: str = "CSZScoreNorm"
    embargo_days: int = Field(
        default=2,
        description="Tail-purge length; must equal label forward depth (frozen rule).",
    )
    preprocessing_rule: str = (
        "all preprocessing statistics are fit on the train segment only"
    )
    data_sha256: Optional[str] = None

    def compute_hash(self) -> str:
        """Deterministic hash of the data contract.

        Deliberately excludes ``episode_role``/``data_sha256`` annotations so
        the hash tracks the *evaluation semantics* (ranges/label/embargo).
        """
        payload = {
            "protocol_version": self.protocol_version,
            "episode_id": self.episode_id,
            "market": self.market,
            "benchmark": self.benchmark,
            "train": self.train.model_dump(),
            "visible_dev": self.visible_dev.model_dump(),
            "sealed_promotion": self.sealed_promotion.model_dump(),
            "final_test": self.final_test.model_dump(),
            "label_expression": self.label_expression,
            "label_norm": self.label_norm,
            "embargo_days": self.embargo_days,
            "preprocessing_rule": self.preprocessing_rule,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @classmethod
    def from_yaml(cls, yaml_path: str, episode_id: str) -> "FrozenSplitManifest":
        """Load one episode from a Protocol-B-style splits YAML file."""
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        meta = doc.get("meta", {})
        episodes = doc.get("episodes", {})
        if episode_id not in episodes:
            raise KeyError(
                f"Episode {episode_id!r} not found in {yaml_path}; "
                f"available: {sorted(episodes)}"
            )
        ep = episodes[episode_id]
        label = doc.get("label", {})
        embargo = doc.get("embargo", {})
        version = meta.get("version")

        def _range(value) -> "SplitRange":
            # Protocol-B YAML stores ranges as ["start", "end"] lists
            if isinstance(value, (list, tuple)):
                return SplitRange(start=str(value[0]), end=str(value[1]))
            return SplitRange(**value)

        return cls(
            protocol_version=f"protocol_b_v{version}",
            episode_id=episode_id,
            episode_role=ep.get("role", "evaluation"),
            market=meta.get("market", "csi300"),
            benchmark=meta.get("benchmark", "SH000300"),
            train=_range(ep["train"]),
            visible_dev=_range(ep["visible_dev"]),
            sealed_promotion=_range(ep["sealed_promotion"]),
            final_test=_range(ep["final_test"]),
            label_expression=label.get("expression", cls.model_fields["label_expression"].default),
            label_norm=label.get("norm", cls.model_fields["label_norm"].default),
            embargo_days=int(embargo.get("days", 2)),
            preprocessing_rule=doc.get("preprocessing", {}).get(
                "rule", cls.model_fields["preprocessing_rule"].default
            ),
            data_sha256=meta.get("data_sha256"),
        )


# =============================================================================
# Budget
# =============================================================================


class ResourceBudget(BaseModel):
    """One resource line: limit + spent (spent <= limit enforced at charge time)."""

    limit: float = Field(ge=0)
    spent: float = Field(default=0.0, ge=0)

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit - self.spent)


class BudgetLedgerState(BaseModel):
    """Serializable snapshot of the whole ledger (persisted in StateStore)."""

    gpu_seconds: ResourceBudget = Field(
        default_factory=lambda: ResourceBudget(limit=float("inf"))
    )
    wall_seconds: ResourceBudget = Field(
        default_factory=lambda: ResourceBudget(limit=float("inf"))
    )
    llm_tokens: ResourceBudget = Field(
        default_factory=lambda: ResourceBudget(limit=float("inf"))
    )
    visible_queries: ResourceBudget = Field(
        default_factory=lambda: ResourceBudget(limit=float("inf"))
    )
    # Per-episode sealed/final budgets: {episode_id: ResourceBudget}
    sealed_queries: Dict[str, ResourceBudget] = Field(default_factory=dict)
    final_queries: Dict[str, ResourceBudget] = Field(default_factory=dict)


class EvaluationCost(BaseModel):
    """Atomic cost of one evaluation, charged to the BudgetLedger."""

    wall_seconds: float = 0.0
    gpu_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    llm_tokens: int = 0
    visible_query_count: int = 0
    sealed_query_count: int = 0


# =============================================================================
# Evidence
# =============================================================================


class MetricDistribution(BaseModel):
    """A metric observed across seeds: mean/std/CI + raw per-seed values."""

    mean: float
    std: float = 0.0
    ci95_low: Optional[float] = None
    ci95_high: Optional[float] = None
    n_seeds: int = 1
    per_seed: List[float] = Field(default_factory=list)

    @classmethod
    def from_samples(cls, samples: List[float]) -> "MetricDistribution":
        import math

        vals = [float(s) for s in samples]
        n = len(vals)
        if n == 0:
            raise ValueError("MetricDistribution.from_samples: empty samples")
        mean = sum(vals) / n
        if n > 1:
            var = sum((v - mean) ** 2 for v in vals) / (n - 1)
            std = math.sqrt(var)
            half = 1.96 * std / math.sqrt(n)
            ci = (mean - half, mean + half)
        else:
            std = 0.0
            ci = (None, None)
        return cls(
            mean=mean, std=std, ci95_low=ci[0], ci95_high=ci[1],
            n_seeds=n, per_seed=vals,
        )


class StaticCheckResult(BaseModel):
    """F0 output: did the candidate pass static inspection?"""

    compiles: bool = True
    leakage_flags: List[str] = Field(default_factory=list)
    schema_errors: List[str] = Field(default_factory=list)
    forbidden_edits: List[str] = Field(
        default_factory=list,
        description="Edits to protected components (backtester/cost/split)",
    )
    required_packages: List[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.compiles
            and not self.leakage_flags
            and not self.schema_errors
            and not self.forbidden_edits
        )


class EvidenceVector(BaseModel):
    """Structured evidence for ONE candidate at ONE fidelity level.

    Replaces the scalar ``combined_score`` as the unit the policy reasons
    about. A candidate accumulates multiple EvidenceVectors over its life
    (F0, F1, possibly several F2 rounds with different seeds); the store
    keeps all of them keyed by (candidate_id, fidelity, eval_id).
    """

    candidate_id: str
    episode_id: str
    eval_id: str = Field(description="unique per evaluation run")
    fidelity: Fidelity
    split_scope: str = Field(
        description="which split was used: 'visible_dev' (sealed never appears here)"
    )
    data_contract_hash: str

    # Core metric distributions (visible split only)
    rank_ic: Optional[MetricDistribution] = None
    icir: Optional[MetricDistribution] = None
    sharpe: Optional[MetricDistribution] = None
    max_drawdown: Optional[MetricDistribution] = None
    turnover: Optional[MetricDistribution] = None

    # Stability across subperiods/regimes: {regime: mean rank_ic}
    regime_stability: Dict[str, float] = Field(default_factory=dict)
    # Worst subperiod RankIC (fragility indicator)
    worst_subperiod_rank_ic: Optional[float] = None

    validity: float = 0.0
    static_check: Optional[StaticCheckResult] = None
    failure_stage: Optional[str] = Field(
        default=None,
        description="compile|data|train|predict|backtest|timeout|oom — None if success",
    )
    error_info: Optional[str] = None

    complexity: Dict[str, Any] = Field(
        default_factory=dict,
        description="n_params, ast_nodes, model_family ... for cost/complexity penalty",
    )
    cost: EvaluationCost = Field(default_factory=EvaluationCost)
    novelty: Optional[float] = Field(
        default=None,
        description="1 - max cosine similarity to certified/search archive code embeddings",
    )
    seed_list: List[int] = Field(default_factory=list)

    model_config = ConfigDict(use_enum_values=False)

    @property
    def is_valid(self) -> bool:
        return self.validity >= 1.0 and self.failure_stage is None


# =============================================================================
# Gate
# =============================================================================


class GateRequest(BaseModel):
    """Frozen promotion request. Once created, the candidate code is immutable."""

    request_id: str
    candidate_id: str
    episode_id: str
    candidate_code_hash: str
    data_contract_hash: str
    protocol_version: str
    query_token: str = Field(
        description="one-time token issued by the BudgetLedger/BudgetedGate"
    )
    seed_list: List[int] = Field(default_factory=list)
    frozen_at: float


class GateVerdict(BaseModel):
    """The ONLY information the sealed side ever returns.

    No numeric metrics. ``margin_band`` is the three-level quantization
    (clear_pass / marginal / clear_fail) approved in the design review;
    it leaks at most ~1.6 bits per query and lets the promotion policy
    learn from near-misses.
    """

    verdict: GateVerdictKind
    reason_code: GateReasonCode
    margin_band: MarginBand = MarginBand.UNKNOWN
    query_cost: int = 1

    model_config = ConfigDict(use_enum_values=False)


# =============================================================================
# Agent-facing decision structures
# =============================================================================


class StructuredAction(BaseModel):
    """What the meta-controller decided: expert + parents + eval resources."""

    expert: ExpertKind
    parent_ids: List[str] = Field(default_factory=list)
    model_family: str = Field(
        default="gbdt",
        description="gbdt | linear | mlp | temporal_transformer | fusion ...",
    )
    edit_target: Optional[str] = Field(
        default=None,
        description="component to modify, e.g. 'cross_sectional_encoder'",
    )
    fidelity: Fidelity = Fidelity.F1_CHEAP
    seed_list: List[int] = Field(default_factory=lambda: [11])
    batch_size: int = Field(default=1, ge=1)
    max_gpu_seconds: Optional[float] = None
    promotion_requested: bool = False
    rationale: str = ""

    model_config = ConfigDict(use_enum_values=False)


class DecisionRecord(BaseModel):
    """Full trace of one agent decision (for offline RL / audit)."""

    decision_id: str
    observation_digest: str = Field(
        description="sha256 of the serialized AgentObservation"
    )
    structured_action: StructuredAction
    state_version: int
    policy_version: str = "heuristic_v0"
    action_log_prob: Optional[float] = Field(
        default=None,
        description="None for heuristic policies; set once an RL policy is live",
    )
    predicted_value: Optional[float] = None
    timestamp: float

    model_config = ConfigDict(use_enum_values=False)


# =============================================================================
# Transition (RL unit)
# =============================================================================


class Transition(BaseModel):
    """One state -> action -> evidence -> next_state step for the RL trainer."""

    transition_id: str
    decision_ref: str = Field(description="decision_id of the DecisionRecord")
    observation_digest: str
    action: StructuredAction
    evidence_refs: List[str] = Field(
        default_factory=list, description="eval_ids of produced EvidenceVectors"
    )
    candidate_ids: List[str] = Field(default_factory=list)
    gate_verdict: Optional[GateVerdict] = None
    reward: Optional[float] = Field(
        default=None, description="delayed reward; filled when it matures"
    )
    costs: EvaluationCost = Field(default_factory=EvaluationCost)
    done: bool = False
    state_version: int
    policy_version: str = "heuristic_v0"

    model_config = ConfigDict(use_enum_values=False)


class PolicyCheckpoint(BaseModel):
    """Versioned policy artifact reference (weights live in artifact storage)."""

    policy_version: str
    kind: Literal["heuristic", "bc", "offline_rl", "online_rl"] = "heuristic"
    artifact_uri: Optional[str] = None
    trained_on_transitions: int = 0
    notes: str = ""


# =============================================================================
# Misc small models
# =============================================================================


class CandidateLineage(BaseModel):
    candidate_id: str
    parent_ids: List[str] = Field(default_factory=list)
    expert: Optional[str] = None
    decision_id: Optional[str] = None
    certified: bool = False


class PaperResult(BaseModel):
    """Final one-shot evaluation output. One-way arrow out; never feeds back."""

    candidate_id: str
    episode_id: str
    rank_ic: float
    icir: float
    sharpe: float
    max_drawdown: float
    turnover: float
    regime_stability: Dict[str, float] = Field(default_factory=dict)
    multi_seed_ci: Dict[str, List[float]] = Field(default_factory=dict)
    total_compute_cost: EvaluationCost = Field(default_factory=EvaluationCost)
    query_costs: Dict[str, int] = Field(default_factory=dict)


class GateConfig(BaseModel):
    """Sealed gate decision thresholds (frozen after power study)."""

    delta_min: float = Field(
        default=0.0091,
        description="minimum detectable RankIC improvement vs incumbent (power-study frozen)",
    )
    marginal_band_width: float = Field(
        default=0.003,
        description="half-width of the 'marginal' band around delta_min",
    )
    max_seed_cv: float = Field(
        default=1.5,
        description="reject if |std/mean| of sealed RankIC across seeds exceeds this",
    )
    min_regimes_positive: float = Field(
        default=0.6,
        description="fraction of subperiods that must show non-negative improvement",
    )
    seeds: List[int] = Field(default_factory=lambda: [101, 202, 303])
