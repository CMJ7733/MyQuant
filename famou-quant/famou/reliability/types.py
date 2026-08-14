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
# Task specification (frozen experiment contract)
# =============================================================================


class CandidateMode(str, Enum):
    """What KIND of candidate the search may propose. Chosen at startup.

    - ``FORMULA``: deterministic weighted combinations of Alpha158 factors.
      No fitting, ~0.06s per candidate, and the winner is a readable factor
      list. Stability evidence comes from subperiods, not seeds.
    - ``MODEL``: fitted predictors (LightGBM / MLP). 40-105s per candidate,
      higher ceiling, opaque.
    - ``MIXED``: both compete in one pool. This is the setting where the
      budget machinery actually bites — with a 700x cost spread, "is this
      candidate worth its compute?" stops being a rhetorical question.

    The mode is not stored separately: it selects ``allowed_model_families``,
    which is already part of the frozen TaskSpec hash. Two runs in different
    modes therefore have different task_spec_hashes and are not comparable —
    which is correct, since they searched different spaces.
    """

    FORMULA = "formula"
    MODEL = "model"
    MIXED = "mixed"

    def families(self) -> List[str]:
        if self is CandidateMode.FORMULA:
            return ["formula"]
        if self is CandidateMode.MODEL:
            return ["gbdt", "linear", "mlp", "temporal_transformer"]
        return ["formula", "gbdt", "linear", "mlp", "temporal_transformer"]


class TaskSpec(BaseModel):
    """What the agent is allowed to search over, and what it must not touch.

    The reason this is a first-class frozen object rather than scattered
    constants: the agent optimises whatever it is scored on. If it can edit
    the portfolio builder, the transaction-cost model or the backtester, the
    cheapest way to raise its score stops being "predict better" and becomes
    "exploit the backtest". Everything under ``Protected`` below is therefore
    fixed here, hashed into every candidate, and enforced by the F0 static
    check.

    Candidates own exactly one thing: the prediction model and its training
    configuration.
    """

    task_id: str = "evoquant_csi300"
    # --- search space (candidates MAY vary these) ---------------------
    universe: str = Field(default="csi300", description="stock pool")
    feature_set: str = Field(default="Alpha158", description="base feature library")
    horizon_days: int = Field(default=1, ge=1, description="holding period")
    prediction_target: str = Field(
        default="cross_sectional_return",
        description="what the model outputs (ranked score per stock per day)",
    )
    allowed_model_families: List[str] = Field(
        default_factory=lambda: ["gbdt", "linear", "mlp", "temporal_transformer"]
    )
    allowed_packages: List[str] = Field(
        default_factory=lambda: [
            "numpy", "pandas", "scipy", "sklearn", "lightgbm", "xgboost", "torch",
        ],
        description="importable in a candidate; anything else fails F0",
    )

    # --- Protected: frozen evaluation machinery ------------------------
    portfolio_builder: str = Field(default="topk_dropout")
    topk: int = Field(default=50, ge=1)
    n_drop: int = Field(default=5, ge=0)
    transaction_cost_bps: float = Field(
        default=15.0, description="round-trip cost in basis points"
    )
    backtest_engine: str = Field(default="qlib.backtest")
    #: identifiers a candidate must not assign to; assigning means it is
    #: rewriting the evaluation protocol rather than the model
    protected_symbols: List[str] = Field(
        default_factory=lambda: [
            "sealed_promotion", "final_test", "transaction_cost", "backtest_config",
            "portfolio_builder", "topk", "n_drop", "embargo", "split_config",
        ]
    )

    # --- resource ceilings --------------------------------------------
    max_candidate_gpu_seconds: float = Field(default=1800.0)
    max_candidate_wall_seconds: float = Field(default=3600.0)

    def compute_hash(self) -> str:
        """Hash of the whole contract; stamped onto every CandidatePackage."""
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @classmethod
    def for_mode(cls, mode: "CandidateMode", **overrides: Any) -> "TaskSpec":
        """Build a spec whose search space matches a candidate mode."""
        return cls(allowed_model_families=CandidateMode(mode).families(), **overrides)

    @property
    def candidate_mode(self) -> "CandidateMode":
        """Which mode this spec's family list corresponds to."""
        families = set(self.allowed_model_families)
        if families == {"formula"}:
            return CandidateMode.FORMULA
        if "formula" in families:
            return CandidateMode.MIXED
        return CandidateMode.MODEL

    def protected_hash(self) -> str:
        """Hash of only the protected half.

        Split out so a paper can state that two runs shared identical
        evaluation machinery even if their search spaces differed.
        """
        payload = {
            "portfolio_builder": self.portfolio_builder,
            "topk": self.topk,
            "n_drop": self.n_drop,
            "transaction_cost_bps": self.transaction_cost_bps,
            "backtest_engine": self.backtest_engine,
        }
        blob = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


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


# =============================================================================
# Candidate package
# =============================================================================


class QuantModelSpec(BaseModel):
    """The structured description of what a candidate actually is.

    Kept alongside the generated code (not instead of it): the code is what
    runs, this is what the policy and the analysis reason over. Having it
    typed is what lets ``edit_target`` mean something — an action can say
    "change the architecture" and a diff can show exactly what moved.
    """

    model_family: str = "gbdt"
    feature_pipeline: Dict[str, Any] = Field(
        default_factory=dict,
        description="feature subset, normalisation, missing-value policy",
    )
    architecture: Dict[str, Any] = Field(
        default_factory=dict, description="model structure / hyperparameters"
    )
    loss: str = "mse"
    optimizer: Dict[str, Any] = Field(default_factory=dict)
    training: Dict[str, Any] = Field(
        default_factory=dict, description="epochs, early stopping, batch size, seeds"
    )

    def component(self, name: str) -> Dict[str, Any]:
        """Read one editable component by ``edit_target`` name."""
        return getattr(self, name, {}) if isinstance(getattr(self, name, None), dict) else {}


class CandidatePackage(BaseModel):
    """Everything needed to evaluate, reproduce and audit one candidate.

    ``task_spec_hash`` and ``data_contract_hash`` travel with the candidate so
    a result can never be silently compared against a different protocol.
    """

    candidate_id: str
    episode_id: str
    code: str
    code_hash: str
    spec: QuantModelSpec = Field(default_factory=QuantModelSpec)
    required_packages: List[str] = Field(default_factory=list)
    parent_ids: List[str] = Field(default_factory=list)
    lineage: Optional["CandidateLineage"] = None
    decision_id: Optional[str] = None
    expert: Optional[str] = None
    data_contract_hash: str = ""
    task_spec_hash: str = ""
    generation_meta: Dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def hash_code(code: str) -> str:
        return hashlib.sha256(code.encode("utf-8")).hexdigest()


class EvalRequest(BaseModel):
    """Resource contract for ONE evaluation.

    Carries the ceilings the action asked for. Without this object
    ``StructuredAction.max_gpu_seconds`` had nowhere to land and was silently
    ignored — the policy could 'choose' a compute budget that never applied.
    """

    request_id: str
    candidate_id: str
    fidelity: Fidelity
    seed_list: List[int] = Field(default_factory=lambda: [11])
    split_scope: Literal["visible_dev"] = Field(
        default="visible_dev",
        description="the visible evaluator can address no other split; "
        "sealed evaluation goes through SealedGateService",
    )
    max_gpu_seconds: Optional[float] = None
    timeout_seconds: Optional[float] = None
    max_boost_rounds: Optional[int] = None
    train_fraction: float = Field(
        default=1.0, gt=0.0, le=1.0,
        description="fraction of the train window used (F1 shortens it)",
    )
    universe_fraction: float = Field(
        default=1.0, gt=0.0, le=1.0,
        description="fraction of the stock pool used (F1 subsamples it)",
    )

    model_config = ConfigDict(use_enum_values=False)


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
    promotion_target_id: Optional[str] = Field(
        default=None,
        description=(
            "Promote an EXISTING candidate using the evidence it already has. "
            "When set, the decision produces no new candidate: without it a "
            "'promote the stable candidate' intent silently promotes a fresh "
            "mutation of that candidate instead."
        ),
    )
    rationale: str = ""

    model_config = ConfigDict(use_enum_values=False)


class DecisionRecord(BaseModel):
    """Full trace of one agent decision (for offline RL / audit)."""

    decision_id: str
    observation_digest: str = Field(
        description="sha256 of the serialized AgentObservation"
    )
    observation_features: Optional[List[float]] = Field(
        default=None,
        description=(
            "encoded observation at decision time. The digest proves WHICH "
            "observation was seen; these are what a policy can learn from. "
            "Stored rather than reconstructed because the archives have moved "
            "on by training time — replaying them would rebuild a different "
            "state than the one the agent actually acted on."
        ),
    )
    encoding_version: Optional[str] = Field(
        default=None,
        description="feature encoder version; mismatched records are skipped",
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
    """One state -> action -> evidence -> next_state step for the RL trainer.

    ``state_version`` is the version the observation was built at and
    ``next_state_version`` is the version the BarrierCommit produced, so the
    (s, a, s') boundary is explicit rather than inferred from ordering.
    """

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
    next_state_version: Optional[int] = Field(
        default=None,
        description="state version after the barrier committed this batch",
    )
    stale: bool = Field(
        default=False,
        description=(
            "the decision was made against an older state version than the "
            "one live at commit time; keep for audit, filter for on-policy RL"
        ),
    )
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


# CandidatePackage forward-references CandidateLineage, which is defined below
# it; resolve now that both exist.
CandidatePackage.model_rebuild()
