"""State and action spaces for the meta-policy.

Two jobs:

- ``ObservationEncoder``: AgentObservation -> fixed-length float vector.
- ``ActionCodec``: StructuredAction <-> a factored discrete action.

Design notes that matter for what the policy can possibly learn:

Features are RATIOS and BOUNDED COUNTS, never raw magnitudes. "1400 GPU
seconds left" means nothing without knowing the initial grant; "0.35 of the
budget left" transfers across episodes with different budgets. Same for
RankIC: what matters is the gain over the incumbent, not the level, because
the level is dominated by the market regime of that episode's dev window. A
policy trained on levels would learn "2014 is a good year", which does not
generalise to E7.

The action space is FACTORED (five independent heads) rather than one flat
categorical over the cross-product. The cross-product is ~6*4*3*4*2 = 576
classes, which is hopeless to fit from a few hundred transitions; the factored
form shares statistical strength across choices that genuinely are close to
independent (fidelity is not really a function of which family you picked).

Bumping ENCODING_VERSION invalidates existing checkpoints on purpose.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from famou.reliability.judge import FailureKind
from famou.reliability.observation import AgentObservation
from famou.reliability.types import ExpertKind, Fidelity, StructuredAction

#: Stamped into every PolicyCheckpoint; a mismatch on load is an error.
#: v2 added the "formula" family — the action space changed shape, so v1
#: checkpoints cannot be reused.
ENCODING_VERSION = "enc_v2"

#: Action head vocabularies. Order is part of the encoding contract.
EXPERTS: Tuple[str, ...] = tuple(e.value for e in ExpertKind)
FAMILIES: Tuple[str, ...] = ("formula", "gbdt", "linear", "mlp", "temporal_transformer")
FIDELITIES: Tuple[int, ...] = (0, 1, 2)
BATCH_BUCKETS: Tuple[int, ...] = (1, 2, 4, 8)
PROMOTION: Tuple[bool, ...] = (False, True)

#: Failure kinds that become observation features, in fixed order.
TRACKED_FAILURES: Tuple[FailureKind, ...] = (
    FailureKind.SYNTAX,
    FailureKind.CONTRACT,
    FailureKind.FORBIDDEN_EDIT,
    FailureKind.LEAKAGE,
    FailureKind.DEPENDENCY,
    FailureKind.SHAPE,
    FailureKind.NAN_OUTPUT,
    FailureKind.DEGENERATE,
    FailureKind.DATA,
    FailureKind.TIMEOUT,
    FailureKind.OOM,
    FailureKind.CRASH,
    FailureKind.NO_IMPROVEMENT,
)

#: Reference budget grants, used only to turn absolutes into ratios when the
#: ledger reports an unbounded (inf) limit.
_REF_GPU_SECONDS = 100_000.0
_REF_VISIBLE = 500.0
_REF_SEALED = 20.0


def _ratio(value: Optional[float], reference: float) -> float:
    """Bounded [0,1] ratio; inf/None -> 1.0 (treat unbounded as 'plenty left')."""
    if value is None:
        return 0.0
    if value == float("inf"):
        return 1.0
    if reference <= 0:
        return 0.0
    return max(0.0, min(1.0, float(value) / reference))


def _squash(x: float, scale: float) -> float:
    """Map an unbounded quantity into [-1, 1] without clipping information."""
    if scale <= 0:
        return 0.0
    z = float(x) / scale
    return z / (1.0 + abs(z))


class ObservationEncoder:
    """AgentObservation -> fixed-length feature vector."""

    version = ENCODING_VERSION

    #: Feature names, in vector order. Kept explicit so a trained policy can be
    #: inspected ("which feature drives promotion?") rather than being a
    #: black box over an anonymous array.
    FEATURE_NAMES: Tuple[str, ...] = (
        # --- budget (ratios: transfer across episodes) -----------------
        "budget_gpu_frac",
        "budget_visible_frac",
        "budget_sealed_frac",
        "budget_sealed_empty",
        # --- archive scale --------------------------------------------
        "n_candidates_norm",
        "n_certified_norm",
        "has_certified",
        "has_incumbent",
        # --- best uncertified candidate vs incumbent -------------------
        "top_gain_vs_incumbent",
        "top_rank_ic_std",
        "top_n_f2_seeds_norm",
        "top_is_f2",
        "top_gate_attempted",
        "top_novelty",
        # --- second best (is there a real choice, or one lucky run?) ---
        "second_gain_vs_incumbent",
        "gap_top_to_second",
        # --- family success rates -------------------------------------
        "family_gbdt_valid_rate",
        "family_mlp_valid_rate",
        "family_formula_valid_rate",
        "family_best_valid_rate",
        "n_families_tried_norm",
        # --- progress / stagnation ------------------------------------
        "state_version_norm",
        "in_flight_norm",
        "frac_candidates_certified",
    ) + tuple(f"fail_{k.value}" for k in TRACKED_FAILURES)

    def __init__(
        self,
        *,
        ref_gpu_seconds: float = _REF_GPU_SECONDS,
        ref_visible: float = _REF_VISIBLE,
        ref_sealed: float = _REF_SEALED,
        candidate_scale: float = 100.0,
    ):
        self.ref_gpu_seconds = ref_gpu_seconds
        self.ref_visible = ref_visible
        self.ref_sealed = ref_sealed
        self.candidate_scale = candidate_scale

    @property
    def dim(self) -> int:
        return len(self.FEATURE_NAMES)

    def encode(self, obs: AgentObservation) -> List[float]:
        budget = obs.remaining_budget or {}
        summary = obs.search_archive_summary or {}
        incumbent = obs.incumbent_rank_ic

        n_cand = float(summary.get("n_candidates", 0))
        n_cert = float(summary.get("n_certified", 0))
        families = summary.get("families") or []

        top = obs.top_visible_evidence[0] if obs.top_visible_evidence else None
        second = obs.top_visible_evidence[1] if len(obs.top_visible_evidence) > 1 else None

        def gain(cand) -> float:
            if cand is None or cand.best_rank_ic is None:
                return 0.0
            # Gain over the incumbent, squashed. The LEVEL is a property of the
            # episode's market regime, not of the search; the gain is what the
            # policy can actually influence.
            return _squash(cand.best_rank_ic - (incumbent or 0.0), 0.02)

        sealed_left = budget.get("sealed_queries", 0.0)
        stats = obs.model_family_stats or {}

        def valid_rate(family: str) -> float:
            entry = stats.get(family) or {}
            return float(entry.get("valid_rate", 0.0))

        best_rate = max((float((v or {}).get("valid_rate", 0.0)) for v in stats.values()),
                        default=0.0)

        vec: List[float] = [
            _ratio(budget.get("gpu_seconds"), self.ref_gpu_seconds),
            _ratio(budget.get("visible_queries"), self.ref_visible),
            _ratio(sealed_left, self.ref_sealed),
            1.0 if sealed_left < 1 else 0.0,

            min(1.0, n_cand / self.candidate_scale),
            min(1.0, n_cert / max(1.0, self.candidate_scale / 10.0)),
            1.0 if obs.certified_candidates else 0.0,
            1.0 if incumbent is not None else 0.0,

            gain(top),
            _squash(top.rank_ic_std or 0.0, 0.02) if top else 0.0,
            min(1.0, (top.n_f2_seeds if top else 0) / 5.0),
            1.0 if (top and top.highest_fidelity >= 2) else 0.0,
            1.0 if (top and top.gate_attempts > 0) else 0.0,
            float(top.novelty) if (top and top.novelty is not None) else 1.0,

            gain(second),
            (gain(top) - gain(second)) if (top and second) else 0.0,

            valid_rate("gbdt"),
            valid_rate("mlp"),
            valid_rate("formula"),
            best_rate,
            min(1.0, len(families) / 4.0),

            min(1.0, obs.state_version / 200.0),
            min(1.0, obs.in_flight_tasks / 16.0),
            (n_cert / n_cand) if n_cand > 0 else 0.0,
        ]

        failures = obs.recent_failure_patterns or {}
        total_failures = max(1.0, float(sum(failures.values())))
        vec.extend(failures.get(kind.value, 0) / total_failures for kind in TRACKED_FAILURES)

        assert len(vec) == self.dim, f"encoder produced {len(vec)} != {self.dim}"
        return vec

    def describe(self, obs: AgentObservation) -> Dict[str, float]:
        """Named features — for debugging and for explaining a policy."""
        return dict(zip(self.FEATURE_NAMES, self.encode(obs)))


class ActionCodec:
    """StructuredAction <-> factored discrete action.

    Heads: expert, model_family, fidelity, batch bucket, promotion flag.
    """

    version = ENCODING_VERSION
    HEADS: Tuple[str, ...] = ("expert", "family", "fidelity", "batch", "promote")

    @property
    def head_sizes(self) -> Tuple[int, ...]:
        return (len(EXPERTS), len(FAMILIES), len(FIDELITIES),
                len(BATCH_BUCKETS), len(PROMOTION))

    # ------------------------------------------------------------------

    @staticmethod
    def _index(seq: Sequence[Any], value: Any, default: int = 0) -> int:
        try:
            return list(seq).index(value)
        except ValueError:
            return default

    def encode(self, action: StructuredAction) -> Tuple[int, ...]:
        batch = int(action.batch_size)
        # Round DOWN to a bucket: a policy asked for 3 rollouts should be
        # recorded as the 2-bucket it can actually be served, not the 4 it
        # cannot.
        bucket = 0
        for i, b in enumerate(BATCH_BUCKETS):
            if batch >= b:
                bucket = i
        return (
            self._index(EXPERTS, action.expert.value),
            self._index(FAMILIES, action.model_family),
            self._index(FIDELITIES, int(action.fidelity.value), default=1),
            bucket,
            1 if action.promotion_requested else 0,
        )

    def decode(
        self,
        indices: Sequence[int],
        *,
        parent_ids: Optional[List[str]] = None,
        seed_list: Optional[List[int]] = None,
        promotion_target_id: Optional[str] = None,
        rationale: str = "",
    ) -> StructuredAction:
        e, f, fid, b, p = (int(i) for i in indices)
        fidelity = Fidelity(FIDELITIES[min(fid, len(FIDELITIES) - 1)])
        # Seeds are not a learned head: multi-seed evidence is what the
        # promotion policy needs, so the count follows the fidelity rather than
        # being something the policy can economise on.
        default_seeds = [11] if fidelity != Fidelity.F2_FULL else [11, 29, 47]
        return StructuredAction(
            expert=ExpertKind(EXPERTS[min(e, len(EXPERTS) - 1)]),
            parent_ids=list(parent_ids or []),
            model_family=FAMILIES[min(f, len(FAMILIES) - 1)],
            fidelity=fidelity,
            seed_list=list(seed_list or default_seeds),
            batch_size=BATCH_BUCKETS[min(b, len(BATCH_BUCKETS) - 1)],
            promotion_requested=bool(PROMOTION[min(p, 1)]),
            promotion_target_id=promotion_target_id,
            rationale=rationale,
        )
