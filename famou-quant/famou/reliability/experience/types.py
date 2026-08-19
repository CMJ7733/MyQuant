"""Types for the experience layer.

Kept deliberately smaller than the original design sketch in one respect:
there is no confidence interval on an aggregated effect. A CI computed over
three supporting cases is not a confidence interval, and claiming statistical
rigour the data does not support is worse than reporting raw counts — the same
reason ``EvidenceVector`` distinguishes "high mean, high variance" from "lower
mean, stable" instead of collapsing both into one score.

What replaces it is ``reliability_weight``: an explicit *ranking* weight, not
a probability, documented as such at its definition.
"""

from __future__ import annotations

import math
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

#: Bump when the retrieval scoring function or filter semantics change, so a
#: study can tell which bundles are comparable. Unlike ENCODING_VERSION this
#: does NOT invalidate policy checkpoints — stage 0/1 retrieval never reaches
#: the policy — but it does partition any A/B analysis of retrieval quality.
RETRIEVAL_VERSION = "retrieval_v0"


class ExperienceType(str, Enum):
    EPISODIC = "episodic"       # index entry pointing at one Transition
    FAILURE = "failure"         # aggregated failure/repair pattern (stage 1)
    SEMANTIC = "semantic"       # induced regularity (stage 3+)
    CERTIFIED = "certified"     # derived from sealed-gate promotions (stage 3+)


class EvidenceLevel(str, Enum):
    """How much weight a record has earned. Ordered weakest to strongest."""

    PROVISIONAL = "provisional"              # single cheap evaluation
    VISIBLE_MULTISEED = "visible_multiseed"  # F2, multiple seeds
    SEALED_CERTIFIED = "sealed_certified"    # survived the sealed gate


#: Ranking multipliers per evidence level. Not probabilities.
_LEVEL_WEIGHT: Dict[EvidenceLevel, float] = {
    EvidenceLevel.PROVISIONAL: 0.3,
    EvidenceLevel.VISIBLE_MULTISEED: 0.7,
    EvidenceLevel.SEALED_CERTIFIED: 1.0,
}

#: Sample count at which the saturating term reaches ~63% of its ceiling.
_SAMPLE_SCALE = 5.0


def reliability_weight(level: EvidenceLevel, sample_count: int) -> float:
    """Retrieval ranking weight in [0, 1]. **Not** a confidence or a p-value.

    Two properties are all that is claimed: it increases with evidence level,
    and it saturates in sample count so the tenth observation of a pattern
    moves it far less than the second. Anything read into it beyond ordering
    is over-reading.
    """
    ceiling = _LEVEL_WEIGHT.get(level, 0.3)
    saturation = 1.0 - math.exp(-max(0, sample_count) / _SAMPLE_SCALE)
    return round(ceiling * saturation, 6)


class QueryType(str, Enum):
    POLICY_DECISION = "policy_decision"
    PARENT_SELECTION = "parent_selection"
    GENERATION = "generation"
    DEBUG = "debug"
    PROMOTION = "promotion"


class ExperienceRecord(BaseModel):
    """One retrievable piece of experience.

    Holds references, not copies: the events themselves stay in the Trajectory
    Store and Search Archive (see the package docstring, invariant E1/E2).
    """

    experience_id: str
    experience_type: ExperienceType

    #: Human-readable rendering. Stage 1 generates this deterministically from
    #: counts; later stages may use an LLM. Never the sole retrieval key (E3).
    statement: str = ""
    #: Structured hard-filter keys — this is what retrieval actually matches on.
    applicability: Dict[str, Any] = Field(default_factory=dict)
    action_pattern: Dict[str, Any] = Field(default_factory=dict)
    outcome_summary: Dict[str, Any] = Field(default_factory=dict)

    # -- provenance: references into the existing stores ------------------
    transition_ids: List[str] = Field(default_factory=list)
    decision_ids: List[str] = Field(default_factory=list)
    candidate_ids: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    contradicting_transition_ids: List[str] = Field(default_factory=list)

    # -- reliability -------------------------------------------------------
    evidence_level: EvidenceLevel = EvidenceLevel.PROVISIONAL
    sample_count: int = 0
    reliability_weight: float = Field(
        default=0.0,
        description="ranking weight in [0,1]; not a probability — see "
                    "reliability_weight()",
    )

    # -- temporal / protocol ----------------------------------------------
    created_at: float = 0.0
    updated_at: float = 0.0
    valid_from_state_version: int = Field(
        description="first state version at which this record may be retrieved; "
                    "enforced by ExperienceIndex.visible_at (invariant E1)",
    )
    episode_ids: List[str] = Field(default_factory=list)
    data_protocol_version: str = ""
    policy_versions: List[str] = Field(default_factory=list)

    # -- safety ------------------------------------------------------------
    split_scope: str = "visible_dev"
    leakage_safe: bool = Field(
        default=True,
        description="False marks a record that must never be retrieved during "
                    "a reporting run",
    )
    deprecated: bool = False

    model_config = ConfigDict(use_enum_values=False)

    def refresh_weight(self) -> None:
        self.reliability_weight = reliability_weight(
            self.evidence_level, self.sample_count
        )


class RetrievalBundle(BaseModel):
    """What one retrieval call returned, recorded whether or not it was used.

    Recording this is the point of stage 0: without it there is no way to ask
    afterwards whether retrieval helped, or which experience a decision was
    standing on. ``consumed_by_policy`` is the A/B key — in stages 0-1 it is
    always False, because the bundle is collected but never fed to the policy.
    """

    query_id: str = Field(default_factory=lambda: f"ret_{uuid.uuid4().hex[:12]}")
    decision_id: Optional[str] = None
    state_version: int
    query_type: QueryType

    query_filters: Dict[str, Any] = Field(default_factory=dict)
    retrieved_experience_ids: List[str] = Field(default_factory=list)
    retrieval_scores: List[float] = Field(default_factory=list)
    evidence_levels: List[str] = Field(default_factory=list)

    n_candidates_considered: int = 0
    token_cost: int = 0
    retrieval_version: str = RETRIEVAL_VERSION
    consumed_by_policy: bool = Field(
        default=False,
        description="whether the retrieved experience actually entered the "
                    "policy's input; False for stages 0-1",
    )
    created_at: float = 0.0

    model_config = ConfigDict(use_enum_values=False)
