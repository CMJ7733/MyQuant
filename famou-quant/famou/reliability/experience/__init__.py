"""Evidence-grounded experience layer (stages 0-4).

What this is
------------
A retrieval layer over what the search has already learned. It is deliberately
NOT a copy of the event log: ``ExperienceRecord`` holds references
(``transition_ids``, ``evidence_ids``, ``candidate_ids``) into the Trajectory
Store and Search Archive, which remain the single source of truth for what
happened. Copying those events into a second store would reintroduce exactly
the dual-source-of-truth drift that invariant C1 removed — and the RL trainer
replays trajectories, so a divergence between the two would be silent.

Two retrievals per decision, and why
------------------------------------
Retrieval feeds the policy, but retrieval is also something the policy
chooses. Taken literally that is circular: the policy would have to decide how
much to retrieve before seeing anything. The resolution is two queries with
different jobs:

1. **Decision-phase** (stage 3): fixed small ``top_k``, always runs, no family
   filter (the action has not been chosen yet). Its *summary* — counts,
   strongest weight, how many patterns are repairable or policy-level — goes
   into ``AgentObservation`` as six features. The statements do not: a policy
   over ~40 features cannot read prose, and letting the observation grow with
   the index would break the fixed-width contract.

2. **Proposal-phase** (stages 2 + 4): size chosen by the policy's retrieval
   head, filtered to the family the action actually picked. Its records reach
   the expert as ``ExperienceGuidance``, and its token cost is charged to the
   batch so the head pays for what it asked for.

So the policy sees a cheap summary and decides how much context to buy for
generation. No circularity, and the cost lands on the decision that incurred
it.

Design invariants (same weight as the ones in ``famou.reliability``)
--------------------------------------------------------------------

E1. **Version isolation.** ``ExperienceIndex.visible_at(n)`` returns only
    records with ``valid_from_state_version <= n``. Experience produced by a
    batch committing at version n+1 is invisible to a decision taken at n.
    This is not merely an anti-lookahead rule for offline RL — it is C1
    applied to retrieval. ``AgentObservation`` is built at the last committed
    version; if retrieval read a live index instead, two replays of the same
    trajectory would feed the policy different context while producing an
    identical ``observation_digest``, and the divergence would never surface
    as an error.

E2. **Writes happen inside the barrier window.** ``ExperienceConsolidator``
    is called by ``BarrierCommit`` under the same ``CommitGuard`` window that
    writes the archives, so indexed experience and archive state always share
    a version number.

E3. **Structured filters first, text second.** ``statement`` is never the sole
    retrieval key. Matching runs on ``applicability`` (model family, failure
    kind, protocol version); free text only ranks or explains. Otherwise a
    single hallucinated summary becomes a retrievable "fact" — and one dressed
    in supporting_experience_ids at that. Stage 1 uses no LLM at all: the
    statement is rendered deterministically from counts.

E4. **No sealed numerics, ever.** Experience may record a gate verdict as
    PROMOTE / REJECT / INCONCLUSIVE plus reason_code, never a sealed score,
    margin distance, or date. See also the cross-episode caveat below.

E5. **final_test never enters.** ``PaperResult`` is not accepted anywhere in
    this package, mirroring ``famou.reliability.final_test``.

E6. **Guidance steers away, never toward.** Constraints are derived from
    failure patterns only. There is deliberately no "this worked, do it
    again": copying past winners collapses a search onto one lineage, and a
    single good visible score is exactly the weak evidence the reliability
    layer exists to discount.

Known gap, deliberately left open
---------------------------------
Cross-episode aggregation of gate verdicts is a leakage channel the
per-episode sealed budget does not bound. A single verdict is ~2 bits; an
experience aggregating verdicts across many candidates and episodes is a
learned model of the gate's decision function, which is shared across
episodes. ``ExperienceRecord.episode_ids`` exists so a reporting run can
exclude patterns derived from its own episode, but nothing enforces that yet
— it lands with the Certified Pattern memory, which is the first memory type
that actually aggregates verdicts. Failure memory reads no verdicts at all,
so the channel is closed for now.
"""

from famou.reliability.experience.consolidator import ExperienceConsolidator
from famou.reliability.experience.failure import (
    FailureMemory,
    ObservedOutcome,
    failure_experience_id,
)
from famou.reliability.experience.guidance import (
    ExperienceGuidance,
    build_guidance,
    derive_constraints,
)
from famou.reliability.experience.index import ExperienceIndex
from famou.reliability.experience.retriever import MemoryRetriever
from famou.reliability.experience.types import (
    EvidenceLevel,
    ExperienceRecord,
    ExperienceType,
    QueryType,
    RetrievalBundle,
    RETRIEVAL_VERSION,
    reliability_weight,
)

__all__ = [
    "ExperienceConsolidator",
    "ExperienceGuidance",
    "ExperienceIndex",
    "ExperienceRecord",
    "ExperienceType",
    "EvidenceLevel",
    "FailureMemory",
    "MemoryRetriever",
    "ObservedOutcome",
    "QueryType",
    "RetrievalBundle",
    "RETRIEVAL_VERSION",
    "build_guidance",
    "derive_constraints",
    "failure_experience_id",
    "reliability_weight",
]
