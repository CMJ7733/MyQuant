"""Experience layer stages 2-4: guidance, observation features, retrieval action.

Stage 2  experience reaches the proposal side and changes what is generated
Stage 3  a summary reaches the policy's observation (ENCODING_VERSION -> v3)
Stage 4  the policy chooses how much to retrieve, and pays for it
"""

from __future__ import annotations

import pytest

from famou.core.state import StateStore
from famou.reliability.experience import (
    ExperienceIndex,
    ExperienceRecord,
    ExperienceType,
    FailureMemory,
    MemoryRetriever,
    ObservedOutcome,
)
from famou.reliability.experience.guidance import (
    ExperienceGuidance,
    build_guidance,
    derive_constraints,
)
from famou.reliability.experts import GBDTExpert, NNExpert, FormulaExpert
from famou.reliability.observation import (
    AgentObservation,
    RetrievedExperienceSummary,
)
from famou.reliability.reward import RewardConfig
from famou.reliability.rl.encoding import (
    ENCODING_VERSION,
    RETRIEVAL_BUCKETS,
    ActionCodec,
    ObservationEncoder,
)
from famou.reliability.types import (
    EvaluationCost,
    EvidenceVector,
    ExpertKind,
    Fidelity,
    StructuredAction,
)


def oom_record(family="gbdt", occurrences=3, kind="oom") -> ExperienceRecord:
    rec = ExperienceRecord(
        experience_id=f"fail::{kind}::{family}",
        experience_type=ExperienceType.FAILURE,
        statement=f"{family} hit {kind} {occurrences}x",
        applicability={"failure_kind": kind, "model_family": family},
        outcome_summary={"occurrences": occurrences, "repairable": False,
                         "policy_level": False},
        sample_count=occurrences,
        valid_from_state_version=0,
    )
    rec.refresh_weight()
    return rec


class TestGuidanceConstraints:
    def test_repeated_oom_constrains_growth(self):
        constraints = derive_constraints([oom_record(occurrences=3)])
        assert constraints["avoid_growth"] is True
        assert constraints["reasons"] == ["oom x3"]

    def test_single_occurrence_is_not_enough(self):
        """One OOM is an accident; a constraint that fires on it would shrink
        the search space on noise."""
        assert derive_constraints([oom_record(occurrences=1)]) == {}

    def test_unrelated_failures_produce_no_constraint(self):
        assert derive_constraints([oom_record(kind="shape", occurrences=9)]) == {}

    def test_timeout_also_constrains_growth(self):
        assert derive_constraints([oom_record(kind="timeout")])["avoid_growth"]

    def test_empty_guidance_is_falsy(self):
        assert not ExperienceGuidance()
        assert build_guidance([oom_record()])

    def test_prompt_block_shows_statements_not_ids_or_scores(self):
        block = build_guidance([oom_record()]).as_prompt_block()
        assert "gbdt hit oom 3x" in block
        assert "fail::oom::gbdt" not in block      # no ids to pattern-match on
        assert "0." not in block                   # no scores


class TestGuidedProposal:
    """Stage 2: guidance actually changes generated candidates."""

    def _action(self, family="gbdt"):
        return StructuredAction(expert=ExpertKind.MUTATE, model_family=family)

    def test_gbdt_capacity_does_not_climb_under_avoid_growth(self):
        parent = GBDTExpert(ExpertKind.EXPLORE, rng_seed=1).propose(
            self._action(), [], iteration=1
        )
        p_params = GBDTExpert.extract_hyperparams(parent)
        guidance = build_guidance([oom_record()])

        # Many seeds: an unconstrained mutation grows roughly half the time,
        # so a single sample would pass by luck.
        for seed in range(12):
            child = GBDTExpert(ExpertKind.MUTATE, rng_seed=seed).propose(
                self._action(), [parent], iteration=2, guidance=guidance
            )
            c_params = GBDTExpert.extract_hyperparams(child)
            assert c_params["num_leaves"] <= p_params["num_leaves"]
            assert c_params["max_depth"] <= p_params["max_depth"]

    def test_formula_never_adds_a_term_under_avoid_growth(self):
        """Formula mode was the one context where retrieval had zero effect,
        because FormulaExpert did not read constraints. Now it does."""
        parent = FormulaExpert(ExpertKind.EXPLORE, rng_seed=1).propose(
            self._action("formula"), [], iteration=1
        )
        p_terms = FormulaExpert.extract_hyperparams(parent)["terms"]
        guidance = build_guidance([oom_record(family="formula")])

        for seed in range(20):
            child = FormulaExpert(ExpertKind.MUTATE, rng_seed=seed).propose(
                self._action("formula"), [parent], iteration=2, guidance=guidance
            )
            c_terms = FormulaExpert.extract_hyperparams(child)["terms"]
            assert len(c_terms) <= len(p_terms)

        """The constraint has to be doing something — otherwise the test above
        proves nothing."""
        parent = GBDTExpert(ExpertKind.EXPLORE, rng_seed=1).propose(
            self._action(), [], iteration=1
        )
        p_leaves = GBDTExpert.extract_hyperparams(parent)["num_leaves"]

        grew = any(
            GBDTExpert.extract_hyperparams(
                GBDTExpert(ExpertKind.MUTATE, rng_seed=s).propose(
                    self._action(), [parent], iteration=2
                )
            )["num_leaves"] > p_leaves
            for s in range(12)
        )
        assert grew

    def test_nn_never_adds_a_layer_under_avoid_growth(self):
        parent = NNExpert(ExpertKind.EXPLORE, rng_seed=1, family="mlp").propose(
            self._action("mlp"), [], iteration=1
        )
        p_dims = NNExpert.extract_hyperparams(parent)["hidden_dims"]
        guidance = build_guidance([oom_record(family="mlp")])

        for seed in range(12):
            child = NNExpert(ExpertKind.MUTATE, rng_seed=seed, family="mlp").propose(
                self._action("mlp"), [parent], iteration=2, guidance=guidance
            )
            dims = NNExpert.extract_hyperparams(child)["hidden_dims"]
            assert len(dims) <= len(p_dims)
            assert max(dims) <= max(p_dims)

    def test_guidance_is_recorded_on_the_candidate(self):
        guidance = build_guidance([oom_record()])
        child = GBDTExpert(ExpertKind.MUTATE, rng_seed=0).propose(
            self._action(), [], iteration=1, guidance=guidance
        )

        assert child.meta["guided_by"] == ["fail::oom::gbdt"]
        assert child.meta["guidance_constraints"]["avoid_growth"] is True

    def test_no_guidance_leaves_no_trace(self):
        child = GBDTExpert(ExpertKind.MUTATE, rng_seed=0).propose(
            self._action(), [], iteration=1
        )
        assert "guided_by" not in child.meta


class TestObservationEncoding:
    """Stage 3: retrieval summary becomes features."""

    def _obs(self, retrieved=None):
        return AgentObservation(
            state_version=1, episode_id="E1", retrieved_experience=retrieved
        )

    def test_encoding_version_tripwire(self):
        """Deliberate tripwire: changing the observation or action encoding
        must be a conscious act, because it invalidates every checkpoint and
        every trajectory collected so far. If this fails, confirm the bump was
        intended and update the expected value."""
        assert ENCODING_VERSION == "enc_v4"

    def test_vector_width_matches_declared_names(self):
        encoder = ObservationEncoder()
        assert len(encoder.encode(self._obs())) == encoder.dim
        assert len(encoder.FEATURE_NAMES) == encoder.dim

    def test_disabled_retrieval_is_distinguishable_from_empty(self):
        """Both look like 'no experience', but only one means the memory layer
        was off — a policy that cannot tell them apart learns from unretrieved
        runs as if memory had been useless."""
        encoder = ObservationEncoder()
        off = encoder.describe(self._obs(None))
        ran_empty = encoder.describe(
            self._obs(RetrievedExperienceSummary(n_retrieved=0, n_available=0))
        )

        assert off["retrieval_ran"] == 0.0
        assert ran_empty["retrieval_ran"] == 1.0

    def test_repairable_and_policy_level_fractions_are_encoded(self):
        encoder = ObservationEncoder()
        feats = encoder.describe(self._obs(RetrievedExperienceSummary(
            n_retrieved=4, n_available=10, max_reliability=0.8,
            n_repairable=3, n_policy_level=1,
        )))

        assert feats["retrieval_frac_repairable"] == pytest.approx(0.75)
        assert feats["retrieval_frac_policy_level"] == pytest.approx(0.25)
        assert feats["retrieval_max_reliability"] == pytest.approx(0.8)

    def test_features_stay_bounded_on_absurd_input(self):
        encoder = ObservationEncoder()
        vec = encoder.encode(self._obs(RetrievedExperienceSummary(
            n_retrieved=9999, n_available=9999, max_reliability=50.0,
            n_repairable=9999, n_policy_level=9999,
        )))
        assert all(-1.0 <= v <= 1.0 for v in vec)


class TestRetrievalAction:
    """Stage 4: retrieval is a choice with a price."""

    def test_codec_round_trips_the_retrieval_head(self):
        codec = ActionCodec()
        for k in RETRIEVAL_BUCKETS:
            action = StructuredAction(expert=ExpertKind.MUTATE, retrieval_top_k=k)
            decoded = codec.decode(codec.encode(action))
            assert decoded.retrieval_top_k == k

    def test_zero_retrieval_is_reachable(self):
        """The control condition: a policy that cannot decline retrieval
        cannot learn that context costs anything."""
        assert RETRIEVAL_BUCKETS[0] == 0
        codec = ActionCodec()
        assert codec.decode([0, 0, 1, 0, 0, 0]).retrieval_top_k == 0

    def test_head_count_and_sizes_agree(self):
        codec = ActionCodec()
        assert len(codec.HEADS) == len(codec.head_sizes) == 6
        assert codec.HEADS[-1] == "retrieval"

    def test_promote_is_read_by_name_not_position(self):
        """Adding the retrieval head moved promote off the end; a positional
        read would silently turn 'retrieval bucket 1' into 'promote'."""
        codec = ActionCodec()
        promote_at = codec.HEADS.index("promote")
        indices = [0, 0, 1, 0, 1, 0]         # promote=1, retrieval=0
        assert indices[promote_at] == 1
        assert codec.decode(indices).promotion_requested is True
        assert codec.decode(indices).retrieval_top_k == 0

    def test_retrieval_tokens_are_charged_in_the_reward(self):
        from famou.reliability.archives import SearchArchive
        from famou.reliability.reward import RewardBuilder
        from famou.reliability.trajectory import TrajectoryStore, build_transition
        from famou.reliability.types import DecisionRecord

        search = SearchArchive(StateStore())
        builder = RewardBuilder(search)
        store = TrajectoryStore()

        def transition(tokens):
            return build_transition(
                decision=DecisionRecord(
                    decision_id="d", observation_digest="o",
                    structured_action=StructuredAction(expert=ExpertKind.MUTATE),
                    state_version=1, timestamp=0.0,
                ),
                candidate_ids=[], evidence_ids=[],
                costs=EvaluationCost(retrieval_tokens=tokens),
            )

        free = builder.mature(store, transition(0), incumbent_rank_ic=None)
        expensive = builder.mature(store, transition(1000), incumbent_rank_ic=None)

        assert expensive < free
        assert free - expensive == pytest.approx(
            RewardConfig().w_retrieval_token * 1000
        )

    def test_retrieval_costs_more_per_token_than_plain_llm_tokens(self):
        """Retrieved context biases what the expert produces, so it is not
        merely tokens; an indiscriminate policy would narrow the search while
        appearing cheap."""
        cfg = RewardConfig()
        assert cfg.w_retrieval_token > cfg.w_llm_token


class TestHeuristicDemonstrations:
    """The BC corpus must contain both retrieval choices."""

    def _obs(self, n_available):
        return AgentObservation(
            state_version=1, episode_id="E1",
            search_archive_summary={"n_candidates": 0},
            retrieved_experience=RetrievedExperienceSummary(
                n_retrieved=min(2, n_available), n_available=n_available
            ),
        )

    def test_declines_retrieval_when_memory_is_empty(self):
        from famou.reliability.strategy import HeuristicMetaPolicy

        assert HeuristicMetaPolicy().act(self._obs(0)).retrieval_top_k == 0

    def test_requests_retrieval_once_memory_exists(self):
        from famou.reliability.strategy import HeuristicMetaPolicy

        assert HeuristicMetaPolicy().act(self._obs(5)).retrieval_top_k > 0

    def test_does_not_gate_on_repairability(self):
        """OOM/timeout are not 'repairable' in the taxonomy, but they are the
        only patterns that currently constrain generation — gating on
        repairability skipped exactly the useful cases."""
        from famou.reliability.strategy import HeuristicMetaPolicy

        obs = AgentObservation(
            state_version=1, episode_id="E1",
            search_archive_summary={"n_candidates": 0},
            retrieved_experience=RetrievedExperienceSummary(
                n_retrieved=1, n_available=1, n_repairable=0, dominant_failure="oom"
            ),
        )
        assert HeuristicMetaPolicy().act(obs).retrieval_top_k > 0


class TestProposalRetrievalIsFamilyFiltered:
    def test_generation_query_narrows_to_the_chosen_family(self):
        """The decision-phase query cannot filter by family — the action has
        not been chosen yet. The proposal query can, and must."""
        index = ExperienceIndex(StateStore())
        memory = FailureMemory()
        for family in ("gbdt", "mlp"):
            memory.observe(
                index,
                ObservedOutcome(
                    candidate_id=f"c_{family}", episode_id="E1",
                    model_family=family,
                    evidence=[EvidenceVector(
                        candidate_id=f"c_{family}", episode_id="E1",
                        eval_id=f"ev_{family}", fidelity=Fidelity.F1_CHEAP,
                        split_scope="visible_dev", data_contract_hash="h",
                        validity=0.0, error_info="out of memory",
                        failure_stage="evaluate",
                    )],
                ),
                valid_from_state_version=0, transition_id="t", decision_id="d",
            )
        retriever = MemoryRetriever(index)

        broad = retriever.retrieve(at_version=0, top_k=8)
        narrow = retriever.retrieve(
            at_version=0, top_k=8, filters={"model_family": "mlp"}
        )

        assert len(broad.retrieved_experience_ids) == 2
        assert narrow.retrieved_experience_ids == ["fail::oom::mlp"]
