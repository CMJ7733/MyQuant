"""Formula candidates and the FORMULA / MODEL / MIXED mode selector.

Also pins three fixes for the same recurring bug: an action that means "act on
THIS existing candidate" implemented as "mutate it into a new one". Each time,
the guard that was supposed to stop the heuristic re-firing never advanced, and
the search spun in place burning iterations.
"""

from __future__ import annotations

import pytest

from famou.core.state import StateStore
from famou.reliability.archives import CertifiedArchive, SearchArchive
from famou.reliability.evaluator import EvidenceBuilder, StaticChecker
from famou.reliability.experts import (
    FACTOR_GROUPS,
    FormulaExpert,
    default_expert_registry,
)
from famou.reliability.observation import ObservationBuilder
from famou.reliability.promotion import PromotionPolicy
from famou.reliability.types import (
    CandidateMode,
    EvaluationCost,
    EvidenceVector,
    ExpertKind,
    Fidelity,
    MetricDistribution,
    StructuredAction,
    TaskSpec,
)


def explore_action(family: str = "formula") -> StructuredAction:
    return StructuredAction(expert=ExpertKind.EXPLORE, model_family=family)


class TestCandidateMode:
    def test_modes_select_disjoint_spaces(self):
        assert CandidateMode.FORMULA.families() == ["formula"]
        assert "formula" not in CandidateMode.MODEL.families()
        assert "formula" in CandidateMode.MIXED.families()
        assert "gbdt" in CandidateMode.MIXED.families()

    def test_mode_round_trips_through_task_spec(self):
        for mode in CandidateMode:
            spec = TaskSpec.for_mode(mode)
            assert spec.candidate_mode is mode

    def test_modes_have_distinct_hashes(self):
        """Two modes searched different spaces, so their results are not
        comparable — and the frozen hash has to say so."""
        hashes = {m: TaskSpec.for_mode(m).compute_hash() for m in CandidateMode}
        assert len(set(hashes.values())) == len(CandidateMode)

    def test_protected_half_is_identical_across_modes(self):
        """The evaluation machinery does not change with the mode, so runs
        remain comparable on that axis even though the search space differs."""
        protected = {TaskSpec.for_mode(m).protected_hash() for m in CandidateMode}
        assert len(protected) == 1

    def test_registry_matches_the_mode(self):
        formula = default_expert_registry(families=CandidateMode.FORMULA.families())
        model = default_expert_registry(families=CandidateMode.MODEL.families())
        mixed = default_expert_registry(families=CandidateMode.MIXED.families())

        assert all(k.startswith("formula") for k in formula)
        assert not any(k.startswith("formula") for k in model)
        assert any(k.startswith("formula") for k in mixed) and "gbdt" in mixed

    def test_unknown_family_is_an_error(self):
        with pytest.raises(ValueError, match="no experts"):
            default_expert_registry(families=["nonexistent"])


class TestFormulaExpert:
    def test_produces_a_valid_candidate(self):
        expert = FormulaExpert(ExpertKind.EXPLORE, rng_seed=0)
        program = expert.propose(explore_action(), [], iteration=1)
        assert StaticChecker().check(program).passed
        terms = expert.extract_hyperparams(program)["terms"]
        assert 3 <= len(terms) <= 8
        assert all({"feature", "weight", "transform"} <= set(t) for t in terms)

    def test_weights_are_l1_normalised(self):
        """RankIC is invariant to positive rescaling, so overall magnitude is
        not a real degree of freedom; fixing it keeps mutation working on the
        relative weights and stops rescaled duplicates looking distinct."""
        expert = FormulaExpert(ExpertKind.EXPLORE, rng_seed=3)
        for _ in range(5):
            program = expert.propose(explore_action(), [], iteration=1)
            terms = expert.extract_hyperparams(program)["terms"]
            assert sum(abs(t["weight"]) for t in terms) == pytest.approx(1.0, abs=0.01)

    def test_factors_come_from_distinct_groups(self):
        """Five flavours of momentum is not a multi-factor score."""
        expert = FormulaExpert(ExpertKind.EXPLORE, rng_seed=7)
        terms = expert.extract_hyperparams(
            expert.propose(explore_action(), [], iteration=1))["terms"]
        groups = [FormulaExpert._group_of(t["feature"]) for t in terms]
        assert len(set(groups)) == len(groups)

    def test_all_referenced_factors_exist_in_groups(self):
        for members in FACTOR_GROUPS.values():
            assert members, "empty factor group"

    def test_mutation_changes_the_formula(self):
        parent = FormulaExpert(ExpertKind.EXPLORE, rng_seed=1).propose(
            explore_action(), [], iteration=1)
        expert = FormulaExpert(ExpertKind.MUTATE, rng_seed=2)
        changed = False
        for i in range(10):
            child = expert.propose(explore_action(), [parent], iteration=2)
            if (expert.extract_hyperparams(child)["terms"]
                    != expert.extract_hyperparams(parent)["terms"]):
                changed = True
                break
        assert changed, "10 mutations produced no change"

    def test_spec_records_zero_fitted_parameters(self):
        """A formula is the interpretable end of the pool: what it is, is the
        factor list, and it fits nothing."""
        expert = FormulaExpert(ExpertKind.EXPLORE, rng_seed=0)
        program = expert.propose(explore_action(), [], iteration=1)
        spec = expert.build_spec(expert.extract_hyperparams(program))
        assert spec.model_family == "formula"
        assert spec.training["fitted_parameters"] == 0
        assert spec.training["deterministic"] is True
        assert spec.feature_pipeline["factors"]

    def test_package_carries_the_factor_list(self):
        expert = FormulaExpert(ExpertKind.EXPLORE, rng_seed=0)
        program = expert.propose(explore_action(), [], iteration=1)
        package = expert.package(program, episode_id="E1")
        assert package.spec.model_family == "formula"
        assert package.code_hash


class TestDeterministicEvidence:
    """A formula re-run with another seed returns the identical number, so
    seed dispersion is structurally zero. Subperiods carry the real question."""

    def _evidence(self, manifest, subperiods, mean_ic=0.03):
        return EvidenceBuilder().build(
            candidate_id="f1", manifest=manifest, fidelity=Fidelity.F2_FULL,
            raw={
                "validity": 1.0,
                "rank_ic": mean_ic,
                "per_seed_rank_ic": [mean_ic],
                "subperiod_rank_ic": subperiods,
                "deterministic": True,
                "factors_used": ["ROC20", "STD20"],
            },
            cost=EvaluationCost(),
        )

    def test_subperiods_populate_regime_stability(self, manifest):
        ev = self._evidence(manifest, [0.02, 0.04, 0.01, 0.05])
        assert len(ev.regime_stability) == 4
        assert ev.worst_subperiod_rank_ic == pytest.approx(0.01)
        assert ev.complexity["deterministic"] is True
        assert ev.complexity["n_factors"] == 2

    def test_promotion_uses_subperiods_for_deterministic_candidates(self, manifest):
        """With only one 'seed', a seed-count gate would reject every formula
        outright — they would never be promotable at all."""
        ev = [self._evidence(manifest, [0.030, 0.031, 0.029, 0.030])]
        decision = PromotionPolicy().evaluate(
            ev, incumbent_rank_ic=0.02, sealed_queries_remaining=3, novelty=0.5)
        assert decision.action == "request_gate"
        assert "subperiods" in decision.reason

    def test_unstable_across_subperiods_asks_for_more(self, manifest):
        ev = [self._evidence(manifest, [0.08, -0.06, 0.05, -0.04])]
        decision = PromotionPolicy().evaluate(
            ev, incumbent_rank_ic=0.02, sealed_queries_remaining=3, novelty=0.5)
        assert decision.action == "more_seeds"

    def test_too_few_subperiods_asks_for_more(self, manifest):
        ev = [self._evidence(manifest, [0.03])]
        decision = PromotionPolicy().evaluate(
            ev, incumbent_rank_ic=0.02, sealed_queries_remaining=3, novelty=0.5)
        assert decision.action == "more_seeds"

    def test_headline_ic_is_compared_not_the_subperiod_mean(self, manifest):
        """Subperiod means and the full-window mean are different quantities,
        and the incumbent is measured on the full window."""
        ev = [self._evidence(manifest, [0.05, 0.05, 0.05, 0.05], mean_ic=0.01)]
        decision = PromotionPolicy().evaluate(
            ev, incumbent_rank_ic=0.03, sealed_queries_remaining=3, novelty=0.5)
        assert decision.action == "skip"
        assert "0.0100" in decision.reason


class TestNoRepeatedNoOpDecisions:
    """Three bugs, one shape: 'act on THIS candidate' implemented as 'mutate
    it', leaving the guard that should stop the heuristic un-advanced."""

    def _archives(self, state_store):
        return SearchArchive(state_store), CertifiedArchive(state_store)

    def test_promotion_check_recorded_even_when_declined(self, state_store):
        """A declined promotion spends no sealed query, so gate_attempts stays
        0. Without a separate record the candidate is re-proposed forever."""
        search, _ = self._archives(state_store)
        search.add_candidate("c1", episode_id="E1", model_family="formula",
                             code_hash="h")
        assert search.promotion_checked_at("c1") == 0
        search.record_promotion_check("c1", 2)
        assert search.promotion_checked_at("c1") == 2
        assert search.gate_attempts("c1") == 0     # nothing was spent

    def test_observation_exposes_the_guard(self, state_store, ledger, manifest):
        search, certified = self._archives(state_store)
        search.add_candidate("c1", episode_id="E1", model_family="formula",
                             code_hash="h")
        search.add_evidence(EvidenceVector(
            candidate_id="c1", episode_id="E1", eval_id="e1",
            fidelity=Fidelity.F2_FULL, split_scope="visible_dev",
            data_contract_hash="h",
            rank_ic=MetricDistribution.from_samples([0.05, 0.051, 0.049]),
            validity=1.0))
        search.record_promotion_check("c1", 1)

        obs = ObservationBuilder(search, certified, ledger).build(
            episode_id="E1", state_version=1)
        cand = obs.top_visible_evidence[0]
        assert cand.n_evidence == 1
        assert cand.promotion_checked_at == 1
        # nothing new to reconsider -> the heuristic must not re-fire
        assert not (cand.promotion_checked_at < cand.n_evidence)

    def test_new_evidence_reopens_the_question(self, state_store, ledger):
        search, certified = self._archives(state_store)
        search.add_candidate("c1", episode_id="E1", model_family="formula",
                             code_hash="h")
        for i in range(2):
            search.add_evidence(EvidenceVector(
                candidate_id="c1", episode_id="E1", eval_id=f"e{i}",
                fidelity=Fidelity.F2_FULL, split_scope="visible_dev",
                data_contract_hash="h",
                rank_ic=MetricDistribution.from_samples([0.05, 0.051, 0.049]),
                validity=1.0))
            if i == 0:
                search.record_promotion_check("c1", 1)

        obs = ObservationBuilder(search, certified, ledger).build(
            episode_id="E1", state_version=1)
        cand = obs.top_visible_evidence[0]
        assert cand.promotion_checked_at < cand.n_evidence   # reconsider

    def test_reeval_target_is_a_distinct_intent(self):
        """'Escalate this candidate to F2' must name the candidate, not become
        a mutation of it — otherwise the original stays at F1 and the branch
        re-fires every iteration."""
        action = StructuredAction(
            expert=ExpertKind.LOCAL_HPO, fidelity=Fidelity.F2_FULL,
            reeval_target_id="cand_7",
        )
        assert action.reeval_target_id == "cand_7"
        assert action.promotion_target_id is None
