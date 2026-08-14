"""TaskSpec, EvalRequest, CandidatePackage, Freeze and the one-shot final test.

The theme is the same throughout: the things a paper claims were fixed must
be structurally impossible to move, not merely documented as fixed.
"""

from __future__ import annotations

import pytest

from famou.core.data import Program
from famou.core.state import StateStore
from famou.reliability.archives import CertifiedArchive
from famou.reliability.budget import BudgetExhausted, BudgetLedger
from famou.reliability.evaluator import FidelityEvaluator, StaticChecker
from famou.reliability.experts import GBDTExpert, NNExpert
from famou.reliability.final_test import (
    ExperimentNotFrozen,
    FinalTestAlreadyRun,
    FinalTestService,
    freeze_experiment,
    get_freeze,
    is_frozen,
)
from famou.reliability.types import (
    CandidatePackage,
    EvalRequest,
    ExpertKind,
    Fidelity,
    PaperResult,
    StructuredAction,
    TaskSpec,
)

VALID_CODE = '''
import argparse
import json

HYPERPARAMS = {"learning_rate": 0.1}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-config", required=True)
    cfg = json.load(open(ap.parse_args().split_config))
    print("FAMOU_RESULT " + json.dumps({"validity": 1.0, "rank_ic": 0.05}))

if __name__ == "__main__":
    main()
'''


def make_program(code: str, pid: str = "cand_x") -> Program:
    return Program(id=pid, code=code, generation=1, iteration=1)


class TestTaskSpec:
    def test_hash_is_deterministic_and_content_sensitive(self):
        a = TaskSpec()
        b = TaskSpec()
        assert a.compute_hash() == b.compute_hash()
        c = TaskSpec(transaction_cost_bps=30.0)
        assert c.compute_hash() != a.compute_hash()

    def test_protected_hash_ignores_search_space(self):
        """Two runs may search different families and still be comparable, as
        long as the evaluation machinery is identical."""
        a = TaskSpec(allowed_model_families=["gbdt"])
        b = TaskSpec(allowed_model_families=["gbdt", "mlp"])
        assert a.protected_hash() == b.protected_hash()
        assert a.compute_hash() != b.compute_hash()

    def test_protected_hash_tracks_cost_model(self):
        a = TaskSpec()
        b = TaskSpec(transaction_cost_bps=30.0)
        assert a.protected_hash() != b.protected_hash()


class TestStaticCheckerDrivenByTaskSpec:
    def test_valid_candidate_passes(self):
        assert StaticChecker(TaskSpec()).check(make_program(VALID_CODE)).passed

    def test_protected_symbol_from_spec_is_enforced(self):
        code = VALID_CODE + "\ntopk = 500\n"
        result = StaticChecker(TaskSpec()).check(make_program(code))
        assert result.forbidden_edits
        assert not result.passed

    def test_tightening_the_spec_tightens_the_check(self):
        """The protected list is part of the frozen contract, so adding to it
        is a protocol amendment rather than an edit inside the checker."""
        code = VALID_CODE + "\nmy_slippage_model = 0.0\n"
        assert StaticChecker(TaskSpec()).check(make_program(code)).passed
        strict = TaskSpec(protected_symbols=["slippage"])
        assert not StaticChecker(strict).check(make_program(code)).passed

    def test_disallowed_import_rejected(self):
        code = "import requests\n" + VALID_CODE
        result = StaticChecker(TaskSpec()).check(make_program(code))
        assert any("allow-list" in e for e in result.schema_errors)
        assert not result.passed

    def test_allowed_import_accepted(self):
        code = "import lightgbm\n" + VALID_CODE
        assert StaticChecker(TaskSpec()).check(make_program(code)).passed

    def test_future_idiom_in_candidate_expression_flagged(self):
        code = VALID_CODE + '\nLEAK = "Ref($close, -1)"\n'
        result = StaticChecker(TaskSpec()).check(make_program(code))
        assert result.leakage_flags

    def test_frozen_label_is_not_a_leak(self, manifest):
        """The frozen label is itself a forward expression. Flagging it would
        make every legitimate candidate fail F0."""
        code = VALID_CODE + f'\nLABEL = "{manifest.label_expression}"\n'
        checker = StaticChecker(
            TaskSpec(), label_expression=manifest.label_expression
        )
        assert not checker.check(make_program(code)).leakage_flags
        # ... but the exemption is exact, not a blanket amnesty
        other = VALID_CODE + '\nSNEAKY = "Ref($close, -5)"\n'
        assert checker.check(make_program(other)).leakage_flags


class TestEvalRequest:
    def _evaluator(self, manifest, ledger, spy=None):
        def run_fn(code, cfg):
            if spy is not None:
                spy.update(cfg)
            return {"validity": 1.0, "rank_ic": 0.05}

        return FidelityEvaluator(manifest, run_fn, ledger)

    def test_action_ceiling_is_applied(self, manifest, ledger):
        spy = {}
        evaluator = self._evaluator(manifest, ledger, spy)
        evaluator.evaluate(
            make_program(VALID_CODE), Fidelity.F2_FULL, max_gpu_seconds=120.0
        )
        assert spy["max_gpu_seconds"] == 120.0

    def test_task_spec_cap_bounds_the_action(self, manifest, ledger):
        """A policy cannot vote itself more compute than the contract allows."""
        spy = {}
        evaluator = self._evaluator(manifest, ledger, spy)
        evaluator.task_spec.max_candidate_gpu_seconds = 300.0
        evaluator.evaluate(
            make_program(VALID_CODE), Fidelity.F2_FULL, max_gpu_seconds=99999.0
        )
        assert spy["max_gpu_seconds"] == 300.0

    def test_f1_actually_reduces_the_work(self, manifest, ledger):
        """F1 used to only cut seeds and boosting rounds, so it was not much
        cheaper than F2. It now shortens the train window and subsamples the
        universe too."""
        f1 = self._evaluator(manifest, ledger).build_request("c", Fidelity.F1_CHEAP,
                                                             seed_list=[11, 29, 47])
        f2 = self._evaluator(manifest, ledger).build_request("c", Fidelity.F2_FULL,
                                                             seed_list=[11, 29, 47])
        assert len(f1.seed_list) == 1 and len(f2.seed_list) == 3
        assert f1.train_fraction < f2.train_fraction
        assert f1.universe_fraction < f2.universe_fraction
        assert f1.max_boost_rounds is not None and f2.max_boost_rounds is None

    def test_split_scope_cannot_name_sealed(self):
        with pytest.raises(Exception):
            EvalRequest(
                request_id="r", candidate_id="c",
                fidelity=Fidelity.F2_FULL, split_scope="sealed_promotion",
            )

    def test_task_spec_reaches_the_harness(self, manifest, ledger):
        spy = {}
        evaluator = self._evaluator(manifest, ledger, spy)
        evaluator.evaluate(make_program(VALID_CODE), Fidelity.F2_FULL)
        assert spy["topk"] == evaluator.task_spec.topk
        assert spy["transaction_cost_bps"] == evaluator.task_spec.transaction_cost_bps


class TestCandidatePackage:
    def test_gbdt_package_splits_components(self):
        expert = GBDTExpert(ExpertKind.MUTATE, rng_seed=0)
        program = expert.propose(
            StructuredAction(expert=ExpertKind.MUTATE, model_family="gbdt"), [], iteration=1
        )
        package = expert.package(
            program, episode_id="E1", data_contract_hash="dch", task_spec_hash="tsh"
        )
        assert package.code_hash == CandidatePackage.hash_code(program.code)
        assert package.data_contract_hash == "dch"
        assert package.task_spec_hash == "tsh"
        spec = package.spec
        assert spec.model_family == "gbdt"
        assert "learning_rate" in spec.optimizer
        assert "num_leaves" in spec.architecture
        assert "learning_rate" not in spec.architecture  # no double-counting

    def test_nn_package_splits_components(self):
        expert = NNExpert(ExpertKind.EXPLORE, rng_seed=0, family="mlp")
        program = expert.propose(
            StructuredAction(expert=ExpertKind.EXPLORE, model_family="mlp"), [], iteration=1
        )
        spec = expert.package(program, episode_id="E1").spec
        assert "hidden_dims" in spec.architecture
        assert "epochs" in spec.training
        assert "learning_rate" in spec.optimizer

    def test_package_round_trips_through_json(self):
        expert = GBDTExpert(ExpertKind.MUTATE, rng_seed=0)
        program = expert.propose(
            StructuredAction(expert=ExpertKind.MUTATE, model_family="gbdt"), [], iteration=1
        )
        package = expert.package(program, episode_id="E1")
        restored = CandidatePackage.model_validate(package.model_dump(mode="json"))
        assert restored.code_hash == package.code_hash
        assert restored.spec.architecture == package.spec.architecture


def final_fn(code, manifest, seeds):
    return {
        "rank_ic": 0.041, "icir": 0.38, "sharpe": 1.2,
        "max_drawdown": -0.18, "turnover": 0.42,
        "regime_stability": {"bull": 0.05, "bear": 0.02},
        "multi_seed_ci": {"rank_ic": [0.035, 0.047]},
    }


class TestFreezeAndFinalTest:
    def _wiring(self, manifest, ledger):
        store = StateStore()
        certified = CertifiedArchive(store)
        certified.add_baseline(
            "init_0", episode_id="E1", model_family="gbdt", code_hash="h0"
        )
        service = FinalTestService(
            manifest, final_fn, state_store=store, ledger=ledger
        )
        return store, certified, service

    def _freeze(self, store, manifest, certified):
        return freeze_experiment(
            state_store=store, manifest=manifest, task_spec=TaskSpec(),
            certified_archive=certified, policy_version="heuristic_v0",
            expert_registry=["gbdt", "mlp"], seeds=[101, 202, 303],
        )

    def test_final_test_requires_a_freeze(self, manifest, ledger):
        store, certified, service = self._wiring(manifest, ledger)
        with pytest.raises(ExperimentNotFrozen):
            service.run(candidate_id="c1", candidate_code="print(1)")

    def test_freeze_snapshots_the_certified_archive(self, manifest, ledger):
        store, certified, _ = self._wiring(manifest, ledger)
        freeze = self._freeze(store, manifest, certified)
        assert "init_0" in freeze.certified_snapshot
        assert freeze.policy_version == "heuristic_v0"
        assert freeze.expert_registry == ["gbdt", "mlp"]
        assert freeze.data_contract_hash == manifest.compute_hash()

    def test_freeze_is_idempotent(self, manifest, ledger):
        """A second call must not re-snapshot a mutated archive."""
        store, certified, _ = self._wiring(manifest, ledger)
        first = self._freeze(store, manifest, certified)
        certified.add_baseline(
            "sneaked_in", episode_id="E1", model_family="mlp", code_hash="h9"
        )
        second = self._freeze(store, manifest, certified)
        assert second.freeze_id == first.freeze_id
        assert "sneaked_in" not in second.certified_snapshot

    def test_final_test_runs_once_and_returns_paper_result(self, manifest, ledger):
        store, certified, service = self._wiring(manifest, ledger)
        self._freeze(store, manifest, certified)
        result = service.run(candidate_id="c1", candidate_code="print(1)")
        assert isinstance(result, PaperResult)
        assert result.rank_ic == pytest.approx(0.041)
        assert result.query_costs["final"] == 1
        assert service.has_run()

    def test_second_final_test_refused(self, manifest, ledger):
        store, certified, service = self._wiring(manifest, ledger)
        self._freeze(store, manifest, certified)
        service.run(candidate_id="c1", candidate_code="print(1)")
        with pytest.raises(FinalTestAlreadyRun):
            service.run(candidate_id="c2", candidate_code="print(2)")

    def test_final_query_budget_is_spent(self, manifest, ledger):
        store, certified, service = self._wiring(manifest, ledger)
        self._freeze(store, manifest, certified)
        assert ledger.remaining(episode_id="E1")["final_queries"] == 1
        service.run(candidate_id="c1", candidate_code="print(1)")
        assert ledger.remaining(episode_id="E1")["final_queries"] == 0

    def test_crashed_run_still_consumes_the_query(self, manifest, ledger):
        """Otherwise a failing evaluator gives unlimited free looks."""
        store, certified, _ = self._wiring(manifest, ledger)
        self._freeze(store, manifest, certified)

        def boom(code, manifest_, seeds):
            raise RuntimeError("harness died")

        service = FinalTestService(manifest, boom, state_store=store, ledger=ledger)
        with pytest.raises(RuntimeError, match="harness died"):
            service.run(candidate_id="c1", candidate_code="print(1)")
        assert ledger.remaining(episode_id="E1")["final_queries"] == 0
        with pytest.raises(FinalTestAlreadyRun):
            service.run(candidate_id="c1", candidate_code="print(1)")

    def test_protocol_drift_after_freeze_is_refused(self, manifest, ledger):
        store, certified, _ = self._wiring(manifest, ledger)
        self._freeze(store, manifest, certified)
        drifted = manifest.model_copy(deep=True)
        drifted.visible_dev.end = "2015-06-30"
        service = FinalTestService(
            drifted, final_fn, state_store=store, ledger=ledger
        )
        with pytest.raises(ExperimentNotFrozen, match="data contract changed"):
            service.run(candidate_id="c1", candidate_code="print(1)")

    def test_service_holds_no_writable_store(self, manifest, ledger):
        """The one-way property is structural: the service has no archive,
        trajectory store or observation builder to write a result into."""
        store, certified, service = self._wiring(manifest, ledger)
        held = vars(service)
        assert not any(
            type(v).__name__
            in {"SearchArchive", "CertifiedArchive", "TrajectoryStore", "ObservationBuilder"}
            for v in held.values()
        )


class TestSearchClosesAtFreeze:
    def test_frozen_episode_refuses_new_decisions(self, manifest, ledger, tmp_path):
        from famou.reliability.final_test import SearchClosed
        from famou.reliability.strategy import ReliabilityAwareQuantStrategy
        from famou.reliability.trajectory import TrajectoryStore

        store = StateStore()
        certified = CertifiedArchive(store)
        strategy = ReliabilityAwareQuantStrategy(
            manifest=manifest,
            fidelity_evaluator=FidelityEvaluator(
                manifest, lambda c, cfg: {"validity": 1.0, "rank_ic": 0.05}, ledger
            ),
            ledger=ledger,
            state_store=store,
            trajectory_store=TrajectoryStore(),
        )
        freeze_experiment(
            state_store=store, manifest=manifest, task_spec=TaskSpec(),
            certified_archive=certified, policy_version="heuristic_v0",
        )
        assert is_frozen(store, "E1")
        assert get_freeze(store, "E1") is not None
        with pytest.raises(SearchClosed):
            strategy.forward_batch(None, [], max_batch_size=1)
