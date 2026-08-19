"""Evaluator: F0 static checks, fidelity ladder, evidence construction."""

from __future__ import annotations

import pytest

from famou.core.data import Program
from famou.reliability.evaluator import (
    EvidenceBuilder,
    FidelityEvaluator,
    StaticChecker,
)
from famou.reliability.experts import GBDTExpert, NNExpert
from famou.reliability.types import (
    ExpertKind,
    Fidelity,
    StructuredAction,
)


def make_program(code: str, pid: str = "cand_x") -> Program:
    return Program(id=pid, code=code, generation=1, iteration=1)


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


class TestStaticChecker:
    def test_valid_candidate_passes(self):
        result = StaticChecker().check(make_program(VALID_CODE))
        assert result.passed
        assert result.compiles

    def test_syntax_error_fails(self):
        result = StaticChecker().check(make_program("def broken(:"))
        assert not result.compiles
        assert not result.passed

    def test_missing_contract_fails(self):
        result = StaticChecker().check(make_program("print('no contract')"))
        assert not result.passed
        assert any("HYPERPARAMS" in e for e in result.schema_errors)
        assert any("FAMOU_RESULT" in e for e in result.schema_errors)

    def test_future_reference_flagged(self):
        code = VALID_CODE + '\nLEAK = "Ref($close, -1)"\n'
        result = StaticChecker().check(make_program(code))
        assert result.leakage_flags
        assert not result.passed

    def test_protected_component_edit_flagged(self):
        code = VALID_CODE + "\nsealed_promotion_start = '2015-01-01'\n"
        result = StaticChecker().check(make_program(code))
        assert result.forbidden_edits
        assert not result.passed


class TestFidelityEvaluator:
    def _run_fn(self, code: str, split_cfg: dict) -> dict:
        # stub harness: deterministic metrics derived from the split config
        n_seeds = len(split_cfg.get("seed_list", [1]))
        return {
            "validity": 1.0,
            "rank_ic": 0.05,
            "per_seed_rank_ic": [0.05 + 0.001 * i for i in range(n_seeds)],
        }

    def test_f0_only_when_static_fails(self, manifest, ledger):
        evaluator = FidelityEvaluator(manifest, self._run_fn, ledger)
        ev = evaluator.evaluate(make_program("def broken(:"), Fidelity.F2_FULL)
        assert ev.fidelity == Fidelity.F0_STATIC
        assert ev.failure_stage == "compile"
        assert not ev.is_valid

    def test_f1_caps_seeds(self, manifest, ledger):
        evaluator = FidelityEvaluator(manifest, self._run_fn, ledger)
        ev = evaluator.evaluate(
            make_program(VALID_CODE), Fidelity.F1_CHEAP, seed_list=[11, 29, 47]
        )
        assert ev.fidelity == Fidelity.F1_CHEAP
        assert ev.is_valid
        assert ev.rank_ic.n_seeds == 1  # F1 preset caps at 1 seed
        assert ev.split_scope == "visible_dev"

    def test_f2_uses_all_seeds(self, manifest, ledger):
        evaluator = FidelityEvaluator(manifest, self._run_fn, ledger)
        ev = evaluator.evaluate(
            make_program(VALID_CODE), Fidelity.F2_FULL, seed_list=[11, 29, 47]
        )
        assert ev.rank_ic.n_seeds == 3
        assert ev.rank_ic.ci95_low is not None

    def test_candidate_exception_becomes_evidence(self, manifest, ledger):
        def boom(code, cfg):
            raise ValueError("shape mismatch in cross_sectional_encoder")

        evaluator = FidelityEvaluator(manifest, boom, ledger)
        ev = evaluator.evaluate(make_program(VALID_CODE), Fidelity.F1_CHEAP)
        assert not ev.is_valid
        assert ev.failure_stage == "train"
        assert "shape mismatch" in ev.error_info

    def test_visible_query_charged(self, manifest, ledger):
        ledger.set_global_limit("visible_queries", 100)
        evaluator = FidelityEvaluator(manifest, self._run_fn, ledger)
        evaluator.evaluate(make_program(VALID_CODE), Fidelity.F1_CHEAP)
        assert ledger.remaining()["visible_queries"] == 99

    def test_split_config_points_at_visible_dev(self, manifest, ledger):
        seen = {}

        def spy(code, cfg):
            seen.update(cfg)
            return {"validity": 1.0, "rank_ic": 0.05}

        evaluator = FidelityEvaluator(manifest, spy, ledger)
        evaluator.evaluate(make_program(VALID_CODE), Fidelity.F1_CHEAP)
        assert seen["dev_start"] == manifest.visible_dev.start
        assert seen["dev_end"] == manifest.visible_dev.end
        # sealed ranges must never appear in the visible evaluator's config
        assert manifest.sealed_promotion.start not in seen.values()


class TestEvidenceBuilder:
    """B2 regression: ICIR is a within-run time-series ratio, so it must come
    from the harness. Reconstructing it from cross-seed dispersion inflates it
    by an order of magnitude for typical seed spreads."""

    def _build(self, manifest, raw):
        from famou.reliability.types import EvaluationCost

        return EvidenceBuilder().build(
            candidate_id="c1",
            manifest=manifest,
            fidelity=Fidelity.F2_FULL,
            raw=raw,
            cost=EvaluationCost(),
        )

    def test_per_seed_icir_used_verbatim(self, manifest):
        ev = self._build(
            manifest,
            {
                "validity": 1.0,
                "per_seed_rank_ic": [0.050, 0.058, 0.062],
                "per_seed_icir": [0.38, 0.44, 0.41],
            },
        )
        assert ev.icir.n_seeds == 3
        assert ev.icir.mean == pytest.approx(0.41, abs=1e-6)
        # and it is NOT the old cross-seed reconstruction
        assert ev.icir.mean < 1.0

    def test_scalar_icir_fallback(self, manifest):
        ev = self._build(
            manifest,
            {"validity": 1.0, "per_seed_rank_ic": [0.05, 0.06], "icir": 0.4},
        )
        assert ev.icir.mean == pytest.approx(0.4)

    def test_no_icir_reported_means_none(self, manifest):
        ev = self._build(
            manifest, {"validity": 1.0, "per_seed_rank_ic": [0.050, 0.058, 0.062]}
        )
        assert ev.icir is None  # never fabricated from rank_ic dispersion
        assert ev.rank_ic.n_seeds == 3


class TestExperts:
    def test_gbdt_expert_produces_valid_candidate(self):
        expert = GBDTExpert(ExpertKind.MUTATE, rng_seed=0)
        action = StructuredAction(expert=ExpertKind.MUTATE, model_family="gbdt")
        program = expert.propose(action, [], iteration=1)
        assert StaticChecker().check(program).passed
        params = expert.extract_hyperparams(program)
        assert "learning_rate" in params

    def test_gbdt_mutation_changes_params(self):
        parent = GBDTExpert(ExpertKind.EXPLORE, rng_seed=1).propose(
            StructuredAction(expert=ExpertKind.EXPLORE, model_family="gbdt"),
            [],
            iteration=1,
        )
        child = GBDTExpert(ExpertKind.MUTATE, rng_seed=2).propose(
            StructuredAction(expert=ExpertKind.MUTATE, model_family="gbdt"),
            [parent],
            iteration=2,
        )
        assert child.parent_id == parent.id
        assert child.generation == parent.generation + 1
        assert (
            GBDTExpert.extract_hyperparams(child)
            != GBDTExpert.extract_hyperparams(parent)
        )

    def test_expert_does_not_inherit_a_foreign_familys_hyperparams(self):
        """A learned policy has an independent family head, so it will ask the
        GBDT expert to mutate an MLP parent (and vice versa). The expert must
        fall back to its own defaults instead of reading keys that family
        never had — that used to raise KeyError inside _mutate."""
        mlp_parent = NNExpert(ExpertKind.EXPLORE, rng_seed=1, family="mlp").propose(
            StructuredAction(expert=ExpertKind.EXPLORE, model_family="mlp"),
            [],
            iteration=1,
        )
        child = GBDTExpert(ExpertKind.MUTATE, rng_seed=2).propose(
            StructuredAction(expert=ExpertKind.MUTATE, model_family="gbdt"),
            [mlp_parent],
            iteration=2,
        )

        assert child.meta["model_family"] == "gbdt"
        assert child.parent_id == mlp_parent.id      # lineage is still recorded
        params = GBDTExpert.extract_hyperparams(child)
        assert "num_leaves" in params and "dropout" not in params
        assert StaticChecker().check(child).passed

        # ...and the reverse direction.
        nn_child = NNExpert(ExpertKind.MUTATE, rng_seed=3, family="mlp").propose(
            StructuredAction(expert=ExpertKind.MUTATE, model_family="mlp"),
            [child],
            iteration=3,
        )
        assert "dropout" in NNExpert.extract_hyperparams(nn_child)

    def test_same_family_parent_with_partial_hyperparams_is_completed(self):
        """A hand-written baseline may declare only some keys; _mutate needs
        the full set, so what the parent provides is merged over the defaults
        rather than replacing them."""
        partial = Program(
            id="hand_written",
            code="HYPERPARAMS = {'num_leaves': 64}\n",
            generation=0,
            iteration=0,
        )
        partial.meta["model_family"] = "gbdt"

        child = GBDTExpert(ExpertKind.MUTATE, rng_seed=4).propose(
            StructuredAction(expert=ExpertKind.MUTATE, model_family="gbdt"),
            [partial],
            iteration=1,
        )

        params = GBDTExpert.extract_hyperparams(child)
        assert {"learning_rate", "num_leaves", "max_depth"} <= set(params)

    def test_nn_expert_produces_valid_candidate(self):
        expert = NNExpert(ExpertKind.EXPLORE, rng_seed=0, family="mlp")
        action = StructuredAction(expert=ExpertKind.EXPLORE, model_family="mlp")
        program = expert.propose(action, [], iteration=1)
        assert StaticChecker().check(program).passed
        params = expert.extract_hyperparams(program)
        assert "hidden_dims" in params

    def test_boolean_hyperparams_render_as_python(self):
        """Regression: HYPERPARAMS used to be rendered with json.dumps, which
        emits `true` — a NameError at import time. GBDT has no booleans so the
        tests passed while every MLP candidate died in the subprocess."""
        expert = NNExpert(ExpertKind.EXPLORE, rng_seed=1, family="mlp")
        program = expert.propose(
            StructuredAction(expert=ExpertKind.EXPLORE, model_family="mlp"),
            [], iteration=1,
        )
        assert "true" not in program.code.split("def ")[0]
        # the real check: it must actually execute as Python
        namespace: dict = {}
        exec(compile(program.code.split("def load_split_config")[0], "<c>", "exec"), namespace)
        assert isinstance(namespace["HYPERPARAMS"]["layer_norm"], bool)

    def test_nn_expert_llm_path(self):
        def fake_llm_edit(parent_code, action):
            return VALID_CODE  # LLM returns a full rewritten candidate

        parent = NNExpert(rng_seed=0).propose(
            StructuredAction(expert=ExpertKind.EXPLORE, model_family="mlp"),
            [],
            iteration=1,
        )
        expert = NNExpert(ExpertKind.MUTATE, llm_edit_fn=fake_llm_edit)
        child = expert.propose(
            StructuredAction(expert=ExpertKind.MUTATE, model_family="mlp"),
            [parent],
            iteration=2,
        )
        assert child.meta["llm_guided"]
        assert StaticChecker().check(child).passed
