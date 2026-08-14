"""Sealed gate: verdict discreteness, margin bands, token replay protection."""

from __future__ import annotations

import hashlib

import pytest

from famou.reliability.promotion import (
    BudgetedGate,
    SealedGateService,
    build_gate_request,
)
from famou.reliability.types import (
    GateConfig,
    GateReasonCode,
    GateVerdictKind,
    MarginBand,
)


def make_eval_fn(per_seed, incumbent=0.05, regime_improvements=None):
    def _fn(code, manifest, seeds):
        return {
            "rank_ic_per_seed": list(per_seed),
            "incumbent_rank_ic": incumbent,
            "regime_improvements": regime_improvements or {},
        }

    return _fn


def make_request(ledger, manifest, code="print('hello')", seeds=None):
    return build_gate_request(
        candidate_id="cand_1",
        candidate_code=code,
        manifest=manifest,
        ledger=ledger,
        seed_list=seeds,
    )


class TestVerdicts:
    def test_clear_improvement_promotes(self, ledger, manifest):
        gate = SealedGateService(manifest, make_eval_fn([0.07, 0.071, 0.069]))
        req = make_request(ledger, manifest)
        verdict = gate.evaluate(req, "print('hello')")
        assert verdict.verdict == GateVerdictKind.PROMOTE
        assert verdict.reason_code == GateReasonCode.ROBUST_IMPROVEMENT
        assert verdict.margin_band == MarginBand.CLEAR_PASS

    def test_no_improvement_rejects(self, ledger, manifest):
        gate = SealedGateService(manifest, make_eval_fn([0.03, 0.031, 0.029]))
        req = make_request(ledger, manifest)
        verdict = gate.evaluate(req, "print('hello')")
        assert verdict.verdict == GateVerdictKind.REJECT
        assert verdict.reason_code == GateReasonCode.NO_IMPROVEMENT
        assert verdict.margin_band == MarginBand.CLEAR_FAIL

    def test_positive_but_underpowered_is_inconclusive(self, ledger, manifest):
        # margin = 0.055 - 0.05 = 0.005 < delta_min = 0.0091
        gate = SealedGateService(manifest, make_eval_fn([0.055, 0.056, 0.054]))
        req = make_request(ledger, manifest)
        verdict = gate.evaluate(req, "print('hello')")
        assert verdict.verdict == GateVerdictKind.INCONCLUSIVE
        assert verdict.reason_code == GateReasonCode.INSUFFICIENT_POWER

    def test_unstable_seeds_reject(self, ledger, manifest):
        # mean 0.07 but huge spread -> CV > 1.5
        gate = SealedGateService(manifest, make_eval_fn([0.15, -0.05, 0.11]))
        req = make_request(ledger, manifest)
        verdict = gate.evaluate(req, "print('hello')")
        assert verdict.verdict == GateVerdictKind.REJECT
        assert verdict.reason_code == GateReasonCode.UNSTABLE_ACROSS_SEEDS

    def test_regime_fragile_rejects(self, ledger, manifest):
        gate = SealedGateService(
            manifest,
            make_eval_fn(
                [0.07, 0.07, 0.07],
                regime_improvements={"bull": 0.02, "bear": -0.01, "range": -0.02},
            ),
        )
        req = make_request(ledger, manifest)
        verdict = gate.evaluate(req, "print('hello')")
        assert verdict.verdict == GateVerdictKind.REJECT
        assert verdict.reason_code == GateReasonCode.REGIME_FRAGILE

    def test_eval_error_is_inconclusive_and_silent(self, ledger, manifest):
        def boom(code, manifest, seeds):
            raise RuntimeError("sealed data path /secret/2015/Q3 exploded")

        gate = SealedGateService(manifest, boom)
        req = make_request(ledger, manifest)
        verdict = gate.evaluate(req, "print('hello')")
        assert verdict.verdict == GateVerdictKind.INCONCLUSIVE
        assert verdict.reason_code == GateReasonCode.EVALUATION_ERROR
        # no exception detail may leak into the verdict
        assert "secret" not in repr(verdict)
        assert "2015" not in repr(verdict)


class TestNoNumericLeakage:
    """The verdict must carry no numeric performance information."""

    @pytest.mark.parametrize("per_seed", [[0.09, 0.09, 0.09], [0.01, 0.01, 0.01]])
    def test_verdict_has_no_numeric_fields(self, ledger, manifest, per_seed):
        gate = SealedGateService(manifest, make_eval_fn(per_seed))
        req = make_request(ledger, manifest)
        verdict = gate.evaluate(req, "print('hello')")
        payload = verdict.model_dump()
        for key, value in payload.items():
            if key == "query_cost":
                continue
            assert not isinstance(value, float), f"numeric field leaked: {key}"
            assert not isinstance(value, int) or isinstance(value, bool), key


class TestProtocolIntegrity:
    def test_code_hash_mismatch_rejects(self, ledger, manifest):
        gate = SealedGateService(manifest, make_eval_fn([0.09, 0.09, 0.09]))
        req = make_request(ledger, manifest, code="print('v1')")
        verdict = gate.evaluate(req, "print('v2')")  # drifted code
        assert verdict.verdict == GateVerdictKind.REJECT
        assert verdict.reason_code == GateReasonCode.PROTOCOL_VIOLATION

    def test_contract_hash_mismatch_rejects(self, ledger, manifest):
        gate = SealedGateService(manifest, make_eval_fn([0.09, 0.09, 0.09]))
        req = make_request(ledger, manifest)
        req.data_contract_hash = hashlib.sha256(b"tampered").hexdigest()
        verdict = gate.evaluate(req, "print('hello')")
        assert verdict.verdict == GateVerdictKind.REJECT
        assert verdict.reason_code == GateReasonCode.PROTOCOL_VIOLATION


class TestBudgetedGate:
    def test_replay_rejected(self, ledger, manifest):
        gate = BudgetedGate(
            SealedGateService(manifest, make_eval_fn([0.09, 0.09, 0.09])), ledger
        )
        req = make_request(ledger, manifest)
        first = gate.evaluate(req, "print('hello')")
        assert first.verdict == GateVerdictKind.PROMOTE
        # replaying the same request must fail (token already consumed)
        second = gate.evaluate(req, "print('hello')")
        assert second.verdict == GateVerdictKind.REJECT
        assert second.reason_code == GateReasonCode.PROTOCOL_VIOLATION

    def test_marginal_band(self, ledger, manifest):
        # margin = 0.059 - 0.05 = 0.009 ≈ delta_min (0.0091) -> marginal
        cfg = GateConfig()
        gate = SealedGateService(
            manifest, make_eval_fn([0.059, 0.059, 0.059]), config=cfg
        )
        req = make_request(ledger, manifest)
        verdict = gate.evaluate(req, "print('hello')")
        assert verdict.margin_band in (MarginBand.MARGINAL, MarginBand.CLEAR_PASS)
