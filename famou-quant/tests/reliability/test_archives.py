"""Archives: search vs certified separation, admission checks, promotion policy."""

from __future__ import annotations

import pytest

from famou.reliability.archives import (
    CertifiedAdmission,
    CertifiedArchive,
    SearchArchive,
)
from famou.reliability.promotion import PromotionPolicy, build_gate_request
from famou.reliability.types import (
    CandidateLineage,
    EvidenceVector,
    Fidelity,
    GateReasonCode,
    GateVerdict,
    GateVerdictKind,
    MarginBand,
    MetricDistribution,
)


def make_evidence(
    candidate_id: str,
    fidelity: Fidelity,
    samples,
    *,
    valid: bool = True,
    manifest_episode: str = "E1",
    contract_hash: str = "h",
) -> EvidenceVector:
    return EvidenceVector(
        candidate_id=candidate_id,
        episode_id=manifest_episode,
        eval_id=f"ev_{candidate_id}_{int(fidelity.value)}",
        fidelity=fidelity,
        split_scope="visible_dev",
        data_contract_hash=contract_hash,
        rank_ic=MetricDistribution.from_samples(list(samples)),
        validity=1.0 if valid else 0.0,
        failure_stage=None if valid else "train",
    )


class TestSearchArchive:
    def test_add_and_retrieve_evidence(self, state_store):
        archive = SearchArchive(state_store)
        archive.add_candidate(
            "c1", episode_id="E1", model_family="gbdt", code_hash="abc"
        )
        archive.add_evidence(make_evidence("c1", Fidelity.F1_CHEAP, [0.04]))
        archive.add_evidence(make_evidence("c1", Fidelity.F2_FULL, [0.05, 0.051, 0.049]))

        f2 = archive.get_evidence("c1", fidelity=Fidelity.F2_FULL)
        assert len(f2) == 1
        assert f2[0].rank_ic.n_seeds == 3

        best = archive.best_evidence("c1")
        assert best.fidelity == Fidelity.F2_FULL

    def test_failed_candidates_produce_evidence(self, state_store):
        archive = SearchArchive(state_store)
        archive.add_candidate("bad", episode_id="E1", model_family="mlp", code_hash="x")
        archive.add_evidence(make_evidence("bad", Fidelity.F1_CHEAP, [0.0], valid=False))
        evs = archive.get_evidence("bad")
        assert len(evs) == 1
        assert evs[0].failure_stage == "train"
        assert archive.best_evidence("bad") is None  # invalid evidence excluded

    def test_family_stats(self, state_store):
        archive = SearchArchive(state_store)
        for i, ok in enumerate([True, True, False]):
            cid = f"g{i}"
            archive.add_candidate(cid, episode_id="E1", model_family="gbdt", code_hash=f"h{i}")
            archive.add_evidence(make_evidence(cid, Fidelity.F1_CHEAP, [0.03], valid=ok))
        stats = archive.family_stats()
        assert stats["gbdt"]["n"] == 3
        assert stats["gbdt"]["n_valid"] == 2

    def test_code_hash_dedup(self, state_store):
        archive = SearchArchive(state_store)
        archive.add_candidate("c1", episode_id="E1", model_family="gbdt", code_hash="dup")
        assert archive.code_hash_exists("dup")
        assert not archive.code_hash_exists("other")


class TestCertifiedAdmission:
    def _setup(self, state_store):
        search = SearchArchive(state_store)
        certified = CertifiedArchive(state_store)
        admission = CertifiedAdmission(search, certified)
        search.add_candidate(
            "cand_1", episode_id="E1", model_family="gbdt",
            code_hash="hash_of_code",
            lineage=CandidateLineage(candidate_id="cand_1", parent_ids=["init_0"]),
        )
        return search, certified, admission

    def _request(self, ledger, manifest, code_hash):
        req = build_gate_request(
            candidate_id="cand_1",
            candidate_code="placeholder",
            manifest=manifest,
            ledger=ledger,
        )
        req.candidate_code_hash = code_hash  # align with archive record
        return req

    def test_promote_admits(self, state_store, ledger, manifest):
        search, certified, admission = self._setup(state_store)
        req = self._request(ledger, manifest, "hash_of_code")
        verdict = GateVerdict(
            verdict=GateVerdictKind.PROMOTE,
            reason_code=GateReasonCode.ROBUST_IMPROVEMENT,
            margin_band=MarginBand.CLEAR_PASS,
        )
        assert admission.verify_and_admit(req, verdict, model_family="gbdt")
        assert certified.is_certified("cand_1")
        # lineage marked certified
        assert search.get_lineage("cand_1").certified

    def test_reject_does_not_admit(self, state_store, ledger, manifest):
        search, certified, admission = self._setup(state_store)
        req = self._request(ledger, manifest, "hash_of_code")
        verdict = GateVerdict(
            verdict=GateVerdictKind.REJECT,
            reason_code=GateReasonCode.NO_IMPROVEMENT,
        )
        assert not admission.verify_and_admit(req, verdict, model_family="gbdt")
        assert not certified.is_certified("cand_1")

    def test_hash_mismatch_rejected(self, state_store, ledger, manifest):
        search, certified, admission = self._setup(state_store)
        req = self._request(ledger, manifest, "different_hash")
        verdict = GateVerdict(
            verdict=GateVerdictKind.PROMOTE,
            reason_code=GateReasonCode.ROBUST_IMPROVEMENT,
        )
        assert not admission.verify_and_admit(req, verdict, model_family="gbdt")
        assert not certified.is_certified("cand_1")

    def test_baseline_seeding(self, state_store):
        certified = CertifiedArchive(state_store)
        certified.add_baseline(
            "init_0", episode_id="E1", model_family="gbdt", code_hash="h0"
        )
        assert certified.is_certified("init_0")
        assert "init_0" in certified.parent_pool()


class TestPromotionPolicy:
    def test_no_evidence_raises_fidelity(self):
        policy = PromotionPolicy()
        decision = policy.evaluate(
            [], incumbent_rank_ic=0.05, sealed_queries_remaining=5
        )
        assert decision.action == "raise_fidelity"

    def test_f1_only_raises_fidelity(self):
        policy = PromotionPolicy()
        ev = [make_evidence("c", Fidelity.F1_CHEAP, [0.06])]
        decision = policy.evaluate(
            ev, incumbent_rank_ic=0.05, sealed_queries_remaining=5
        )
        assert decision.action == "raise_fidelity"

    def test_few_f2_seeds_asks_more_seeds(self):
        policy = PromotionPolicy(min_f2_seeds=3)
        ev = [make_evidence("c", Fidelity.F2_FULL, [0.08])]
        decision = policy.evaluate(
            ev, incumbent_rank_ic=0.05, sealed_queries_remaining=5
        )
        assert decision.action == "more_seeds"

    def test_high_variance_asks_more_seeds(self):
        policy = PromotionPolicy(min_f2_seeds=3, max_rank_ic_cv=0.5)
        ev = [make_evidence("c", Fidelity.F2_FULL, [0.02, 0.10, 0.05])]
        decision = policy.evaluate(
            ev, incumbent_rank_ic=0.03, sealed_queries_remaining=5
        )
        assert decision.action == "more_seeds"

    def test_below_incumbent_skips(self):
        policy = PromotionPolicy()
        ev = [make_evidence("c", Fidelity.F2_FULL, [0.04, 0.041, 0.039])]
        decision = policy.evaluate(
            ev, incumbent_rank_ic=0.05, sealed_queries_remaining=5
        )
        assert decision.action == "skip"

    def test_stable_improvement_requests_gate(self):
        policy = PromotionPolicy()
        ev = [make_evidence("c", Fidelity.F2_FULL, [0.07, 0.071, 0.069])]
        decision = policy.evaluate(
            ev, incumbent_rank_ic=0.05, sealed_queries_remaining=5, novelty=0.5
        )
        assert decision.action == "request_gate"

    def test_no_budget_skips(self):
        policy = PromotionPolicy()
        ev = [make_evidence("c", Fidelity.F2_FULL, [0.07, 0.071, 0.069])]
        decision = policy.evaluate(
            ev, incumbent_rank_ic=0.05, sealed_queries_remaining=0
        )
        assert decision.action == "skip"

    def test_low_novelty_skips(self):
        policy = PromotionPolicy(min_novelty=0.1)
        ev = [make_evidence("c", Fidelity.F2_FULL, [0.07, 0.071, 0.069])]
        decision = policy.evaluate(
            ev, incumbent_rank_ic=0.05, sealed_queries_remaining=5, novelty=0.01
        )
        assert decision.action == "skip"
