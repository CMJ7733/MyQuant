"""CertifiedOnlyPopulation: parent pool = certified members only."""

from __future__ import annotations

import hashlib

from famou.core.data import Program
from famou.reliability.archives import CertifiedArchive
from famou.reliability.population import CertifiedOnlyPopulation


def _prog(pid: str, code: str) -> Program:
    return Program(id=pid, code=code, generation=0, iteration=0)


class TestCertifiedOnlyPopulation:
    def test_uncertified_candidate_excluded(self, state_store):
        certified = CertifiedArchive(state_store)
        pop_module = CertifiedOnlyPopulation(certified)
        result = pop_module.update_population({}, _prog("c1", "print('uncertified')"))
        assert result == {}  # not added

    def test_certified_candidate_included(self, state_store):
        certified = CertifiedArchive(state_store)
        code = "print('baseline')"
        certified.add_baseline(
            "init_0", episode_id="E1", model_family="gbdt",
            code_hash=hashlib.sha256(code.encode()).hexdigest(),
        )
        pop_module = CertifiedOnlyPopulation(certified)
        result = pop_module.update_population({}, _prog("init_0", code))
        assert "certified" in result
        assert result["certified"][0].id == "init_0"

    def test_no_duplicates(self, state_store):
        certified = CertifiedArchive(state_store)
        code = "print('x')"
        certified.add_baseline(
            "b", episode_id="E1", model_family="gbdt",
            code_hash=hashlib.sha256(code.encode()).hexdigest(),
        )
        pop_module = CertifiedOnlyPopulation(certified)
        pop = pop_module.update_population({}, _prog("b", code))
        pop = pop_module.update_population(pop, _prog("b", code))
        assert len(pop["certified"]) == 1
