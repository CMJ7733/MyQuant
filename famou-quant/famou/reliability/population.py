"""Certified-only population module.

Implements the "Search Archive ≠ Certified Archive" split at the famou
population layer: the *live population* (what selection modules see as
parents) contains only certified members — initial baselines plus
gate-promoted candidates. Everything else still lands in the experiment
archive (search memory) via the Evolver, just not in the parent pool.

Membership check is by ``code_hash`` against the CertifiedArchive, so it
works regardless of how program IDs are renamed across islands.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from famou.core.data import Program
from famou.modules.population.base import PopulationModule
from famou.reliability.archives import CertifiedArchive


class CertifiedOnlyPopulation(PopulationModule):
    """Population = certified members only (baselines + gate-promoted)."""

    def __init__(self, certified_archive: CertifiedArchive, name: Optional[str] = None):
        super().__init__(name)
        self._certified = certified_archive

    def update_population(
        self,
        current_population: Dict[str, List[Program]],
        new_program: Program,
        experiment_config=None,
        island_id: int = 0,
    ) -> Dict[str, List[Program]]:
        import hashlib

        code_hash = hashlib.sha256(new_program.code.encode("utf-8")).hexdigest()
        members = self._certified.members()
        is_certified = any(m.get("code_hash") == code_hash for m in members.values())
        if not is_certified:
            # Search-archive-only candidate: recorded by the Evolver in the
            # experiment archive, but not eligible as a parent.
            return current_population

        pop = {k: list(v) for k, v in current_population.items()}
        bucket = pop.setdefault("certified", [])
        if all(p.id != new_program.id for p in bucket):
            bucket.append(new_program)
        return pop
