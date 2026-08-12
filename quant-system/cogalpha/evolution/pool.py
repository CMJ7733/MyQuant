"""Parent pool, candidate pool, elitism and the plateau stopping rule.

Three pieces of bookkeeping the paper specifies but does not name:

* the **parent pool** — qualified alphas, capped at ``parent_pool_size``, carrying
  the previous generation's top two elites forward unchanged (§3.4);
* the **candidate pool** — every elite ever produced, de-duplicated, which is the
  run's actual output (§3.4);
* the **plateau rule** — stop when the elite pool's mean stops improving across two
  consecutive windows by more than ``plateau_delta`` (§B.4).

De-duplication is content-addressed on ``alpha_id``, which hashes the *canonical*
source — function name, docstrings, comments and whitespace normalised away (see
:func:`~cogalpha.types.canonical_code`).  That makes "how much of the search space
did we actually cover" a countable quantity rather than a claim, which is the
measurement ``reflection.md`` names as missing from the paper.  Hashing raw source
would not: generated names carry a random suffix, so the same alpha under two labels
would count twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from cogalpha.config import EvolutionConfig, FitnessConfig
from cogalpha.fitness.thresholds import combined_score
from cogalpha.types import Alpha, AlphaTier


@dataclass
class CandidatePool:
    """Every elite alpha the run has produced, best first."""

    alphas: Dict[str, Alpha] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)

    def add(self, alphas: Sequence[Alpha], use_abs_ic: bool = True) -> int:
        """Add elites, keeping the better copy on an id collision.

        An id collision means identical code, which can legitimately arrive twice
        (two agents converging, or an elite carried forward and re-evaluated).
        Keeping the higher-scoring copy makes the pool independent of arrival order.
        """
        added = 0
        for alpha in alphas:
            if alpha.fitness is None:
                continue
            score = combined_score(alpha.fitness, use_abs_ic)
            existing = self.scores.get(alpha.alpha_id)
            if existing is None:
                added += 1
            elif score <= existing:
                continue
            self.alphas[alpha.alpha_id] = alpha
            self.scores[alpha.alpha_id] = score
        return added

    def top(self, n: Optional[int] = None) -> List[Alpha]:
        """Best ``n`` candidates by combined score, or all of them when ``n`` is None."""
        ordered = sorted(
            self.alphas.values(),
            key=lambda a: self.scores.get(a.alpha_id, float("-inf")),
            reverse=True,
        )
        return ordered if n is None else ordered[:n]

    def mean_score(self) -> float:
        """Mean score of the whole pool. Not used by the plateau rule -- see
        :class:`PlateauStopper` for why the cumulative mean is the wrong signal."""
        finite = [s for s in self.scores.values() if np.isfinite(s)]
        return float(np.mean(finite)) if finite else float("nan")

    def __len__(self) -> int:
        return len(self.alphas)


def build_parent_pool(
    qualified: Sequence[Alpha],
    previous_elites: Sequence[Alpha],
    cfg: EvolutionConfig,
    fitness_cfg: FitnessConfig,
) -> List[Alpha]:
    """Assemble the next generation's parent pool.

    Order of operations matters.  The carried-forward elites are inserted *before*
    the cap is applied, so elitism cannot be silently undone by a generation that
    produced more than ``parent_pool_size`` qualified alphas — which is the whole
    point of "the top two elite alphas from the previous generation are always
    carried forward" (§B.4).
    """
    use_abs = fitness_cfg.use_abs_ic
    carried = _top_by_score(previous_elites, cfg.elitism_carry, use_abs)
    carried_ids = {a.alpha_id for a in carried}

    rest = [a for a in qualified if a.alpha_id not in carried_ids]
    rest = _top_by_score(rest, max(cfg.parent_pool_size - len(carried), 0), use_abs)

    pool = carried + rest
    seen: set = set()
    unique: List[Alpha] = []
    for alpha in pool:
        if alpha.alpha_id in seen:
            continue
        seen.add(alpha.alpha_id)
        unique.append(alpha)
    return unique[: cfg.parent_pool_size]


def _top_by_score(alphas: Sequence[Alpha], n: int, use_abs_ic: bool) -> List[Alpha]:
    if n <= 0:
        return []
    scored = [a for a in alphas if a.fitness is not None]
    scored.sort(key=lambda a: combined_score(a.fitness, use_abs_ic), reverse=True)  # type: ignore[arg-type]
    return scored[:n]


class PlateauStopper:
    """Early stop on a flat elite-pool trajectory (§B.4).

    The rule as written: track the elite-pool performance and compare two
    consecutive windows of length ``plateau_win``,
    ``delta = mean(curr) - mean(prev)``; if ``delta <= 0.001`` the run terminates.

    One clarification the paper leaves open: what "elite-pool performance" is per
    generation.  Here it is the mean combined score of the elites *produced in that
    generation* rather than of the cumulative pool.  The cumulative mean is
    strongly autocorrelated — adding one alpha to a pool of 80 barely moves it — so
    a cumulative reading would trip the plateau rule almost immediately regardless
    of whether the search was still finding things.
    """

    def __init__(self, window: int = 3, delta: float = 0.001) -> None:
        self.window = max(1, window)
        self.delta = delta
        self.history: List[float] = []
        self.last_delta: float = float("nan")

    def observe(self, generation_score: float) -> None:
        """Record one generation's elite score. NaN (no elites) is recorded as-is."""
        self.history.append(float(generation_score))

    def should_stop(self) -> bool:
        """True when the last two windows show no meaningful improvement.

        Needs ``2 * window`` observations before it can fire, so an early run is never
        stopped for lack of evidence.
        """
        if len(self.history) < 2 * self.window:
            return False
        recent = self.history[-self.window :]
        previous = self.history[-2 * self.window : -self.window]

        curr = _nanmean(recent)
        prev = _nanmean(previous)
        if not (np.isfinite(curr) and np.isfinite(prev)):
            # No elites in either window is not evidence of convergence; it is
            # evidence the gate is too tight, and stopping would hide that.
            return False

        self.last_delta = curr - prev
        return self.last_delta <= self.delta

    def reason(self) -> str:
        """Human-readable stop reason, archived in ``summary.json``."""
        return (
            f"elite-pool mean improved by {self.last_delta:+.5f} over "
            f"{self.window} generations, at or below the {self.delta} plateau threshold"
        )


def _nanmean(values: Sequence[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def generation_elite_score(
    elites: Sequence[Alpha],
    fitness_cfg: FitnessConfig,
) -> float:
    """Mean combined score of this generation's elites, or NaN if there were none."""
    scores = [
        combined_score(a.fitness, fitness_cfg.use_abs_ic)
        for a in elites
        if a.fitness is not None
    ]
    finite = [s for s in scores if np.isfinite(s)]
    return float(np.mean(finite)) if finite else float("nan")
