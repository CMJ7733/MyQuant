"""Tier assignment: percentile gate plus absolute floor (§3.4, Appendix A.4).

The rule, verbatim from the paper: an alpha whose five metrics *all* exceed the
65th percentile among the alphas of the same generation is *qualified*; above the
80th percentile it is *elite*.  Each metric additionally has a minimum bound "to
prevent the dominance of outliers": IC and RankIC ≥ 0.005, ICIR and RankICIR ≥
0.05, MI ≥ 0.02, with 0.01 / 0.1 / 0.02 for elite.

Why the floor matters, concretely: percentiles are computed *within a generation*.
A generation where every alpha is noise still has a 65th percentile, so without a
floor a fifth of pure noise would be promoted to parents every round and the
search would drift on rank alone.  The floor is what makes the gate absolute.

On sign
-------
``use_abs_ic`` (default on) tiers on |IC| rather than IC.  A factor's sign is a
free parameter — negate the returned series and IC flips — so a strongly negative
IC is a strongly *informative* factor written in the inconvenient direction.  The
paper does not discuss this; scoring signed IC discards half of every generation's
information for a reason that a single ``-`` fixes.  ``fitness.use_abs_ic: false``
restores the strict reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from cogalpha.config import FitnessConfig
from cogalpha.types import Alpha, AlphaTier, Fitness

#: Metrics that participate in tiering, in canonical order.
TIER_METRICS: Tuple[str, ...] = Fitness.TIER_METRICS


@dataclass
class TierOutcome:
    """Result of tiering one generation."""

    qualified: List[Alpha] = field(default_factory=list)
    elite: List[Alpha] = field(default_factory=list)
    plain: List[Alpha] = field(default_factory=list)
    cutoffs: Dict[str, float] = field(default_factory=dict)
    elite_cutoffs: Dict[str, float] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    reasons: Dict[str, str] = field(default_factory=dict)

    @property
    def n_scored(self) -> int:
        """Alphas that entered tiering. Note ``qualified`` includes elites, so this
        counts ``qualified + plain`` rather than summing all three lists."""
        return len(self.qualified) + len(self.plain)


def tier_values(fitness: Fitness, use_abs_ic: bool = True) -> Dict[str, float]:
    """The five metric values as tiering sees them.

    Only the correlation metrics take an absolute value; MI is already
    non-negative, and ICIR/RankICIR inherit the sign convention of their IC, so
    taking |ICIR| alongside |IC| keeps the pair consistent.
    """
    values = fitness.tier_values()
    if not use_abs_ic:
        return dict(values)
    out = dict(values)
    for key in ("ic", "icir", "rank_ic", "rank_icir"):
        v = out.get(key, float("nan"))
        out[key] = abs(v) if v == v else v
    return out


def percentile_cutoffs(
    population: Sequence[Fitness],
    percentile: float,
    use_abs_ic: bool = True,
) -> Dict[str, float]:
    """Per-metric percentile over a generation, ignoring non-finite values."""
    cutoffs: Dict[str, float] = {}
    for metric in TIER_METRICS:
        sample = [
            tier_values(f, use_abs_ic)[metric]
            for f in population
            if f is not None and np.isfinite(tier_values(f, use_abs_ic)[metric])
        ]
        cutoffs[metric] = (
            float(np.percentile(sample, percentile)) if sample else float("inf")
        )
    return cutoffs


def _passes(
    values: Dict[str, float],
    cutoffs: Dict[str, float],
    floors: Dict[str, float],
) -> Tuple[bool, str]:
    """All five metrics must clear both the percentile and the floor.

    Returns ``(passed, reason)``. The reason names the **first** failing metric, and
    it lands in the archive verbatim, which is what lets a run's rejection histogram
    say *which* metric is the binding constraint rather than just "not qualified".

    Failing fast on the first metric means the reason is the first in ``TIER_METRICS``
    order, not necessarily the worst offender -- fine for diagnosis, but do not read
    it as "this was the only problem".
    """
    for metric in TIER_METRICS:
        value = values.get(metric, float("nan"))
        if not np.isfinite(value):
            return False, f"{metric} is not finite"
        floor = floors.get(metric)
        if floor is not None and value < floor:
            return False, f"{metric}={value:.4g} below the floor {floor:g}"
        cutoff = cutoffs.get(metric, float("inf"))
        if not np.isfinite(cutoff):
            # An undefined cutoff means no alpha in this generation had a finite
            # value for the metric. Passing would promote on missing evidence.
            return False, f"{metric} percentile cutoff undefined"
        if value < cutoff:
            return False, f"{metric}={value:.4g} below the generation cutoff {cutoff:.4g}"
    return True, "all five metrics cleared the cutoff and the floor"


def combined_score(fitness: Fitness, use_abs_ic: bool = True) -> float:
    """A single scalar for ranking within a tier.

    Not part of the paper's specification — the paper tiers and does not rank
    inside a tier.  But elitism ("the top two elite alphas") and the plateau rule
    ("mean of the elite pool") both need an order, so one is defined here and used
    consistently: the mean of the five metrics after scaling each to a comparable
    range.  IC-type metrics are multiplied by 20 and IR-type by 2 so that a
    "good" value of each (IC 0.05, ICIR 0.5, MI 0.5) contributes about equally;
    without the scaling MI alone would decide every ordering.
    """
    values = tier_values(fitness, use_abs_ic)
    weights = {"ic": 20.0, "rank_ic": 20.0, "icir": 2.0, "rank_icir": 2.0, "mi": 1.0}
    parts = [
        weights[m] * values[m]
        for m in TIER_METRICS
        if np.isfinite(values.get(m, float("nan")))
    ]
    if not parts:
        return float("-inf")
    return float(np.mean(parts))


def assign_tiers(
    alphas: Sequence[Alpha],
    cfg: FitnessConfig,
) -> TierOutcome:
    """Tier one generation of scored alphas in place.

    Alphas without a fitness (rejected by the quality checker) are left as
    ``INVALID`` and excluded from the percentile computation — including them
    would let a generation full of broken code lower the bar for the survivors.
    """
    scored = [a for a in alphas if a.fitness is not None and a.fitness.is_finite()]
    outcome = TierOutcome()

    if not scored:
        for alpha in alphas:
            if alpha.fitness is None:
                continue
            alpha.tier = AlphaTier.INVALID
            outcome.reasons[alpha.alpha_id] = "no finite metrics in this generation"
        return outcome

    population = [a.fitness for a in scored]
    q_cut = percentile_cutoffs(population, cfg.qualified_percentile, cfg.use_abs_ic)
    e_cut = percentile_cutoffs(population, cfg.elite_percentile, cfg.use_abs_ic)
    outcome.cutoffs = q_cut
    outcome.elite_cutoffs = e_cut

    for alpha in scored:
        values = tier_values(alpha.fitness, cfg.use_abs_ic)  # type: ignore[arg-type]
        score = combined_score(alpha.fitness, cfg.use_abs_ic)  # type: ignore[arg-type]
        outcome.scores[alpha.alpha_id] = score

        # Test elite first: the elite gate is strictly stricter, so an alpha clearing
        # it clears the qualified gate too and the second test is only needed for its
        # reason string.
        elite_ok, elite_why = _passes(values, e_cut, cfg.elite_min_bounds)
        qual_ok, qual_why = _passes(values, q_cut, cfg.qualified_min_bounds)

        if elite_ok:
            alpha.tier = AlphaTier.ELITE
            outcome.elite.append(alpha)
            # An elite alpha is by construction also qualified, and the paper
            # passes qualified alphas to the next generation -- so elites belong
            # in the parent pool too, not only in the final candidate pool.
            # Consequence for callers: `outcome.qualified` is a superset of
            # `outcome.elite`, and filtering on `tier == QUALIFIED` exactly will
            # silently drop the best alphas of the generation.
            outcome.qualified.append(alpha)
            outcome.reasons[alpha.alpha_id] = f"elite: {elite_why}"
        elif qual_ok:
            alpha.tier = AlphaTier.QUALIFIED
            outcome.qualified.append(alpha)
            outcome.reasons[alpha.alpha_id] = f"qualified: {qual_why}"
        else:
            # PLAIN, not INVALID: it ran cleanly and is leakage-free, it merely did
            # not score well enough. Adaptive Generation uses these as its "did not
            # work" examples, which needs the distinction from a checker rejection.
            alpha.tier = AlphaTier.PLAIN
            outcome.plain.append(alpha)
            outcome.reasons[alpha.alpha_id] = f"not qualified: {qual_why}"

    # Sorted best-first so that elitism (`top 2`) and parent-pool truncation can just
    # slice; several callers rely on this order.
    outcome.qualified.sort(key=lambda a: outcome.scores[a.alpha_id], reverse=True)
    outcome.elite.sort(key=lambda a: outcome.scores[a.alpha_id], reverse=True)
    outcome.plain.sort(key=lambda a: outcome.scores[a.alpha_id], reverse=True)
    return outcome


def worst_invalid(
    alphas: Iterable[Alpha],
    n: int,
    cfg: Optional[FitnessConfig] = None,
) -> List[Alpha]:
    """The ``n`` worst-performing invalid alphas, for Adaptive Generation (§3.5).

    "Worst-performing invalid" spans two populations: alphas that scored but fell
    below the gate (rankable by score), and alphas the checker rejected outright
    (not rankable, but the most instructive failures). Scored-but-poor come first
    so the feedback names a concrete numeric failure, with rejected ones filling
    the remainder.
    """
    items = list(alphas)
    use_abs = cfg.use_abs_ic if cfg is not None else True

    scored_poor = [
        a
        for a in items
        if a.fitness is not None
        and a.fitness.is_finite()
        and a.tier in (AlphaTier.PLAIN, AlphaTier.INVALID)
    ]
    scored_poor.sort(key=lambda a: combined_score(a.fitness, use_abs))  # type: ignore[arg-type]

    rejected = [a for a in items if a.rejected_at is not None]

    out = scored_poor[:n]
    if len(out) < n:
        out.extend(rejected[: n - len(out)])
    return out
