"""The Seven-Level Agent Hierarchy (§3.1, Appendix A.1, Figure 2).

Twenty-one task-specific agents, arranged macro to micro.  Each agent's ``focus``
text is the paper's own description of that agent, and ``probe`` is the concrete
research question handed to the LLM.

The point of the hierarchy is explicit *partitioning* of the search space: an
unconstrained agent falls into path dependence and re-emits the same few
formulas, which is exactly the "formula stacking" the paper criticises in prior
LLM work.  Splitting the space along a practitioner's factor taxonomy buys
breadth for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

#: Golden-ratio constant used to select a subset of agents per run (§B.8).
GOLDEN_RATIO = 0.6180339887498949


@dataclass(frozen=True)
class AgentSpec:
    """One task-specific agent."""

    name: str
    level: int
    layer: str
    focus: str
    probe: str

    @property
    def slug(self) -> str:
        """Lowercase name without the ``Agent`` prefix, for filenames and log keys."""
        return self.name.replace("Agent", "", 1).lower()


LAYERS: Dict[int, str] = {
    1: "Market Structure & Cycle Layer",
    2: "Extreme Risk & Fragility Layer",
    3: "Price-Volume Dynamics Layer",
    4: "Price-Volatility Behavior Layer",
    5: "Multi-Scale Complexity Layer",
    6: "Stability & Regime-Gating Layer",
    7: "Geometric & Fusion Layer",
}

LAYER_DESCRIPTIONS: Dict[int, str] = {
    1: (
        "Explores large-scale temporal structures such as long-term trends, market "
        "phases, and cyclical state transitions inferred from daily OHLCV dynamics."
    ),
    2: (
        "Models tail-risk exposure, crash precursors, and systemic fragility patterns "
        "that signal potential regime breakdowns or stress accumulation."
    ),
    3: (
        "Captures the interactions between price and trading activity -- liquidity, "
        "order imbalance, and coherence between price movement and volume behavior."
    ),
    4: (
        "Analyzes trend persistence, short-term reversals, volatility clustering, and "
        "asymmetric price dynamics as core sources of predictive alpha."
    ),
    5: (
        "Measures cross-scale irregularities, fractal roughness, drawdown-recovery "
        "geometry, and long-memory characteristics in time-series structures."
    ),
    6: (
        "Assesses temporal stability and constructs adaptive gating mechanisms that "
        "regulate signal activation under varying market conditions."
    ),
    7: (
        "Focuses on geometric pattern representation (candlestick morphology) and "
        "multi-factor fusion, combining independent signals into coherent composites."
    ),
}


HIERARCHY: Tuple[AgentSpec, ...] = (
    # ------------------------------------------------------ Level I (2 agents)
    AgentSpec(
        name="AgentMarketCycle",
        level=1,
        layer=LAYERS[1],
        focus=(
            "Long-term trends, market phases, and cyclical state transitions inferred "
            "from daily OHLCV dynamics."
        ),
        probe=(
            "Where does the stock sit within its own multi-month cycle, and does that "
            "position predict the next leg? Consider phase relative to long-horizon "
            "highs and lows, the transition between accumulation and distribution, and "
            "how far price has travelled through the current phase."
        ),
    ),
    AgentSpec(
        name="AgentVolatilityRegime",
        level=1,
        layer=LAYERS[1],
        focus="Volatility regime identification and transitions between regimes.",
        probe=(
            "Which volatility regime is the stock in, and is it about to switch? "
            "Contrast short- and long-horizon realised volatility, the persistence of "
            "the current level, and how returns behave conditional on the regime."
        ),
    ),
    # ----------------------------------------------------- Level II (2 agents)
    AgentSpec(
        name="AgentTailRisk",
        level=2,
        layer=LAYERS[2],
        focus="Tail-risk exposure and the shape of the loss distribution.",
        probe=(
            "How exposed is the stock to left-tail outcomes, and is that exposure "
            "compensated? Consider worst-case returns over a window, downside "
            "semi-deviation, and the ratio of extreme to typical moves."
        ),
    ),
    AgentSpec(
        name="AgentCrashPredictor",
        level=2,
        layer=LAYERS[2],
        focus="Crash precursors and systemic fragility that precede regime breakdown.",
        probe=(
            "What observable pattern precedes a sharp decline? Look for stress "
            "accumulation: successive lower closes on rising volume, widening ranges "
            "with failing follow-through, or gap behaviour that erodes support."
        ),
    ),
    # ---------------------------------------------------- Level III (4 agents)
    AgentSpec(
        name="AgentLiquidity",
        level=3,
        layer=LAYERS[3],
        focus="Liquidity conditions and price impact per unit of traded volume.",
        probe=(
            "How much price movement does a unit of volume buy? Thin liquidity means "
            "small trades move prices far, which market microstructure theory links to "
            "short-term reversal and an illiquidity premium. Normalise by dollar volume "
            "so the measure is comparable across price levels."
        ),
    ),
    AgentSpec(
        name="AgentOrderImbalance",
        level=3,
        layer=LAYERS[3],
        focus="Directional order imbalance inferred from OHLCV geometry.",
        probe=(
            "Without an order book, what does the bar tell you about buy/sell pressure? "
            "Consider where the close sits inside the day's range, the balance of "
            "up-volume against down-volume, and the persistence of that imbalance."
        ),
    ),
    AgentSpec(
        name="AgentPriceVolumeCoherence",
        level=3,
        layer=LAYERS[3],
        focus="Coherence between price movement and volume behaviour.",
        probe=(
            "Do price and volume agree? Moves on expanding volume confirm; moves on "
            "shrinking volume are fragile. Quantify the agreement with a rolling "
            "association between returns and volume changes."
        ),
    ),
    AgentSpec(
        name="AgentVolumeStructure",
        level=3,
        layer=LAYERS[3],
        focus="The structure of trading activity itself, independent of direction.",
        probe=(
            "Is participation unusual? Compare current volume with its own recent "
            "distribution, examine the concentration of volume across days, and look "
            "for absorption -- heavy volume that fails to move price."
        ),
    ),
    # ----------------------------------------------------- Level IV (5 agents)
    AgentSpec(
        name="AgentDailyTrend",
        level=4,
        layer=LAYERS[4],
        focus="Directional persistence and multi-day momentum strength.",
        probe=(
            "How sustained is the current direction? Consider the fraction of recent "
            "days that moved the same way, the ratio of net travel to total travel "
            "(trend efficiency), and momentum measured across two horizons."
        ),
    ),
    AgentSpec(
        name="AgentReversal",
        level=4,
        layer=LAYERS[4],
        focus="Mean reversion and correction of short-term overreaction.",
        probe=(
            "Has the stock overshot? Consider deviation from a short moving average, "
            "the size of the last move relative to its typical size, and the tendency "
            "of transient mispricings to correct within days."
        ),
    ),
    AgentSpec(
        name="AgentRangeVol",
        level=4,
        layer=LAYERS[4],
        focus="Range-based volatility dynamics, including compression-expansion cycles.",
        probe=(
            "Is the daily range contracting or expanding? Range compression tends to "
            "precede expansion. Use high-low ranges scaled by price and compare the "
            "current range with its own recent average."
        ),
    ),
    AgentSpec(
        name="AgentLagResponse",
        level=4,
        layer=LAYERS[4],
        focus="Delayed price adjustment and lagged feedback among volatility, volume and returns.",
        probe=(
            "Does information arrive with a lag? Look for returns that respond to "
            "yesterday's volume or volatility shock, and for cross-lagged relations "
            "that indicate slow diffusion of information."
        ),
    ),
    AgentSpec(
        name="AgentVolAsymmetry",
        level=4,
        layer=LAYERS[4],
        focus="Asymmetric volatility between upward and downward moves.",
        probe=(
            "Is the stock's volatility skewed? Compare realised volatility computed on "
            "down days against up days, or the average magnitude of negative versus "
            "positive returns, to capture skewed risk behaviour."
        ),
    ),
    # ------------------------------------------------------ Level V (2 agents)
    AgentSpec(
        name="AgentDrawdown",
        level=5,
        layer=LAYERS[5],
        focus="Depth, duration and recovery geometry of cumulative losses.",
        probe=(
            "What is the shape of the current drawdown? Consider depth against a "
            "trailing peak, how long the stock has been underwater, and the speed of "
            "recovery -- resilience is itself a signal."
        ),
    ),
    AgentSpec(
        name="AgentFractal",
        level=5,
        layer=LAYERS[5],
        focus="Multi-scale roughness and long-memory characteristics.",
        probe=(
            "How rough is the price path across scales? Compare variability measured at "
            "two horizons (a scaling exponent), or count direction changes per window, "
            "to separate persistent trends from noisy churn."
        ),
    ),
    # ----------------------------------------------------- Level VI (2 agents)
    AgentSpec(
        name="AgentRegimeGating",
        level=6,
        layer=LAYERS[6],
        focus=(
            "Adaptive gates that modulate signal activation depending on volatility, "
            "trend or liquidity states."
        ),
        probe=(
            "Take a signal that works only sometimes and gate it. Build a state "
            "indicator from past data alone, then activate the signal only in the "
            "favourable state. The gate must use no information from the current or "
            "future bar."
        ),
    ),
    AgentSpec(
        name="AgentStability",
        level=6,
        layer=LAYERS[6],
        focus="Temporal consistency and persistence in returns or derived signals.",
        probe=(
            "How stable is the stock's behaviour? Consider the autocorrelation of "
            "returns, the variability of a rolling statistic, or the smoothness of the "
            "price path -- stable names deserve more weight."
        ),
    ),
    # ---------------------------------------------------- Level VII (4 agents)
    AgentSpec(
        name="AgentBarShape",
        level=7,
        layer=LAYERS[7],
        focus="Candlestick geometry -- body, shadow and symmetry -- as continuous descriptors.",
        probe=(
            "Encode the bar's geometry numerically rather than as a named pattern: body "
            "share of the range, upper versus lower shadow, the asymmetry between them, "
            "and how those descriptors evolve over a few days."
        ),
    ),
    AgentSpec(
        name="AgentCreative",
        level=7,
        layer=LAYERS[7],
        focus="Non-linear transformations, reparametrisations, or soft gating.",
        probe=(
            "Take a plain quantity and change its geometry: bound it with tanh or "
            "arctan, take a signed log, or blend two views with a soft weight. The aim "
            "is a better-behaved distribution, not obfuscation -- state why the "
            "transformation helps."
        ),
    ),
    AgentSpec(
        name="AgentComposite",
        level=7,
        layer=LAYERS[7],
        focus="Fusing independent factors into coherent composites, emphasising synergy and orthogonality.",
        probe=(
            "Combine two economically distinct signals -- say a liquidity measure and a "
            "trend measure -- so the composite is informative where neither is alone. "
            "Scale each part before combining so one does not dominate by units."
        ),
    ),
    AgentSpec(
        name="AgentHerding",
        level=7,
        layer=LAYERS[7],
        focus="Collective crowding behaviour and directional alignment within OHLCV dynamics.",
        probe=(
            "Is the crowd aligned? Consider streaks of same-signed returns paired with "
            "rising volume, the share of the range travelled in one direction, or "
            "acceleration in participation -- consensus intensity that later unwinds."
        ),
    ),
)

assert len(HIERARCHY) == 21, "the paper specifies exactly 21 task-specific agents"

BY_NAME: Dict[str, AgentSpec] = {a.name: a for a in HIERARCHY}


def by_level(level: int) -> List[AgentSpec]:
    """Agents on one hierarchy level (1-7)."""
    return [a for a in HIERARCHY if a.level == level]


def get_agent(name: str) -> AgentSpec:
    """Look up an agent by exact name; raises KeyError listing valid names."""
    if name not in BY_NAME:
        raise KeyError(f"unknown agent '{name}'; have {sorted(BY_NAME)}")
    return BY_NAME[name]


def select_agents(
    n: int,
    seed: int = 42,
    use_golden_ratio: bool = True,
    pool: Sequence[AgentSpec] = HIERARCHY,
) -> List[AgentSpec]:
    """Pick ``n`` of the 21 agents for one run.

    The paper selects 13 agents "applying the golden ratio, which is commonly used
    in quantitative finance for balanced allocation" (§B.8) without giving the
    mechanics.  We implement it as a low-discrepancy (additive-recurrence)
    sequence over the agent list: successive indices step by the golden ratio, so
    the chosen subset spreads across levels instead of clustering, and no agent
    repeats.  With ``use_golden_ratio=False`` the selection is a plain shuffle,
    which is the control arm for checking the choice matters.
    """
    if n <= 0:
        return []
    items = list(pool)
    if n >= len(items):
        return items

    if not use_golden_ratio:
        import random

        rng = random.Random(seed)
        picked = items[:]
        rng.shuffle(picked)
        return picked[:n]

    # Additive recurrence: x_{k+1} = (x_k + phi) mod 1, mapped onto the list and
    # de-duplicated by walking forward on collision.
    total = len(items)
    offset = (seed % 1000) * GOLDEN_RATIO % 1.0
    chosen: List[AgentSpec] = []
    used: set[int] = set()
    x = offset
    guard = 0
    while len(chosen) < n and guard < total * 10:
        guard += 1
        idx = int(x * total) % total
        while idx in used:
            idx = (idx + 1) % total
        used.add(idx)
        chosen.append(items[idx])
        x = (x + GOLDEN_RATIO) % 1.0
    return chosen


def hierarchy_summary() -> str:
    """Human-readable table of the hierarchy, used by ``cogalpha inspect``."""
    lines = []
    for level in sorted(LAYERS):
        agents = ", ".join(a.name for a in by_level(level))
        lines.append(f"Level {level:>2}  {LAYERS[level]}")
        lines.append(f"          {agents}")
    return "\n".join(lines)
