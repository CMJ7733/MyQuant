"""Deterministic offline backend.

This is not a language model imitation.  It is a *code generator with the same
interface as one*: given the role tag on a call, it emits a well-formed response
of the shape that role's parser expects — including, for generation roles, real
Python alpha code assembled from a template bank.

That makes the mock useful for two things the real backend cannot do cheaply:

1. the test suite gets a full 24-generation search with no network and no cost;
2. every non-LLM part of the system (sandbox, metrics, tiering, elitism, plateau
   stopping, archiving) is verified against a *fixed* generator, so a regression
   there cannot hide behind model randomness.

Its alphas are drawn from hand-written OHLCV expressions of genuinely varying
quality, so tiering has something to discriminate.  Roughly a fifth of the bank
is deliberately defective — syntax errors, all-NaN output, constants, forward
shifts — so the quality checker's rejection paths are exercised on every run.

Capacity limit
--------------
The bank is finite: 12 sound templates x 6 window choices is on the order of ~70
distinct structures.  Because de-duplication is on canonical source
(:func:`~cogalpha.types.canonical_code`), a long mock run *will* exhaust it and
later generations report mostly ``duplicates_dropped``.  That is the mock running
out of ideas, not the search failing — a real model has no such ceiling.  Read
``unique_structures`` on a mock run as a fraction of ~70, not of the number
generated, and keep mock runs short.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Dict, List, Optional

from cogalpha.llm.base import LLMClient, LLMResponse

# --------------------------------------------------------------------------- bank
# Each template is a complete function body.  ``{w}``/``{w2}`` are window sizes and
# ``{name}`` the function name, filled per draw so mutation produces real variants.

_GOOD_TEMPLATES: List[str] = [
    # Liquidity impact -- the paper's Listing 1 family.
    """def {name}(df):
    \"\"\"Liquidity impact: price rise (high-close) per unit of traded volume.

    A large positive value means the stock moved up strongly while volume stayed
    low, indicating thin liquidity and a higher expected short-term return.
    Formula: (high - close) / (volume + eps).
    \"\"\"
    out = df.copy()
    eps = 1e-9
    out['{name}'] = (out['high'] - out['close']) / (out['volume'] + eps)
    return out['{name}']
""",
    # Dollar-volume normalised impact with a bounded transform -- Listing 3 family.
    """def {name}(df):
    \"\"\"Impact proxy: absolute daily move per dollar volume, tanh-bounded.

    Normalising by dollar volume removes the price-level effect so the factor is
    comparable across names; tanh caps tail noise.
    \"\"\"
    import numpy as np
    out = df.copy()
    eps = 1e-9
    abs_move = (out['close'] - out['open']).abs()
    dollar_vol = out['volume'] * out['close']
    raw = abs_move / (dollar_vol + eps)
    out['{name}'] = np.tanh(raw * 1e6)
    return out['{name}']
""",
    # Short-horizon reversal (Level IV, AgentReversal).
    """def {name}(df):
    \"\"\"Short-term reversal: negative of the trailing {w}-day return.

    Transient overreaction tends to correct, so recent losers are expected to
    outperform over the next horizon.
    \"\"\"
    out = df.copy()
    eps = 1e-12
    past = out['close'].shift({w})
    out['{name}'] = -(out['close'] / (past + eps) - 1.0)
    return out['{name}']
""",
    # Range compression (Level IV, AgentRangeVol).
    """def {name}(df):
    \"\"\"Range compression: today's true range against its {w}-day average.

    A contracting range precedes expansion; low values flag coiled volatility.
    \"\"\"
    out = df.copy()
    eps = 1e-12
    rng = (out['high'] - out['low']) / (out['close'] + eps)
    out['{name}'] = -(rng / (rng.rolling({w}, min_periods=max(2, {w} // 2)).mean() + eps))
    return out['{name}']
""",
    # Price-volume coherence (Level III).
    """def {name}(df):
    \"\"\"Price-volume coherence: rolling correlation of returns and volume change.

    Negative coherence means moves happen on shrinking volume -- a fragile trend.
    \"\"\"
    out = df.copy()
    ret = out['close'].pct_change()
    dvol = out['volume'].pct_change()
    out['{name}'] = -ret.rolling({w}, min_periods=max(3, {w} // 2)).corr(dvol)
    return out['{name}']
""",
    # Candle geometry (Level VII, AgentBarShape).
    """def {name}(df):
    \"\"\"Bar shape: upper-shadow share of the daily range.

    A long upper shadow marks rejected highs -- selling pressure into strength.
    \"\"\"
    out = df.copy()
    eps = 1e-12
    span = (out['high'] - out['low']) + eps
    upper = out['high'] - out[['open', 'close']].max(axis=1)
    out['{name}'] = -(upper / span)
    return out['{name}']
""",
    # Volatility asymmetry (Level IV, AgentVolAsymmetry).
    """def {name}(df):
    \"\"\"Volatility asymmetry: downside minus upside realised volatility over {w} days.

    Skewed downside risk earns a premium, so high asymmetry predicts higher return.
    \"\"\"
    import numpy as np
    out = df.copy()
    ret = out['close'].pct_change()
    down = ret.where(ret < 0, 0.0)
    up = ret.where(ret > 0, 0.0)
    mp = max(3, {w} // 2)
    out['{name}'] = (
        down.rolling({w}, min_periods=mp).std() - up.rolling({w}, min_periods=mp).std()
    )
    return out['{name}']
""",
    # Drawdown geometry (Level V, AgentDrawdown).
    """def {name}(df):
    \"\"\"Drawdown depth against the trailing {w}-day high.

    Deep but recovering drawdowns carry a resilience premium.
    \"\"\"
    out = df.copy()
    eps = 1e-12
    peak = out['close'].rolling({w}, min_periods=max(2, {w} // 2)).max()
    out['{name}'] = out['close'] / (peak + eps) - 1.0
    return out['{name}']
""",
    # Regime gating (Level VI) -- a branch, which a formula tree cannot express.
    """def {name}(df):
    \"\"\"Regime-gated reversal: the {w}-day reversal, active only in calm markets.

    Gating suppresses the signal when {w2}-day volatility is above its median, on
    the view that reversal decays in turbulent regimes.
    \"\"\"
    out = df.copy()
    eps = 1e-12
    ret = out['close'].pct_change()
    vol = ret.rolling({w2}, min_periods=max(3, {w2} // 2)).std()
    gate = (vol <= vol.expanding(min_periods={w2}).median().shift(1)).astype(float)
    rev = -(out['close'] / (out['close'].shift({w}) + eps) - 1.0)
    out['{name}'] = rev * gate
    return out['{name}']
""",
    # Volume structure (Level III, AgentVolumeStructure).
    """def {name}(df):
    \"\"\"Volume surprise: log volume against its {w}-day mean.

    Unusual participation without a matching price move signals absorption.
    \"\"\"
    import numpy as np
    out = df.copy()
    lv = np.log(out['volume'].clip(lower=1.0))
    mp = max(2, {w} // 2)
    out['{name}'] = -(lv - lv.rolling({w}, min_periods=mp).mean())
    return out['{name}']
""",
    # Market cycle (Level I).
    """def {name}(df):
    \"\"\"Cycle position: distance from the {w}-day mid-range, sign-flipped.

    Names near the bottom of their cycle range are expected to mean-revert up.
    \"\"\"
    out = df.copy()
    eps = 1e-12
    mp = max(2, {w} // 2)
    hi = out['high'].rolling({w}, min_periods=mp).max()
    lo = out['low'].rolling({w}, min_periods=mp).min()
    out['{name}'] = -((out['close'] - lo) / (hi - lo + eps) - 0.5)
    return out['{name}']
""",
    # Tail risk (Level II).
    """def {name}(df):
    \"\"\"Tail exposure: worst {w}-day return relative to realised volatility.

    A mild tail given the volatility level indicates a sturdier name.
    \"\"\"
    out = df.copy()
    eps = 1e-12
    ret = out['close'].pct_change()
    mp = max(3, {w} // 2)
    out['{name}'] = ret.rolling({w}, min_periods=mp).min() / (
        ret.rolling({w}, min_periods=mp).std() + eps
    )
    return out['{name}']
""",
]

#: Deliberately broken templates, so rejection paths run on every mock search.
_BAD_TEMPLATES: List[Dict[str, str]] = [
    {
        "kind": "syntax",
        "code": """def {name}(df):
    \"\"\"Momentum over {w} days.\"\"\"
    out = df.copy(
    out['{name}'] = out['close'] / out['close'].shift({w}) - 1.0
    return out['{name}']
""",
    },
    {
        "kind": "constant",
        "code": """def {name}(df):
    \"\"\"Constant placeholder -- carries no cross-sectional information.\"\"\"
    out = df.copy()
    out['{name}'] = 1.0
    return out['{name}']
""",
    },
    {
        "kind": "all_nan",
        "code": """def {name}(df):
    \"\"\"Ratio that divides by zero everywhere, producing NaN.\"\"\"
    out = df.copy()
    zero = out['close'] - out['close']
    out['{name}'] = (out['high'] - out['low']) / zero
    return out['{name}']
""",
    },
    {
        "kind": "leakage",
        "code": """def {name}(df):
    \"\"\"Anticipates the next {w} days by shifting the close backwards.\"\"\"
    out = df.copy()
    eps = 1e-12
    out['{name}'] = out['close'].shift(-{w}) / (out['close'] + eps) - 1.0
    return out['{name}']
""",
    },
    {
        "kind": "runtime",
        "code": """def {name}(df):
    \"\"\"References a column that does not exist in the panel.\"\"\"
    out = df.copy()
    out['{name}'] = out['turnover_rate'] / out['close']
    return out['{name}']
""",
    },
]

_STEM_RE = re.compile(r'"""([^\n"]+)')


def _stem_from_template(template: str) -> str:
    """Derive a function-name stem from the template's docstring summary.

    Keeps the generated name honest — ``factor_liquidity_impact_20d`` really is the
    liquidity template — without duplicating the stem as separate data that can
    drift out of sync with the code it names.
    """
    match = _STEM_RE.search(template)
    if match is None:  # pragma: no cover - every template has a docstring
        return "alpha"
    words = re.findall(r"[a-z]+", match.group(1).lower())
    return "_".join(words[:2]) if words else "alpha"


class MockLLMClient(LLMClient):
    """Deterministic stand-in for a code-generating LLM.

    Determinism is **per run**, not per prompt: the RNG for a call is seeded from
    ``mock_seed``, a hash of the prompt, *and* the call index.  So replaying a run
    from the same seed reproduces it exactly, while two identical prompts still
    give different alphas -- which is what a real model at temperature 0.7-1.2
    does, and what the search loop needs.

    Seeding on the prompt alone (the obvious choice) makes every mutation of a
    given parent return the same child, so a 96-child generation collapses to one
    distinct alpha and de-duplication discards the rest.  Observed directly: 35 of
    36 breeding attempts dropped as duplicates.
    """

    def __init__(
        self,
        model: str = "mock",
        default_temperature: float = 0.8,
        max_tokens: int = 4096,
        seed: int = 0,
        recorder=None,
        max_calls: Optional[int] = None,
        bad_rate: float = 0.2,
        max_concurrency: int = 1,
    ) -> None:
        super().__init__(
            model=model,
            default_temperature=default_temperature,
            max_tokens=max_tokens,
            recorder=recorder,
            max_calls=max_calls,
            max_concurrency=max_concurrency,
        )
        self.seed = seed
        self.bad_rate = bad_rate

    # ------------------------------------------------------------------ helpers

    def _rng(self, prompt: str) -> random.Random:
        digest = hashlib.sha1(
            f"{self.seed}:{self.n_calls}:{prompt}".encode("utf-8")
        ).hexdigest()
        return random.Random(int(digest[:16], 16))

    def _alpha_block(self, rng: random.Random, n: int) -> str:
        blocks: List[str] = []
        for _ in range(n):
            window = rng.choice([3, 5, 10, 20, 30, 60])
            window2 = rng.choice([10, 20, 40])

            if rng.random() < self.bad_rate:
                template = rng.choice(_BAD_TEMPLATES)["code"]
            else:
                template = rng.choice(_GOOD_TEMPLATES)

            stem = _stem_from_template(template)
            name = f"factor_{stem}_{window}d_{rng.randrange(1000, 9999)}"
            code = template.format(name=name, w=window, w2=window2)
            blocks.append(f"```python\n{code}```")
        return "\n\n".join(blocks)

    # ---------------------------------------------------------------- interface

    def _complete(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        rng = self._rng(prompt)
        role = _infer_role(prompt, system)

        if role == "generate":
            n = _requested_count(prompt, default=5)
            text = self._alpha_block(rng, n)
        elif role in {"mutate", "crossover"}:
            text = self._alpha_block(rng, 1)
        elif role == "code_quality":
            # Static checks upstream already catch real breakage; the mock reviewer
            # passes everything so the deterministic path is the interesting one.
            text = "VERDICT: PASS\nNo blocking issues found."
        elif role == "judge":
            # Reject a deterministic minority so the improvement loop is exercised.
            if rng.random() < 0.15:
                text = (
                    "VERDICT: REVISE\n"
                    "The economic rationale is thin: the normalisation does not make "
                    "the factor comparable across price levels."
                )
            else:
                text = "VERDICT: PASS\nLogically consistent and economically meaningful."
        elif role in {"repair", "improve"}:
            text = self._alpha_block(rng, 1)
        elif role == "analyse":
            text = (
                "The effective alphas normalise by a volume or volatility scale, which "
                "makes them comparable across names; the ineffective ones are raw price "
                "differences dominated by price level, or constants with no dispersion."
            )
        else:  # pragma: no cover - defensive
            text = "OK"

        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(text) // 4)
        return LLMResponse(
            text=text,
            model=self.model,
            temperature=temperature,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            finish_reason="stop",
            raw={"mock_role": role},
        )


def _infer_role(prompt: str, system: Optional[str]) -> str:
    """Recover the calling role from the prompt's role marker.

    Every prompt template in :mod:`cogalpha.prompts` starts with a
    ``[ROLE: xxx]`` line precisely so the mock (and any log reader) can tell the
    stages apart without brittle keyword matching.
    """
    haystack = f"{system or ''}\n{prompt}"
    marker = "[ROLE:"
    idx = haystack.find(marker)
    if idx >= 0:
        end = haystack.find("]", idx)
        if end > idx:
            return haystack[idx + len(marker) : end].strip().lower()
    return "generate"


def _requested_count(prompt: str, default: int) -> int:
    """Parse ``[COUNT: n]`` if the prompt declares how many alphas it wants."""
    marker = "[COUNT:"
    idx = prompt.find(marker)
    if idx < 0:
        return default
    end = prompt.find("]", idx)
    try:
        return max(1, min(20, int(prompt[idx + len(marker) : end].strip())))
    except ValueError:
        return default
