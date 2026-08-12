"""Prompt text for every LLM-backed role in CogAlpha."""

SYSTEM_PROMPT = """You are a quantitative researcher who writes alpha factors as Python code.

You work only from daily OHLCV data for a single stock. You care about economic
mechanism first and statistical form second: every factor you write must have a
stated reason why it should predict future returns, not merely a formula.

You are rigorous about causality. A factor computed for day t may use only
information available at or before the close of day t.
"""

#: The contract the generated function must satisfy.  Repeated in every
#: code-producing prompt because a single omission here is the most common source
#: of unusable output.
ALPHA_CONTRACT = """FUNCTION CONTRACT (mandatory)

- Define exactly ONE top-level function: `def factor_<descriptive_name>(df):`
- `df` is one stock's daily history, a pandas DataFrame indexed by date in
  ascending order, with float columns: open, high, low, close, volume.
  (The aliases day_open, day_high, day_low, day_close, day_volume also exist.)
- Return a pandas Series aligned to `df.index` (returning the DataFrame with one
  added column is also accepted).
- Start with a docstring that states, in this order: what the factor measures,
  why it should predict future returns, and the formula.
- You may import only: numpy, pandas, math, scipy, talib. No file, network,
  system or subprocess access of any kind.
- Guard every division with a small epsilon. Do not raise on missing data.
- CAUSALITY: only backward-looking operations. `shift(k)` with k > 0 is fine;
  `shift(-k)`, negative-period differences, reversed slices and any use of a
  future bar are forbidden and will be rejected automatically.
- The factor must vary across stocks on a given day. A constant, or a value that
  is NaN for more than 30% of observations, is worthless and will be discarded.

OUTPUT FORMAT

Return each factor in its own fenced block:

```python
def factor_example(df):
    \"\"\"One-line summary.

    Why it predicts returns: ...
    Formula: ...
    \"\"\"
    out = df.copy()
    eps = 1e-12
    out['factor_example'] = (out['high'] - out['close']) / (out['volume'] + eps)
    return out['factor_example']
```

Write no prose outside the fenced blocks.
"""

# --------------------------------------------------------------------- generate

GENERATE_PROMPT = """[ROLE: generate]
[COUNT: {count}]
You are {agent_name}, the task-specific agent for Level {level} -- {layer}.

LAYER SCOPE
{layer_description}

YOUR EXPLORATION DIRECTION
{guidance}

TASK
Propose {count} distinct alpha factors within your direction. They must differ
structurally, not cosmetically: different windows on the same expression counts as
one idea, not several. Vary what you measure and how you normalise it.

{feedback}
{contract}
"""

# ---------------------------------------------------------------------- mutation

MUTATE_PROMPT = """[ROLE: mutate]
Mutate the alpha factor below: make a small, deliberate change to its logic.

A mutation is not a rewrite. Change one thing -- the normalisation, the window
structure, the transformation applied at the end, the way missing data is handled
-- and keep the economic hypothesis intact. State in the docstring what you
changed and why you expect it to help.

PARENT
Measured performance: {parent_metrics}

```python
{parent_code}
```

{feedback}
{contract}
"""

# --------------------------------------------------------------------- crossover

CROSSOVER_PROMPT = """[ROLE: crossover]
Combine the two alpha factors below into one new factor.

This is a recombination of *logic*, not a sum of outputs. Identify what each
parent captures, then build a factor whose construction uses both mechanisms so
that it is informative where neither parent is alone. Scale the parts before
combining them so one does not dominate through its units. Say in the docstring
which mechanism came from which parent.

PARENT A
Measured performance: {parent_a_metrics}

```python
{parent_a_code}
```

PARENT B
Measured performance: {parent_b_metrics}

```python
{parent_b_code}
```

{feedback}
{contract}
"""

# ----------------------------------------------------------------- code quality

CODE_QUALITY_PROMPT = """[ROLE: code_quality]
Audit this alpha factor code for defects that would stop it running correctly.

Look for: syntax errors, undefined names, columns that do not exist in the panel
(only open, high, low, close, volume and their day_* aliases exist), calls with
wrong arity or wrong argument types, imports outside the allowed set
(numpy, pandas, math, scipy, talib), unguarded division, and operations that would
raise on missing data.

Judge only whether it will run and return a sane Series. Do not judge whether the
idea is good -- that is a later stage.

```python
{code}
```

Reply with exactly one of:
VERDICT: PASS
VERDICT: FAIL
followed by one line per issue found.
"""

REPAIR_PROMPT = """[ROLE: repair]
Fix this alpha factor. Preserve its intent exactly; change only what is broken.

REPORTED ISSUES
{issues}

```python
{code}
```

{contract}
"""

# ----------------------------------------------------------------------- judge

JUDGE_PROMPT = """[ROLE: judge]
Evaluate this alpha factor on three criteria.

1. Logically consistent -- correct operator ordering, coherent data flow, no
   degenerate expressions (e.g. a term that cancels to a constant).
2. Technically correct -- valid use of rolling windows, transforms and TA-Lib
   calls; windows long enough to be meaningful and short enough to leave data.
3. Economically meaningful -- rests on a stated market mechanism rather than an
   arbitrary combination of columns. A fabricated indicator with no rationale
   fails this criterion even if the code is flawless.

```python
{code}
```

Reply with exactly one of:
VERDICT: PASS
VERDICT: REVISE
followed by your reasoning, and if REVISE, the specific change required.
"""

IMPROVE_PROMPT = """[ROLE: improve]
Refine this alpha factor to address the assessment below.

Restructure the formula, adjust window parameters, replace dubious
transformations or remove redundant operations as needed -- but preserve the
original modelling intent. The goal is a factor with the same hypothesis and
better financial interpretability, not a different factor.

ASSESSMENT
{assessment}

```python
{code}
```

{contract}
"""

# -------------------------------------------------------------------- analysis

ANALYSE_PROMPT = """[ROLE: analyse]
Below are alpha factors from the last generation with their measured performance.
Explain concisely why the effective ones worked and the ineffective ones did not.

Metrics: IC and RankIC measure association with the forward return; ICIR and
RankICIR measure how stable that association is over time; MI captures non-linear
dependence. A factor can have a decent IC and still be useless if its ICIR is
near zero -- that means the relationship flips sign across periods.

EFFECTIVE
{valid_block}

INEFFECTIVE
{invalid_block}

Write at most six sentences. Be specific about construction choices -- what kind
of normalisation, what horizon, what transformation -- not generic advice. Your
answer is fed back into the next round of generation, so state what to do and what
to avoid.
"""

