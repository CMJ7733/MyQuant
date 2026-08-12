# Architecture — read this first

You are taking over ~9,600 lines implementing the CogAlpha paper. This document is
the map: how data flows, which contracts must not be broken, and where the traps
are. `README.md` covers *what the system does and what it measured*; this covers
*how it is put together and how to change it safely*.

---

## 1. The one object everything revolves around

`cogalpha/types.py` → `Alpha`. Every stage produces, filters, scores or archives
`Alpha` instances. Its fields, and who writes them:

| Field | Written by | Notes |
|---|---|---|
| `code` | generator / evolution operators | Python source, exactly one top-level `def` |
| `name` | parser, then `checker` | Must equal the function name in `code` |
| `rationale` | parser (from docstring) | The interpretability payload of §4.4 |
| `lineage` | generator / operators | Provenance: op, parents, agent, generation |
| `fitness` | `FitnessEvaluator` | `None` until scored; `None` means "never scored" |
| `tier` | `assign_tiers` | INVALID → PLAIN → QUALIFIED → ELITE |
| `checks` | `QualityChecker` | Append-only audit trail, one per stage |
| `rejected_at` / `reject_reason` | `QualityChecker` | Non-`None` means dead |

**`alpha_id` is the de-duplication key and it hashes `canonical_code(code)`, not
`code`.** Canonicalisation strips the function name, docstrings, comments and
whitespace. If you change `canonical_code`, you change what counts as "the same
alpha", which changes the parent pool, the candidate pool and the
`unique_structures` metric all at once. There is a comment there explaining why
raw-source hashing failed; read it before touching it.

Use `display_id` (`name-hash`) in logs and filenames, `alpha_id` for identity.

---

## 2. Data flow, end to end

```
                     configs/*.yaml  +  configs/llm.yaml (git-ignored)
                                    │
                          config.merge_configs()
                                    │
                            CogAlphaConfig
                                    │
      ┌─────────────────────────────┼──────────────────────────────┐
      │                             │                              │
 data.load_panel()          llm.build_client()          FitnessEvaluator
      │                             │                              │
    Panel  ──────────────────────── │ ─────────────────────────────┤
      │                             │                              │
      └──────────► CogAlphaSearch (evolution/loop.py) ◄────────────┘
                          │
        ┌─────────────────┼──────────────────┐
        │                 │                   │
  AlphaGenerator   ThinkingEvolution   AdaptiveGeneration
  (agents/)        (evolution/)        (evolution/)
        │                 │                   │
        └────────┬────────┘                   │
                 ▼                            │
          QualityChecker ──────────────────────┘
          (quality/checker.py)   feedback for next generation
                 │
                 ▼
          FitnessEvaluator  ──►  sandbox child process
          (fitness/evaluate.py)      │
                 │                   ├─ apply_alpha        (compute factor)
                 ▼                   ├─ check_numeric      (stability)
            assign_tiers             ├─ leakage probe      (causality)
          (fitness/thresholds.py)    └─ ic_series / MI     (metrics)
                 │
        ┌────────┴────────┐
        ▼                 ▼
   parent pool      CandidatePool  ──►  RunArchive  ──►  runs/<ts>-<name>/
   (next gen)       (the output)                              │
                                                        compose.py (Table 1)
```

### The one call that crosses a process boundary

`FitnessEvaluator.evaluate()` → `SandboxRunner.run()` → forked child →
`evaluation_job()`. Everything about the sandbox exists to make that boundary safe
and cheap:

- The **panel does not cross it**. `fork` gives the child copy-on-write access.
  Only a small dict of scalars comes back. A factor matrix for 300 names × 3400 days
  is 8 MB; returning raw values for a 96-child generation would move ~800 MB.
- **One child per batch, results streamed per alpha.** A hang is therefore
  attributable to a specific alpha, which the parent kills and then restarts the
  worker on the remainder.
- `evaluation_job` does metrics **and** stability **and** the leakage probe in one
  visit, because the probe already needs a second factor computation and splitting
  the stages would triple the dominant cost of a generation.

If you add a per-alpha measurement, add it inside `evaluation_job` and return it in
the payload dict. Do not add a second sandbox pass.

---

## 3. Contracts you must not break

### 3.1 The alpha function contract

Stated in `prompts/templates.py::ALPHA_CONTRACT`, enforced by
`quality/audit.py::audit_code`, relied on by `quality/sandbox.py::apply_alpha`.

An alpha is `def factor_x(df) -> Series`, where `df` is **one instrument's** daily
history, date-indexed ascending, with columns `open/high/low/close/volume` plus
`day_*` aliases. Per-instrument (not panel-wide) is deliberate: the paper's listings
call `talib.EMA(df['day_close'])`, which only makes sense on a single series, and it
keeps all cross-sectional work inside the evaluator.

Change the contract and you must change all four of: the prompt text, the audit, the
sandbox's frame construction, and `Panel.instrument_frame`.

### 3.2 The label convention

`Panel.label(horizon)` = `open[t+1+h] / open[t+1] - 1`. Both legs strictly after *t*
(§4.1 buys at the open). The naive `close[t+h]/close[t]` is already tradable at *t*'s
close and is what most leaky reimplementations use.

**The backtest payoff is a different series**: a *one-day* return, because §B.2
rebalances daily. Using the h-day label there counts every return h times — measured
difference was IR +3.18 vs +0.86 on the same alpha. `FitnessEvaluator.__init__`
builds both; don't unify them.

### 3.3 Coverage is measured against the universe, never the rectangle

`Panel.universe_mask()` is the denominator for every coverage/NaN ratio, in both
`quality/numeric.py` and `compose.py::build_feature_matrix`. CSI300 over 2011–2024 is
748 tickers of which ~300 trade daily, so the wide layout is ~60% empty before an
alpha runs. Measuring against the rectangle rejects every alpha ever written — this
bug was hit twice, in two different modules.

### 3.4 Selection never sees the test split

`cli.py::cmd_search` slices the panel at the end of the fitness window before
handing it to the evaluator. That is structural, not conventional: an alpha cannot
read the test period even by accident. `data.fit_split` defaults to `train`, and the
preflight explains why `valid` is too short.

---

## 4. Module reference: what to read, in what order

Read in this order to build a mental model:

1. **`types.py`** — the vocabulary. 15 minutes, pays for itself.
2. **`config.py`** — every knob, each default citing its paper section. Skim the
   docstrings; they carry the measurements that justify non-obvious defaults.
3. **`data/panel.py`** — `Panel`, the label, `universe_mask`.
4. **`quality/checker.py`** — the 8-stage pipeline; the clearest single view of
   how a generation is processed.
5. **`fitness/evaluate.py`** — where the sandbox and the metrics meet.
6. **`evolution/loop.py`** — the schedule (24 gens × 3 sub-cycles × 13 agents).
7. Everything else on demand.

| Package | Responsibility | Entry points |
|---|---|---|
| `data/` | Panel, label, providers | `load_panel`, `Panel.label`, `Panel.universe_mask` |
| `llm/` | Backends + transcript | `build_client`, `MockLLMClient`, `CallRecorder` |
| `agents/` | 21 agents, 5 guidance modes, parsing | `select_agents`, `AlphaGenerator.generate`, `parse_alphas` |
| `prompts/` | All prompt text | `GENERATE_PROMPT`, `ALPHA_CONTRACT`, … |
| `quality/` | Audit → sandbox → numeric → leakage | `QualityChecker.check`, `SandboxRunner.run` |
| `fitness/` | Metrics, tiers, backtest | `FitnessEvaluator.evaluate`, `assign_tiers`, `run_backtest` |
| `evolution/` | Operators, feedback, pools, loop | `CogAlphaSearch.run`, `ThinkingEvolution.breed` |
| `compose.py` | 20 factors → one prediction | `compose`, `build_feature_matrix` |
| `archive.py` | Run directory I/O | `RunArchive.save_run`, `RunArchive.load` |
| `monitor/` | Live dashboard, reads the archive only | `RunReader.poll`, `server.serve` |
| `cli.py` | Commands + preflight | `main`, `preflight` |

---

## 5. Common tasks

**Add a task-specific agent.** Append an `AgentSpec` to `HIERARCHY` in
`agents/hierarchy.py`. The `assert len(HIERARCHY) == 21` will fire — update it and
say why in the commit. `focus` and `probe` go straight into the prompt.

**Add a fitness metric.** Compute it in `fitness/metrics.py`; add the field to
`Fitness` in `types.py`; if it should gate selection, add it to `Fitness.TIER_METRICS`
and to both bound dicts in `FitnessConfig`. Metrics not in `TIER_METRICS` are
reported but do not select.

**Change a threshold.** `FitnessConfig` only. Do not hard-code it. Run
`scripts/calibrate_real.py` afterwards to see where the new value sits in the real
distribution — a floor above the p80 of genuine alphas produces an empty pool.

**Add an evolution operator.** Add to `OPERATORS` and `_OP_TO_ENUM` in
`evolution/operators.py`, a branch in `ThinkingEvolution._apply`, a member in
`EvolutionOp` (`types.py`), and a weight in `EvolutionConfig.op_weights`.

**Swap the LLM backend.** Subclass `LLMClient` (`llm/base.py`), implement
`_complete`, register it in `llm/factory.py::build_client`. Recording and budget
enforcement live in the base class — you get both for free.

**Support another market.** Add a provider in `data/registry.py` returning a
`Panel`. The rest of the system knows nothing about markets.

---

## 6. Traps

These are all bugs that were hit and fixed. They will bite again if the surrounding
code is rewritten carelessly.

1. **`RLIMIT_AS` cannot bound this process.** numpy+pandas+qlib reserve ~5 GB of
   *virtual* space on a 128-core host (glibc opens up to 8×ncores 64 MB arenas) while
   RSS is ~450 MB. A 4 GB cap killed 16/16 alphas with `exited with code -15`. Memory
   is enforced by the parent's RSS watchdog. See `sandbox.py::_install_limits`.
2. **The leakage probe must truncate by calendar date, not by row count.** Delisted
   names end at different dates; dropping the last N rows per instrument deletes
   old data for old names and reports "the value changed". That false-positived 14 of
   16 clean alphas. See `fitness/evaluate.py::_run_probe`.
3. **MI in nats is not the paper's MI.** Genuine alphas measure 0.0006–0.0223 nats;
   the paper's 0.02 floor would reject 22 of 23. Default scale is
   `corr_equivalent`. See `metrics.py::mutual_information`.
4. **LightGBM at the paper's `lr=0.0001` is deliberately under-fitted** (total
   shrinkage 0.1 over 1000 trees) and scores below Ridge. Kept as the default for
   fidelity; `--lgbm-lr` overrides. See `compose.py::make_model`.
5. **The rolling fit needs an embargo of `horizon + 1` days.** Without it the
   composition trains on returns that had not happened at the cutoff, and every
   number inflates invisibly. See `compose.py::rolling_predict`.
6. **The mock's template bank is finite (~70 structures).** A long mock run
   exhausts it and reports mostly `duplicates_dropped`. That is the mock running out
   of ideas, not the search failing.
7. **`assign_tiers` puts elites in `qualified` too.** Elites are by construction
   qualified and belong in the next parent pool as well as the candidate pool.
   Filtering `tier == QUALIFIED` exactly will silently drop them.
8. **A leaky factor makes every composition metric excellent.** One leaky column
   drove Ridge to RankIC 0.98. `cli.py::_implausible` warns above |RankIC| 0.20.
   Keep that warning; it guards the one failure direction nobody questions.

---

## 7. Debugging a run

Everything needed is in the archive; no re-running required.

```bash
cogalpha report --run runs/<dir>      # stage-by-stage deaths, operators, LLM cost
cogalpha monitor --run runs/<dir>     # the same, live and clickable, in a browser
```

The monitor is the faster path when a run is still going: it surfaces the funnel and
the per-agent yield without you having to parse anything. It reads the archive only —
`cogalpha/monitor/` imports nothing from the search — so attaching to a running job
is free and cannot perturb it. Note the two coupling points it does have, which will
break it silently if changed: it depends on the **field names in
`GenerationRecord.to_dict()`** and on **`tags.role` / `tags.agent`** being present on
every LLM call. If you add a call site, tag it.

Then, by symptom:

| Symptom | Look at |
|---|---|
| No candidates | `report`'s stage table. One stage >35% is systematic, not attrition |
| Everything dies at `leakage_unit_test` | The probe message names the % of changed cells. If `max abs diff 0` and cells merely appeared/vanished, suspect a panel-shape bug, not real leakage |
| Everything dies at `numeric_stability` | `alphas.jsonl` → `checks[].payload` has `nan_ratio`, `coverage`, `mean_distinct_per_day` |
| Everything dies at `judge` | `llm_calls.jsonl`, filter `tags.role == "judge"`; the model may be refusing a valid pattern |
| `duplicates_dropped` dominates | Generator has run out of ideas — real LLM: raise temperature or vary guidance; mock: expected |
| Metrics implausibly good | A leaky factor. `cogalpha evaluate` each candidate and read its leakage verdict |
| Run stops early | `summary.json` → `stopped_early` gives the agent and the reason (plateau vs budget) |

`llm_calls.jsonl` is JSONL with `tags.role` on every entry, so
`jq 'select(.tags.role=="mutate")'` gets you the full mutation transcript.

---

## 8. What is not done

- **No run has ever used a real LLM.** The apparatus is verified; the paper's claims
  about prompt efficacy are not.
- No ablation (§4.3) or threshold sweep (§4.7). `generations.jsonl` records what
  they need.
- No test suite. `configs/synthetic.yaml` is the de-facto smoke test:
  `cogalpha search --config configs/synthetic.yaml` should finish in ~20 s with a
  non-empty candidate pool.
- CSI300 only.
