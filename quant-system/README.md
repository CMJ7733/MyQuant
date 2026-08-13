# CogAlpha

An implementation of **Cognitive Alpha Mining via LLM-Driven Code-Based Evolution**
(Liu et al., ACL 2026, pp. 11715–11749). The paper's own source is not public; this
is a from-scratch implementation of the method it describes, plus the measurement
apparatus needed to tell whether it is working.

Alphas are **Python functions**, not formula strings. Twenty-one task-specific
agents propose them, a multi-agent quality checker filters them, five metrics score
them, and LLM-driven mutation and crossover evolve the survivors. This implementation
is organized in the repository's `cogalpha/` package.

```
Raw OHLCV → 7-Level Hierarchy → Quality Checker → 5-Metric Evaluation
                 ↑                                        ↓
            Candidate Pool ← 5-M Eval ← Checker ← Thinking Evolution
```

## Setup

Activate the `quant` environment and work from the package directory:

```bash
conda activate quant
cd /path/to/quant/F4Q/quant-system
python -m pip install -e '.[monitor,dev]'
```

Every example below assumes those three lines. This install provides the declared core
dependencies (NumPy, pandas, SciPy, and PyYAML), the monitor dependencies (FastAPI and
Uvicorn), and the test dependencies (pytest and httpx). Install the `llm`, `model`,
`qlib`, or `ta` extras separately when the corresponding command needs them; they are
not part of the monitor setup above.

If `conda activate` is unavailable in a non-interactive shell, use
`conda run -n quant python -m cogalpha.cli ...` instead. On the current macOS setup,
that command resolves to `/opt/homebrew/Caskroom/miniforge/base/envs/quant/bin/python`.

The editable install also provides the shorter `cogalpha` entry point:

```bash
cogalpha inspect hierarchy          # equivalent to python -m cogalpha.cli inspect hierarchy
```

The examples keep the explicit module form so they do not depend on the console entry
point.

To install through the Tsinghua PyPI mirror explicitly:

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>
```

## Quick start

Four commands, in the order they are worth running. None of them needs an LLM
credential.

```bash
# 1. offline smoke test: synthetic panel, deterministic mock backend, ~20 s
python -m cogalpha.cli search --config configs/synthetic.yaml --out runs/

# 2. watch it (or any run) in a browser
python -m cogalpha.cli monitor --run runs/ --port 8080

# 3. real CSI300 data, no LLM: score the 23 hand-written baselines
python -m cogalpha.cli evaluate 'seeds/L[1-7]_*.py' \
    --config configs/paper_csi300.yaml --split test

# 4. the reading Table 1 reports: 20 factors → one prediction
python -m cogalpha.cli compose 'seeds/L[1-7]_*.py' \
    --config configs/paper_csi300.yaml --model ridge --model lightgbm --split test
```

## Running a real search

`search` needs an LLM. Put the endpoint in a **git-ignored** file:

```bash
cp configs/llm.yaml.example configs/llm.yaml
$EDITOR configs/llm.yaml          # api_base, model, api_key
```

Then start small and grow. Each step is a real command with a real cost:

```bash
# a. bounded trial: 3 agents x 8 generations, hard ceiling of 4000 LLM calls
python -m cogalpha.cli search --config configs/quick.yaml --llm-config configs/llm.yaml

# b. watch it from another shell
python -m cogalpha.cli monitor --run runs/ --port 8080

# c. the paper's full specification: 13 agents x 24 generations x 3 sub-cycles
#    ~30k sandbox evaluations (~15 CPU-hours) and ~100k LLM calls / ~230M tokens
python -m cogalpha.cli search --config configs/paper_csi300.yaml --llm-config configs/llm.yaml

# d. combine the run's 20 candidates and compare against Table 1
python -m cogalpha.cli compose --run runs/<timestamp>-csi300 --model lightgbm --split test

# e. what happened, offline
python -m cogalpha.cli report --run runs/<timestamp>-csi300
```

Credentials can also come from the environment instead of a file, which keeps them
off disk entirely:

```bash
export COGALPHA_API_BASE=https://qianfan.baidubce.com/v2
export COGALPHA_API_KEY=...
python -m cogalpha.cli search --config configs/quick.yaml --llm-provider openai
```

## Command reference

Six commands:

| Command | What it does |
|---|---|
| `search` | Run the working stream and archive every generation |
| `monitor` | Serve a live browser dashboard for a run (or replay a finished one) |
| `evaluate` | Score standalone alpha `.py` files on any split |
| `compose` | Combine candidates into one prediction, scored like Table 1 |
| `report` | Derive diagnostics from an archive — offline, no model calls |
| `inspect` | Print the hierarchy, guidance modes, resolved config, or data checks |

`search`, `evaluate`, `compose`, and `inspect` take `--config YAML` and repeatable
`--set SECTION.KEY=VALUE` overrides (parsed as YAML so types survive the shell).
`search` also takes `--llm-config`, `--llm-provider`, `--llm-model`,
`--llm-api-base`, and `--max-llm-calls`.

### search

```bash
# offline, deterministic
python -m cogalpha.cli search --config configs/synthetic.yaml --out runs/

# real endpoint, named run, quiet
python -m cogalpha.cli search --config configs/quick.yaml --llm-config configs/llm.yaml \
    --out runs/ --name trial-A --quiet

# override the schedule without editing a config
python -m cogalpha.cli search --config configs/paper_csi300.yaml --llm-config configs/llm.yaml \
    --set evolution.agents_per_run=5 --set evolution.generations=12 \
    --set evolution.max_llm_calls=20000

# measure fitness on a different window (preflight will object to short ones)
python -m cogalpha.cli search --config configs/paper_csi300.yaml --split train
```

`search` runs a **preflight** and refuses to start on a window that cannot support
selection, reporting which check failed. That is not defensive padding: on this data
the 243-day validation window lets a raw-price control (a pure size proxy carrying no
information) score RankIC 0.0825, above every genuine alpha tested, so selecting there
would promote noise for 24 generations. `--no-preflight` overrides — read what the
checks said first.

### monitor

```bash
python -m cogalpha.cli monitor --run runs/                    # newest run in runs/
python -m cogalpha.cli monitor --run runs/20260810-123307-csi300
python -m cogalpha.cli monitor --run runs/ --port 9000 --interval 0.5 --open
```

The monitor is a Material 3-style Chinese dashboard. All 21 agents in the
seven-level hierarchy remain clickable, including agents that were not selected for
the current run. Each agent opens a right-side detail panel with overview, recent
activity, and generation history; generation, Alpha, and LLM-call rows can be opened
again for their archived details. Live runs update through SSE, with automatic
polling fallback when a tunnel or proxy cannot carry the event stream.

Reads the archive only, so it is safe against a running search and replays a finished
one identically. From a laptop, tunnel rather than exposing it:

```bash
ssh -N -L 8080:127.0.0.1:8080 user@host      # then open http://127.0.0.1:8080
```

### evaluate

Scores standalone `.py` files. Re-scoring a candidate on a later split is how alpha
decay becomes visible.

```bash
# one file
python -m cogalpha.cli evaluate seeds/L1_impact_high_close.py \
    --config configs/paper_csi300.yaml --split test

# a run's candidates on the holdout, as JSON
python -m cogalpha.cli evaluate 'runs/<dir>/candidates/*.py' \
    --config configs/paper_csi300.yaml --split valid --json /tmp/valid.json

# the same candidates on test -- compare the two to see decay
python -m cogalpha.cli evaluate 'runs/<dir>/candidates/*.py' \
    --config configs/paper_csi300.yaml --split test

# skip the backtest when you only want the five metrics
python -m cogalpha.cli evaluate 'seeds/*.py' --config configs/paper_csi300.yaml --no-backtest
```

Quote the globs. An unquoted pattern that matches nothing in the current directory
reaches the command literally, which is handled, but quoting is what you mean.

### compose

Table 1 scores multi-factor combinations of 20 alphas, not single alphas.

```bash
# a run's candidates
python -m cogalpha.cli compose --run runs/<dir> --model lightgbm --split test

# several models side by side, plus JSON
python -m cogalpha.cli compose --run runs/<dir> \
    --model mean --model ridge --model lightgbm --split test --json /tmp/compose.json

# hand-written files instead of a run
python -m cogalpha.cli compose 'seeds/L[1-7]_*.py' --config configs/paper_csi300.yaml \
    --model ridge --split valid --top-n 20

# a LightGBM learning rate that actually converges (the paper's 1e-4 under-fits)
python -m cogalpha.cli compose --run runs/<dir> --model lightgbm --lgbm-lr 0.05
```

`--model mean` is the equal-weight control: if it scores as well as the model, the
model is not adding anything.

### report

```bash
python -m cogalpha.cli report --run runs/<dir>
```

Stage-by-stage attrition, per-operator yield, elite trajectory, LLM cost by role.
Entirely offline — no model calls, no re-running.

### inspect

```bash
python -m cogalpha.cli inspect hierarchy              # 21 agents, 7 levels
python -m cogalpha.cli inspect hierarchy --n 13 --seed 42   # which ones a run picks
python -m cogalpha.cli inspect guidance               # the 5 paraphrasing modes
python -m cogalpha.cli inspect config --config configs/paper_csi300.yaml   # resolved, key redacted
python -m cogalpha.cli inspect data --config configs/paper_csi300.yaml     # preflight all 3 splits
```

`inspect data` is the cheapest way to check a data window before committing hours to
it. `inspect config` never prints the API key.

Exit codes: `0` success, `1` no command, `2` configuration/data/credential problem
(including a failed preflight), `130` interrupted.

## Recalibrating on your own data

If you change market, horizon or universe, the thresholds need re-checking — they were
derived from measurement, not taste:

```bash
python scripts/calibrate_real.py 10      # horizon in trading days
```

Scores 23 hand-written alphas plus a noise and a leakage control across all three
splits, and prints the percentile distribution of each metric. A qualified floor above
the p80 of genuine alphas gives an empty pool; a floor below the noise controls admits
anything.

## Paper → module map


| Paper | Module |
|---|---|
| Seven-Level Agent Hierarchy, 21 agents (§3.1) | `cogalpha/agents/hierarchy.py` |
| Diversified Guidance, 5 modes (§3.2) | `cogalpha/agents/guidance.py` |
| Multi-Agent Quality Checker (§3.3, A.3) | `cogalpha/quality/checker.py` |
| Static audit + sandbox | `cogalpha/quality/audit.py`, `sandbox.py` |
| Numerical stability, NaN/distinct checks (A.3) | `cogalpha/quality/numeric.py` |
| Temporal Leakage Unit Test (A.3) | `cogalpha/quality/leakage.py` |
| Five metrics: IC/ICIR/RankIC/RankICIR/MI (§3.4, B.3) | `cogalpha/fitness/metrics.py` |
| Percentile tiers + floors (§3.4, A.4) | `cogalpha/fitness/thresholds.py` |
| top-50/drop-5 backtest, AER/IR (B.2) | `cogalpha/fitness/backtest.py` |
| Adaptive Generation (§3.5) | `cogalpha/evolution/adaptive.py` |
| Thinking Evolution, 3 operators (§3.6) | `cogalpha/evolution/operators.py` |
| Schedule: 13 agents × 24 gens × 3 sub-cycles (B.4, B.8) | `cogalpha/evolution/loop.py` |
| Multi-factor combination — the Table 1 reading (§4.2) | `cogalpha/compose.py` |
| Run archive and report | `cogalpha/archive.py` |

## What the dashboard shows

The 21-agent matrix by hierarchy level (running / done / queued, plus agents that did
not participate in the current run), the quality checker funnel for the
current generation stage by stage, the elite-score trajectory with agent boundaries
marked, LLM cost broken down by role, and how close the running agent is to the §B.4
plateau stop. Clicking drills all the way down: agent → generation → alpha (code, all
eight check verdicts, lineage) → the individual LLM call, prompt and response verbatim.

Three automatic warnings fire when a run is producing nothing useful: one checker stage
swallowing more than 35% of output (systematic failure, not attrition), duplicates
outnumbering new alphas 3:1 (the generator is repeating itself), and no elite at all
after eight generations.

Two things to know:

- **Agents run sequentially**, so exactly one of the 21 is active at any instant.
  That is the paper's schedule (§B.4 says nothing about parallelism), not a defect.
  A generation on real data takes 30–60 s, so the display advances at that pace.
- **It binds to `127.0.0.1` by default.** `llm_calls.jsonl` holds every prompt the
  system sends, and the detail endpoints serve it verbatim; `--host 0.0.0.0` publishes
  that and prints a warning. Tunnel instead (see the `monitor` examples above).

## Where the paper is underspecified, and what we did


The paper omits several things that determine whether the numbers mean anything.
Each choice below was made from measurement on real CSI300 data, not from taste.

**Leakage detection (A.3 states the stage exists; the rules are not published).**
Two layers. A static AST scan for `shift(-k)`, `bfill`, reversed slices, forward
positional indexing. Then a **truncation probe**: recompute the factor on a panel
cut at date *T* and compare against the full-panel run restricted to *t ≤ T*. A
causal factor is bit-identical; any difference proves information flowed backwards.
Verified on 9 controls: 4/4 leaky caught, 5/5 clean passed. The whole-sample z-score
case is invisible to the AST scan (0 findings) and caught only by the probe (30.86%
of historical values changed) — which is why both layers exist.

**MI scale.** The paper sets an MI floor of 0.02 (A.4) without naming its estimator.
Measured on real data, genuine alphas span 0.0006–0.0223 **nats** (median 0.0029),
so a 0.02-nat floor rejects 22 of 23. We map MI through the Gaussian relation to
`sqrt(1 - exp(-2·MI))`, putting it on the same [0,1) axis as |IC|; the same alphas
then span 0.033–0.209 and the floor behaves as a floor. The estimator (quantile
histogram + Miller–Madow correction) was validated against the closed form
`-0.5·ln(1-ρ²)`: at RankIC 0.357, theory 0.06814 vs measured 0.06825.

**Coverage denominator.** CSI300 membership over 2011–2024 spans 748 tickers of
which ~300 trade on any day, so the wide `(date × instrument)` layout is ~60% empty
before an alpha runs. Measuring the §B.4 30%-NaN limit against that rectangle
rejects **every** alpha ever written. Coverage is therefore measured against
`Panel.universe_mask()`. The seed alpha reads 0.0% missing instead of 60%.

**Which split fitness uses.** `train` (2189 days), not `valid` (243). See the
preflight note above; the paper's ICIR floors of 0.05/0.1 also sit at the p50/p80 of
the train-window distribution and would be unreachable on 243 days.

**Backtest payoff.** §B.2 rebalances daily, so the per-day P&L is a **one-day**
return. Feeding it the 10-day label counts each return ten times: the same alpha
reads AER +1.23 / IR +3.18 with the 10-day label versus AER +0.11 / IR +0.86 with
the daily return — the first being arithmetically impossible at RankIC 0.08.

**Sandbox limits.** `setrlimit(RLIMIT_AS)` cannot be used: importing numpy, pandas
and qlib on a 128-core host reserves ~5 GB of *virtual* address space (glibc opens
up to 8×ncores 64 MB arenas) while resident memory is ~450 MB. A 4 GB AS cap killed
16 of 16 alphas with `exited with code -15`. Memory is enforced by a parent-side RSS
watchdog polling `/proc/<pid>/statm`; the child still gets `RLIMIT_CPU`,
`RLIMIT_FSIZE=0`, `RLIMIT_NPROC`, a filtered `__import__`, and severed sockets.

**Rolling-fit embargo (not discussed in the paper).** A 10-day label formed on day
*t* is only observable at *t+11*. A model retrained at cutoff *T* is therefore
trained on labels up to *T−11*. Without this the composition learns from returns
that had not happened yet, and every downstream number is inflated invisibly.

**Structural de-duplication.** `alpha_id` hashes the *canonical* source — function
name, docstrings, comments and whitespace normalised away. Hashing raw source made
de-duplication useless: generated names carry a random suffix, so one real run
admitted three byte-identical-logic candidates with identical metrics and reported
102 of 102 alphas as structurally unique.

**Sub-cycle semantics.** A sub-cycle boundary resets the parent pool but not the
candidate pool, and re-seeds with a full `initial_pool_size` batch. "Initiates the
search 3 times" (B.4) only means something if a restart starts from fresh
task-generated alphas.

## Measured results

### Composition, CSI300 test 2021-01 – 2024-12, 20 hand-written alphas

These use **hand-written** factors, not LLM-generated ones. The purpose is to
validate the composition pipeline against the paper's scale before attributing
anything to the LLM.

| Model | IC | RankIC | ICIR | RankICIR | AER | IR |
|---|---|---|---|---|---|---|
| equal-weight mean | −0.0021 | +0.0061 | −0.013 | +0.037 | −0.068 | −0.833 |
| Ridge (α=10, §B.4) | **+0.0295** | **+0.0327** | +0.190 | +0.201 | +0.014 | +0.153 |
| LightGBM (lr=1e-4, §B.4) | +0.0117 | +0.0188 | +0.086 | +0.114 | −0.011 | −0.116 |
| LightGBM (lr=0.05) | +0.0194 | +0.0251 | +0.187 | +0.232 | +0.044 | +0.722 |

Paper Table 1, same universe and horizon: CogAlpha IC +0.0591 / RankIC +0.0814;
Alpha158 +0.0358 / +0.0402; LightGBM +0.0269 / +0.0412; Linear +0.0165 / +0.0211.

Reading these honestly:

- **Ridge on 20 hand-written factors lands between the paper's Linear and LightGBM
  rows.** That is the right neighbourhood, and it is what validates the pipeline.
- **The equal-weight control is worthless (RankIC 0.006).** So the model is doing the
  work, not the averaging — a check Table 1 does not report.
- **LightGBM at the paper's lr=0.0001 scores *below* Ridge**, reversing the paper's
  ordering. `0.0001 × 1000 trees` is a total shrinkage of 0.1, so the ensemble
  travels a tenth of the way from its initial constant to the signal; at lr=0.05 it
  overtakes Ridge on RankICIR and IR. We keep the paper's value as the default and
  expose `--lgbm-lr`.
- Not a reproduction of the CogAlpha row: that requires LLM-generated alphas, and
  the paper trains on a different qlib snapshot.

### Single factors, same window

A hand-written low-volatility factor scores AER +0.1071 / IR +0.858 on the test
split — inside the band of the paper's *baselines* (LightGBM 0.0878/1.0980,
Alpha158 0.0946/0.8556), which is the check that the backtest is calibrated.

The paper's own Listing 1/2/3 liquidity lineage scores RankIC 0.0182 / 0.0153 /
0.0164 on train+valid (paper: 0.0061 / 0.0021 / 0.0087) — same order, different
ranking, different data snapshot. All three go **negative** on 2021–2024
(−0.037 / −0.041 / −0.029), which is alpha decay stated plainly and the reason
`compose` selects on a holdout.

### Cost, extrapolated from measurement

Sandbox: 1.86 s per alpha (fork overhead ~1.2 s amortised per batch). A full
paper-spec run is 13 agents × 24 generations × 96 children ≈ 30k evaluations
≈ 15 CPU-hours, ~1 h at 16 workers. LLM: ~100k calls, ~230M tokens. Set
`evolution.max_llm_calls` on a metered endpoint.

## Data

qlib binary snapshot at `~/.qlib/qlib_data/cn_data_2026` — `chenditc/investment_data`
release 2026-08-08, calendar 2000-01-04 … 2026-08-07 (6445 trading days), CSI
300/500/1000 membership, OHLCV + adjustment factor. Prices are backward-adjusted;
no second adjustment is applied. Splits follow §B.1 exactly: train 2011–2019,
valid 2020, test 2021-01 … 2024-12.

The older `~/.qlib/qlib_data/cn_data` is a **frozen** snapshot backing another
project's reproducibility claims and is never touched.

Providers: `qlib`, `synthetic` (plants the paper's seed alpha at a calibrated
strength so a known answer is recoverable), `noise` (negative control), `csv`.

## Layout

```
cogalpha/
  types.py         Alpha, Fitness, Lineage, CheckReport, canonical_code
  config.py        six config sections; every default cites its paper section
  cli.py           search / compose / evaluate / report / inspect + preflight
  compose.py       multi-factor combination (the Table 1 reading)
  archive.py       run directory: config, generations, alphas, transcript, report
  data/            panel + label construction, providers
  llm/             OpenAI-compatible client, deterministic mock, call recorder
  agents/          21 agents, 5 guidance modes, generation, output parsing
  prompts/         every prompt, tagged [ROLE: x] for transcript filtering
  quality/         audit → sandbox → numeric → leakage → checker pipeline
  fitness/         metrics, tiers, backtest, sandboxed evaluation
  monitor/         live dashboard: reader (tails JSONL) + server (SSE) + one HTML file
configs/           synthetic (offline) / quick (bounded real) / paper_csi300
seeds/             23 hand-written baselines + 2 noise + 1 leaky control
scripts/           calibrate_real.py — the threshold calibration on real data
```

An archived run holds `config.json` (secrets redacted), `panel.json` (provenance
only, never data), `generations.jsonl`, `alphas.jsonl` (**including rejects and why**),
`llm_calls.jsonl`, `candidates/*.py` with lineage headers, and `report.md`.
Rejected alphas are kept deliberately: a run that loses 80% of its output at one
stage has a prompt problem, and that is invisible if only survivors are recorded.

## Status

Implemented and exercised on real data: the full method (§3.1–§3.6), the checker,
the metrics, tiering, the backtest, composition, archiving, and the CLI.

Not yet done:

- **Never run against a real LLM.** Every run so far used the deterministic mock, so
  the paper's central claims — that the hierarchy's prompts elicit useful factors,
  that the checker's pass rate is workable, that the evolution operators add value —
  are **unverified**. What is verified is that the apparatus around them does not
  lie.
- No ablation (§4.3) or threshold-sensitivity (§4.7) sweep. `generations.jsonl`
  records what these need; the sweeps have not been run.
- CSI300 only. The snapshot is A-share; CSI500 is available, S&P500/HSI/HSCI need
  another source.
- No test suite.

## Caveats

Research code. The paper's own disclaimer applies: nothing here is investment
advice, backtests omit real trading frictions, and an automatically generated factor
needs a human to check its economics before it is trusted. LLM sampling means an
exact re-run is impossible by construction — the transcript in `llm_calls.jsonl` is
what makes a run auditable instead.
