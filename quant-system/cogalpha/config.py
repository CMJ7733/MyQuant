"""Configuration objects, with defaults taken verbatim from the paper.

Every default here is traceable to a section of the paper or its appendix; the
docstrings name the source so a reader can check a number without re-reading the
PDF.  Overrides come from a YAML file via :func:`load_config`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

#: Values that look like an unedited template. Treated as "unset" so that copying
#: llm.yaml.example without editing fails with a useful message.
_PLACEHOLDERS = frozenset(
    {
        "",
        "xxx",
        "xxxx",
        "put-your-key-here",
        "your-key-here",
        "changeme",
        "todo",
        "none",
        "null",
        "<unset>",
    }
)


def _is_placeholder(value: Optional[str]) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in _PLACEHOLDERS



@dataclass
class DataConfig:
    """Dataset and label construction.

    Defaults describe the paper's main experiment: CSI300, 10-day forward
    return, chronological train/valid/test split of 2011-2019 / 2020 /
    2021-2024 (§4.1, Appendix B.1).
    """

    provider: str = "qlib"
    """``qlib`` | ``synthetic`` | ``noise`` | ``csv``."""

    market: str = "csi300"
    horizon: int = 10
    """Prediction horizon in trading days (10 by default; the paper also uses 30)."""

    train: Tuple[str, str] = ("2011-01-01", "2019-12-31")
    valid: Tuple[str, str] = ("2020-01-01", "2020-12-31")
    test: Tuple[str, str] = ("2021-01-01", "2024-12-01")

    fit_split: str = "train"
    """Which split the search computes fitness on.

    ``train`` (2011-2019, 2189 trading days) rather than ``valid`` (2020, 243
    days), because the short window cannot support selection.  Measured on real
    CSI300 data with 23 hand-written alphas plus two noise controls:

    ==========================  =====  =======  ==================  ===============
    window                          T  SE(IC)  real |RankIC| p50   noise |RankIC|
    ==========================  =====  =======  ==================  ===============
    train 2011-2019              2189  0.0030              0.0086          0.0019
    valid 2020                    243  0.0099              0.0129          0.0825
    train+valid 2011-2020        2432  0.0029              0.0093          0.0099
    ==========================  =====  =======  ==================  ===============

    On the 2020 window the raw-price-level control scores RankIC 0.0825 -- above
    every real alpha -- because 243 days is not enough to separate a size proxy
    from a signal.  Selecting on it would promote noise for 24 generations.  The
    paper's own metric floors corroborate the long window: its ICIR floors of
    0.05/0.1 sit at the p50/p80 of the train-window distribution, and would be
    unreachable on a 243-day window.

    ``valid`` is then a genuine holdout used once, by ``cogalpha compose``, to
    choose among finished candidates; ``test`` is touched only for the final report.
    """

    qlib_provider_uri: str = "~/.qlib/qlib_data/cn_data_2026"
    """Local qlib bin directory.

    Points at the 2026-08-08 ``chenditc/investment_data`` snapshot (calendar
    2000-01-04 .. 2026-08-07), which covers the paper's 2021-2024 test window.
    Deliberately *not* ``cn_data``: that directory is the frozen snapshot the
    EvoQuant paper's reproducibility claims rest on and must stay untouched.
    """

    qlib_region: str = "cn"
    csv_path: Optional[str] = None


    # Synthetic provider knobs (offline development and unit tests).
    synth_n_instruments: int = 60
    synth_n_days: int = 700
    synth_seed: int = 7
    synth_signal_strength: float = 0.0012
    """Planted-signal amplitude, so a known-good alpha is recoverable in tests.

    Calibrated on a 300-name, 500-day panel: at 0.0012 the paper's seed alpha
    ``(high - close) / volume`` scores RankIC 0.076 / MI 0.090, the same order of
    magnitude as the paper's headline 0.0814, so percentile gates and metric floors
    behave the way they would on real data.  Pure noise (0.0) scores RankIC -0.004.
    """

    price_columns: Tuple[str, ...] = ("open", "high", "low", "close", "volume")
    """The OHLCV bandwidth the paper restricts itself to (§3, and
    ``reflection.md`` fragile assumption #2 notes this is a self-imposed limit)."""


@dataclass
class LLMConfig:
    """LLM backend, shaped after famou-v2's ``LLMConfig`` (api_base/api_key/model).

    The paper runs every agent on ``gpt-oss-120b`` served locally, sampling the
    temperature of task/evolution agents from {0.7 ... 1.2} while pinning quality
    checker agents at 0.8 and max tokens at 4096 (§4.1).
    """

    provider: str = "openai"
    """``openai`` (any OpenAI-compatible endpoint: Qianfan, vLLM, ...) | ``mock``.

    ``mock`` is a deterministic offline backend used by the test suite and by
    ``--llm-provider mock`` smoke runs; it never reaches the network.
    """

    model: str = "gpt-oss-120b"
    api_base: Optional[str] = None
    """Full base URL, e.g. ``https://qianfan.baidubce.com/v2`` or
    ``http://127.0.0.1:8000/v1``.  Falls back to ``$COGALPHA_API_BASE``."""

    api_key: Optional[str] = None
    """Prefer leaving this unset and using ``key_set``/env so secrets stay out of
    version control."""

    api_key_env: str = "COGALPHA_API_KEY"
    api_base_env: str = "COGALPHA_API_BASE"

    key_set: Optional[str] = None
    """Path to a famou-style ``key_set.yaml``; the entry named by ``key_set_name``
    supplies ``api_base``/``api_key``/``model_names``.

    Prefer ``configs/llm.yaml`` (see ``configs/llm.yaml.example``) for new setups;
    this exists so an existing famou key file can be reused as-is.
    """

    key_set_name: Optional[str] = None

    max_tokens: int = 4096
    timeout: int = 120
    max_retries: int = 3

    task_temperatures: Tuple[float, ...] = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2)
    checker_temperature: float = 0.8

    max_concurrency: int = 4
    mock_seed: int = 0
    """Seed for the deterministic mock backend used by tests."""

    def describe(self) -> str:
        """One line for logs, with the key reduced to a fingerprint.

        Printed at the start of every run so the transcript records *which*
        endpoint produced it, without recording the credential.
        """
        if self.provider == "mock":
            return f"provider=mock model={self.model} seed={self.mock_seed} (offline)"
        key = self.api_key or ""
        fingerprint = f"{key[:4]}...{key[-4:]} ({len(key)} chars)" if len(key) >= 8 else "<unset>"
        return (
            f"provider={self.provider} model={self.model} "
            f"api_base={self.api_base or '<unset>'} api_key={fingerprint}"
        )


    def resolve_secrets(self) -> "LLMConfig":
        """Fill ``api_base``/``api_key`` from ``key_set`` then environment.

        Precedence: explicit field > key_set file > environment variable.  Returns
        ``self`` so it can be chained; mutates in place because the config object
        is the single thing handed to the client factory.

        Placeholder values from the tracked template are treated as unset — copying
        ``llm.yaml.example`` and forgetting to edit it should produce a clear error
        about a missing key, not a 401 from the endpoint.
        """
        import os

        if _is_placeholder(self.api_key):
            self.api_key = None

        if self.key_set:
            path = Path(self.key_set).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"llm.key_set not found: {path}")
            with open(path, "r", encoding="utf-8") as fh:
                entries = yaml.safe_load(fh) or {}
            if not isinstance(entries, dict) or not entries:
                raise ValueError(f"llm.key_set is empty or not a mapping: {path}")
            name = self.key_set_name or next(iter(entries))
            if name not in entries:
                raise KeyError(
                    f"llm.key_set_name '{name}' not in {path}; have {sorted(entries)}"
                )
            entry = entries[name] or {}
            if self.api_base is None:
                self.api_base = entry.get("api_base")
            if self.api_key is None and not _is_placeholder(entry.get("api_key")):
                self.api_key = entry.get("api_key")
            models = entry.get("model_names") or []
            if self.model not in models and models and self.model == "gpt-oss-120b":
                # Only override the *default* model, never an explicit choice.
                self.model = models[0]

        if self.api_base is None:
            self.api_base = os.environ.get(self.api_base_env) or None
        if self.api_key is None:
            self.api_key = os.environ.get(self.api_key_env) or None
        return self




@dataclass
class QualityConfig:
    """Multi-agent quality checker (§3.3, Appendix A.3)."""

    max_repair_rounds: int = 2
    """Code Repair Agent attempts before discarding ("several attempts", A.3)."""

    max_improve_rounds: int = 1
    """Logic Improvement Agent attempts before discarding."""

    enable_judge: bool = True
    enable_llm_repair: bool = True

    nan_ratio_limit: float = 0.30
    """Alphas with >30% NaN are discarded (§B.4)."""

    min_distinct_per_day: int = 5
    """"distinct values per day" check from A.3: guards constant/degenerate output."""

    min_coverage: float = 0.60
    """Complement of the NaN limit, enforced per evaluated day."""

    abs_value_limit: float = 1e12
    """Overflow guard; values beyond this are treated as numerically unstable."""

    exec_timeout_s: float = 60.0
    memory_limit_mb: int = 8192
    """Resident-memory ceiling for a sandbox worker.

    Enforced by polling ``/proc/<pid>/statm`` rather than ``setrlimit(RLIMIT_AS)``:
    importing numpy, pandas and qlib on a 128-core host reserves ~5 GB of *virtual*
    address space (glibc opens up to ``8 x ncores`` 64 MB malloc arenas) while
    resident memory is ~450 MB.  An address-space cap that admits the interpreter
    cannot constrain an alpha, and one that constrains an alpha kills the import --
    which is exactly what a 4096 MB ``RLIMIT_AS`` did on the first real-data run:
    every alpha came back as ``worker exited with code -15``.
    """


    allowed_imports: Tuple[str, ...] = (
        "numpy",
        "np",
        "pandas",
        "pd",
        "math",
        "talib",
        "scipy",
        "scipy.stats",
        "scipy.special",
    )
    """Import allow-list for the static audit; anything else is rejected."""

    leakage_shift_probe: bool = True
    """Run the causality probe of :mod:`cogalpha.quality.leakage`."""

    leakage_tail_days: int = 40
    """How many trailing days get perturbed by the causality probe."""


@dataclass
class FitnessConfig:
    """Fitness evaluation and tiering (§3.4, Appendix A.4).

    ``qualified_percentile``/``elite_percentile`` default to the (65, 80) pair
    that §4.7 finds best: a looser gate keeps the parent pool large and delays
    premature convergence.

    The metric floors are the paper's own numbers, and a calibration on real
    CSI300 data (2011-2019, 23 hand-written alphas spanning all seven levels)
    confirms they sit where a floor should:

    ==========  ============  =========  =========  =========
    metric      real p50      real p80   qual floor elite floor
    ==========  ============  =========  =========  =========
    |IC|            0.0095      0.0162       0.005       0.01
    |ICIR|          0.0606      0.1247       0.05         0.1
    |RankIC|        0.0086      0.0253       0.005       0.01
    |RankICIR|      0.0640      0.1313       0.05         0.1
    ==========  ============  =========  =========  =========

    So the qualified floor admits roughly the better half of genuine alphas and the
    elite floor roughly the top fifth -- consistent with the 65/80 percentiles, and
    evidence the paper's constants were tuned on a comparable universe rather than
    picked arbitrarily.  The one exception is MI; see ``mi_scale``.
    """

    qualified_percentile: float = 65.0
    elite_percentile: float = 80.0

    qualified_min_bounds: Dict[str, float] = field(
        default_factory=lambda: {
            "ic": 0.005,
            "rank_ic": 0.005,
            "icir": 0.05,
            "rank_icir": 0.05,
            "mi": 0.02,
        }
    )
    elite_min_bounds: Dict[str, float] = field(
        default_factory=lambda: {
            "ic": 0.01,
            "rank_ic": 0.01,
            "icir": 0.1,
            "rank_icir": 0.1,
            "mi": 0.02,
        }
    )

    use_abs_ic: bool = True
    """Score |IC| rather than signed IC.

    The paper's own evolution example keeps sign implicitly positive, but a
    factor's sign is a free parameter (flip the code and the sign flips), so
    tiering on the absolute value avoids discarding a perfectly good inverted
    signal.  Set ``False`` to reproduce a strict sign-sensitive reading.
    """

    mi_bins: int = 10
    """Bin count for the histogram estimator in :mod:`cogalpha.fitness.metrics`."""

    mi_scale: str = "corr_equivalent"
    """``corr_equivalent`` | ``nats``.

    The paper's MI floor is 0.02 (§A.4) but its estimator is never stated, and MI
    in nats is not comparable across estimators.  Measured on real CSI300
    2011-2019, genuine alphas span 0.0006-0.0223 nats (p50 = 0.0029), so a
    0.02-nat floor would reject all but the single best of 23 -- the floor cannot
    have been meant in nats on this scale.  Mapping MI through the Gaussian
    relation ``sqrt(1 - exp(-2 MI))`` puts it on the same [0,1) axis as |IC|, where
    the same alphas span 0.033-0.209 (p50 = 0.076) and 0.02 behaves as a floor.
    """

    min_names_per_day: int = 10
    """Cross-sections thinner than this are dropped from the IC series."""

    rank_ic_method: str = "spearman"

    compute_backtest: bool = False
    """Compute AER/IR via the top-50/drop-5 simulation of §B.2.

    Off during search (it costs more than the five metrics and does not affect
    tiering); switched on when reporting a final candidate.
    """

    top_k: int = 50
    drop_n: int = 5
    open_cost: float = 0.0005
    close_cost: float = 0.0015
    min_fee: float = 5.0
    trading_days_per_year: int = 252


@dataclass
class EvolutionConfig:
    """Thinking Evolution and the search schedule (§3.5, §3.6, §B.4, §B.8)."""

    initial_pool_size: int = 80
    """Minimum number of alphas the task-specific agents produce (§B.4)."""

    parent_pool_size: int = 32
    children_multiplier: int = 3
    """Children pool = 3x parent pool -> at least 96 evolved alphas per generation."""

    generations: int = 24
    sub_cycles: int = 3
    """24 generations split into 3 inner sub-cycles of 8 (§B.4)."""

    agents_per_run: int = 13
    """The paper selects 13 of the 21 agents per run using the golden ratio (§B.8)."""

    golden_ratio_selection: bool = True
    alphas_per_agent: int = 6
    """"approximately 5-6 alpha factors" per selected agent (§B.8)."""

    inject_every: int = 2
    """Every 2 generations, freshly generated agent alphas are filtered and
    injected into the parent pool (§B.4)."""

    elitism_carry: int = 2
    """Top-2 elites of the previous generation are carried forward unchanged."""

    op_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "mutation": 1.0,
            "crossover": 1.0,
            "crossover_then_mutation": 1.0,
        }
    )
    """The three evolution types of §3.6, sampled with these relative weights."""

    guidance_modes: Tuple[str, ...] = (
        "light",
        "moderate",
        "creative",
        "divergent",
        "concrete",
    )
    """Diversified Guidance paraphrasing modes (§3.2, Appendix A.2)."""

    adaptive_valid_samples: int = 2
    adaptive_invalid_samples: int = 2
    """Per generation, 2 random valid + 2 worst invalid alphas become guiding
    samples for the next prompt (§3.5)."""

    plateau_window: int = 3
    plateau_delta: float = 0.001
    """Early stop when the elite-pool mean improves by <= 0.001 across two
    consecutive windows (§B.4)."""

    max_llm_calls: Optional[int] = None
    """Hard budget; ``None`` means unlimited.  A cheap defence against the cost
    blow-up flagged in ``reflection.md``."""

    dedup: bool = True
    seed: int = 42


@dataclass
class RunConfig:
    """Where the run archive goes."""

    out_dir: str = "runs"
    run_name: Optional[str] = None
    save_every_generation: bool = True
    log_level: str = "INFO"
    top_candidates: int = 20
    """Size of the final reported candidate set; the paper composes 20 alphas."""


@dataclass
class CogAlphaConfig:
    """The whole configuration: six independent sections.

    Build it with :func:`load_config` (single file) or :func:`merge_configs` (a run
    config plus a credentials overlay plus CLI overrides). Nothing in the library
    reads global state -- this object is the only input.
    """

    data: DataConfig = field(default_factory=DataConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Nested plain dict. Redaction happens in :mod:`cogalpha.archive`, not here."""
        return asdict(self)


def _coerce(section_cls, payload: Dict[str, Any]):
    """Build a dataclass from ``payload``, tolerating unknown keys loudly."""
    known = {f.name: f for f in fields(section_cls)}
    unknown = set(payload) - set(known)
    if unknown:
        raise ValueError(
            f"{section_cls.__name__}: unknown config keys {sorted(unknown)}"
        )
    kwargs: Dict[str, Any] = {}
    for key, value in payload.items():
        target = known[key]
        # Tuple-typed fields arrive from YAML as lists.
        if isinstance(value, list) and "Tuple" in str(target.type):
            value = tuple(value)
        kwargs[key] = value
    return section_cls(**kwargs)


def load_config(path: Optional[str | Path] = None, **overrides) -> CogAlphaConfig:
    """Load a config from YAML, then apply ``section.key=value`` overrides.

    >>> cfg = load_config(None, evolution={"generations": 4})
    >>> cfg.evolution.generations
    4
    """
    payload: Dict[str, Any] = {}
    if path is not None:
        payload = _read_yaml(path)

    for section, value in overrides.items():
        if not isinstance(value, dict):
            raise ValueError(f"override for '{section}' must be a mapping")
        payload.setdefault(section, {}).update(value)

    sections: Dict[str, Any] = {}
    for f in fields(CogAlphaConfig):
        sub = payload.pop(f.name, {}) or {}
        if not is_dataclass(f.type) and not isinstance(sub, dict):
            raise ValueError(f"config section '{f.name}' must be a mapping")
        sections[f.name] = _coerce(f.default_factory().__class__, sub)  # type: ignore[misc]

    if payload:
        raise ValueError(f"unknown config sections: {sorted(payload)}")
    return CogAlphaConfig(**sections)


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        hint = ""
        if resolved.name == "llm.yaml":
            hint = (
                "\nThis file holds your endpoint and API key and is deliberately "
                "git-ignored. Create it from the tracked template:\n"
                f"  cp {resolved.parent / 'llm.yaml.example'} {resolved}"
            )
        raise FileNotFoundError(f"config file not found: {resolved}{hint}")
    with open(resolved, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def merge_configs(base: Optional[str | Path], *overlays: Optional[str | Path], **overrides):
    """Load a base config, then layer YAML files over it, then keyword overrides.

    Used by the CLI so that the run configuration and the credentials live in
    separate files: ``--config configs/paper_csi300.yaml --llm-config configs/llm.yaml``.
    Keeping them apart is what lets the run config be committed while the endpoint
    file stays out of version control.

    Later overlays win, and merging is per-section rather than whole-file, so an
    ``llm.yaml`` that sets only ``llm.api_key`` leaves every other section intact.
    """
    payload: Dict[str, Any] = _read_yaml(base) if base is not None else {}

    for overlay in overlays:
        if overlay is None:
            continue
        for section, values in _read_yaml(overlay).items():
            if isinstance(values, dict):
                payload.setdefault(section, {}).update(values)
            else:
                payload[section] = values

    for section, values in overrides.items():
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ValueError(f"override for '{section}' must be a mapping")
        payload.setdefault(section, {}).update(values)

    sections: Dict[str, Any] = {}
    for f in fields(CogAlphaConfig):
        sub = payload.pop(f.name, {}) or {}
        sections[f.name] = _coerce(f.default_factory().__class__, sub)  # type: ignore[misc]
    if payload:
        raise ValueError(f"unknown config sections: {sorted(payload)}")
    return CogAlphaConfig(**sections)

