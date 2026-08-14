"""Multi-fidelity evaluation for reliability-gated candidates.

- ``StaticChecker`` (F0): AST-level checks — compiles, candidate contract
  present, no future-data idioms, no edits to protected components.
- ``EvidenceBuilder``: converts a raw evaluator result dict into an
  EvidenceVector (merging per-seed metrics into distributions).
- ``FidelityEvaluator``: wraps the famou ``ExecutionEnvironment`` to run a
  candidate at F1 (cheap: fewer seeds, capped rounds) or F2 (full visible
  dev, multi-seed), charging the BudgetLedger atomically.

The evaluator NEVER touches sealed data — the split config it builds only
ever points at train + visible_dev. Sealed evaluation lives exclusively in
``promotion.SealedGateService``.
"""

from __future__ import annotations

import ast
import time
import uuid
from typing import Any, Dict, List, Optional

from famou.core.data import Program
from famou.reliability.budget import BudgetLedger
from famou.reliability.types import (
    EvalRequest,
    EvaluationCost,
    EvidenceVector,
    Fidelity,
    FrozenSplitManifest,
    MetricDistribution,
    StaticCheckResult,
    TaskSpec,
)


# =============================================================================
# F0: static checks
# =============================================================================


class StaticChecker:
    """F0 static inspection of a candidate Program.

    Driven by the TaskSpec rather than hard-coded constants: the protected
    symbol list and the package allow-list are part of the frozen experiment
    contract, so tightening them is a protocol amendment with a new hash, not
    an edit buried in the checker.

    Checks:
    - parses / compiles
    - candidate contract present (HYPERPARAMS, FAMOU_RESULT output)
    - no future-reference idioms in candidate-authored expressions
    - no assignment to protected protocol names (backtester / cost / splits)
    - no imports outside the allowed set
    """

    #: idioms that indicate lookahead in qlib-style expressions
    FUTURE_IDIOMS = ("Ref($close, -", "Ref($open, -", ".shift(-", "shift(-")

    #: stdlib / harness-provided modules that are never install requirements
    _STDLIB = {
        "argparse", "json", "math", "os", "sys", "time", "pathlib",
        "typing", "collections", "dataclasses", "functools", "itertools",
        "random", "logging", "famou_candidate_runtime",
    }

    def __init__(
        self,
        task_spec: Optional[TaskSpec] = None,
        label_expression: str = "",
    ):
        self.task_spec = task_spec or TaskSpec()
        #: the frozen label is itself a forward expression and lives in the
        #: data contract, not the TaskSpec; it is exempt from the leak scan
        self.label_expression = label_expression

    def check(self, program: Program) -> StaticCheckResult:
        result = StaticCheckResult()

        # 1. compile
        try:
            tree = ast.parse(program.code)
            compile(tree, filename=f"<{program.id}>", mode="exec")
        except SyntaxError as e:
            result.compiles = False
            result.schema_errors.append(f"SyntaxError: {e}")
            return result

        # 2. contract
        has_hyper = any(
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "HYPERPARAMS" for t in n.targets)
            for n in tree.body
        )
        if not has_hyper:
            result.schema_errors.append("missing HYPERPARAMS assignment")
        if "FAMOU_RESULT" not in program.code:
            result.schema_errors.append("missing FAMOU_RESULT output line")
        if "--split-config" not in program.code:
            result.schema_errors.append("missing --split-config argument parsing")

        # 3. leakage idioms.
        #    The frozen label itself is a forward expression (Ref($close,-2)/
        #    Ref($close,-1)-1) and legitimately appears in the harness, so only
        #    candidate-authored string constants are scanned — not the whole
        #    file — and the label expression is exempt.
        label = self._label_expression()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            text = node.value
            if label and label in text:
                continue  # the frozen label, not a candidate invention
            for idiom in self.FUTURE_IDIOMS:
                if idiom in text:
                    result.leakage_flags.append(
                        f"future-reference idiom {idiom!r} in candidate expression"
                    )

        # 4. protected components: candidates may *read* protocol values from
        #    the split config but must not redefine them.
        protected = {s.lower() for s in self.task_spec.protected_symbols}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                name = ""
                if isinstance(target, ast.Name):
                    name = target.id
                elif isinstance(target, ast.Attribute):
                    name = target.attr
                if not name:
                    continue
                lowered = name.lower()
                if any(marker in lowered for marker in protected):
                    result.forbidden_edits.append(
                        f"assignment to protected name {name!r}"
                    )

        # 5. imports must be inside the frozen allow-list
        imported = self._scan_imports(tree)
        result.required_packages = sorted(imported)
        allowed = set(self.task_spec.allowed_packages)
        for pkg in sorted(imported - allowed):
            result.schema_errors.append(
                f"package {pkg!r} is not in the TaskSpec allow-list"
            )
        return result

    def _label_expression(self) -> str:
        return self.label_expression or ""

    @classmethod
    def _scan_imports(cls, tree: ast.AST) -> set:
        pkgs = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                pkgs.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                pkgs.add(node.module.split(".")[0])
        return pkgs - cls._STDLIB


# =============================================================================
# Evidence construction
# =============================================================================


class EvidenceBuilder:
    """Build an EvidenceVector from a raw evaluator result dict."""

    def build(
        self,
        *,
        candidate_id: str,
        manifest: FrozenSplitManifest,
        fidelity: Fidelity,
        raw: Dict[str, Any],
        cost: EvaluationCost,
        static_check: Optional[StaticCheckResult] = None,
        seed_list: Optional[List[int]] = None,
        failure_stage: Optional[str] = None,
        novelty: Optional[float] = None,
    ) -> EvidenceVector:
        per_seed = raw.get("per_seed_rank_ic") or []
        rank_ic = None
        if per_seed:
            rank_ic = MetricDistribution.from_samples([float(v) for v in per_seed])
        elif raw.get("rank_ic") is not None:
            rank_ic = MetricDistribution.from_samples([float(raw["rank_ic"])])

        # ICIR must be mean(daily IC) / std(daily IC) — a *time-series* ratio
        # computed inside one training run. It is therefore reported by the
        # harness, never reconstructed here: an earlier version divided each
        # seed's mean IC by the across-seed std, which is a different quantity
        # (and an order of magnitude larger for typical seed spreads).
        icir = None
        per_seed_icir = raw.get("per_seed_icir") or []
        if per_seed_icir:
            icir = MetricDistribution.from_samples([float(v) for v in per_seed_icir])
        elif raw.get("icir") is not None:
            icir = MetricDistribution.from_samples([float(raw["icir"])])

        # Subperiod ICs: the stability evidence for DETERMINISTIC candidates.
        # A formula re-run with another seed returns the identical number, so
        # cross-seed dispersion is structurally zero and says nothing about
        # whether the candidate holds up. Contiguous subperiods do.
        subperiod = [float(v) for v in (raw.get("subperiod_rank_ic") or [])]
        regime_stability = {f"sub{i}": v for i, v in enumerate(subperiod)}
        worst_sub = min(subperiod) if subperiod else None

        complexity: Dict[str, Any] = {"model_family": raw.get("model_family", "unknown")}
        if raw.get("deterministic"):
            complexity["deterministic"] = True
        if raw.get("factors_used"):
            complexity["n_factors"] = len(raw["factors_used"])
            complexity["factors"] = list(raw["factors_used"])

        return EvidenceVector(
            candidate_id=candidate_id,
            episode_id=manifest.episode_id,
            eval_id=f"ev_{uuid.uuid4().hex[:12]}",
            fidelity=fidelity,
            split_scope="visible_dev",
            data_contract_hash=manifest.compute_hash(),
            rank_ic=rank_ic,
            icir=icir,
            regime_stability=regime_stability,
            worst_subperiod_rank_ic=worst_sub,
            validity=float(raw.get("validity", 0.0)),
            static_check=static_check,
            failure_stage=failure_stage,
            error_info=raw.get("error_info"),
            complexity=complexity,
            cost=cost,
            novelty=novelty,
            seed_list=seed_list or [],
        )


# =============================================================================
# F1/F2 evaluator
# =============================================================================


class FidelityEvaluator:
    """Runs candidates at F1/F2 fidelity through the famou env abstraction.

    ``run_fn`` is the harness hook with signature::

        run_fn(program_code, split_config: dict) -> dict  # raw metrics

    It is injected (rather than hard-coding qlib) so the evaluator is
    testable with a stub and the qlib wiring lives in the example layer.

    Thread-safety: one instance is shared by all rollout workers. It holds no
    per-evaluation state, and the ledger it charges serialises check-and-charge
    under a lock, so concurrent evaluate() calls are safe. (This is also why
    ``_ReliabilityEvaluate`` refuses to deep-copy it.)
    """

    #: how each fidelity level shrinks the evaluation
    FIDELITY_PRESETS = {
        Fidelity.F1_CHEAP: {
            "max_seeds": 1,
            "max_boost_rounds": 200,
            "train_fraction": 0.5,
            "universe_fraction": 0.5,
        },
        Fidelity.F2_FULL: {
            "max_seeds": None,
            "max_boost_rounds": None,
            "train_fraction": 1.0,
            "universe_fraction": 1.0,
        },
    }

    def __init__(
        self,
        manifest: FrozenSplitManifest,
        run_fn,
        ledger: Optional[BudgetLedger] = None,
        task_spec: Optional[TaskSpec] = None,
    ):
        self.manifest = manifest
        self._run_fn = run_fn
        self._ledger = ledger
        self.task_spec = task_spec or TaskSpec()
        self._checker = StaticChecker(
            self.task_spec, label_expression=manifest.label_expression
        )
        self._builder = EvidenceBuilder()

    # ------------------------------------------------------------------

    def build_request(
        self,
        candidate_id: str,
        fidelity: Fidelity,
        *,
        seed_list: Optional[List[int]] = None,
        max_gpu_seconds: Optional[float] = None,
    ) -> EvalRequest:
        """Turn a fidelity choice into a concrete resource contract.

        The preset sets the shape of the evaluation; the action's own ceiling
        may only tighten it, never exceed the TaskSpec cap.
        """
        preset = self.FIDELITY_PRESETS.get(fidelity, self.FIDELITY_PRESETS[Fidelity.F2_FULL])
        seeds = list(seed_list or [11])
        if preset["max_seeds"] is not None:
            seeds = seeds[: preset["max_seeds"]]

        cap = self.task_spec.max_candidate_gpu_seconds
        gpu_cap = cap if max_gpu_seconds is None else min(float(max_gpu_seconds), cap)
        return EvalRequest(
            request_id=f"er_{uuid.uuid4().hex[:12]}",
            candidate_id=candidate_id,
            fidelity=fidelity,
            seed_list=seeds,
            max_gpu_seconds=gpu_cap,
            timeout_seconds=self.task_spec.max_candidate_wall_seconds,
            max_boost_rounds=preset["max_boost_rounds"],
            train_fraction=preset["train_fraction"],
            universe_fraction=preset["universe_fraction"],
        )

    def evaluate(
        self,
        program: Program,
        fidelity: Fidelity,
        *,
        seed_list: Optional[List[int]] = None,
        novelty: Optional[float] = None,
        request: Optional[EvalRequest] = None,
        max_gpu_seconds: Optional[float] = None,
    ) -> EvidenceVector:
        if request is None:
            request = self.build_request(
                program.id,
                fidelity,
                seed_list=seed_list,
                max_gpu_seconds=max_gpu_seconds,
            )

        # ---- F0 gate: static check always runs first -------------------
        static = self._checker.check(program)
        if request.fidelity == Fidelity.F0_STATIC or not static.passed:
            return self._builder.build(
                candidate_id=program.id,
                manifest=self.manifest,
                fidelity=Fidelity.F0_STATIC,
                raw={
                    "validity": 0.0,
                    "error_info": "; ".join(
                        static.schema_errors
                        + static.leakage_flags
                        + static.forbidden_edits
                    ),
                },
                cost=EvaluationCost(),
                static_check=static,
                failure_stage="compile" if not static.compiles else "static_check",
                novelty=novelty,
            )

        # ---- F1/F2: build split config (visible only!) -----------------
        split_cfg = {
            "train_start": self.manifest.train.start,
            "train_end": self.manifest.train.end,
            "dev_start": self.manifest.visible_dev.start,
            "dev_end": self.manifest.visible_dev.end,
            "embargo_days": self.manifest.embargo_days,
            "seed_list": list(request.seed_list),
            "fidelity": int(request.fidelity.value),
            "train_fraction": request.train_fraction,
            "universe_fraction": request.universe_fraction,
            "max_gpu_seconds": request.max_gpu_seconds,
            "timeout_seconds": request.timeout_seconds,
            "universe": self.task_spec.universe,
            "topk": self.task_spec.topk,
            "n_drop": self.task_spec.n_drop,
            "transaction_cost_bps": self.task_spec.transaction_cost_bps,
        }
        if request.max_boost_rounds is not None:
            split_cfg["max_boost_rounds"] = request.max_boost_rounds

        started = time.time()
        failure_stage = None
        try:
            raw = self._run_fn(program.code, split_cfg)
        except TimeoutError:
            raw = {"validity": 0.0, "error_info": "evaluation timeout"}
            failure_stage = "timeout"
        except MemoryError:
            raw = {"validity": 0.0, "error_info": "out of memory"}
            failure_stage = "oom"
        except Exception as e:  # candidate bug — evidence, not rollout failure
            raw = {"validity": 0.0, "error_info": f"{type(e).__name__}: {e}"}
            failure_stage = "train"
        wall = time.time() - started

        cost = EvaluationCost(
            wall_seconds=wall,
            gpu_seconds=float(raw.get("gpu_seconds", wall if raw.get("used_gpu") else 0.0)),
            llm_tokens=int(raw.get("llm_tokens", 0)),
            visible_query_count=1,
        )
        if self._ledger is not None:
            self._ledger.charge(cost, episode_id=self.manifest.episode_id)

        return self._builder.build(
            candidate_id=program.id,
            manifest=self.manifest,
            fidelity=request.fidelity,
            raw=raw,
            cost=cost,
            static_check=static,
            seed_list=list(request.seed_list),
            failure_stage=failure_stage,
            novelty=novelty,
        )
