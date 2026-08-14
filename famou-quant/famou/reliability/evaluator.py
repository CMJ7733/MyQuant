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
    EvaluationCost,
    EvidenceVector,
    Fidelity,
    FrozenSplitManifest,
    MetricDistribution,
    StaticCheckResult,
)


# =============================================================================
# F0: static checks
# =============================================================================


class StaticChecker:
    """F0 static inspection of a candidate Program.

    Checks:
    - parses / compiles
    - candidate contract present (HYPERPARAMS, FAMOU_RESULT output)
    - no future-reference idioms (negative Ref/shift on price fields)
    - no edits to protected components (backtester / cost / split config)
    """

    #: idioms that indicate lookahead in qlib-style expressions
    FUTURE_IDIOMS = ('Ref($close, -', 'Ref($open, -', ".shift(-", "shift(-")

    #: substrings that must not appear modified by candidates
    PROTECTED_MARKERS = (
        "sealed_promotion",
        "final_test",
        "transaction_cost",
        "backtest_config",
    )

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

        # 3. leakage idioms
        for idiom in self.FUTURE_IDIOMS:
            if idiom in program.code:
                result.leakage_flags.append(f"future-reference idiom: {idiom!r}")

        # 4. protected components: candidates may *read* split ranges from the
        #    config but must not redefine protected protocol constants.
        for marker in self.PROTECTED_MARKERS:
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        name = ""
                        if isinstance(target, ast.Name):
                            name = target.id
                        elif isinstance(target, ast.Attribute):
                            name = target.attr
                        if marker in name.lower():
                            result.forbidden_edits.append(
                                f"assignment to protected name {name!r}"
                            )

        # 5. required packages (best-effort import scan)
        result.required_packages = sorted(self._scan_imports(tree))
        return result

    @staticmethod
    def _scan_imports(tree: ast.AST) -> set:
        pkgs = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                pkgs.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                pkgs.add(node.module.split(".")[0])
        # stdlib / harness-provided modules are not install requirements
        stdlib = {
            "argparse", "json", "math", "os", "sys", "time", "pathlib",
            "typing", "collections", "dataclasses", "functools", "itertools",
            "famou_candidate_runtime",
        }
        return pkgs - stdlib


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

        icir = None
        if rank_ic is not None and rank_ic.std > 0:
            icir_vals = [
                v / rank_ic.std for v in (rank_ic.per_seed or [rank_ic.mean])
            ]
            icir = MetricDistribution.from_samples(icir_vals)

        return EvidenceVector(
            candidate_id=candidate_id,
            episode_id=manifest.episode_id,
            eval_id=f"ev_{uuid.uuid4().hex[:12]}",
            fidelity=fidelity,
            split_scope="visible_dev",
            data_contract_hash=manifest.compute_hash(),
            rank_ic=rank_ic,
            icir=icir,
            validity=float(raw.get("validity", 0.0)),
            static_check=static_check,
            failure_stage=failure_stage,
            error_info=raw.get("error_info"),
            complexity={"model_family": raw.get("model_family", "unknown")},
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
    """

    FIDELITY_PRESETS = {
        Fidelity.F1_CHEAP: {"max_seeds": 1, "max_boost_rounds": 200},
        Fidelity.F2_FULL: {"max_seeds": None, "max_boost_rounds": None},
    }

    def __init__(
        self,
        manifest: FrozenSplitManifest,
        run_fn,
        ledger: Optional[BudgetLedger] = None,
    ):
        self.manifest = manifest
        self._run_fn = run_fn
        self._ledger = ledger
        self._checker = StaticChecker()
        self._builder = EvidenceBuilder()

    def evaluate(
        self,
        program: Program,
        fidelity: Fidelity,
        *,
        seed_list: Optional[List[int]] = None,
        novelty: Optional[float] = None,
    ) -> EvidenceVector:
        # ---- F0 gate: static check always runs first -------------------
        static = self._checker.check(program)
        if fidelity == Fidelity.F0_STATIC or not static.passed:
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
        preset = self.FIDELITY_PRESETS[fidelity]
        seeds = list(seed_list or [11])
        if preset["max_seeds"] is not None:
            seeds = seeds[: preset["max_seeds"]]
        split_cfg = {
            "train_start": self.manifest.train.start,
            "train_end": self.manifest.train.end,
            "dev_start": self.manifest.visible_dev.start,
            "dev_end": self.manifest.visible_dev.end,
            "embargo_days": self.manifest.embargo_days,
            "seed_list": seeds,
            "fidelity": int(fidelity.value),
        }
        if preset["max_boost_rounds"] is not None:
            split_cfg["max_boost_rounds"] = preset["max_boost_rounds"]

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
            gpu_seconds=wall if raw.get("used_gpu") else 0.0,
            visible_query_count=1,
        )
        if self._ledger is not None:
            self._ledger.charge(cost, episode_id=self.manifest.episode_id)

        return self._builder.build(
            candidate_id=program.id,
            manifest=self.manifest,
            fidelity=fidelity,
            raw=raw,
            cost=cost,
            static_check=static,
            seed_list=seeds,
            failure_stage=failure_stage,
            novelty=novelty,
        )
