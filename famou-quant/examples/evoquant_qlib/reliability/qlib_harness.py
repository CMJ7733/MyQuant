"""Real qlib harness: the ``run_fn`` the FidelityEvaluator calls.

``FidelityEvaluator`` is deliberately agnostic about how a candidate runs —
it takes ``run_fn(code, split_config) -> dict``. This module is that hook for
qlib, and the same factory serves the sealed gate: only the split range in the
config differs, so a promoted candidate is re-scored by identical machinery.

Candidates execute in a SUBPROCESS, not in-process. Three reasons, all of
which have bitten this kind of pipeline before:

- a candidate that segfaults, leaks memory or calls ``sys.exit`` takes down
  one worker, not the search;
- timeouts are enforceable (an in-process infinite loop is not killable);
- the candidate cannot reach into the evolver's objects — it talks over stdout
  through one ``FAMOU_RESULT`` line, which is also what makes its output
  auditable.

The subprocess gets ``famou_candidate_runtime`` on PYTHONPATH, which is what
supplies the frozen dataset/label/preprocessing rules.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

HERE = Path(__file__).resolve().parent
RESULT_PREFIX = "FAMOU_RESULT"


class HarnessError(RuntimeError):
    """The candidate ran but did not honour the output contract."""


def _parse_result(stdout: str) -> Dict[str, Any]:
    """Take the LAST FAMOU_RESULT line.

    Last, not first: a candidate may legitimately print progress, and if it
    somehow emits several results the final one is what its main() decided.
    """
    payload = None
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(RESULT_PREFIX):
            payload = line[len(RESULT_PREFIX):].strip()
    if payload is None:
        raise HarnessError("candidate produced no FAMOU_RESULT line")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as e:
        raise HarnessError(f"FAMOU_RESULT is not valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise HarnessError(f"FAMOU_RESULT must be a JSON object, got {type(parsed).__name__}")
    return parsed


def make_run_fn(
    *,
    provider_uri: str,
    python_executable: Optional[str] = None,
    runtime_dir: Optional[Path] = None,
    default_timeout: float = 3600.0,
    num_threads: int = 4,
    extra_env: Optional[Dict[str, str]] = None,
) -> Callable[[str, Dict[str, Any]], Dict[str, Any]]:
    """Build the ``run_fn`` for a FidelityEvaluator (or a sealed gate).

    Args:
        provider_uri: qlib data directory. Injected into every split config so
            a candidate cannot point itself at a different snapshot.
        python_executable: interpreter for the subprocess (default: current).
        runtime_dir: where ``famou_candidate_runtime.py`` lives.
        default_timeout: wall-clock ceiling when the request does not set one.
        num_threads: per-candidate LightGBM threads. Keep modest — parallelism
            belongs at the candidate level, and a single model is already
            memory-bandwidth bound past ~4 threads.
    """
    runtime_path = Path(runtime_dir or HERE).resolve()
    if not (runtime_path / "famou_candidate_runtime.py").exists():
        raise FileNotFoundError(
            f"famou_candidate_runtime.py not found in {runtime_path}"
        )
    interpreter = python_executable or sys.executable

    def run_fn(code: str, split_config: Dict[str, Any]) -> Dict[str, Any]:
        cfg = dict(split_config)
        # The harness owns these: a candidate must not be able to choose its
        # data snapshot or its thread budget.
        cfg["provider_uri"] = provider_uri
        cfg.setdefault("num_threads", num_threads)
        timeout = float(cfg.get("timeout_seconds") or default_timeout)

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(runtime_path)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        env.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        env.setdefault("OMP_NUM_THREADS", str(num_threads))
        if extra_env:
            env.update(extra_env)

        started = time.time()
        with tempfile.TemporaryDirectory(prefix="famou_cand_") as tmp:
            tmp_path = Path(tmp)
            script = tmp_path / "candidate.py"
            script.write_text(code, encoding="utf-8")
            cfg_path = tmp_path / "split_config.json"
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

            try:
                proc = subprocess.run(
                    [interpreter, str(script), "--split-config", str(cfg_path)],
                    capture_output=True, text=True, timeout=timeout,
                    env=env, cwd=tmp,
                )
            except subprocess.TimeoutExpired:
                # Surfaced as TimeoutError so FidelityEvaluator tags the
                # evidence failure_stage="timeout" rather than "train".
                raise TimeoutError(
                    f"candidate exceeded {timeout:.0f}s"
                ) from None

            wall = time.time() - started
            if proc.returncode != 0:
                tail = (proc.stderr or "").strip().splitlines()[-12:]
                return {
                    "validity": 0.0,
                    "error_info": f"exit {proc.returncode}: " + " | ".join(tail),
                    "wall_seconds": wall,
                }
            try:
                result = _parse_result(proc.stdout)
            except HarnessError as e:
                tail = (proc.stderr or "").strip().splitlines()[-8:]
                return {
                    "validity": 0.0,
                    "error_info": f"{e}; stderr: {' | '.join(tail)}",
                    "wall_seconds": wall,
                }

        result.setdefault("wall_seconds", wall)
        result.setdefault("validity", 1.0)
        # GPU accounting: only the NN families touch a device, and only when
        # one is present. Reported so the BudgetLedger charges honestly rather
        # than billing CPU time as GPU time.
        if result.get("used_gpu") and "gpu_seconds" not in result:
            result["gpu_seconds"] = wall
        return result

    return run_fn


def make_sealed_eval_fn(
    *,
    provider_uri: str,
    manifest,
    incumbent_code: Optional[str] = None,
    **kwargs,
) -> Callable[..., Dict[str, Any]]:
    """Build the sealed-gate hook: the SAME runner pointed at sealed_promotion.

    Returns the signature SealedGateService expects::

        sealed_eval_fn(candidate_code, manifest, seeds) -> {
            "rank_ic_per_seed": [...], "incumbent_rank_ic": float, ...
        }

    The raw numbers produced here never leave the gate object — it quantises
    them into a verdict plus a coarse margin band.
    """
    run_fn = make_run_fn(provider_uri=provider_uri, **kwargs)

    #: (incumbent code hash, seeds) -> sealed RankIC.
    #: The margin is candidate_IC - incumbent_IC measured on the SAME sealed
    #: segment: the incumbent's *visible* IC is a different time period, and
    #: differencing across regimes measures the market, not the model. But the
    #: incumbent is fixed for the episode, so retraining it on every gate query
    #: recomputes an identical number — that is half the sealed compute for
    #: nothing. Cached here, inside the sealed side, so no value escapes.
    incumbent_cache: Dict[tuple, Optional[float]] = {}

    def sealed_eval_fn(candidate_code: str, man, seeds) -> Dict[str, Any]:
        seed_tuple = tuple(seeds)
        cfg = {
            "train_start": man.train.start,
            "train_end": man.train.end,
            # The only line that differs from the visible evaluator.
            "dev_start": man.sealed_promotion.start,
            "dev_end": man.sealed_promotion.end,
            "embargo_days": man.embargo_days,
            "seed_list": list(seeds),
            "label_expression": man.label_expression,
        }
        out = run_fn(candidate_code, cfg)
        per_seed = out.get("per_seed_rank_ic") or []
        payload: Dict[str, Any] = {"rank_ic_per_seed": per_seed}

        if incumbent_code is not None:
            # Key on the code itself: if the incumbent is ever made dynamic
            # (e.g. the best certified member rather than a fixed baseline),
            # a new incumbent gets a new entry instead of a stale hit.
            key = (hashlib.sha256(incumbent_code.encode()).hexdigest(), seed_tuple)
            if key not in incumbent_cache:
                base = run_fn(incumbent_code, cfg)
                base_seeds = base.get("per_seed_rank_ic") or []
                incumbent_cache[key] = (
                    sum(base_seeds) / len(base_seeds) if base_seeds else None
                )
            payload["incumbent_rank_ic"] = incumbent_cache[key]
        return payload

    return sealed_eval_fn
