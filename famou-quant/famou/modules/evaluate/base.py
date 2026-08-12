"""
Evaluation modules for Famou 2.0.

Evaluates a single program using custom evaluator functions via ExecutionEnvironment.
"""

import time
from typing import Callable, Dict, Any, Optional

from famou.core.data import Context, RolloutResult, normalize_error_info_value
from famou.core.protocol import Module, RequiresEnv
from famou.core.types import RolloutStatus
from famou.infrastructure.env import ExecutionEnvironment
from famou.utils.trace_utils import (
    append_evaluate_trace,
    attach_debug_trace,
    build_evaluate_trace,
    snapshot_program,
    to_serializable,
)


class EvaluateModule(Module, RequiresEnv):
    """
    Evaluation module using user-provided evaluator functions.

    Evaluates a single program by calling a custom evaluator function that returns
    evaluation metrics, combined_score, and validity.

    Data Flow:
    - Input: result.generated_program (single Program)
    - Call evaluate_fn(program_path, gpu_ids=...) -> enriches program in place
    - Output: Program enriched with metrics, combined_score, validity, error_info

    Dependencies (Injected):
        env: ExecutionEnvironment - Injected by RolloutEngine (default: LocalEnv)

    Args:
        name: Module name (default: class name)
        evaluate_fn: User evaluator function (REQUIRED)

    Evaluator Function Signature:
        Your evaluator function should accept:
        - program_path: str (path to the generated program file)
        - gpu_ids: Optional[List[int]] (allocated GPU IDs, if GPU is configured)

        Example:
            def evaluate(program_path: str, gpu_ids: Optional[List[int]] = None):
                import os
                import subprocess

                # Set GPU environment variable if GPUs are allocated
                env = os.environ.copy()
                if gpu_ids is not None:
                    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

                # Run the program
                result = subprocess.run(
                    ['python', program_path],
                    env=env,
                    capture_output=True,
                    timeout=60
                )

                return {
                    "combined_score": 0.95,
                    "validity": 1.0,
                    "custom_metric": result.returncode,
                }
    """

    env: ExecutionEnvironment  # Injected by RolloutEngine

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        evaluate_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    ):
        super().__init__(name)
        self.evaluate_fn = evaluate_fn

    def validate_input(self, context: Context, result: RolloutResult) -> None:
        """Validate that a program is available and evaluator is configured."""
        if not result.generated_program:
            raise ValueError(
                f"{self.name}: No program to evaluate. "
                "Make sure GenerateModule has run first."
            )

        if self.evaluate_fn is None:
            raise ValueError(
                f"{self.name}: evaluate_fn is required. "
                "Provide evaluate_fn=your_evaluator when creating the module."
            )

        if not callable(self.evaluate_fn):
            raise ValueError(
                f"{self.name}: evaluate_fn must be callable, got {type(self.evaluate_fn)}"
            )

    def execute(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        """Execute program using custom evaluator function via execution environment."""
        program = result.generated_program
        extension_map = {
            "python": ".py",
            "cpp": ".cpp",
            "c": ".c",
            "java": ".java",
            "js": ".js",
            "rust": ".rs",
            "go": ".go",
            "typescript": ".ts",
            "bash": ".sh",
        }

        extension = program.file_extension or extension_map.get(context.language, ".py")
        program.file_extension = extension
        pending_debug_attempt = program.meta.pop("_pending_debug_attempt", None)
        eval_result = None
        started_at = None
        completed_at = None
        caught_error = None
        effective_timeout = getattr(self.env, "default_timeout", None)

        request_payload = {
            "program_id": program.id,
            "parent_id": program.parent_id,
            "language": context.language,
            "extension": program.file_extension,
            "timeout": effective_timeout,
            "required_packages": program.required_packages or [],
            "evaluator_required_packages": list(
                getattr(self.evaluate_fn, "required_packages", []) or []
            ),
        }

        try:
            # Execute program via environment abstraction
            # The env handles temp file creation/cleanup and GPU allocation
            self.log_info(f"Evaluating {program.id}")
            started_at = time.time()
            eval_result = self.env.execute_with_evaluator(
                program_code=program.code,
                evaluate_fn=self.evaluate_fn,
                language=context.language,
                extension=program.file_extension,
                timeout=effective_timeout,
                required_packages=program.required_packages,
            )
            completed_at = time.time()

            # Validate that required fields are present
            if "combined_score" not in eval_result:
                raise ValueError(
                    f"Evaluator must return 'combined_score'. "
                    f"Got keys: {list(eval_result.keys())}"
                )
            if "validity" not in eval_result:
                raise ValueError(
                    f"Evaluator must return 'validity'. "
                    f"Got keys: {list(eval_result.keys())}"
                )

            # Extract metrics from flat format
            # All keys except reserved ones become metrics
            reserved_keys = {"combined_score", "validity", "error_info"}
            program.metrics = {
                k: v for k, v in eval_result.items()
                if k not in reserved_keys
            }

            # Set evaluation results
            program.combined_score = eval_result["combined_score"]
            program.validity = eval_result["validity"]
            program.error_info = normalize_error_info_value(
                eval_result.get("error_info")
            )

            self.log_info(
                f"Evaluated {program.id}",
                program_id=program.id,
                combined_score=program.combined_score,
                validity=program.validity,
            )

        except Exception as e:
            caught_error = e
            if completed_at is None:
                completed_at = time.time()
            # Handle evaluator errors
            # Set program as invalid with error info
            program.validity = 0.0
            program.error_info = normalize_error_info_value(
                f"Evaluator error: {str(e)}"
            )
            program.combined_score = 0.0

            self.log_error(
                f"Evaluator failed for {program.id}: {e}",
                program_id=program.id,
            )

            # 连续失败计数：挂在 env 上，以在同一 worker 的多次 rollout 间持久化
            _consecutive_key = "_evaluator_consecutive_errors"
            consecutive = getattr(self.env, _consecutive_key, 0) + 1
            setattr(self.env, _consecutive_key, consecutive)
            max_consecutive = getattr(self, "max_consecutive_errors", 10)
            if consecutive >= max_consecutive:
                self.log_error(
                    f"Evaluator failed {consecutive} times consecutively, escalating to FATAL",
                    program_id=program.id,
                    consecutive_errors=consecutive,
                )
                result.status = RolloutStatus.FATAL
                result.error_message = (
                    f"Evaluator consecutive failure limit ({max_consecutive}) reached: {e}"
                )
                return result
        else:
            # 成功时重置计数器
            setattr(self.env, "_evaluator_consecutive_errors", 0)
        if program.error_info:
            # Keep the TAIL of the error: tracebacks / compiler logs carry the
            # real failure at the end. This is the only place error_info is
            # capped; prompts render it in full.
            program.error_info = (
                program.normalized_error_info
                if len(program.normalized_error_info) <= 5000
                else program.normalized_error_info[-5000:]
            )

        evaluate_trace = build_evaluate_trace(
            module_name=self.name,
            request=request_payload,
            started_at=started_at,
            completed_at=completed_at,
            raw_result=eval_result,
            parsed={
                "combined_score": program.combined_score,
                "validity": program.validity,
                "error_info": program.error_info,
                "metrics": program.metrics,
            },
            error=caught_error,
        )
        append_evaluate_trace(program, evaluate_trace)
        if isinstance(pending_debug_attempt, int):
            attach_debug_trace(
                program,
                attempt=pending_debug_attempt,
                field_name="evaluate_trace",
                trace=evaluate_trace,
            )

        if pending_debug_attempt is not None:
            debug_history = program.meta.setdefault("debug_history", [])
            attempt_record = {
                "attempt": pending_debug_attempt,
                "evaluation_result": {
                    "combined_score": program.combined_score,
                    "validity": program.validity,
                    "error_info": program.error_info,
                    "metrics": to_serializable(program.metrics),
                },
                "output_program": snapshot_program(program),
            }
            if (
                debug_history
                and isinstance(debug_history[-1], dict)
                and debug_history[-1].get("attempt") == pending_debug_attempt
            ):
                debug_history[-1].update(attempt_record)
            else:
                debug_history.append(attempt_record)

        return result

    def validate_output(self, context: Context, result: RolloutResult) -> None:
        """Validate that program was evaluated."""
        if result.generated_program and result.generated_program.combined_score is None:
            self.log_warning(
                f"{self.name}: Program does not have combined_score set. "
                "Check evaluator function."
            )
