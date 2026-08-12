"""
RayTaskEnv: GPU on-demand via Ray task.

When configured with type=ray_task, the worker actor does NOT hold GPU resources.
Instead, RayTaskEnv dynamically submits a Ray task with num_gpus at execution time,
and releases the GPU immediately when the task completes.
"""

import logging
from typing import Any, Dict, List, Optional

from famou.config.settings import EnvConfig

logger = logging.getLogger(__name__)


def _ray_task_execute(
    env_config: EnvConfig,
    program_code: str,
    evaluate_fn_bytes: bytes,
    language: str,
    extension: Optional[str],
    timeout: Optional[float],
    required_packages: Optional[List[str]],
) -> Dict[str, Any]:
    """
    GPU task execution body.

    Constructs the inner env via EnvFactory._create_from_config(env_config) —
    a stateless call that avoids class-variable race conditions.
    """
    import cloudpickle
    from famou.infrastructure.env.env_factory import EnvFactory

    evaluate_fn = cloudpickle.loads(evaluate_fn_bytes)
    # Inside the Ray task, fall back to process env for actual execution.
    # Without this override, _create_from_config would create another RayTaskEnv
    # (since env_config.type == "ray_task"), causing infinite recursion.
    inner_config = env_config.model_copy(update={"type": "process"})
    real_env = EnvFactory._create_from_config(inner_config)

    return real_env.execute_with_evaluator(
        program_code, evaluate_fn, language, extension, timeout, required_packages
    )


class RayTaskEnv:
    """Env implementation: submits evaluator to a resource-carrying Ray task for execution."""

    def __init__(self, env_config: EnvConfig):
        self._env_config = env_config
        self.default_timeout = env_config.timeout
        self._gpu = env_config.gpu

    def _make_remote_fn(self):
        """Build a ray.remote wrapper with all resources from env_config."""
        import ray

        resources = {}
        cpu = getattr(self._env_config, 'cpu', 1.0)
        if cpu > 0:
            resources["num_cpus"] = cpu
        if self._gpu > 0:
            resources["num_gpus"] = self._gpu
        mem = getattr(self._env_config, 'mem', None)
        if mem:
            resources["memory"] = _parse_memory(mem)
        return ray.remote(**resources)(_ray_task_execute)

    def execute_with_evaluator(
        self,
        program_code: str,
        evaluate_fn,
        language: str = "python",
        extension: Optional[str] = None,
        timeout: Optional[float] = None,
        required_packages: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute program code with the provided evaluator function.

        Submits a Ray task that runs the code and returns the execution results.
        """
        import cloudpickle

        evaluate_fn_bytes = cloudpickle.dumps(evaluate_fn)
        ref = self._make_remote_fn().remote(
            self._env_config, program_code, evaluate_fn_bytes,
            language, extension, timeout, required_packages
        )
        import ray
        return ray.get(ref)

    def prepare_dependencies(
        self,
        required_packages: List[str],
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Prepare environment dependencies by running a no-op task.

        This ensures all required packages are installed before actual execution.
        """
        import cloudpickle

        noop = cloudpickle.dumps(lambda _: {"combined_score": 0.0, "validity": 1.0})
        ref = self._make_remote_fn().remote(
            self._env_config, "pass\n", noop, "python", ".py", timeout, required_packages
        )
        import ray
        return ray.get(ref)

    def close(self) -> None:
        """Clean up resources used by the Ray task environment."""
        pass


def _parse_memory(mem_str: str) -> int:
    """Parse memory string (e.g. '512M', '2G') to bytes for Ray resource."""
    mem_str = mem_str.strip().upper()
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    for suffix, multiplier in units.items():
        if mem_str.endswith(suffix):
            return int(float(mem_str[:-1]) * multiplier)
    return int(mem_str)
