"""
Configuration models for Famou 2.0.

Provides a hierarchical config structure:
- FamouConfig: Top-level entry point
  - ExperimentConfig: What to optimize (problem definition, execution scope)
  - InfraConfig: How to run (infrastructure services)
  - ModulesConfig: Algorithm parameters (optional)

Each infrastructure service has its own config:
- LLMConfig, StorageConfig, LoggerConfig, EnvConfig, EmbeddingConfig
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from famou.core.types import Language


# =============================================================================
# Infrastructure Configs
# =============================================================================


class LLMConfig(BaseModel):
    """
    Configuration for LLM provider.

    Fields:
        provider: LLM provider name (e.g., "openai", "anthropic")
        api_base: API base URL
        model: Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet")
        api_key: API key for authentication
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds
        max_retries: Maximum transport-level retry attempts for one LLM request
    """

    provider: str = Field(default="openai", description="LLM provider (openai, anthropic, etc.)")
    api_base: str = Field(default="https://api.openai.com", description="LLM API base URL")
    model: str = Field(default="gpt-4o", description="Model identifier")
    api_key: str = Field(description="API key for authentication")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=2000, gt=0, description="Maximum response tokens")
    timeout: int = Field(default=60, gt=0, description="Request timeout (seconds)")
    max_retries: int = Field(default=3, gt=0, description="Maximum transport-level retry attempts")
    fallbacks: Optional[List["LLMConfig"]] = Field(
        default=None,
        description="Fallback model configs tried in order when the primary model fails",
    )

    class Config:
        """extra fields allowed"""
        extra = "allow"


# Resolve the self-referential type annotation required by Pydantic v2
LLMConfig.model_rebuild()


class StorageConfig(BaseModel):
    """
    Configuration for storage service.

    Fields:
        type: Storage type (e.g., "local", "s3")
        base_path: Base path for storage (system directory for dual-write)
        bucket: S3 bucket name (for s3 type)
        dual_write: Enable dual-write to both base and user paths
        user_path: Path for user-facing redacted storage (defaults to ./famou_data)
        llm_request_log_dir: Dedicated directory for llm_requests.log
    """

    type: str = Field(default="local", description="Storage type (local, s3, etc.)")
    base_path: str = Field(default="./famou_data", description="Base path for storage")
    precise_path: Optional[str] = Field(
        default=None,
        description=(
            "Exact output directory path for a single experiment run. "
            "When set, Famou writes directly to this directory instead of "
            "creating {base_path}/{experiment_id}/"
        ),
    )
    # S3-specific (optional)
    bucket: Optional[str] = Field(default=None, description="S3 bucket name")
    region: Optional[str] = Field(default=None, description="S3 region")
    # Dual-write configuration
    dual_write: bool = Field(
        default=False,
        description="Enable dual-write to both base and user paths"
    )
    user_path: Optional[str] = Field(
        default=None,
        description="User path for dual-write (defaults to './famou_data')"
    )
    precise_user_path: Optional[str] = Field(
        default=None,
        description="Exact output directory for user-facing redacted storage in dual-write mode"
    )
    llm_request_log_dir: Optional[str] = Field(
        default=None,
        description="Dedicated directory for llm_requests.log (defaults to experiment directory)"
    )

    class Config:
        """extra fields allowed"""
        extra = "allow"


class LoggerConfig(BaseModel):
    """
    Configuration for logger.

    Fields:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        file_enabled: Whether to write logs to file
        log_dir: Directory for log files (optional, defaults to storage path)
    """
    type: str = Field(default="local", description="Logger type (local, remote)")
    level: str = Field(default="INFO", description="Logging level")
    file_enabled: bool = Field(default=True, description="Enable file logging")
    log_dir: Optional[str] = Field(default=None, description="Log directory (defaults to storage path)")


class EnvConfig(BaseModel):
    """
    Configuration for execution environment.

    Fields:
        type: Environment type (e.g., "local", "docker", "remote")
        timeout: Default execution timeout in seconds
        sandbox: Whether to use sandboxed execution
        gpu_ids: Available GPU IDs (e.g., [0,1,2,3]). Empty list = no GPU.
        gpu_num_each_run: Number of GPUs per evaluation task (e.g., 1.0 = 1 GPU)
    """

    type: str = Field(default="local", description="Environment type (local, process, thread, ray_task)")
    timeout: int = Field(default=30, gt=0, description="Execution timeout (seconds)")
    sandbox: bool = Field(default=False, description="Use sandboxed execution")

    # Resource configuration
    cpu: float = Field(
        default=1.0,
        gt=0,
        description="CPU cores allowed (fractional allowed)"
    )
    gpu: float = Field(
        default=0.0,
        ge=0,
        description="Number of GPUs required (0 = CPU only)"
    )
    mem: Optional[str] = Field(
        default=None,
        description='Memory limit, e.g. "512M", "2G"'
    )

    # Extra parameters passed to the evaluator function
    input_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra parameters passed to the evaluator function (e.g., competition_name, data_dir)"
    )

    class Config:
        """extra fields allowed"""
        extra = "allow"


class EvaluatorConfig(BaseModel):
    """
    Configuration for evaluator.

    Supports two modes:
    - local: Use local evaluator.py file
    - hybrid: Use remote evaluation service via famou_sdk

    Note: Evaluation timeout comes from EnvConfig.timeout (via env.default_timeout).

    Fields:
        type: Evaluator type ("local" or "hybrid")
        service_url: Remote evaluator service URL (for hybrid mode)
        api_key: API key for authentication (for hybrid mode)
        experiment_id: Experiment ID for remote service (for hybrid mode)
    """

    type: str = Field(default="local", description="Evaluator type: local/hybrid")

    # Remote evaluator config (for hybrid mode)
    service_url: Optional[str] = Field(default=None, description="Remote evaluator service URL")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    experiment_id: Optional[str] = Field(default=None, description="Experiment ID for remote service")

    # Timeout protection config
    system_protection_timeout: int = Field(
        default=86400, gt=0,
        description="System protection timeout (24h)"
    )
    user_timeout_delta: int = Field(
        default=600, gt=0,
        description="User timeout buffer (10min)"
    )

    # Network retry config
    network_retry_attempts: int = Field(default=3, gt=0, description="Network retry attempts")
    network_retry_delay: float = Field(default=30.0, gt=0, description="Base retry delay (exponential backoff)")
    network_timeout: float = Field(default=30.0, gt=0, description="Single request timeout")
    polling_interval: int = Field(default=10, gt=0, description="Status polling interval (seconds)")

    class Config:
        """extra fields allowed"""
        extra = "allow"


class EmbeddingConfig(BaseModel):
    """
    Configuration for embedding service.

    Fields:
        provider: Embedding provider (e.g., "openai")
        model: Model identifier
        api_key: API key (optional, defaults to LLM api_key)
        api_base: API base URL (optional)
    """

    provider: str = Field(default="openai", description="Embedding provider")
    model: str = Field(default="text-embedding-3-small", description="Model identifier")
    api_key: Optional[str] = Field(default=None, description="API key (defaults to LLM api_key)")
    api_base: Optional[str] = Field(default=None, description="API base URL")

    class Config:
        """extra fields allowed"""
        extra = "allow"


class MonitorConfig(BaseModel):
    """
    Configuration for monitoring service.

    Fields:
        enabled: Whether to enable monitoring
        type: Monitoring backend type (e.g., "wandb")
        wandb: WandB-specific configuration
    """
    enabled: bool = Field(default=False, description="Enable monitoring")
    type: str = Field(default="wandb", description="Monitoring backend type")
    wandb: Optional['WandBMonitorConfig'] = Field(default=None, description="WandB monitor configuration")

    class Config:
        """extra fields allowed"""
        extra = "allow"


class WandBMonitorConfig(BaseModel):
    """
    Configuration for WandB monitoring.

    Fields:
        project: WandB project name
        entity: WandB entity (username or team)
        run_name: Run name (null for auto-generation)
        group: Group name for grouping runs
        tags: List of tags for categorization
        notes: Run notes
        async_mode: Enable asynchronous logging
        num_workers: Number of async worker threads
        queue_size: Max queue size for async logging
        track_metrics: What metrics to track
        track_artifacts: Track artifacts (best programs, etc.)
        track_lineage_tree: Track lineage tree visualization
        checkpoint_interval: Record metrics every N iterations
    """
    project: str = Field(default="famou-experiments", description="WandB project name")
    entity: Optional[str] = Field(default=None, description="WandB entity (username or team)")
    run_name: Optional[str] = Field(default=None, description="Run name (null for auto)")
    group: Optional[str] = Field(default=None, description="Group name for grouping runs")
    tags: Optional[list[str]] = Field(default_factory=list, description="Tags for categorization")
    notes: Optional[str] = Field(default=None, description="Run notes")
    async_mode: bool = Field(default=True, description="Enable async logging")
    num_workers: int = Field(default=2, gt=0, description="Number of async workers")
    queue_size: int = Field(default=100, gt=0, description="Max queue size")
    track_metrics: Optional[Dict[str, bool]] = Field(
        default_factory=lambda: {
            "performance": True,
            "population": True,
            "diversity": True,
            "cost": True,
        },
        description="What metrics to track"
    )
    track_artifacts: bool = Field(default=True, description="Track artifacts")
    track_lineage_tree: bool = Field(default=True, description="Track lineage tree")
    checkpoint_interval: int = Field(default=1, gt=0, description="Record metrics every N iterations")

    class Config:
        """extra fields allowed"""
        extra = "allow"


class LangfuseConfig(BaseModel):
    """
    Configuration for Langfuse observability platform.

    Fields:
        enabled: Whether Langfuse tracing is enabled
        public_key: Langfuse public API key
        secret_key: Langfuse secret API key
        base_url: Langfuse API endpoint
    """

    enabled: bool = Field(default=False, description="Enable Langfuse tracing")
    public_key: Optional[str] = Field(default=None, description="Langfuse public key")
    secret_key: Optional[str] = Field(default=None, description="Langfuse secret key")
    base_url: str = Field(default="https://cloud.langfuse.com", description="Langfuse API endpoint")

class BackendConfig(BaseModel):
    """
    Configuration for backend services used by scheduler.
    
    """
    mode: str = Field(default="threadpool", description="Backend mode: threadpool/ray")
    ray_host: str = Field(default="", description="Ray host")

class InfraConfig(BaseModel):
    """
    Configuration for all infrastructure services.

    Groups LLM, storage, logger, env, embedding, langfuse, and monitor configs.
    """

    llm: LLMConfig = Field(description="LLM service configuration")
    storage: StorageConfig = Field(default_factory=StorageConfig, description="Storage service configuration")
    logger: LoggerConfig = Field(default_factory=LoggerConfig, description="Logger configuration")
    env: EnvConfig = Field(default_factory=EnvConfig, description="Execution environment configuration")
    embedding: Optional[EmbeddingConfig] = Field(default=None, description="Embedding service configuration")
    langfuse: Optional[LangfuseConfig] = Field(default=None, description="Langfuse observability configuration")
    monitor: Optional[MonitorConfig] = Field(default=None, description="Monitoring service configuration")
    backend: BackendConfig = Field(default_factory=BackendConfig, description="Backend service configuration")
    evaluator: Optional[EvaluatorConfig] = Field(default=None, description="Evaluator configuration")


# =============================================================================
# Experiment Config
# =============================================================================


class IslandConfig(BaseModel):
    """
    Configuration for island model (multi-population evolution).

    Fields:
        num_islands: Number of evolutionary islands (1 = single population)
        population_size: Total population size across all islands (higher priority)
        island_size: Population size per island when population_size is not set
        migration_interval: Migrate best programs every N iterations
        migration_size: Number of programs to migrate between islands
        migration_topology: Island topology (ring, full, star)
        reset_interval: reset worst-performing islands every N iterations
        reset_size: Programs to reset per island (defaults to island_size)
    """

    num_islands: int = Field(default=1, gt=0, description="Number of islands (1 = single population)")
    population_size: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Total population size across all islands. If set, overrides island_size. "
            "When it is not divisible by num_islands, the remainder is assigned one-by-one "
            "to the earliest islands so capacities stay as balanced as possible."
        ),
    )
    island_size: int = Field(default=5, gt=0, description="Population size per island")
    migration_topology: str = Field(default="ring", description="Topology: ring, full, star")
    migration_interval: Optional[int] = Field(default=None, ge=1, description="Migration interval (None disables)")
    migration_size: Optional[int] = Field(default=1, gt=0, description="Programs to migrate per interval")
    reset_interval: Optional[int] = Field(default=None, ge=1, description="reset interval (None disables)")
    reset_size: Optional[int] = Field(default=1, gt=0, description="Programs to reset per interval")

    @property
    def effective_island_size(self) -> int:
        """
        Backwards-compatible representative island capacity.

        For uneven splits, returns the capacity of island 0, which is also the
        maximum per-island capacity under the current deterministic balancing rule.
        """
        return self.get_island_size(0)

    @property
    def effective_population_size(self) -> int:
        """Resolved total population capacity."""
        if self.population_size is not None:
            return self.population_size
        return self.island_size * self.num_islands

    @property
    def island_sizes(self) -> List[int]:
        """
        Resolved capacity for every island.

        If population_size is not set, all islands share island_size. Otherwise,
        population_size is split as evenly as possible, with any remainder
        distributed to lower-index islands.
        """
        if self.population_size is None:
            return [self.island_size] * self.num_islands

        base_size = self.population_size // self.num_islands
        remainder = self.population_size % self.num_islands
        return [
            base_size + (1 if island_id < remainder else 0)
            for island_id in range(self.num_islands)
        ]

    def get_island_size(self, island_id: int) -> int:
        """Return the resolved capacity for one island."""
        if island_id < 0 or island_id >= self.num_islands:
            raise IndexError(
                f"island_id must be in [0, {self.num_islands - 1}], got {island_id}"
            )
        return self.island_sizes[island_id]


class ExperimentConfig(BaseModel):
    """
    Experiment configuration - defines WHAT to optimize.

    This contains problem definition and execution scope,
    but NOT infrastructure settings (those are in InfraConfig).

    Fields:
        name: Human-readable experiment name
        task_description: Description of the optimization problem/task
        language: Programming language
        strategy: Strategy name for rollout pipeline (e.g., "standard", "exploration")

        programs: Path to initial program file(s) - optional, can use CLI args instead
        evaluator: Path to evaluator module - optional, can use CLI args instead

        max_iterations: Maximum number of iterations to run
        max_workers: Maximum parallel workers for streaming evolution
        checkpoint_interval: Save checkpoint every N completed rollouts
        seed: Random seed for reproducibility

        island: Island model configuration

        production_info: Extensible production context metadata (e.g. is_resume, task_source)
    """

    # Problem definition
    name: str = Field(description="Experiment name")
    task_description: str = Field(description="Description of the optimization problem/task")
    language: Language = Field(default=Language.PYTHON, description="Programming language")
    strategy: str = Field(default="standard", description="Strategy name for rollout pipeline")
    run_user_id: str = Field(default="", description="User ID who initiated the experiment run")

    # Code paths (optional, can be overridden by CLI args)
    programs: Optional[str] = Field(
        default=None,
        description="Path to initial program file(s), comma-separated for multiple seeds"
    )
    evaluator: Optional[str] = Field(
        default=None,
        description="Path to evaluator module (Python file with 'evaluate' function)"
    )

    # Execution scope
    max_iterations: int = Field(default=10, gt=0, description="Maximum iterations")
    max_workers: int = Field(default=4, gt=0, description="Maximum parallel workers")
    checkpoint_interval: int = Field(default=5, gt=0, description="Checkpoint interval (rollouts)")
    feasibility_check: bool = Field(
        default=False,
        description=(
            "Whether to require at least one initial program to achieve validity=1.0 "
            "after initial enrichment before evolution starts"
        ),
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")

    # Island model
    island: IslandConfig = Field(default_factory=IslandConfig, description="Island model configuration")

    # Debug loop (auto-injected by Evolver into every code-producing rollout)
    debug: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": False, "max_iterations": 3, "dynamic": False},
        description=(
            "Debug loop configuration (Evolver-level injection). "
            "enabled (bool): activate debug loop. "
            "max_iterations (int): max repair attempts per buggy program. "
            "dynamic (bool): dynamically adjust trigger probability based on historical success rate. "
            "When dynamic=True: total_cnt<=5 always triggers; total_cnt>5 triggers with "
            "rate=min(1.0, success_rate+0.2) where success_rate=validity==1.0 after debug."
        )
    )

    # Data leak detection (auto-injected after GenerateModule, before EvaluateModule)
    data_leak_detector: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": False},
        description=(
            "Data leak detection configuration (Evolver-level injection). "
            "enabled (bool): activate leak detection. "
            "Uses the global LLM settings."
        )
    )

    # Runtime context augmentation (LLM-extracted time constraint + diagnostic print requirement)
    runtime_context: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": False},
        description=(
            "Runtime context augmentation: enabled (bool, default False). "
            "When enabled, makes a one-time LLM call at experiment start to extract "
            "the per-test-case time constraint from evaluator code, then appends it "
            "along with a stderr diagnostic print requirement to task_description."
        )
    )

    # Strategy router (auto-select strategy based on task direction)
    strategy_router: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": False},
        description=(
            "Strategy router: enabled (bool), direction "
            "(optional: machine_learning/combinatorial_optimization/math/other), "
            "direction2strategy (optional partial dict override for direction->strategy mapping; "
            "unspecified directions fall back to the built-in defaults)"
        )
    )

    # Production context (extensible key-value pairs for run-time context)
    production_info: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Production context metadata. Extensible key-value pairs for run-time context, "
            "e.g. is_resume (bool): whether initial programs come from a previous experiment, "
            "task_source (str), priority (str), etc."
        )
    )

    # Convenience properties for backwards compatibility
    @property
    def num_islands(self) -> int:
        """Number of islands (convenience accessor)."""
        return self.island.num_islands

    @property
    def island_size(self) -> int:
        """Backwards-compatible representative island size accessor."""
        return self.island.effective_island_size

    @property
    def island_sizes(self) -> List[int]:
        """Resolved capacity for every island."""
        return self.island.island_sizes

    @property
    def population_size(self) -> int:
        """Total population size across all islands."""
        return self.island.effective_population_size

    def get_island_size(self, island_id: int) -> int:
        """Return the resolved capacity for one island."""
        return self.island.get_island_size(island_id)

    @property
    def migration_interval(self) -> int:
        """Migration interval (convenience accessor)."""
        return self.island.migration_interval

    @property
    def migration_size(self) -> int:
        """Migration size (convenience accessor)."""
        return self.island.migration_size

    @property
    def migration_topology(self) -> str:
        """Migration topology (convenience accessor)."""
        return self.island.migration_topology

    @property
    def reset_interval(self) -> Optional[int]:
        """reset interval (convenience accessor)."""
        return self.island.reset_interval

    @property
    def reset_size(self) -> Optional[int]:
        """reset size (convenience accessor)."""
        return self.island.reset_size


# =============================================================================
# Modules Config
# =============================================================================


class ModulesConfig(BaseModel):
    """
    Optional configuration for module parameters.

    These parameters are passed to modules when instantiated.
    Modules are still instantiated in code (type-safe),
    but parameters can come from this config.

    Example YAML:
        modules:
          select:
            k: 1
            num_inspirations: 2
          population:
            max_size: 20

    Usage in code:
        rollout = Rollout(
            modules=[
                EliteSelect(**config.modules.select),
                ...
            ]
        )
    """

    select: Dict[str, Any] = Field(default_factory=dict, description="Select module parameters")
    generate: Dict[str, Any] = Field(default_factory=dict, description="Generate module parameters (e.g. write_mode: full_write|diff_write, default full_write)")
    evaluate: Dict[str, Any] = Field(default_factory=dict, description="Evaluate module parameters")
    judge: Dict[str, Any] = Field(default_factory=dict, description="Judge module parameters")
    population: Dict[str, Any] = Field(default_factory=dict, description="Population module parameters")
    strategy: Dict[str, Any] = Field(default_factory=dict, description="Strategy-level parameters")


# =============================================================================
# Top-Level Config
# =============================================================================


class FamouConfig(BaseModel):
    """
    Top-level configuration for Famou.

    Combines experiment definition, infrastructure services,
    and module parameters.

    Example:
        config = FamouConfig.from_yaml("config.yaml")
        infra = InfraFactory.create(config.infrastructure)
        evolver = Evolver(
            config=config.experiment,
            infrastructure=infra,
            ...
        )
    """

    experiment: ExperimentConfig = Field(description="Experiment configuration")
    infrastructure: InfraConfig = Field(description="Infrastructure configuration")
    modules: ModulesConfig = Field(default_factory=ModulesConfig, description="Module parameters")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.

        Uses json mode to properly serialize enums (e.g., Language) as their string values
        instead of enum objects.
        """
        return self.model_dump(mode='json')

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FamouConfig":
        """Create config from dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "FamouConfig":
        """
        Load config from YAML file.

        Args:
            yaml_path: Path to YAML file

        Returns:
            FamouConfig instance
        """
        import yaml
        from pathlib import Path

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, yaml_path: str) -> None:
        """
        Save config to YAML file.

        Args:
            yaml_path: Path to save YAML file
        """
        import yaml
        from pathlib import Path

        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False, allow_unicode=True)
