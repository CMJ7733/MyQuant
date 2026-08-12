"""
Infrastructure Factory for Famou 2.0.

Provides factory methods to instantiate infrastructure services from config.
Supports both automatic instantiation (via type/provider field) and
explicit instantiation (direct class usage).

Example (automatic):
    config = FamouConfig.from_yaml("config.yaml")
    infra = InfraFactory.create_all(config.infrastructure)
    # infra.llm, infra.storage, infra.logger, infra.env, infra.embedding

Example (explicit):
    llm_client = OpenAIClient(
        api_key=config.infrastructure.llm.api_key,
        model=config.infrastructure.llm.model,
    )
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
from langfuse import Langfuse

from famou.config.settings import (
    EmbeddingConfig,
    EnvConfig,
    EvaluatorConfig,
    InfraConfig,
    LangfuseConfig,
    LLMConfig,
    LoggerConfig,
    MonitorConfig,
    StorageConfig,
    BackendConfig
)
from famou.infrastructure.embedding import EmbeddingClient, OpenAIEmbeddingClient, MockEmbeddingClient
from famou.infrastructure.env import EnvFactory
from famou.infrastructure.llm import GeminiClient, LLMClient, MockLLMClient, OpenAIClient
from famou.infrastructure.llm.fallback_client import FallbackLLMClient
from famou.infrastructure.logger import LocalLogger, Logger
from famou.infrastructure.storage import DataService, LocalStorage
from famou.infrastructure.storage.dual_write_storage import DualWriteLocalStorage
from famou.infrastructure.backend import BackendExecutor, ThreadPoolBackend

# Import RayBackend if available (requires ray package)
try:
    from famou.infrastructure.backend import RayBackend
except ImportError:
    RayBackend = None


@dataclass
class Infrastructure:
    """
    Container for all infrastructure services.

    Created by InfraFactory.create_all() for convenience.
    """

    llm: LLMClient
    storage: DataService
    logger: Logger
    env: EnvFactory
    embedding: Optional[EmbeddingClient] = None
    monitor: Optional['MonitoringBackend'] = None  # Optional monitoring backend
    langfuse: Optional[Langfuse] = None  # type: ignore
    backend: Optional[BackendExecutor] = None
    evaluator_config: Optional[EvaluatorConfig] = None  # Evaluator config for Strategy


class InfraFactory:
    """
    Factory for creating infrastructure services from config.

    Supports automatic instantiation based on type/provider fields,
    while still allowing explicit instantiation when needed.
    """

    # ==========================================================================
    # LLM Client
    # ==========================================================================

    @staticmethod
    def create_llm(config: LLMConfig, langfuse: Optional["Langfuse"] = None) -> LLMClient:  # type: ignore
        """
        Create LLM client from config.

        If config.fallbacks is set, returns a FallbackLLMClient that wraps the
        primary client followed by each fallback client in order.

        Args:
            config: LLM configuration with provider field
            langfuse: Optional Langfuse client for observability

        Returns:
            LLMClient instance (FallbackLLMClient when fallbacks are configured)

        Raises:
            ValueError: If provider is not supported
        """
        primary = InfraFactory._create_single_llm(config, langfuse)
        if not config.fallbacks:
            return primary
        fallback_clients = [
            InfraFactory._create_single_llm(fb_config, langfuse)
            for fb_config in config.fallbacks
        ]
        return FallbackLLMClient(clients=[primary] + fallback_clients)  # type: ignore

    @staticmethod
    def _create_single_llm(config: LLMConfig, langfuse: Optional["Langfuse"] = None) -> LLMClient:  # type: ignore
        """
        Create a single LLM client from config without fallback wrapping.

        Args:
            config: LLM configuration with provider field
            langfuse: Optional Langfuse client for observability

        Returns:
            LLMClient instance

        Raises:
            ValueError: If provider is not supported
        """
        provider = config.provider.lower()

        if provider == "openai":
            return OpenAIClient(
                api_key=config.api_key,
                model=config.model,
                api_base=config.api_base,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
                max_retries=config.max_retries,
                langfuse=langfuse,
            )
        elif provider in ("google", "gemini"):
            return GeminiClient(
                api_key=config.api_key,
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
                max_retries=config.max_retries,
                langfuse=langfuse,
            )
        elif provider == "mock":
            return MockLLMClient(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
                max_retries=config.max_retries,
                langfuse=langfuse,
            )
        # Add more providers here as needed
        # elif provider == "anthropic":
        #     return AnthropicClient(..., langfuse=langfuse)
        else:
            raise ValueError(
                f"Unknown LLM provider: {config.provider}. "
                f"Supported providers: openai, google, mock"
            )

    # ==========================================================================
    # Storage Service
    # ==========================================================================

    @staticmethod
    def create_storage(config: StorageConfig) -> DataService:
        """
        Create storage service from config.

        Args:
            config: Storage configuration with type field

        Returns:
            DataService instance

        Raises:
            ValueError: If storage type is not supported
        """
        storage_type = config.type.lower()

        if storage_type == "local":
            if config.dual_write:
                # Dual-write mode supports precise_path and precise_user_path
                # When precise_path is set, use it as base_path (don't use default)
                # When precise_user_path is set, use it as user_path (don't use default)
                effective_base_path = config.base_path if config.base_path else "./famou_system"
                effective_user_path = config.user_path if config.user_path else "./famou_data"

                return DualWriteLocalStorage(
                    base_path=effective_base_path,
                    user_path=effective_user_path,
                    precise_path=getattr(config, "precise_path", None),
                    precise_user_path=getattr(config, "precise_user_path", None),
                    llm_request_log_dir=getattr(config, "llm_request_log_dir", None),
                )
            return LocalStorage(
                base_path=config.base_path,
                precise_path=getattr(config, "precise_path", None),
                llm_request_log_dir=getattr(config, "llm_request_log_dir", None),
            )
        # Add more storage types here as needed
        # elif storage_type == "s3":
        #     return S3Storage(bucket=config.bucket, region=config.region)
        else:
            raise ValueError(
                f"Unknown storage type: {config.type}. "
                f"Supported types: local"
            )

    # ==========================================================================
    # Logger
    # ==========================================================================

    @staticmethod
    def create_logger(
        config: LoggerConfig,
        name: str = "famou",
        experiment_id: Optional[str] = None,
        storage_path: Optional[str] = None,
        resolved_log_dir: Optional[str] = None,
        extra_log_dirs: Optional[list] = None,
        extra_jsonl_dirs: Optional[list] = None,
        extra_log_levels: Optional[set] = None,
    ) -> Logger:
        """
        Create logger from config.

        Args:
            config: Logger configuration
            name: Logger name (typically experiment name)
            experiment_id: Experiment ID for log file naming
            storage_path: Base storage path (used if log_dir not specified)
            extra_log_dirs: Additional directories to write log files to (for dual-write)
            extra_jsonl_dirs: Additional directories to write JSONL files to (overrides extra_log_dirs for JSONL)
            extra_log_levels: If set, only these log levels are written to extra directories

        Returns:
            Logger instance
        """
        # Determine log directory
        logger_type = config.type.lower()
        log_dir = resolved_log_dir or config.log_dir
        if log_dir is None and storage_path and experiment_id:
            log_dir = f"{storage_path}/{experiment_id}"
        if logger_type == "local":
            return LocalLogger(
                name=name,
                experiment_id=experiment_id,
                log_dir=log_dir,
                level=config.level,
                file_enabled=config.file_enabled,
                extra_log_dirs=extra_log_dirs,
                extra_jsonl_dirs=extra_jsonl_dirs,
                extra_log_levels=extra_log_levels,
            )
        else:
            raise ValueError(
                f"Unknown logger type: {config.type}. "
                f"Supported types: local"
            )

    # ==========================================================================
    # Execution Environment
    # ==========================================================================

    @staticmethod
    def create_env(config: EnvConfig) -> EnvFactory:
        """
        Create execution environment from config.
        
        Delegates to EnvFactory which creates famou_sdk environments.

        Args:
            config: Environment configuration with type field and GPU settings

        Returns:
            EnvFactory

        Raises:
            ValueError: If environment type is not supported
        """
        EnvFactory.set_config(config)
        return EnvFactory

    # ==========================================================================
    # Embedding Client
    # ==========================================================================

    @staticmethod
    def create_embedding(
        config: EmbeddingConfig,
    ) -> EmbeddingClient:
        """
        Create embedding client from config.

        Args:
            config: Embedding configuration with provider field
            llm_api_key: Fallback API key from LLM config

        Returns:
            EmbeddingClient instance

        Raises:
            ValueError: If provider is not supported
        """
        provider = config.provider.lower()

        # Use embedding-specific api_key, or fall back to LLM api_key
        api_key = config.api_key
        if not api_key:
            raise ValueError("Embedding API key required (either in embedding config or LLM config)")

        if provider == "openai":
            return OpenAIEmbeddingClient(
                api_key=api_key,
                model=config.model,
                api_base=config.api_base,
            )
        # Add more providers here as needed
        elif provider == "mock":
            return MockEmbeddingClient()
        else:
            raise ValueError(
                f"Unknown embedding provider: {config.provider}. "
                f"Supported providers: openai, mock"
            )

    # ==========================================================================
    # Monitor (WandB)
    # ==========================================================================

    @staticmethod
    def create_monitor(
        config: MonitorConfig,
    ) -> Optional['MonitoringBackend']:
        """
        Create monitor backend from config.

        Args:
            config: Monitor configuration

        Returns:
            MonitoringBackend instance or None if disabled/unavailable
        """
        if not config.enabled:
            return None

        try:
            # Try to import monitoring components
            from famou.infrastructure.monitor.base import MonitoringBackend
            from famou.infrastructure.monitor.wandb_monitor import WandBMonitor

            if config.type.lower() == "wandb":
                if config.wandb is None:
                    return None

                return WandBMonitor(
                    project=config.wandb.project,
                    entity=config.wandb.entity,
                    config=None,  # Config will be logged separately
                    tags=config.wandb.tags,
                    notes=config.wandb.notes,
                    name=config.wandb.run_name,
                    group=config.wandb.group,
                    async_mode=config.wandb.async_mode,
                    num_workers=config.wandb.num_workers,
                    queue_size=config.wandb.queue_size,
                )
            else:
                raise ValueError(
                    f"Unknown monitor type: {config.type}. "
                    f"Supported types: wandb"
                )

        except ImportError as e:
            # If wandb or other dependencies are not installed, just disable monitoring
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Monitoring disabled due to missing dependencies: {e}. "
                f"Install with: pip install wandb"
            )
            return None

    # ==========================================================================
    # Langfuse Observability
    # ==========================================================================

    @staticmethod
    def create_langfuse(config: LangfuseConfig) -> Optional[Any]:  # type: ignore
        """
        Create Langfuse client from config using Langfuse v3+ API.

        Uses get_client() to initialize the global Langfuse client singleton.
        The client is configured via environment variables or explicit initialization.

        Args:
            config: Langfuse configuration

        Returns:
            Langfuse client or None if disabled

        Raises:
            ValueError: If enabled but missing credentials
            ImportError: If langfuse package not installed
        """
        if not config.enabled:
            return None

        if not config.public_key or not config.secret_key:
            raise ValueError("Langfuse enabled but missing credentials (public_key or secret_key)")
        
        langfuse_client = Langfuse(
            public_key=config.public_key,
            secret_key=config.secret_key,
            base_url=config.base_url,

        )

        # Initialize and return the global client singleton
        return langfuse_client

    @staticmethod
    def create_backend(config: BackendConfig) -> BackendExecutor:
        """
        Create backend executor from config.

        Args:
            config: Backend configuration

        Returns:
            BackendExecutor instance

        Raises:
            ValueError: If invalid backend type provided
        """
        return BackendExecutor.create(config)

    # ==========================================================================
    # Create All Infrastructure
    # ==========================================================================

    @classmethod
    def create_all(
        cls,
        config: InfraConfig,
        experiment_name: str = "famou",
        experiment_id: Optional[str] = None,
    ) -> Infrastructure:
        """
        Create all infrastructure services from config.

        This is the main convenience method for creating all services at once.

        Args:
            config: Full infrastructure configuration
            experiment_name: Name for logger
            experiment_id: Experiment ID for logger and storage paths

        Returns:
            Infrastructure container with all services

        Example:
            config = FamouConfig.from_yaml("config.yaml")
            infra = InfraFactory.create_all(
                config.infrastructure,
                experiment_name=config.experiment.name,
                experiment_id=experiment_id,
            )
        """
        # Create Langfuse client first (optional, needed for LLM instrumentation)
        langfuse = None
        if config.langfuse:
            try:
                langfuse = cls.create_langfuse(config.langfuse)
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to initialize Langfuse: {e}")

        # Create LLM client (with langfuse for instrumentation)
        llm = cls.create_llm(config.llm, langfuse=langfuse)

        # Create storage service
        storage = cls.create_storage(config.storage)

        # Create logger
        precise_path = getattr(config.storage, "precise_path", None)
        precise_user_path = getattr(config.storage, "precise_user_path", None)
        dual_write = getattr(config.storage, "dual_write", False)

        # Determine log paths for system and user
        if dual_write and precise_user_path:
            # Dual-write mode: primary log goes to precise_path (system)
            # Text logs also written to precise_user_path (user) via extra_log_dirs
            # JSONL (experiment.jsonl) only written to system dir, NOT user dir
            # Only INFO level logs are written to user dir (ERROR/WARNING are system-only)
            resolved_log_dir = precise_path if not config.logger.log_dir else None
            logger_storage_path = precise_user_path
            extra_log_dirs = [precise_user_path] if precise_user_path and precise_user_path != precise_path else None
            # JSONL should NOT be written to the user directory
            extra_jsonl_dirs = []
            extra_log_levels = {"INFO"}
        elif precise_path:
            resolved_log_dir = precise_path if not config.logger.log_dir else None
            logger_storage_path = str(Path(precise_path).parent) if precise_path else config.storage.base_path
            extra_log_dirs = None
            extra_jsonl_dirs = None
            extra_log_levels = None
        else:
            resolved_log_dir = None
            logger_storage_path = config.storage.base_path
            extra_log_dirs = None
            extra_jsonl_dirs = None
            extra_log_levels = None

        logger = cls.create_logger(
            config.logger,
            name=experiment_name,
            experiment_id=experiment_id,
            storage_path=logger_storage_path,
            resolved_log_dir=resolved_log_dir,
            extra_log_dirs=extra_log_dirs,
            extra_jsonl_dirs=extra_jsonl_dirs,
            extra_log_levels=extra_log_levels,
        )
        # Resolve llm_requests.log directory: explicit config takes precedence,
        # otherwise derive from storage/logger paths with dual-write logic.
        llm_request_log_dir = getattr(config.storage, "llm_request_log_dir", None)
        if llm_request_log_dir:
            llm_log_dir = llm_request_log_dir
        else:
            llm_log_dir = resolved_log_dir or config.logger.log_dir
            # In dual-write mode, llm_requests.log belongs in the system dir (precise_path),
            # not in the user-visible dir (precise_user_path).
            if dual_write and precise_path:
                llm_log_dir = precise_path
            if llm_log_dir is None and logger_storage_path and experiment_id:
                llm_log_dir = f"{logger_storage_path}/{experiment_id}"
        if llm_log_dir and hasattr(llm, "add_jsonl_request_hook"):
            llm.add_jsonl_request_hook(str(Path(llm_log_dir) / "llm_requests.log"))

        # Inject logger into FallbackLLMClient so fallback events are visible in logs
        if isinstance(llm, FallbackLLMClient):
            llm._logger = logger

        # Create execution environment
        env = cls.create_env(config.env)

        # Create embedding client (optional)
        embedding = None
        if config.embedding:
            embedding = cls.create_embedding(
                config.embedding,
            )

        # Create monitor (optional)
        monitor = None
        if config.monitor:
            monitor = cls.create_monitor(config.monitor)
        
        backend = cls.create_backend(config.backend)

        return Infrastructure(
            llm=llm,
            storage=storage,
            logger=logger,
            env=env,
            embedding=embedding,
            monitor=monitor,
            langfuse=langfuse,
            backend=backend,
            evaluator_config=config.evaluator,
        )
