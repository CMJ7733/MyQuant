"""
Configuration models for Famou 2.0.

Hierarchical config structure:
- FamouConfig: Top-level entry point (use this to load from YAML)
  - ExperimentConfig: What to optimize (problem definition, execution scope)
  - InfraConfig: How to run (infrastructure services)
  - ModulesConfig: Algorithm parameters (optional)

Infrastructure configs:
- LLMConfig, StorageConfig, LoggerConfig, EnvConfig, EmbeddingConfig, MonitorConfig

Example:
    from famou.config import FamouConfig
    config = FamouConfig.from_yaml("config.yaml")
"""

from famou.config.settings import (
    # Top-level
    FamouConfig,
    # Experiment
    ExperimentConfig,
    IslandConfig,
    # Infrastructure
    InfraConfig,
    LLMConfig,
    StorageConfig,
    LoggerConfig,
    EnvConfig,
    EmbeddingConfig,
    MonitorConfig,
    WandBMonitorConfig,
    # Modules
    ModulesConfig,
)

__all__ = [
    # Top-level config
    "FamouConfig",
    # Experiment config
    "ExperimentConfig",
    "IslandConfig",
    # Infrastructure configs
    "InfraConfig",
    "LLMConfig",
    "StorageConfig",
    "LoggerConfig",
    "EnvConfig",
    "EmbeddingConfig",
    "MonitorConfig",
    "WandBMonitorConfig",
    # Module config
    "ModulesConfig",
]
