"""
Monitoring infrastructure for Famou 2.0.

This module provides real-time monitoring and visualization for evolutionary experiments,
similar to training monitoring in PyTorch.

Components:
- base: Abstract interfaces and constants
- async_logger: Asynchronous logging queue (non-blocking)
- wandb_monitor: WandB backend implementation
- metrics_collector: Metrics collection from experiments
- artifact_logger: Code and visualization logging
- visualizations: Chart generation (lineage trees, heatmaps, etc.)

Usage:
    # In config.yaml
    infrastructure:
      monitor:
        type: wandb
        wandb:
          project: "famou-experiments"
"""

from famou.infrastructure.monitor.base import (
    MonitoringBackend,
    MetricPrefix,
    LogType,
)

from famou.infrastructure.monitor.metrics_collector import (
    EvolutionMetricsCollector,
    PerformanceMetrics,
    PopulationMetrics,
    DiversityMetrics,
    CostTracker,
    ProgramData,
    IslandMetrics,
)

from famou.infrastructure.monitor.stderr_filter import (
    FilteredStderr,
    install_stderr_filter,
    uninstall_stderr_filter,
)

__all__ = [
    # Core interfaces
    "MonitoringBackend",
    "MetricPrefix",
    "LogType",
    # Metrics collectors
    "EvolutionMetricsCollector",
    "PerformanceMetrics",
    "PopulationMetrics",
    "DiversityMetrics",
    "CostTracker",
    "ProgramData",
    "IslandMetrics",
    # Stderr filtering
    "FilteredStderr",
    "install_stderr_filter",
    "uninstall_stderr_filter",
]
