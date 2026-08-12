"""
LocalLogger - Multi-output logging for Famou experiments.

Provides three output streams:
1. Console output (using Python's logging module)
2. Plain text .log file (human-readable, same as console)
3. Structured JSONL file (machine-readable, for analysis)

Implements the minimal Logger protocol with only core methods (info, warning,
error, debug) plus utility methods (program, stats). Framework-specific
formatting is handled by the calling code (e.g., Evolver helper methods).

Supports multi-line logging with traditional format (timestamp + level on each line).
"""

import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

from famou.core.data import Program


class LocalLogger:
    """
    Multi-output logger implementing the minimal Logger protocol.

    Logs to:
    - Console: Traditional Python logging format
    - experiment.log: Plain text file (human-readable, same as console)
    - experiment.jsonl: Structured JSONL (machine-readable, for analysis)

    Features:
    - Multi-line logging support (each line gets timestamp + level)
    - Context-aware formatting (iteration, island, etc.)
    - Full datetime timestamps
    - Traditional logging format

    Implements core methods (info, warning, error, debug) plus utility methods
    (program, stats). Framework-specific formatting is handled by calling code.

    Example:
        >>> logger = LocalLogger(
        ...     name="circle_packing",
        ...     experiment_id="exp_123",
        ...     log_dir="./famou_data/exp_123",
        ...     level="INFO"
        ... )
        >>> # Basic logging with structured context
        >>> logger.info("Starting evolution", iteration=1, island_id=0, max_iterations=100)
        >>>
        >>> # Multi-line logging
        >>> logger.info("[ITER 1/20] abc123 ← def456\\n  Score: 0.8234\\n  Time: 2.34s")
        >>>
        >>> # Program display
        >>> logger.program(best_program, title="Best Program")
        >>>
        >>> # Statistics display
        >>> logger.stats({"best_score": 0.85, "avg_score": 0.67}, title="Population Stats")

        # Console output example:
        2026-01-12 12:34:56.789 INFO     [Iter 1/100] [island_id=0] Starting evolution
        2026-01-12 12:34:56.789 INFO     [ITER 1/20] abc123 ← def456
        2026-01-12 12:34:56.789 INFO       Score: 0.8234
        2026-01-12 12:34:56.789 INFO       Time: 2.34s
    """

    def __init__(
        self,
        name: str = "famou",
        experiment_id: Optional[str] = None,
        log_dir: Optional[str] = None,
        level: str = "INFO",
        console_enabled: bool = True,
        file_enabled: bool = True,
        jsonl_enabled: bool = True,
        extra_log_dirs: Optional[list] = None,
        extra_jsonl_dirs: Optional[list] = None,
        extra_log_levels: Optional[Set[str]] = None,
    ):
        """
        Initialize LocalLogger with multi-output support.

        Args:
            name: Logger name (displayed in logs)
            experiment_id: Experiment ID (added to all log entries)
            log_dir: Directory to store log files (default: ./logs)
            level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
            console_enabled: Enable console output (default: True)
            file_enabled: Enable plain text .log file output (default: True)
            jsonl_enabled: Enable structured .jsonl file output (default: True)
            extra_log_dirs: Additional directories to write log files to (for dual-write)
            extra_jsonl_dirs: Additional directories to write JSONL files to (overrides extra_log_dirs for JSONL).
                              If None, JSONL extra dirs follow extra_log_dirs.
            extra_log_levels: If set, only these log levels are written to extra directories.
                              Level names are uppercase strings: "DEBUG", "INFO", "WARNING", "ERROR".
                              None means all levels are written (default).
        """
        self.name = name
        self.experiment_id = experiment_id
        self.console_enabled = console_enabled
        self.file_enabled = file_enabled
        self.jsonl_enabled = jsonl_enabled
        self._extra_log_levels = extra_log_levels

        # Log level
        self.level = getattr(logging, level.upper(), logging.INFO)

        # Setup Python logger for console output
        self.py_logger = logging.getLogger(f"famou.{name}")
        self.py_logger.setLevel(self.level)
        self.py_logger.handlers.clear()  # Clear any existing handlers

        if console_enabled:
            # Console handler with custom formatter
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            formatter = logging.Formatter(
                fmt="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console_handler.setFormatter(formatter)
            self.py_logger.addHandler(console_handler)

        # Setup file logging
        self.text_log_file = None
        self.jsonl_log_file = None
        # Extra log directories for dual-write
        self._extra_text_log_files: list = []
        self._extra_jsonl_log_files: list = []

        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            # Plain text log file (human-readable)
            if file_enabled:
                self.text_log_file = log_path / "experiment.log"
                self._write_text_header()

            # JSONL log file (machine-readable)
            if jsonl_enabled:
                self.jsonl_log_file = log_path / "experiment.jsonl"
                self._write_jsonl_header()

        # Determine effective extra JSONL dirs:
        # - If extra_jsonl_dirs is explicitly provided, use it
        # - Otherwise, fall back to extra_log_dirs (backward compatible)
        effective_extra_jsonl_dirs = extra_jsonl_dirs if extra_jsonl_dirs is not None else extra_log_dirs

        # Setup extra log directories for dual-write
        if extra_log_dirs:
            for extra_dir in extra_log_dirs:
                extra_path = Path(extra_dir)
                extra_path.mkdir(parents=True, exist_ok=True)

                if file_enabled:
                    extra_text_file = extra_path / "experiment.log"
                    self._extra_text_log_files.append(extra_text_file)
                    self._write_text_header_to(extra_text_file)

        # Setup extra JSONL directories (may differ from text log dirs)
        if effective_extra_jsonl_dirs:
            for extra_dir in effective_extra_jsonl_dirs:
                extra_path = Path(extra_dir)
                extra_path.mkdir(parents=True, exist_ok=True)

                if jsonl_enabled:
                    extra_jsonl_file = extra_path / "experiment.jsonl"
                    self._extra_jsonl_log_files.append(extra_jsonl_file)
                    self._write_jsonl_header_to(extra_jsonl_file)

    def _write_text_header(self) -> None:
        """Write header to plain text log file."""
        if not self.text_log_file:
            return
        self._write_text_header_to(self.text_log_file)

    def _write_text_header_to(self, text_file: Path) -> None:
        """Write header to a specific text log file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        header_lines = [
            "=" * 80,
            f"Famou Evolution Log",
            f"Logger: {self.name}",
            f"Experiment ID: {self.experiment_id}",
            f"Log Level: {logging.getLevelName(self.level)}",
            f"Started: {timestamp}",
            "=" * 80,
            "",
        ]

        try:
            with open(text_file, "a") as f:
                f.write("\n".join(header_lines) + "\n")
        except Exception:
            pass

    def _write_jsonl_header(self) -> None:
        """Write header to JSONL log file."""
        if not self.jsonl_log_file:
            return
        self._write_jsonl_header_to(self.jsonl_log_file)

    def _write_jsonl_header_to(self, jsonl_file: Path) -> None:
        """Write header to a specific JSONL log file."""
        header = {
            "event": "log_start",
            "timestamp": time.time(),
            "logger": self.name,
            "experiment_id": self.experiment_id,
            "level": logging.getLevelName(self.level),
        }

        try:
            with open(jsonl_file, "a") as f:
                f.write(json.dumps(header) + "\n")
        except Exception:
            pass

    def info(self, message: str, **context: Any) -> None:
        """
        Log info-level message.

        Args:
            message: The message to log
            **context: Additional context (e.g., iteration=1, score=0.85)
        """
        if self.level <= logging.INFO:
            self._log("INFO", message, context, "blue")

    def warning(self, message: str, system_only: bool = False, **context: Any) -> None:
        """Log warning-level message."""
        if self.level <= logging.WARNING:
            self._log("WARNING", message, context, "yellow", system_only=system_only)

    def error(self, message: str, **context: Any) -> None:
        """Log error-level message."""
        if self.level <= logging.ERROR:
            self._log("ERROR", message, context, "red")

    def debug(self, message: str, **context: Any) -> None:
        """Log debug-level message."""
        if self.level <= logging.DEBUG:
            self._log("DEBUG", message, context, "dim")

    def _log(self, level: str, message: str, context: Dict[str, Any], color: str, system_only: bool = False) -> None:
        """
        Internal logging method.

        Logs to console (Rich formatted), plain text file, and JSONL file.

        Supports multi-line messages: if message contains newlines, each line
        is logged separately with proper indentation.
        """
        timestamp = time.time()

        # Check if message contains newlines (multi-line)
        if "\n" in message:
            lines = message.split("\n")
            # Log first line with full context
            self._log_single_line(level, lines[0], context, color, timestamp, system_only=system_only)
            # Log subsequent lines with indentation, no context
            for line in lines[1:]:
                if line:  # Skip empty lines
                    # Add indentation prefix (2 spaces)
                    indented_line = f"  {line}"
                    self._log_single_line(level, indented_line, {}, color, timestamp, system_only=system_only)
        else:
            # Single-line message
            self._log_single_line(level, message, context, color, timestamp, system_only=system_only)

        # JSONL file output (structured JSON, always single entry per log call)
        if self.jsonl_enabled and self.jsonl_log_file:
            self._write_jsonl_log(level, message, context, timestamp)

        # Extra JSONL files
        extra_allowed = self._extra_log_levels is None or level in self._extra_log_levels
        if not system_only and extra_allowed:
            for extra_jsonl in self._extra_jsonl_log_files:
                self._write_jsonl_log_to(extra_jsonl, level, message, context, timestamp)

    def _log_single_line(
        self, level: str, message: str, context: Dict[str, Any], color: str, timestamp: float,
        system_only: bool = False
    ) -> None:
        """Log a single line to console and text file."""
        # Format message with context
        formatted_message = self._format_message(message, context)

        # Console output (using Python logger)
        if self.console_enabled:
            log_level = getattr(logging, level.upper())
            self.py_logger.log(log_level, formatted_message)

        # Plain text file output (write directly to avoid duplication)
        if self.file_enabled and self.text_log_file:
            self._write_text_log(formatted_message, level, timestamp)

        # Extra text files (skip when system_only to avoid writing to user-visible dirs)
        extra_allowed = self._extra_log_levels is None or level in self._extra_log_levels
        if not system_only and extra_allowed:
            for extra_text in self._extra_text_log_files:
                self._write_text_log_to(extra_text, formatted_message, level, timestamp)

    def _format_iteration_header(
        self, iteration: int, island_id: Optional[int], max_iterations: Optional[int]
    ) -> str:
        """Format iteration header consistently."""
        if max_iterations is None:
            if island_id is not None:
                return f"[Island {island_id} | Iter {iteration}]"
            return f"[Iter {iteration}]"

        if island_id is not None:
            return f"[Island {island_id} | Iter {iteration}/{max_iterations}]"
        return f"[Iter {iteration}/{max_iterations}]"

    def _format_program_summary(self, program: Program) -> str:
        """
        Format program as concise summary (no code).

        Includes: program_id, generation, iteration, island, score, metrics, error_info
        """
        parts = [f"ID={program.id}"]

        if program.generation is not None:
            parts.append(f"gen={program.generation}")

        if program.iteration is not None:
            parts.append(f"iter={program.iteration}")

        # Extract island from metadata
        island_id = program.meta.get("assigned_island")
        if island_id is not None:
            parts.append(f"island={island_id}")

        if program.combined_score is not None:
            parts.append(f"score={program.combined_score:.4f}")

        if program.validity is not None:
            parts.append(f"validity={program.validity:.2f}")

        if program.metrics:
            metrics_str = ", ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in program.metrics.items()
            )
            parts.append(f"metrics=[{metrics_str}]")

        if program.has_error_details:
            # Truncate long errors for readability
            error_preview = (
                program.normalized_error_info[:100] + "..."
                if len(program.normalized_error_info) > 100
                else program.normalized_error_info
            )
            parts.append(f"error={error_preview}")

        return ", ".join(parts)

    def _format_message(self, message: str, context: Dict[str, Any]) -> str:
        """
        Format message with context (no Rich markup).

        Detects iteration/island context and formats concisely.
        Returns plain text string with context.
        """
        if not context:
            return message

        # Make a copy of context to avoid modifying original
        ctx = dict(context)

        # Detect and format iteration context
        iteration_header = ""
        if "iteration" in ctx:
            iteration = ctx.pop("iteration")
            island_id = ctx.pop("island_id", None)
            max_iterations = ctx.pop("max_iterations", None)
            iteration_header = self._format_iteration_header(iteration, island_id, max_iterations)

        # Format remaining context (plain text, no colors)
        # Keys that should never be truncated (basic metadata)
        no_truncate_keys = {"error", "experiment_id", "program_id", "parent_id", "child_id"}
        ctx_parts = []
        if ctx:
            for k, v in sorted(ctx.items()):
                if isinstance(v, float):
                    formatted = f"{k}={v:.4f}"
                elif isinstance(v, int):
                    formatted = f"{k}={v}"
                elif isinstance(v, str):
                    # Never truncate important keys (errors, IDs, metadata)
                    if k.lower() in no_truncate_keys or "error" in k.lower() or len(v) <= 200:
                        formatted = f"{k}={v}"
                    else:
                        # Truncate very long non-essential strings (>200 chars)
                        formatted = f"{k}={v[:197]}..."
                else:
                    formatted = f"{k}={v}"
                ctx_parts.append(formatted)

        # Assemble final message
        parts = []
        if iteration_header:
            parts.append(iteration_header)
        if ctx_parts:
            parts.append(f"[{' '.join(ctx_parts)}]")
        parts.append(message)

        return " ".join(parts)

    def _write_text_log(self, message: str, level: str, timestamp: float) -> None:
        """
        Write message to plain text log file with timestamp and level.

        Args:
            message: The formatted message (already includes context)
            level: Log level (INFO, WARNING, ERROR, DEBUG)
            timestamp: Unix timestamp
        """
        if not self.text_log_file:
            return
        self._write_text_log_to(self.text_log_file, message, level, timestamp)

    def _write_text_log_to(self, text_file: Path, message: str, level: str, timestamp: float) -> None:
        """Write message to a specific text log file."""
        try:
            dt = datetime.datetime.fromtimestamp(timestamp)
            ts_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            log_line = f"{ts_str} {level:8s} {message}"
            with open(text_file, "a") as f:
                f.write(log_line + "\n")
        except Exception:
            pass

    def _write_jsonl_log(
        self, level: str, message: str, context: Dict[str, Any], timestamp: float
    ) -> None:
        """
        Write structured log entry to JSONL file.

        Format: One JSON object per line, easy to parse with jq, pandas, etc.
        """
        if not self.jsonl_log_file:
            return
        self._write_jsonl_log_to(self.jsonl_log_file, level, message, context, timestamp)

    def _write_jsonl_log_to(
        self, jsonl_file: Path, level: str, message: str, context: Dict[str, Any], timestamp: float
    ) -> None:
        """Write structured log entry to a specific JSONL file."""
        log_entry = {
            "timestamp": timestamp,
            "datetime": datetime.datetime.fromtimestamp(timestamp).isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
        }

        if self.experiment_id:
            log_entry["experiment_id"] = self.experiment_id

        log_entry.update(context)

        try:
            with open(jsonl_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass

    def program(self, program: Program, title: Optional[str] = None) -> None:
        """
        Log program metadata (no code snippets).

        Logs concise program summary with ID, generation, iteration, island,
        score, metrics, and error_info (if present).
        """
        timestamp = time.time()

        # Format program summary
        summary = self._format_program_summary(program)

        # Determine log level based on score and errors
        if program.is_buggy:
            log_level = logging.ERROR
            level_name = "ERROR"
        elif program.combined_score is not None:
            if program.combined_score >= 0.5:
                log_level = logging.INFO
                level_name = "INFO"
            elif program.combined_score >= 0.3:
                log_level = logging.WARNING
                level_name = "WARNING"
            else:
                log_level = logging.WARNING
                level_name = "WARNING"
        else:
            log_level = logging.INFO
            level_name = "INFO"

        # Build message with title
        title_str = title or "Program"
        message = f"{title_str}: {summary}"

        # Console output (using Python logger)
        if self.console_enabled:
            self.py_logger.log(log_level, message)

        # Plain text file output
        if self.file_enabled and self.text_log_file:
            self._write_text_log(message, level_name, timestamp)
        for extra_text in self._extra_text_log_files:
            self._write_text_log_to(extra_text, message, level_name, timestamp)

        # JSONL file output
        if self.jsonl_enabled and self.jsonl_log_file:
            self._log_program_jsonl(program, title)
        for extra_jsonl in self._extra_jsonl_log_files:
            self._log_program_jsonl_to(extra_jsonl, program, title)

    def _log_program_jsonl_to(self, jsonl_file: Path, program: Program, title: Optional[str] = None) -> None:
        """Log program details to a specific JSONL file."""
        log_entry = {
            "timestamp": time.time(),
            "datetime": datetime.datetime.now().isoformat(),
            "event": "program_display",
            "logger": self.name,
            "title": title,
            "program_id": program.id,
            "generation": program.generation,
            "iteration": program.iteration,
            "combined_score": program.combined_score,
            "validity": program.validity,
            "has_error": program.is_buggy,
            "error_info": program.normalized_error_info,
            "metrics": program.metrics,
        }

        if "assigned_island" in program.meta:
            log_entry["island_id"] = program.meta["assigned_island"]

        if self.experiment_id:
            log_entry["experiment_id"] = self.experiment_id

        try:
            with open(jsonl_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass

    def stats(self, stats: Dict[str, Any], title: str = "Statistics") -> None:
        """
        Display statistics.

        Console: Plain text list
        Text file: Plain text list
        JSONL file: Structured JSON
        """
        if not stats:
            return

        # Console output (plain text list)
        if self.console_enabled:
            self._display_stats_console(stats, title)

        # Plain text file output
        if self.file_enabled and self.text_log_file:
            self._write_stats_text(stats, title)
        for extra_text in self._extra_text_log_files:
            self._write_stats_text_to(extra_text, stats, title)

        # JSONL file output
        if self.jsonl_enabled and self.jsonl_log_file:
            self._log_stats_jsonl(stats, title)
        for extra_jsonl in self._extra_jsonl_log_files:
            self._log_stats_jsonl_to(extra_jsonl, stats, title)

    def _display_stats_console(self, stats: Dict[str, Any], title: str) -> None:
        """Display stats as plain text list."""
        # Log title
        self.py_logger.info(f"=== {title} ===")

        # Log each stat
        for key, value in stats.items():
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            elif isinstance(value, int):
                formatted_value = f"{value:,}"
            else:
                formatted_value = str(value)
            self.py_logger.info(f"  {key}: {formatted_value}")

    def _write_stats_text(self, stats: Dict[str, Any], title: str) -> None:
        """Write stats to plain text file."""
        if not self.text_log_file:
            return
        self._write_stats_text_to(self.text_log_file, stats, title)

    def _write_stats_text_to(self, text_file: Path, stats: Dict[str, Any], title: str) -> None:
        """Write stats to a specific text file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        lines = [
            f"{timestamp} INFO    === {title} ===",
        ]

        for key, value in stats.items():
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            elif isinstance(value, int):
                formatted_value = f"{value:,}"
            else:
                formatted_value = str(value)
            lines.append(f"{timestamp} INFO      {key}: {formatted_value}")

        try:
            with open(text_file, "a") as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            pass

    def _log_stats_jsonl(self, stats: Dict[str, Any], title: str) -> None:
        """Log stats to JSONL file."""
        if not self.jsonl_log_file:
            return
        self._log_stats_jsonl_to(self.jsonl_log_file, stats, title)

    def _log_stats_jsonl_to(self, jsonl_file: Path, stats: Dict[str, Any], title: str) -> None:
        """Log stats to a specific JSONL file."""
        log_entry = {
            "timestamp": time.time(),
            "datetime": datetime.datetime.now().isoformat(),
            "event": "statistics",
            "logger": self.name,
            "title": title,
            "stats": stats,
        }

        if self.experiment_id:
            log_entry["experiment_id"] = self.experiment_id

        try:
            with open(jsonl_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass

    def get_log_path(self) -> Dict[str, Optional[Path]]:
        """Get the log file paths."""
        return {
            "text": self.text_log_file,
            "jsonl": self.jsonl_log_file,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LocalLogger(name={self.name}, "
            f"experiment_id={self.experiment_id}, "
            f"text_log={self.text_log_file}, "
            f"jsonl_log={self.jsonl_log_file})"
        )
