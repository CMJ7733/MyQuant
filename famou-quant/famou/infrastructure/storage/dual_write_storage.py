"""
Dual-write storage implementation.

Writes data to both base (system) and user paths simultaneously.
User path stores reference-only metadata filtered by USER_VISIBLE_PATHS whitelist.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from famou.core.data import Program

# Import FamouConfig for type hints (avoid circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from famou.config.settings import FamouConfig

# JSONPath-style whitelist for user-visible fields.
# Only fields listed here will be written to the user path.
# Nested paths use dot notation (e.g. "meta.experiment_id").
USER_VISIBLE_PATHS: Dict[str, List[str]] = {
    "program": [
        "$.id",
        "$.generation",
        "$.iteration",
        "$.parent_id",
        "$.created_at",
        "$.language",
        "$.file_extension",
        "$.combined_score",
        "$.validity",
        "$.error_info",
        "$.metrics",
        "$.meta.experiment_id",
        "$.meta.island_id",
    ],
    "result": [
        "$.rollout_id",
        "$.experiment_id",
        "$.iteration",
        "$.island_id",
        "$.status",
        "$.failed_module",
        "$.error_message",
        "$.selection.parent_id",
        "$.selection.inspiration_ids",
        "$.generated_program_id",
        "$.stats.score",
        "$.stats.validity",
        "$.stats.has_error",
        "$.stats.generation",
        "$.stats.execution_time",
        "$.stats.total_module_time",
        "$.stats.total_retries",
        "$.created_at",
        "$.completed_at",
    ],
}


class DualWriteLocalStorage:
    """
    Dual-write storage that writes to both base (system) and user paths.

    - Base path (system): Full data with all fields
    - User path: Reference-only metadata for programs (no code, metrics, etc.)

    Implements: DataService protocol (structural subtyping)

    Directory Structure:
        famou_system/                  # Full data (base_path)
        ├── {experiment_id}/
        │   ├── programs/
        │   │   └── {program_id}.json  # Contains all fields
        │   │   └── {program_id}.py    # Code file
        │   ├── results/
        │   │   └── {rollout_id}.json
        │   ├── config.yaml
        │   └── experiment_checkpoint_*.json
        │
        famou_data/                    # Reference data (user_path)
        ├── {experiment_id}/
        │   ├── programs/
        │   │   └── {program_id}.json  # Reference metadata only
        │   │   └── {program_id}.py    # Code file
        │   └── results/
        │       └── {rollout_id}.json
    """

    # Language to file extension mapping
    LANGUAGE_EXTENSIONS: Dict[str, str] = {
        "python": ".py",
        "javascript": ".js",
        "java": ".java",
        "cpp": ".cpp",
        "go": ".go",
        "rust": ".rs",
    }

    def __init__(
        self,
        base_path: str = "./famou_system",
        user_path: str = "./famou_data",
        precise_path: Optional[str] = None,
        precise_user_path: Optional[str] = None,
        llm_request_log_dir: Optional[str] = None,
    ):
        """
        Initialize dual-write storage.

        Args:
            base_path: Path for full data storage (system directory)
            user_path: Path for redacted data storage (user directory)
            precise_path: Exact output directory for system storage (bypasses experiment_id subdir)
            precise_user_path: Exact output directory for user storage (bypasses experiment_id subdir)
            llm_request_log_dir: Dedicated directory for llm_requests.log
        """
        # Import here to avoid circular import
        from famou.infrastructure.storage.local_storage import LocalStorage

        # When precise_path is set, use it as base_path (for fallback when not using precise)
        self.base_path = Path(precise_path) if precise_path else Path(base_path)
        self.user_path = Path(precise_user_path) if precise_user_path else Path(user_path)
        self.precise_path = Path(precise_path) if precise_path else None
        self.precise_user_path = Path(precise_user_path) if precise_user_path else None
        self.llm_request_log_dir = llm_request_log_dir

        # Both directories need to exist for proper operation (only if not using precise paths)
        if self.precise_path is None:
            self.base_path.mkdir(parents=True, exist_ok=True)
        if self.precise_user_path is None:
            self.user_path.mkdir(parents=True, exist_ok=True)

        # Create LocalStorage instances for both paths
        # Pass precise_path as base_path so LocalStorage can handle experiment_id subdir correctly
        system_base = str(precise_path) if precise_path else str(base_path)
        self.system_storage = LocalStorage(base_path=system_base, precise_path=precise_path)
        user_base = str(precise_user_path) if precise_user_path else str(user_path)
        self.user_storage = LocalStorage(base_path=user_base, precise_path=precise_user_path)

    def _filter_by_paths(self, data: Dict, paths: List[str]) -> Dict:
        """
        Filter a dict by JSONPath-style whitelist.

        Args:
            data: Source dictionary to filter
            paths: List of JSONPath patterns like "$.id", "$.meta.experiment_id"

        Returns:
            Filtered dict containing only the whitelisted paths
        """
        result: Dict[str, Any] = {}
        for path in paths:
            # Strip leading "$.", e.g. "$.meta.experiment_id" -> "meta.experiment_id"
            parts = path[2:].split(".") if path.startswith("$.") else path.split(".")
            current = data
            for i, part in enumerate(parts):
                if not isinstance(current, dict) or part not in current:
                    break
                if i == len(parts) - 1:
                    # Last segment: set the value, building intermediate dicts as needed
                    target = result
                    for j in range(len(parts) - 1):
                        key = parts[j]
                        if key not in target:
                            target[key] = {}
                        target = target[key]
                    target[part] = current[part]
                else:
                    current = current[part]
        return result

    def _get_code_file_extension(self, language: str, program: Optional[Program] = None) -> str:
        """Get file extension for a given language."""
        # Check for custom extension in program meta
        if program and program.file_extension:
            return program.file_extension
        return self.LANGUAGE_EXTENSIONS.get(language.lower(), ".py")

    def _get_programs_dir(self, experiment_id: str, is_system: bool = False) -> Path:
        """Get programs directory for experiment."""
        if is_system and self.precise_path is not None:
            base = self.precise_path
        elif not is_system and self.precise_user_path is not None:
            base = self.precise_user_path
        else:
            base = self.base_path if is_system else self.user_path

        # When using precise paths, don't add experiment_id subdirectory
        if (is_system and self.precise_path is not None) or (not is_system and self.precise_user_path is not None):
            programs_dir = base / "programs"
        else:
            programs_dir = base / experiment_id / "programs"
        programs_dir.mkdir(parents=True, exist_ok=True)
        return programs_dir

    def _get_results_dir(self, experiment_id: str, is_system: bool = False) -> Path:
        """Get results directory for experiment."""
        if is_system and self.precise_path is not None:
            base = self.precise_path
        elif not is_system and self.precise_user_path is not None:
            base = self.precise_user_path
        else:
            base = self.base_path if is_system else self.user_path

        # When using precise paths, don't add experiment_id subdirectory
        if (is_system and self.precise_path is not None) or (not is_system and self.precise_user_path is not None):
            results_dir = base / "results"
        else:
            results_dir = base / experiment_id / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    def _get_experiment_dir(self, experiment_id: str, is_system: bool = False) -> Path:
        """Get experiment directory."""
        if is_system and self.precise_path is not None:
            base = self.precise_path
        elif not is_system and self.precise_user_path is not None:
            base = self.precise_user_path
        else:
            base = self.base_path if is_system else self.user_path

        # When using precise paths, don't add experiment_id subdirectory
        if (is_system and self.precise_path is not None) or (not is_system and self.precise_user_path is not None):
            exp_dir = base
        else:
            exp_dir = base / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    # ========================================================================
    # Program Storage (Dual-Write)
    # ========================================================================

    def save_program(self, program: Program) -> None:
        """
        Save program to both base (system) and user paths.

        Base path (famou_system): Full program with all fields
        User path (famou_data): Reference-only metadata (no code, metrics, etc.)
        """
        experiment_id = program.meta.get("experiment_id")
        if not experiment_id:
            raise ValueError("Program must have experiment_id in meta")

        # 1. Save FULL data to system path
        self.system_storage.save_program(program)

        # 2. Save reference-only data to user path
        programs_dir = self._get_programs_dir(experiment_id, is_system=False)
        program_stem = self.system_storage._format_program_storage_stem(program)

        # Filter by JSONPath whitelist
        program_data = program.model_dump(mode='json')
        reference_data = self._filter_by_paths(program_data, USER_VISIBLE_PATHS["program"])

        # Save reference JSON
        program_file = programs_dir / f"{program_stem}.json"
        program_file.write_text(json.dumps(reference_data, indent=2, ensure_ascii=False))

        # Save code file (same for both paths)
        extension = self._get_code_file_extension(str(program.language), program)
        code_file = programs_dir / f"{program_stem}{extension}"
        code_file.write_text(program.code)

    def load_program(self, program_id: str) -> Program:
        """Load from system path (full data)."""
        return self.system_storage.load_program(program_id)

    def list_programs(
        self,
        experiment_id: str,
        iteration: Optional[int] = None,
        generation: Optional[int] = None,
    ) -> List[Program]:
        """List from system path (full data)."""
        return self.system_storage.list_programs(experiment_id, iteration, generation)

    # ========================================================================
    # RolloutResult Storage (Dual-Write)
    # ========================================================================

    def save_result(self, result) -> None:
        """Save to system path only (results are not needed in user path)."""
        self.system_storage.save_result(result)

    def load_result(self, rollout_id: str):
        """Load from system path (full data)."""
        return self.system_storage.load_result(rollout_id)

    # ========================================================================
    # Experiment Storage (Dual-Write)
    # ========================================================================

    def save_experiment(self, experiment) -> None:
        """Save to system path only (checkpoint files are not needed in user path)."""
        self.system_storage.save_experiment(experiment)

    def load_experiment(self, experiment_id: str, iteration: int = -1):
        """Load from system path (full data)."""
        return self.system_storage.load_experiment(experiment_id, iteration)

    # ========================================================================
    # Config Storage (Dual-Write)
    # ========================================================================

    def save_config(self, experiment_id: str, config: "FamouConfig") -> None:
        """Save config to system path only (config files are not needed in user path)."""
        self.system_storage.save_config(experiment_id, config)

    def load_config(self, experiment_id: str) -> "FamouConfig":
        """Load config from system path (full config)."""
        return self.system_storage.load_config(experiment_id)

    def save_json_artifact(self, experiment_id: str, filename: str, payload: dict) -> None:
        """Save JSON artifact to system path only (internal data, not needed in user path)."""
        self.system_storage.save_json_artifact(experiment_id, filename, payload)

    def __repr__(self) -> str:
        return (
            f"DualWriteLocalStorage(base={self.base_path}, user={self.user_path}, "
            f"precise_path={self.precise_path}, precise_user_path={self.precise_user_path})"
        )

    def get_experiment_dir(self, experiment_id: str) -> Path:
        """Return the resolved experiment directory for system path without mutating contents."""
        if self.precise_path is not None:
            return self.precise_path
        return self.base_path / experiment_id
