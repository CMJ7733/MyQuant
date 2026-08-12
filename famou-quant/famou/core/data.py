"""
Core data models for Famou 2.0.

Defines the main data structures:
- Program: A single program/solution with code, metadata, and evaluation results
- Context: Experiment-level context (shared across rollouts, mostly read-only)
- RolloutResult: Per-rollout state (built progressively by modules during execution)
- Experiment: Global experiment state for tracking evolution progress
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Set, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from famou.core.types import Language, RolloutStatus
from famou.utils.id_gen import generate_program_id, get_timestamp

if TYPE_CHECKING:
    from famou.config.settings import ExperimentConfig
    from famou.core.protocol import Module

# Import at runtime for Pydantic (no circular dependency since accessors only imports Program in TYPE_CHECKING)
from famou.core.accessors import IslandAccessor, PopulationAccessor, StateAccessor
from famou.core.state import StateStore


_EMPTY_ERROR_INFO_TEXT = {"", "{}", "[]", "null", "none"}


def normalize_error_info_value(value: Any) -> Optional[str]:
    """Normalize evaluator error payloads into a display-ready string or None."""
    if value is None:
        return None

    if isinstance(value, dict):
        if not value:
            return None
        text = json.dumps(value, ensure_ascii=False)
    elif isinstance(value, list):
        if not value:
            return None
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value).strip()

    text = text.strip()
    if text.lower() in _EMPTY_ERROR_INFO_TEXT:
        return None
    return text


class StateUpdate(BaseModel):
    """A single deferred state mutation collected by StateUpdateCollector."""
    kind: Literal["set_global", "set_island", "delete_global", "delete_island"]
    keys: Tuple[str, ...]
    value: Any = None
    island_id: Optional[int] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class StateUpdateCollector:
    """
    Declarative collector for state mutations during a rollout.

    Modules call set/set_island/delete/delete_island on this collector
    instead of writing directly to StateStore. The Evolver applies
    collected updates after the rollout completes.
    """

    def __init__(self, island_id: int):
        self._island_id = island_id
        self._updates: List[StateUpdate] = []

    def set(self, *keys: str, value: Any) -> None:
        """Record a global state set operation."""
        self._updates.append(StateUpdate(kind="set_global", keys=keys, value=value))

    def set_island(self, *keys: str, value: Any) -> None:
        """Record an island state set operation (island_id bound at creation)."""
        self._updates.append(StateUpdate(kind="set_island", keys=keys, value=value, island_id=self._island_id))

    def delete(self, *keys: str) -> None:
        """Record a global state delete operation."""
        self._updates.append(StateUpdate(kind="delete_global", keys=keys))

    def delete_island(self, *keys: str) -> None:
        """Record an island state delete operation."""
        self._updates.append(StateUpdate(kind="delete_island", keys=keys, island_id=self._island_id))

    @property
    def updates(self) -> List[StateUpdate]:
        """Return the ordered list of collected updates."""
        return list(self._updates)

    def __repr__(self) -> str:
        return f"StateUpdateCollector(island_id={self._island_id}, pending={len(self._updates)})"


class ModuleStat(BaseModel):
    """
    Statistics for a single module execution within a rollout.

    Tracks execution time, success/failure status, and retry information
    for debugging, analytics, and performance optimization.

    Fields:
        module_name: Name of the module (e.g., "TopKSelect", "MutationGenerate")
        execution_time: Total execution time in seconds
        success: Whether the module completed successfully
        error_message: Error message if module failed (None if success)
        retry_count: Number of retry attempts before success/failure (0 = no retries)
        langfuse_observation_id: Langfuse span/observation ID for this module (None if Langfuse disabled)
    """

    module_name: str
    execution_time: float = Field(ge=0, description="Execution time in seconds")
    success: bool = True
    error_message: Optional[str] = None
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")
    langfuse_observation_id: Optional[str] = Field(default=None, description="Langfuse observation ID")

    model_config = ConfigDict(validate_assignment=True)


class Program(BaseModel):
    """
    A single program/solution in the evolutionary process.

    Programs are mutable during creation/enrichment, then treated as immutable
    after being added to archive/population. Programs are shared across archive
    and multiple island populations for memory efficiency.

    Fields:
        id: Unique identifier (format: {iteration}_{generation}_{uuid})
        code: The program's source code
        generation: Genealogical depth (0=seed, 1=child of seed, 2=grandchild, etc.)
        iteration: When this program was created (0, 1, 2, ...)
        path: Optional file path if program is stored separately

        language: Programming language (default: python)
        meta: Additional metadata (flexible dict)
        created_at: Unix timestamp when program was created
        parent_id: Parent program ID (for tracking genealogy)

        system_prompt: System prompt sent to LLM (if generated by LLM)
        prompt: User prompt sent to LLM (if generated by LLM)
        response: The full LLM response
        thinking: LLM's reasoning/thinking (for models like o1)

        combined_score: Final combined fitness score (set by EvaluateModule)
        metrics: Dictionary of evaluation metrics (set by EvaluateModule)
        validity: Program validity score 0.0-1.0 from user-evaluator (set by EvaluateModule)
        error_info: Error information if execution failed (set by EvaluateModule)

        llm_feedback: Optional LLM code review feedback (set by JudgeModule)

        feature_vector: Unified feature vector for similarity/diversity (set by JudgeModule)
        post_time: Rollout-side timing summary mirrored onto Program for per-program analysis
    """

    # Core identity
    id: str = Field(default_factory=lambda: generate_program_id(0, 0))
    code: str
    generation: int = Field(ge=0, description="Genealogical depth (0=seed, 1=child, ...)")
    iteration: int = Field(ge=0, description="When created (0, 1, 2, ...)")
    path: Optional[str] = None

    # Metadata
    language: Language = Language.PYTHON
    file_extension: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=get_timestamp)
    parent_id: Optional[str] = None 

    # LLM tracking (for programs generated by LLM)
    system_prompt: Optional[str] = None  # System prompt sent to LLM
    prompt: Optional[str] = None  # User prompt sent to LLM
    response: Optional[str] = None  # Full LLM response
    thinking: Optional[str] = None  # For o1-style models

    # Required packages (extracted from LLM response for Python programs)
    required_packages: Optional[List[str]] = None  # List of required package names

    # Evaluation results (set by EvaluateModule ONLY)
    combined_score: Optional[float] = None  # Overall fitness score from user-evaluator
    metrics: Dict[str, Any] = Field(default_factory=dict)  # Evaluation metrics from user-evaluator
    validity: Optional[float] = None  # Program validity score 0.0-1.0 from user-evaluator
    error_info: Optional[str] = None  # Error information from user-evaluator

    # LLM feedback (set by JudgeModule - optional)
    llm_feedback: Optional[Dict[str, Any]] = None  # LLM code review feedback

    # Feature vector (set by JudgeModule - optional)
    feature_vector: Optional[List[float]] = None  # Unified feature vector for similarity/diversity

    # Rollout timing snapshot mirrored from RolloutResult.stats for alignment
    post_time: Optional[Dict[str, Any]] = None

    # Other info for personalized strategy and others
    meta: Dict[str, Any] = Field(default_factory=dict)


    # Validation and Documentation
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "id": "0_0_a1b2c3d4",
                "code": "def sort(arr):\n    return sorted(arr)",
                "generation": 0,
                "iteration": 0,
                "combined_score": 0.95,
                "validity": 1.0,
            }
        },
    )

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        return (
            f"Program(id={self.id}, gen={self.generation}, "
            f"iter={self.iteration}, score={self.combined_score})"
        )

    @property
    def is_buggy(self) -> bool:
        """Whether this program is buggy (validity < 1.0)."""
        return self.validity is not None and self.validity < 1.0

    @property
    def normalized_error_info(self) -> Optional[str]:
        """Return a normalized error string when meaningful details exist."""
        return normalize_error_info_value(self.error_info)

    @property
    def has_error_details(self) -> bool:
        """Whether this program has meaningful error text to display."""
        return self.normalized_error_info is not None

    @property
    def stdout(self) -> Optional[str]:
        """Standard output from program execution (stored in meta)."""
        return self.meta.get("stdout")

    @property
    def operator(self) -> Optional[str]:
        """The operator/module that created this program (stored in meta)."""
        return self.meta.get("operator")

    @property
    def plan(self) -> Optional[str]:
        """Strategy/plan description that generated this code (stored in meta)."""
        return self.meta.get("plan")


class Context(BaseModel):
    """
    Experiment-level context shared across all rollouts.

    This is owned by the Experiment and contains global information that
    rollouts need to READ from (but not modify). Think of it as the
    "environment" for rollouts.

    Fields:
        experiment_id: Unique experiment identifier

        task_description: Description of the optimization task/problem

        accessor: Read-only accessor for island population queries (O(1) lookups)
        island_accessor: Read-only accessor for island-visible programs with lineage tracking
        state: Shared state store for stateful strategies (mutable, thread-safe)

        experiment_config: Full ExperimentConfig for accessing settings like island_size
        metadata: Additional flexible metadata

        created_at: When context was created
        updated_at: Last update time (e.g., when population changed)

    Note:
        With the prompt registry system, system-level instructions are now in templates.
        Only the task description is needed in Context - templates handle the rest.
        The experiment_config provides access to all experiment settings (island_size,
        migration parameters, etc.) for modules that need them.

        State Access Patterns:
        - context.accessor: Read-only population snapshot (IMMUTABLE, thread-safe)
        - context.island_accessor: Read-only island-visible programs (IMMUTABLE, enforces isolation)
        - context.state: Shared mutable state store (MUTABLE, thread-safe with locks)

        For backward compatibility, the population field is still available.
    """

    # Identity
    experiment_id: str

    # Task description (used by generation/judge modules via templates)
    task_description: str = Field(
        description="Description of the optimization task/problem to solve"
    )
    # Population accessor (preferred way to access population)
    island_id: int = Field(
        default=0,
        ge=0,
        description="Island ID this context snapshot belongs to"
    )
    accessor: Optional[PopulationAccessor] = Field(
        default=None,
        description="Read-only accessor for efficient population queries with O(1) ID lookups"
    )

    # Island accessor (island-scoped access with visibility filtering)
    island_accessor: Optional["IslandAccessor"] = Field(
        default=None,
        description="Read-only accessor for island-visible programs with lineage tracking"
    )

    # Shared state store (immutable snapshot, serialization-safe)
    state: Optional[StateAccessor] = Field(
        default=None,
        description="Immutable state snapshot for read-only access in modules"
    )

    # Configuration
    experiment_config: Optional[ExperimentConfig] = Field(
        default=None,
        description="Full experiment configuration for accessing settings like island_size"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Population diversity (average pairwise cosine distance)
    diversity: float = Field(
        default=0.0,
        description="Population diversity metric (average pairwise cosine distance)"
    )
    iteration: int = Field(
        default=0,
        ge=0,
        description="Current iteration number"
    )

    # Timestamps
    created_at: float = Field(default_factory=get_timestamp)
    updated_at: float = Field(default_factory=get_timestamp)
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    # ==== Properties and Helper Methods ====

    @property
    def language(self) -> str:
        """
        Resolve language from experiment_config.

        Returns:
            Language as string (e.g., "python"), defaults to "python" if not configured
        """
        if self.experiment_config is not None:
            return str(self.experiment_config.language)
        return "python"

    def get_all_programs(self) -> List[Program]:
        """
        Get all programs across buckets in the current island (flattened).

        Uses accessor for efficient access if available, otherwise falls back to population.
        """
        if self.accessor:
            return self.accessor.get_all()

        # Fallback to population for backward compatibility
        all_programs = []
        for bucket_progs in self.population.values():
            all_programs.extend(bucket_progs)
        return all_programs

    def get_program_by_id(self, program_id: str) -> Optional[Program]:
        """
        Find a program by ID across buckets in the current island.

        Uses accessor for O(1) lookup if available, otherwise falls back to O(n) search.
        """
        if self.accessor:
            return self.accessor.get_by_id(program_id)

        # Fallback to population for backward compatibility
        for bucket_progs in self.population.values():
            for program in bucket_progs:
                if program.id == program_id:
                    return program
        return None

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        total_programs = len(self.get_all_programs())
        return (
            f"Context(exp={self.experiment_id}, "
            f"island_id={self.island_id}, "
            f"programs={total_programs})"
        )


class SelectionData(BaseModel):
    """
    Structured output from SelectModule.

    This class encapsulates all selection results in a type-safe, extensible way.
    It replaces the previous separate fields (selected_parent_id, inspiration_ids)
    with a structured object that can grow with new features.

    Core Fields (current):
        parent_id: ID of the program to modify/mutate
        inspiration_ids: Additional program IDs for LLM reference

    Extensible Fields (future):
        experiences: Selected experience IDs for reuse in generation
        skills: Agent skills to apply during generation
        memories: Retrieved memories for context-aware generation

    The `extra` field allows for forward compatibility - new selection types
    can be added without breaking existing code.

    Example:
        >>> # Simple selection
        >>> selection = SelectionData(parent_id="prog_123")
        >>>
        >>> # Selection with inspirations
        >>> selection = SelectionData(
        ...     parent_id="prog_123",
        ...     inspiration_ids=["prog_456", "prog_789"]
        ... )
        >>>
        >>> # Future: selection with experiences
        >>> selection = SelectionData(
        ...     parent_id="prog_123",
        ...     inspiration_ids=["prog_456"],
        ...     experiences=["exp_fixing_bugs", "exp_optimization"]
        ... )
    """

    # Core selection (required)
    parent_id: str = Field(description="ID of program to modify/mutate")
    inspiration_ids: Optional[List[str]] = Field(
        default_factory=None,
        description="Additional program IDs for LLM reference/examples"
    )

    # Extensible: future selection types
    experiences: Optional[List[Any]] = Field(
        default=None,
        description="Selected experience IDs for reuse in generation"
    )
    knowledge: Optional[List[Any]] = Field(
        default=None,
        description="Selected knowledge IDs for reuse in generation"
    )
    # Catch-all for unknown future needs (forward compatibility)
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional selection data (extensible)"
    )

    model_config = ConfigDict(validate_assignment=True)

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        return (
            f"SelectionData(parent={self.parent_id}, "
            f"inspirations={len(self.inspiration_ids)}, "
            f"experiences={len(self.experiences) if self.experiences else 0})"
        )


class RolloutResult(BaseModel):
    """
    Per-rollout state built progressively by modules.

    **One Program Per Rollout** for Now.

    Each rollout gets its own RolloutResult instance that modules
    mutate as they execute. This is the "working document" that
    gets built up through the pipeline.

    Pipeline progression:
    1. SelectModule: Reads context.accessor → Writes result.selection (SelectionData)
    2. GenerateModule: Reads result.selection.parent_id → Writes result.generated_program (ONE Program)
    3. EvaluateModule: Enriches result.generated_program → Adds metrics, validity, combined_score, error_info
    4. JudgeModule: Enriches result.generated_program → Adds llm_feedback (optional)
    5. Final: Compute result.stats

    Fields:
        rollout_id: Unique identifier for this rollout
        iteration: Iteration number
        rollout_name: Name of the Rollout pipeline that was executed (from Rollout.name)

        selection: Selection output from SelectModule (parent_id, inspirations, experiences, skills)
        generated_program: Single program created and enriched by modules

        stats: Statistics about execution (score, status, timing)

        created_at: When rollout started
        completed_at: When rollout finished (None if still running)
    """

    # Identity
    rollout_id: str
    experiment_id: Optional[str] = Field(
        default=None,
        description="Experiment ID this rollout belongs to"
    )
    iteration: int = Field(ge=0)
    island_id: int = Field(
        default=0,
        ge=0,
        description="Island ID this rollout belongs to (0 for single-island mode)"
    )
    rollout_attempt: int = Field(
        default=1,
        ge=1,
        description="Attempt number for the same formal iteration"
    )
    rollout_name: str = Field(
        default="",
        description="Name of the Rollout pipeline that was executed (from Rollout.name)"
    )

    # Working state (built progressively by modules)
    selection: Optional[SelectionData] = Field(
        default=None,
        description=(
            "Selection output from SelectModule containing parent_id, "
            "inspiration_ids, and extensible fields (experiences, skills, etc.)"
        )
    )

    # Single generated program (enriched in place by Evaluate/Judge modules)
    generated_program: Optional[Program] = Field(
        default=None,
        description="Single program created by GenerateModule, enriched by later modules"
    )

    # Rollout status (set by RolloutEngine)
    status: RolloutStatus = Field(
        default=RolloutStatus.SUCCESS,
        description="Rollout pipeline status: SUCCESS or FAILED"
    )
    failed_module: Optional[str] = Field(
        default=None,
        description="Name of module that caused rollout failure (if failed)"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if rollout failed"
    )

    # Per-module execution statistics
    module_stats: List[ModuleStat] = Field(
        default_factory=list,
        description="Per-module execution statistics (timing, success/failure, retries)"
    )

    # Final outputs (computed at end of rollout)
    stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Statistics: score, execution_status, timing, etc."
    )

    # Timestamps
    created_at: float = Field(default_factory=get_timestamp)
    completed_at: Optional[float] = Field(
        default=None,
        description="When rollout finished (None if still running)"
    )

    # Langfuse tracing (optional observability)
    langfuse_trace_id: Optional[str] = Field(
        default=None,
        description="Langfuse trace ID for this rollout (None if Langfuse disabled)"
    )
    langfuse_trace_url: Optional[str] = Field(
        default=None,
        description="Langfuse trace URL for viewing in UI (None if Langfuse disabled)"
    )

    strategy_state: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional strategy state snapshot captured after this rollout completes. "
            "Used for rollout-level traceability and resume debugging."
        ),
    )
    llm_request_logs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Raw llm_requests.log-style entries collected during this rollout. "
            "Used to replay worker-side LLM requests into driver-local logs."
        ),
    )

    state_updates: Optional[StateUpdateCollector] = Field(
        default=None, exclude=True,
        description="State mutations collected by modules, applied by Evolver after rollout completes"
    )

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    def finalize(self) -> None:
        """
        Finalize the result after all modules have executed.

        Computes:
        - stats (score, validity, timing)
        - feature_vector (default from metrics if not set)
        - completed_at timestamp

        Called by RolloutEngine when pipeline completes.
        """
        # Compute statistics for the single generated program
        if self.generated_program:
            self.stats = {
                "score": self.generated_program.combined_score,
                "validity": self.generated_program.validity,
                "has_error": self.generated_program.is_buggy,
                "generation": self.generated_program.generation,
                "execution_time": (
                    self.completed_at - self.created_at
                    if self.completed_at
                    else time.time() - self.created_at
                ), 
            }

            # Fill default feature_vector if feature is not set. 
            if self.generated_program.feature_vector is None:
                self.generated_program.feature_vector = self._compute_default_features()
        else:
            # No program generated
            self.stats = {
                "score": None,
                "validity": None,
                "has_error": True,
                "generation": None,
                "execution_time": 0.0,
            }

        # Add module stats summary to stats dict
        if self.module_stats:
            self.stats["module_stats"] = [s.model_dump() for s in self.module_stats]
            self.stats["total_module_time"] = sum(
                s.execution_time for s in self.module_stats
            )
            self.stats["total_retries"] = sum(s.retry_count for s in self.module_stats)
            failed_modules = [s.module_name for s in self.module_stats if not s.success]
            if failed_modules:
                self.stats["failed_modules"] = failed_modules

        if self.generated_program:
            self.generated_program.post_time = {
                "execution_time": self.stats.get("execution_time"),
                "total_module_time": self.stats.get("total_module_time"),
                "module_stats": self.stats.get("module_stats", []),
            }

        # Mark as complete
        self.completed_at = get_timestamp()

    def _compute_default_features(self) -> List[float]:
        """
        Compute default feature vector from metrics + combined_score.

        Extracts all numeric values from the program's metrics dictionary
        and prepends combined_score. Used when feature is not set. 
        Returns:
            List of floats: [combined_score, metric1, metric2, ...]
        """
        program = self.generated_program
        features: List[float] = []

        # Add combined_score first
        if program.combined_score is not None:
            features.append(program.combined_score)

        # Add all numeric values from metrics (sorted by key for consistency)
        for key in sorted(program.metrics.keys()):
            value = program.metrics[key]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                features.append(float(value))

        return features

    def to_compact_dict(self) -> Dict[str, Any]:
        """
        Convert RolloutResult to compact representation.

        Stores Program IDs instead of full Program objects to reduce size.
        Full Programs are reconstructed from archive during load.

        Returns:
            Dict with compact representation
        """
        return {
            "rollout_id": self.rollout_id,
            "experiment_id": self.experiment_id,
            "rollout_name": self.rollout_name,
            "iteration": self.iteration,
            "island_id": self.island_id,
            "rollout_attempt": self.rollout_attempt,
            "status": self.status,
            "failed_module": self.failed_module,
            "error_message": self.error_message,
            "langfuse_trace_id": self.langfuse_trace_id,
            "langfuse_trace_url": self.langfuse_trace_url,
            # Store selection as dict (serializes SelectionData)
            "selection": self.selection.model_dump() if self.selection else None,
            "generated_program_id": self.generated_program.id if self.generated_program else None,
            "program": (
                self.generated_program.model_dump(mode="json")
                if self.generated_program
                else None
            ),
            "strategy_state": self.strategy_state,
            "llm_request_logs": self.llm_request_logs,
            # Statistics and timestamps (small)
            "stats": self.stats,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_compact_dict(
        cls, data: Dict[str, Any], archive_dict: Dict[str, Program]
    ) -> "RolloutResult":
        """
        Reconstruct RolloutResult from compact representation.

        Args:
            data: Compact representation from to_compact_dict()
            archive_dict: Dict of all programs (id → Program) for lookup

        Returns:
            Fully reconstructed RolloutResult

        Raises:
            ValueError: If a Program ID is not found in archive_dict
        """
        # Reconstruct selection from dict (handles both new and old format)
        selection = None
        selection_data = data.get("selection")
        if selection_data:
            # New format: selection is a SelectionData dict
            selection = SelectionData(**selection_data)
        else:
            # Backward compatibility: handle old selected_parent_id/inspiration_ids format
            parent_id = data.get("selected_parent_id")
            if parent_id:
                selection = SelectionData(
                    parent_id=parent_id,
                    inspiration_ids=data.get("inspiration_ids", [])
                )

        # Reconstruct generated_program from ID
        generated_program = None
        program_data = data.get("program")
        if program_data:
            generated_program = Program.model_validate(program_data)
        else:
            generated_program_id = data.get("generated_program_id")
            if generated_program_id:
                if generated_program_id not in archive_dict:
                    raise ValueError(f"Program {generated_program_id} not found in archive")
                generated_program = archive_dict[generated_program_id]

        return cls(
            rollout_id=data["rollout_id"],
            experiment_id=data.get("experiment_id"),
            iteration=data["iteration"],
            island_id=data.get("island_id", 0),
            rollout_attempt=data.get("rollout_attempt", 1),
            rollout_name=data.get("rollout_name", "unknown"),
            status=data.get("status", "success"),
            failed_module=data.get("failed_module"),
            error_message=data.get("error_message"),
            selection=selection,
            generated_program=generated_program,
            strategy_state=data.get("strategy_state", {}),
            llm_request_logs=list(data.get("llm_request_logs", [])),
            stats=data.get("stats", {}),
            created_at=data["created_at"],
            completed_at=data.get("completed_at"),
        )

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        score = self.generated_program.combined_score if self.generated_program else None
        status = "complete" if self.completed_at else "in_progress"
        has_program = self.generated_program is not None
        return (
            f"RolloutResult(iter={self.iteration}, status={status}, "
            f"has_program={has_program}, score={score})"
        )


class Experiment(BaseModel):
    """
    Global experiment state.

    Represents an entire evolutionary experiment, containing:
    - Current populations for all islands
    - Archive of all programs ever created (for analysis)
    - History of all rollouts
    - Configuration

    Fields:
        id: Unique experiment identifier
        name: Human-readable experiment name

        island_populations: Current populations for all islands
        archive: All programs ever created (cold storage)

        current_iteration: Current iteration number
        rollout_history: History of all completed rollouts

        config: Experiment configuration (stored as dict)

        created_at: When experiment was created
        updated_at: Last update time
    """

    # Identity
    id: str
    name: str

    # Population management (unified structure)
    island_populations: Dict[int, Dict[str, List[Program]]] = Field(
        default_factory=lambda: {0: {}},
        description=(
            "Unified population storage: {island_id: {bucket_id: [programs]}}. "
            "Single-island mode uses island 0. "
            "Supports flexible structures: flat (TopK), grid (MAP-Elites), clusters, etc."
        )
    )
    num_islands: int = Field(
        default=1,
        ge=1,
        description="Number of islands (1 for single-island, >1 for multi-island evolution)"
    )

    archive: Dict[str, Program] = Field(
        default_factory=dict, description="All programs ever created (ID -> Program mapping)"
    )

    # Island visibility layer (defines which programs each island can see)
    island_index: Dict[int, Set[str]] = Field(
        default_factory=dict,
        description="Maps island_id to set of visible program IDs (visibility/access control layer)"
    )

    # Best program tracking
    best_program_id: Optional[str] = Field(
        default=None,
        description="ID of the best program in archive (highest combined_score)"
    )
    best_program_score: Optional[float] = Field(
        default=None,
        description="Score of the best program (for quick access without loading program)"
    )

    # State
    current_iteration: int = Field(default=0, ge=0)
    rollout_history: List[RolloutResult] = Field(default_factory=list)

    # Configuration (stored as dict for flexibility)
    config: Dict[str, Any] = Field(default_factory=dict)

    # Resume tracking (set when experiment was resumed/forked from another)
    resume_from: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Origin experiment if this was resumed/forked. "
            "Contains {experiment_id: str, iteration: int}"
        )
    )

    # Strategy state (persists across rollouts and checkpoints)
    strategy_state: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Persistent state for stateful strategies. "
            "Stores StateStore data (global_state, island_state) for resume support. "
            "Automatically synced from StateStore at checkpoint time."
        )
    )

    # Strategy checkpoint (custom strategy-level state for dump_state/load_state)
    strategy_checkpoint: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Custom strategy checkpoint state from Strategy.dump_state(). "
            "Restored via Strategy.load_state() on resume."
        )
    )

    # Strategy router runtime decision (persisted for stable resume behavior)
    strategy_router_decision: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Persisted strategy router decision for this experiment run. "
            "Contains direction/strategy/source so resume can reuse the same route "
            "without re-calling the LLM."
        )
    )

    # Timestamps
    created_at: float = Field(default_factory=get_timestamp)
    updated_at: float = Field(default_factory=get_timestamp)

    model_config = ConfigDict(validate_assignment=True)

    # ==== Properties and Helper Methods ====

    def get_island_population(self, island_id: int) -> Dict[str, List[Program]]:
        """
        Get population for a specific island.

        Args:
            island_id: Island identifier (0-based)

        Returns:
            Island's population: {bucket_id: [programs]}

        Raises:
            ValueError: If island_id doesn't exist
        """
        if island_id not in self.island_populations:
            raise ValueError(
                f"Island {island_id} does not exist. "
                f"Available islands: {list(self.island_populations.keys())}"
            )
        return self.island_populations[island_id]

    def set_island_population(
        self,
        island_id: int,
        population: Dict[str, List[Program]]
    ) -> None:
        """
        Set population for a specific island.

        Args:
            island_id: Island identifier (0-based)
            population: New population structure

        Raises:
            ValueError: If island_id exceeds num_islands
        """
        if island_id >= self.num_islands:
            raise ValueError(
                f"Island {island_id} exceeds num_islands={self.num_islands}"
            )
        self.island_populations[island_id] = population
        self.updated_at = get_timestamp()

    def sync_island_index(self, island_id: Optional[int] = None) -> None:
        """
        Sync island_index from island_populations.

        Used for backward compatibility when loading old checkpoints
        that don't have island_index, or when island_index gets out of sync.

        Args:
            island_id: If specified, sync only this island. If None, sync all islands.

        Example:
            >>> # Sync all islands
            >>> experiment.sync_island_index()
            >>>
            >>> # Sync specific island
            >>> experiment.sync_island_index(island_id=0)
        """
        if island_id is not None:
            # Sync specific island
            if island_id in self.island_populations:
                visible_ids = set()
                for bucket_programs in self.island_populations[island_id].values():
                    visible_ids.update(p.id for p in bucket_programs)
                self.island_index[island_id] = visible_ids
        else:
            # Sync all islands
            for isl_id, island_pop in self.island_populations.items():
                visible_ids = set()
                for bucket_programs in island_pop.values():
                    visible_ids.update(p.id for p in bucket_programs)
                self.island_index[isl_id] = visible_ids

    def get_all_programs(self) -> List[Program]:
        """
        Get all programs across all islands and buckets (flattened).

        Returns:
            Flat list of all programs
        """
        all_programs = []
        for island_pop in self.island_populations.values():
            for bucket_progs in island_pop.values():
                all_programs.extend(bucket_progs)
        return all_programs

    def get_program_by_id(self, program_id: str) -> Optional[Program]:
        """
        Find a program by ID across all islands and buckets.

        Args:
            program_id: Program identifier

        Returns:
            Program if found, None otherwise
        """
        for island_pop in self.island_populations.values():
            for bucket_progs in island_pop.values():
                for program in bucket_progs:
                    if program.id == program_id:
                        return program
        return None

    def get_best_program(self) -> Optional[Program]:
        """
        Get the best program from current population (across all islands and buckets).

        Returns:
            Program with highest combined_score, or None if population is empty
        """
        all_programs = self.get_all_programs()
        if not all_programs:
            return None

        return max(all_programs, key=lambda p: p.combined_score or 0.0)

    def get_best_overall(self) -> Optional[Program]:
        """
        Get the best program from entire archive.

        Returns:
            Program with highest combined_score from archive, or None if empty
        """
        if not self.archive:
            return None

        return max(self.archive, key=lambda p: p.combined_score or 0.0)

    @property
    def best_program(self) -> Optional[Program]:
        """
        Get the best program from the tracked best_program_id.

        This is a property that looks up the program from the archive using
        the tracked best_program_id. Returns None if best_program_id is not set
        or the program is not found in archive.

        Returns:
            Best Program object, or None if not found
        """
        if self.best_program_id is None:
            return None
        return self.archive.get(self.best_program_id)

    def update_best_program(self, program: Program) -> None:
        """
        Update best program tracking if the given program has a higher score.

        Args:
            program: Program to check against current best

        Updates:
            best_program_id: Set to program.id if program has higher score
            best_program_score: Set to program.combined_score if updated
        """
        if program.combined_score is None:
            return

        if (
            self.best_program_score is None
            or program.combined_score > self.best_program_score
        ):
            self.best_program_id = program.id
            self.best_program_score = program.combined_score

    # ========== Island Management (Stage 1) ==========

    def initialize_islands(self, num_islands: int) -> None:
        """
        Initialize island populations for multi-island evolution.

        Args:
            num_islands: Number of islands to create

        Note:
            - Distributes existing programs across islands (round-robin)
            - Updates self.num_islands
            - If num_islands=1, keeps all programs in island 0
        """
        self.num_islands = num_islands

        if num_islands == 1:
            # Single island mode - ensure island 0 exists
            if 0 not in self.island_populations:
                self.island_populations = {0: {}}
            return

        # Get all existing programs
        all_programs = self.get_all_programs()

        # Create empty islands
        self.island_populations = {}
        for island_id in range(num_islands):
            self.island_populations[island_id] = {"population": []}

        # Distribute programs across islands (round-robin)
        for i, program in enumerate(all_programs):
            island_id = i % num_islands
            self.island_populations[island_id]["population"].append(program)


    def migrate_between_islands(
        self, source_id: int, target_id: int, num_programs: int = 1
    ) -> None:
        """
        Migrate best programs from source island to target island.

        Args:
            source_id: Source island ID
            target_id: Target island ID
            num_programs: Number of best programs to migrate

        Note:
            - Copies (not moves) programs to preserve source island
            - Works with "population" bucket by default
        """
        source_pop = self.island_populations.get(source_id, {}).get("population", [])
        if not source_pop:
            return

        # Get best programs from source
        sorted_programs = sorted(
            source_pop, key=lambda p: p.combined_score or 0.0, reverse=True
        )
        migrants = sorted_programs[:num_programs]

        # Add to target island's "population" bucket (copy, not move)
        if target_id not in self.island_populations:
            self.island_populations[target_id] = {"population": []}

        if "population" not in self.island_populations[target_id]:
            self.island_populations[target_id]["population"] = []

        self.island_populations[target_id]["population"].extend(migrants)

        self.updated_at = get_timestamp()

    # ========== Compact Serialization (Stage 1 Storage Redesign) ==========

    def to_compact_dict(self) -> Dict[str, Any]:
        """
        Convert Experiment to compact representation for efficient checkpointing.

        Compact format stores:
        - Full Program objects only in archive (single source of truth)
        - Only Program IDs in populations and rollout history
        - This reduces checkpoint file size by 10-100x

        Returns:
            Dict with compact representation (ready for JSON serialization)

        Example:
            >>> exp = Experiment(...)
            >>> compact = exp.to_compact_dict()
            >>> # compact["island_population_ids"] = {"0": {"population": ["prog_1", ...]}}
            >>> # compact["archive_ids"] = ["prog_1", "prog_2", ...]
        """
        # Convert populations to ID lists: {island_id: {bucket_id: [program_ids]}}
        island_population_ids = {
            str(island_id): {
                bucket_id: [p.id for p in programs]
                for bucket_id, programs in island_pop.items()
            }
            for island_id, island_pop in self.island_populations.items()
        }

        # Convert rollout history to just IDs (rollouts stored in separate files)
        rollout_ids = [r.rollout_id for r in self.rollout_history]

        # Convert island_index to serializable format: {island_id: [program_ids]}
        island_index_serialized = {
            str(island_id): list(program_ids)
            for island_id, program_ids in self.island_index.items()
        }

        compact = {
            "id": self.id,
            "name": self.name,
            # Store IDs only (reconstruct from archive on load)
            "island_population_ids": island_population_ids,
            # Island visibility layer
            "island_index": island_index_serialized,
            # Archive: store only program IDs for true compact format
            "archive_ids": list(self.archive.keys()),
            # Rollouts: store only IDs (results stored in separate files)
            "rollout_ids": rollout_ids,
            # State
            "current_iteration": self.current_iteration,
            # Best program tracking
            "best_program_id": self.best_program_id,
            "best_program_score": self.best_program_score,
            # Config and timestamps
            "config": self.config,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

        # Include resume_from if present (for forked experiments)
        if self.resume_from:
            compact["resume_from"] = self.resume_from

        # Include strategy state if present
        if self.strategy_state:
            compact["strategy_state"] = self.strategy_state

        # Include strategy checkpoint if present
        if self.strategy_checkpoint:
            compact["strategy_checkpoint"] = self.strategy_checkpoint

        # Include strategy router decision if present
        if self.strategy_router_decision:
            compact["strategy_router_decision"] = self.strategy_router_decision

        return compact

    @classmethod
    def from_compact_dict(
        cls,
        data: Dict[str, Any],
        archive_dict: Optional[Dict[str, Program]] = None,
        rollout_history: Optional[List[RolloutResult]] = None
    ) -> "Experiment":
        """
        Reconstruct Experiment from compact representation.

        Rebuilds full populations by looking up Program IDs in archive.

        Args:
            data: Compact representation from to_compact_dict()
            archive_dict: Optional dict of {program_id: Program} for external loading.
                         If None, attempts to load from data["archive"] (old format).
            rollout_history: Optional list of RolloutResult for external loading.
                           If None, attempts to load from data["rollout_history"] (old format).

        Returns:
            Fully reconstructed Experiment with populated collections

        Raises:
            ValueError: If a Program ID is not found in archive
        """
        # Support both old format (archive with full programs) and new format (archive_ids)
        if archive_dict is None:
            # Old format: archive contains full Program objects
            archive_data = data.get("archive", [])
            if archive_data:
                programs = [Program.model_validate(p) for p in archive_data]
                archive_dict = {p.id: p for p in programs}
            else:
                # New format: archive_ids contains only program IDs
                # This requires external loading - archive_dict must be provided
                raise ValueError(
                    "Compact format uses archive_ids. "
                    "Please provide archive_dict parameter with loaded programs."
                )
        else:
            # Use provided archive_dict directly (already in Dict format)
            archive = archive_dict

        # Reconstruct island_populations from IDs: {island_id: {bucket_id: [program_ids]}}
        island_population_ids = data.get("island_population_ids", {})
        island_populations = {}
        for island_id_str, bucket_dict in island_population_ids.items():
            island_id = int(island_id_str)  # JSON keys are strings
            island_populations[island_id] = {}
            for bucket_id, program_ids in bucket_dict.items():
                island_populations[island_id][bucket_id] = []
                for pid in program_ids:
                    if pid not in archive_dict:
                        raise ValueError(
                            f"Program {pid} not found in archive during island reconstruction"
                        )
                    island_populations[island_id][bucket_id].append(archive_dict[pid])

        # For backward compatibility, support old current_population_ids format
        current_population_ids = data.get("current_population_ids")
        if current_population_ids:
            # Old format: current_population_ids was stored separately
            # Initialize island_populations[0] if not present
            if 0 not in island_populations:
                island_populations[0] = {}
                for bucket_id, program_ids in current_population_ids.items():
                    island_populations[0][bucket_id] = []
                    for pid in program_ids:
                        if pid not in archive_dict:
                            raise ValueError(f"Program {pid} not found in archive during reconstruction")
                        island_populations[0][bucket_id].append(archive_dict[pid])

        # Reconstruct island_index from IDs: {island_id: set(program_ids)}
        island_index_data = data.get("island_index", {})
        island_index = {}
        if island_index_data:
            # New format: island_index is present in checkpoint
            for island_id_str, program_ids in island_index_data.items():
                island_index[int(island_id_str)] = set(program_ids)
        else:
            # Old format: island_index not present, build from island_populations
            for island_id, island_pop in island_populations.items():
                visible_ids = set()
                for bucket_programs in island_pop.values():
                    visible_ids.update(p.id for p in bucket_programs)
                island_index[island_id] = visible_ids

        # Support both old format (rollout_history with full data) and new format (rollout_ids)
        if rollout_history is None:
            # Old format: rollout_history contains compact RolloutResult data
            rollout_history_data = data.get("rollout_history", [])
            if rollout_history_data:
                # Reconstruct from compact format in data
                rollout_history = [
                    RolloutResult.from_compact_dict(result_data, archive_dict)
                    for result_data in rollout_history_data
                ]
            else:
                # New format: rollout_ids contains only rollout IDs
                # This requires external loading - rollout_history must be provided
                rollout_ids = data.get("rollout_ids", [])
                if rollout_ids:
                    raise ValueError(
                        "Compact format uses rollout_ids. "
                        "Please provide rollout_history parameter with loaded results."
                    )
                else:
                    rollout_history = []
        # If rollout_history is provided, use it directly

        return cls(
            id=data["id"],
            name=data["name"],
            archive=archive,
            island_index=island_index,
            island_populations=island_populations,
            current_iteration=data["current_iteration"],
            rollout_history=rollout_history,
            config=data["config"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            best_program_id=data.get("best_program_id"),
            best_program_score=data.get("best_program_score"),
            resume_from=data.get("resume_from"),
            strategy_state=data.get("strategy_state", {}),
            strategy_checkpoint=data.get("strategy_checkpoint", {}),
            strategy_router_decision=data.get("strategy_router_decision", {}),
        )

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        total_programs = len(self.get_all_programs())
        return (
            f"Experiment(id={self.id}, name={self.name}, "
            f"iter={self.current_iteration}, programs={total_programs}, "
            f"archive={len(self.archive)})"
        )

class Rollout(BaseModel):
    """
    Encapsulates a reusable evolutionary pipeline.

    A Rollout is a sequence of modules that defines a single iteration of evolution.
    It provides a clean abstraction for users to define and share pipeline strategies.

    **Design Philosophy:**
    - Rollout is a lightweight data container (no execution logic)
    - RolloutEngine handles execution and dependency injection
    - Users define pipelines declaratively, framework handles plumbing

    **Typical Pipeline Structure:**
    1. SelectModule - Choose parents from population (optional)
    2. GenerateModule - Create new programs (required)
    3. EvaluateModule - Execute and score programs (required)
    4. JudgeModule - Provide LLM feedback (optional)

    Fields:
        modules: List of modules to execute in sequence
        name: Name for this rollout strategy

    Example:
        >>> # Define reusable pipeline strategies
        >>> standard_flow = Rollout(
        ...     modules=[
        ...         TopKSelect(k=2),
        ...         MutationGenerate(num_children_per_parent=2),
        ...         EvaluateModule(evaluate_fn=my_evaluator),
        ...     ],
        ...     name="standard"
        ... )
        >>>
        >>> experimental_flow = Rollout(
        ...     modules=[
        ...         RandomSelect(k=3),
        ...         MutationGenerate(temperature=0.9),
        ...         EvaluateModule(evaluate_fn=my_evaluator),
        ...         LLMJudge(criteria=["elegance", "efficiency"]),
        ...     ],
        ...     name="experimental"
        ... )
    """

    modules: List["Module"] = Field(min_length=1, description="Modules to execute in sequence")
    name: str = Field(default="rollout", description="Rollout name")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("modules")
    @classmethod
    def validate_pipeline(cls, modules: List["Module"]) -> List["Module"]:
        """
        Validate that pipeline has required module types.

        Checks:
        - Pipeline has at least one GenerateModule
        - Pipeline has at least one EvaluateModule

        Raises:
            ValueError: If pipeline is invalid
        """
        # Import here to avoid circular dependency
        from famou.modules.generate import GenerateModule
        from famou.modules.evaluate import EvaluateModule

        has_generate = any(isinstance(m, GenerateModule) for m in modules)
        has_evaluate = any(isinstance(m, EvaluateModule) for m in modules)

        if not has_generate:
            module_names = [m.name for m in modules]
            raise ValueError(
                "Rollout pipeline must include at least one GenerateModule. "
                f"Current modules: {module_names}"
            )

        if not has_evaluate:
            module_names = [m.name for m in modules]
            raise ValueError(
                "Rollout pipeline must include at least one EvaluateModule. "
                f"Current modules: {module_names}"
            )

        return modules

    def __repr__(self) -> str:
        """Concise representation."""
        module_names = [m.name for m in self.modules]
        return f"Rollout(name={self.name}, modules={module_names})"

class WorkBatch(BaseModel):
    """
    A batch of rollouts for concurrent execution.

    **Note**: This class is defined for potential future use (batch execution support).
    The current Strategy Protocol (v2.1) returns single Rollouts via forward().

    WorkBatch unifies single-rollout and multi-rollout patterns, enabling
    strategies to decide:
    - How many rollouts to execute in parallel
    - Whether to wait for batch completion before the next forward() call
    - Why these rollouts were chosen (metadata/decision rationale)

    Design Philosophy:
    ------------------
    - Single rollout is a batch of size 1 (unified abstraction)
    - Concurrency hint is advisory (Evolver may cap at max_workers)
    - Barrier flag enables batch synchronization (wait for completion)
    - Meta dict stores decision rationale for debugging/analysis

    Fields:
        rollouts: List of rollouts to execute concurrently
        concurrency_hint: User-suggested parallelism level (None = use all)
        barrier: If True, Evolver waits for batch completion before next forward()
        meta: Decision metadata (strategy type, phase, sampling weights, etc.)

    Examples:
        >>> # Single rollout (most common case)
        >>> batch = WorkBatch.single(standard_rollout, phase="exploit")

        >>> # Parallel exploration batch
        >>> batch = WorkBatch(
        ...     rollouts=[explore_rollout] * 8,
        ...     concurrency_hint=8,
        ...     barrier=True,
        ...     meta={"phase": "explore", "batch_size": 8}
        ... )

        >>> # Adaptive mixture (decision logged in meta)
        >>> rollout = random.choice([exploit_rollout, explore_rollout])
        >>> batch = WorkBatch.single(
        ...     rollout,
        ...     strategy="mixture",
        ...     chosen=rollout.name,
        ...     weights={"exploit": 0.8, "explore": 0.2}
        ... )
    """

    rollouts: List[Rollout] = Field(min_length=1, description="List of rollouts to execute")
    concurrency_hint: Optional[int] = Field(
        default=None,
        ge=1,
        description="User-suggested parallelism level (None = use all)"
    )
    barrier: bool = Field(
        default=True,
        description="Wait for batch completion before next forward()"
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Decision metadata (strategy type, phase, etc.)"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def single(cls, rollout: Rollout, **meta) -> "WorkBatch":
        """
        Create a single-rollout batch (convenience method).

        Args:
            rollout: The rollout to execute
            **meta: Decision metadata (phase, strategy_type, etc.)

        Returns:
            WorkBatch with single rollout and concurrency_hint=1

        Example:
            >>> batch = WorkBatch.single(
            ...     standard_rollout,
            ...     phase="exploit",
            ...     strategy="standard"
            ... )
        """
        return cls(
            rollouts=[rollout],
            concurrency_hint=1,
            barrier=True,
            meta=meta
        )

    def __len__(self) -> int:
        """Return number of rollouts in batch."""
        return len(self.rollouts)

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        rollout_names = [r.name for r in self.rollouts]
        return (
            f"WorkBatch(size={len(self)}, "
            f"hint={self.concurrency_hint}, "
            f"barrier={self.barrier}, "
            f"rollouts={rollout_names})"
        )


# Rebuild Context model to resolve forward references (ExperimentConfig)
# Import at runtime to avoid circular dependency
from famou.config.settings import ExperimentConfig  # noqa: E402
Context.model_rebuild()

